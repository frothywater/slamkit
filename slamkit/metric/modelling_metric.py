from collections import defaultdict
import logging
logger = logging.getLogger(__name__)

import torchaudio
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence


class ModellingMetricDataset(Dataset):
    def __init__(self, path, sep="_", subfolder=True):
        super().__init__()
        self.data = []
        
        # === custom .pt token loading ===
        if path.endswith('.pt'):
            self.data_type = "tensor"
            # match "real/fake" pairs
            self.tensor_dict = torch.load(path)

            # "real/fake" pairs
            real_keys = [k for k in self.tensor_dict.keys() if k.endswith("_real")]
            for real_key in real_keys:
                fake_key = real_key[:-len("real")] + "fake"
                if fake_key in self.tensor_dict:
                    self.data += [real_key, fake_key]
        else:
            self.data_type = "audio"
            if subfolder:
                for f in Path(path).iterdir():
                    if f.is_dir():
                        self.data += sorted(list(f.glob("*.wav")), key=lambda x: int(x.name.split(sep)[0]))
            else:
                self.data += sorted(list(Path(path).glob("*.wav")), key=lambda x: int(x.name.split(sep)[0]))

    def __len__(self):
        return len(self.data) // 2

    def __getitem__(self, idx):
        # === custom .pt token loading ===
        pos_file, neg_file = self.data[2 * idx], self.data[2 * idx + 1]
        if self.data_type == "tensor":
            pos, neg = self.tensor_dict[pos_file], self.tensor_dict[neg_file]
            if pos.dim() == 2:
                pos = pos.squeeze(0)
            if neg.dim() == 2:
                neg = neg.squeeze(0)
            return pos, neg, pos.shape[-1], neg.shape[-1]
        else:    
            pos = torchaudio.load(str(pos_file))[0][0]
            neg = torchaudio.load(str(neg_file))[0][0]
            return pos, neg, pos.shape[-1], neg.shape[-1]


class TSPMetricDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.data = []
        
        # === custom .pt token loading ===
        self.data_dict = torch.load(path)
        # A dictionary to hold lists of tokens for each utterance ID
        # Format: {utt_id: {'AE': [tensor1, tensor2], 'JE': [tensor3]}}
        utterance_groups = defaultdict(lambda: defaultdict(list))

        for key, tokens in self.data_dict.items():
            try:
                # Split "series-speaker-id" into 3 parts
                series, speaker, utt_id = key.split('-', 2)
            except ValueError:
                logger.warning(f"Skipping malformed key: {key}")
                continue

            # Ensure series is one of the expected types
            if series not in ('AE', 'JE'):
                logger.warning(f"Skipping key with unknown series '{series}': {key}")
                continue

            # Clean and store the token tensor
            cleaned_tokens = tokens if tokens.dim() == 1 else tokens.squeeze(0)
            utterance_groups[utt_id][series].append(cleaned_tokens)

        # Finalize into a list of tuples, ensuring each group is valid
        final_groups = []
        for utt_id, groups in utterance_groups.items():
            # A valid group must have at least one of each type
            if 'AE' in groups and 'JE' in groups:
                final_groups.append((utt_id, groups['AE'], groups['JE']))
            else:
                logger.warning(f"Skipping utterance ID '{utt_id}' due to missing AE or JE utterances.")

        self.groups = final_groups

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        utt_id, typical_tokens, atypical_tokens = self.groups[idx]
        return typical_tokens, atypical_tokens, utt_id


def tsp_collate(batch):
    assert len(batch) == 1, "Batch size must be 1 for TSP metric"
    typical_tokens, atypical_tokens, utt_id = batch[0]
    typical_lengths = torch.tensor([t.shape[0] for t in typical_tokens])
    atypical_lengths = torch.tensor([t.shape[0] for t in atypical_tokens])
    typical_tokens = pad_sequence(typical_tokens, batch_first=True, padding_value=0)
    atypical_tokens = pad_sequence(atypical_tokens, batch_first=True, padding_value=0)
    return typical_tokens, atypical_tokens, typical_lengths, atypical_lengths, utt_id

def tsp_metric(model, dataset, used_token_modality, mean_nll: bool=True,
               batch_size: int = 1, num_workers=8, pin_memory=True):
    from torchmetrics.functional import auroc
    
    dl = DataLoader(dataset, batch_size=batch_size, collate_fn=tsp_collate ,num_workers=num_workers, pin_memory=pin_memory)

    auc_scores_for_batch = []
    # Iterate over each utterance group in the batch
    for group_item in tqdm(dl):
        typical_tokens, atypical_tokens, typical_lengths, atypical_lengths, utt_id = group_item
        # 1. Compute scores (negative loss) for all utterances in the group
        typical_scores = model.log_likelihood(typical_tokens, typical_lengths, used_token_modality=used_token_modality, mean_nll=mean_nll)
        atypical_scores = model.log_likelihood(atypical_tokens, atypical_lengths, used_token_modality=used_token_modality, mean_nll=mean_nll)

        # 2. Prepare predictions and targets for AUC calculation
        # Predictions are all the scores from the model
        preds = torch.cat([typical_scores, atypical_scores])

        # Targets are the ground-truth labels (1 for typical, 0 for atypical)
        typical_labels = torch.ones(typical_scores.size(0), device=model.device)
        atypical_labels = torch.zeros(atypical_scores.size(0), device=model.device)
        targets = torch.cat([typical_labels, atypical_labels]).long()

        # 3. Compute AUC for this specific utterance group
        # This measures how well the typical samples are ranked above the atypical ones
        group_auc = auroc(preds, targets, task="binary")
        auc_scores_for_batch.append(group_auc)

    # 4. Compute the average AUC for the entire batch
    if not auc_scores_for_batch:
        # Return 0.5 (random chance) if the batch was empty or invalid
        return torch.tensor(0.5, device=model.device)
        
    mean_auc = torch.stack(auc_scores_for_batch).mean()
    return mean_auc


class SalmonDataset(Dataset):
    def __init__(self, path, part):
        self.data = []
        self.salmon_path = Path(path)
        dir_path = self.salmon_path / part
        paths = list(dir_path.glob("*.wav"))

        max_sample_index = -1
        for path in paths:
            stem = str(path.stem)
            parts = stem.split("_")
            sample_index = int(parts[1])
            if sample_index > max_sample_index:
                max_sample_index = sample_index

        self.data = [[] for _ in range(max_sample_index + 1)]

        for path in paths:
            stem = str(path.stem)
            parts = stem.split("_")
            sample_index = int(parts[1])
            self.data[sample_index].append(str(path))

        for sample_list in self.data:
            sample_list.sort()

        self.data = [lst for lst in self.data if lst]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_files = self.data[idx]
        pos = torchaudio.load(sample_files[0])[0][0]
        neg = torchaudio.load(sample_files[1])[0][0]
        return pos, neg, pos.shape[-1], neg.shape[-1]


def pad_collate(batch):
    pos, neg, l_pos, l_neg = zip(*batch)
    # pad with silence
    pos = pad_sequence(pos, batch_first=True, padding_value=0)
    neg = pad_sequence(neg, batch_first=True, padding_value=0)
    return pos, neg, torch.tensor(l_pos), torch.tensor(l_neg)


def modelling_metric(model, dataset, used_token_modality, mean_nll: bool=True,
                     batch_size: int = 1, num_workers=8, pin_memory=True):
    dl = DataLoader(dataset, batch_size=batch_size, collate_fn=pad_collate ,num_workers=num_workers, pin_memory=pin_memory)
    res_list = []

    for sample_files in tqdm(dl):
        pos, neg, l_pos, l_neg = sample_files
        pos, neg = pos.to(model.device), neg.to(model.device)
        l_pos, l_neg = l_pos.to(model.device), l_neg.to(model.device)
        with torch.no_grad():
            pos_likelihood = model.log_likelihood(pos, l_pos, used_token_modality=used_token_modality, mean_nll=mean_nll)
            neg_likelihood = model.log_likelihood(neg, l_neg, used_token_modality=used_token_modality, mean_nll=mean_nll)
        res = torch.zeros_like(pos_likelihood)
        res[pos_likelihood > neg_likelihood] = 1
        res[pos_likelihood == neg_likelihood] = 0.5
        res[pos_likelihood < neg_likelihood] = 0

        res_list.append(res)

    res_list = torch.cat(res_list)
    return res_list.float().mean().cpu().item()


def salmon(model, salmon_path, used_token_modality, mean_nll, parts, batch_size, num_workers=8, pin_memory=True):
    if parts[0] == "all":
        parts = ['bg_alignment/', 'bg_all_consistency/', 'bg_domain_consistency/', 'gender_consistency/',
                 'rir_consistency/', 'sentiment_alignment/', 'sentiment_consistency/', 'speaker_consistency/']

    out = dict()
    for part in parts:
        dataset = SalmonDataset(salmon_path, part)
        assert len(dataset) > 0, f"no samples found for {part}"
        cur_res = modelling_metric(model, dataset, used_token_modality, mean_nll, batch_size, num_workers, pin_memory)
        logging.info(f"SALMon - {part}: {cur_res:.4f}")
        out[part] = cur_res

    return out


def swuggy(model, data_path, used_token_modality, mean_nll=True,
           batch_size=1, num_workers=8, pin_memory=True, subfolder=False):
    dataset = ModellingMetricDataset(data_path, sep='_', subfolder=subfolder)
    assert len(dataset) > 0, f"no samples found for {data_path}"
    res = modelling_metric(model, dataset, used_token_modality, mean_nll, batch_size, num_workers, pin_memory)
    logging.info(f"sWUGGY: {res:.4f}")
    return {'sWUGGY': res}


def sblimp(model, data_path, used_token_modality,  mean_nll=True,
           batch_size=1, num_workers=8, pin_memory=True, subfolder=False):
    dataset = ModellingMetricDataset(data_path, sep='+', subfolder=subfolder)
    assert len(dataset) > 0, f"no samples found for {data_path}"
    res = modelling_metric(model, dataset, used_token_modality, mean_nll, batch_size, num_workers, pin_memory)
    logging.info(f"sBLIMP: {res:.4f}")
    return {'sBLIMP': res}

def storycloze(model, data_path, used_token_modality, mean_nll=True,
               batch_size=1, num_workers=8, pin_memory=True, subfolder=False):
    dataset = ModellingMetricDataset(data_path, sep='_', subfolder=subfolder)
    assert len(dataset) > 0, f"no samples found for {data_path}"
    res = modelling_metric(model, dataset, used_token_modality, mean_nll, batch_size, num_workers, pin_memory)
    logging.info(f"StoryCloze: {res:.4f}")
    return {'StoryCloze': res}

def tsp(model, data_path, used_token_modality,  mean_nll=True,
        batch_size=1, num_workers=8, pin_memory=True):
    dataset = TSPMetricDataset(data_path)
    assert len(dataset) > 0, f"no samples found for {data_path}"
    res = tsp_metric(model, dataset, used_token_modality, mean_nll, batch_size, num_workers, pin_memory)
    logging.info(f"TSP: {res:.4f}")
    return {'TSP': res}
