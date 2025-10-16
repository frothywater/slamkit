```bash
# Prepare tokens
python cli/prepare_tokens.py data_path=data/12hz/features/ls-dev-clean.jsonl out_path=data/12hz/tokens ;\
python cli/prepare_tokens.py data_path=data/12hz/features/ls-test-clean.jsonl out_path=data/12hz/tokens ;\
python cli/prepare_tokens.py data_path=data/12hz/features/train/ls-train.jsonl out_path=data/12hz/tokens/train ;\
python cli/prepare_tokens.py data_path=data/12hz/features/train/libri-light-medium.jsonl out_path=data/12hz/tokens/train

python cli/prepare_tokens.py data_path=data/25hz/features/ls-dev-clean.jsonl out_path=data/25hz/tokens ;\
python cli/prepare_tokens.py data_path=data/25hz/features/ls-test-clean.jsonl out_path=data/25hz/tokens ;\
python cli/prepare_tokens.py data_path=data/25hz/features/train/libri-light-medium.jsonl out_path=data/25hz/tokens/train
# python cli/prepare_tokens.py data_path=data/25hz/features/train/ls-train.jsonl out_path=data/25hz/tokens/train ;\

# Train
# +training_args.max_steps=17625
python cli/train.py data.train_path="data/12hz/tokens/train/*.jsonl" data.val_path=data/12hz/tokens/ls-dev-clean.jsonl model=slam training_args.per_device_train_batch_size=8 training_args.gradient_accumulation_steps=16 training_args.output_dir=outputs/12hz data.packing=true model.config_args.attn_implementation=flash_attention_2 tokeniser.feature_extractor.num_units=12800

python cli/train.py data.train_path="data/25hz/tokens/train/*.jsonl" data.val_path=data/25hz/tokens/ls-dev-clean.jsonl model=slam training_args.per_device_train_batch_size=8 training_args.gradient_accumulation_steps=16 training_args.output_dir=outputs/25hz data.packing=true model.config_args.attn_implementation=flash_attention_2 tokeniser.feature_extractor.num_units=12800

# Inference
python cli/eval.py tokeniser=unit_hubert_25 tokeniser.feature_extractor.num_units=12800 metric=generate batch_size=32 model.pretrained_model=outputs/12hz/checkpoint-4053 metric.data_path=/home/huangzj/ssl-resynth/exp/aug25/fsq12800_128d_vae6l_l12_l69_longwin/asr/tokens/test-clean_local_indices.pt metric.prompt_length=30 metric.num_files=32 +metric.generate_kwargs.repetition_penalty=1.0 metric.generate_kwargs.temperature=0.8 metric.generate_kwargs.top_k=128

python cli/eval.py tokeniser=unit_hubert_25 tokeniser.feature_extractor.num_units=12800 metric=generate batch_size=32 model.pretrained_model=outputs/25hz/checkpoint-3640 metric.data_path=/home/huangzj/ssl-resynth/exp/aug25/fsq12800_128d_vae6l_l12_l69_longwin_25hz/asr/tokens/test-clean_local_indices.pt metric.prompt_length=50 metric.num_files=32 +metric.generate_kwargs.repetition_penalty=1.0 metric.generate_kwargs.temperature=1.2 metric.generate_kwargs.top_k=12800 +metric.generate_kwargs.top_p=0.9


# Eval
python cli/eval.py tokeniser=unit_hubert_25 tokeniser.feature_extractor.num_units=12800 metric=swuggy_inter metric.data_path=/home/huangzj/ssl-resynth/exp/aug25/fsq12800_128d_vae6l_l12_l69_longwin/lm/tokens/swuggy_local_indices.pt batch_size=64 model.pretrained_model=outputs/12hz/checkpoint-4053
python cli/eval.py tokeniser=unit_hubert_25 tokeniser.feature_extractor.num_units=12800 metric=sblimp metric.data_path=/home/huangzj/ssl-resynth/exp/aug25/fsq12800_128d_vae6l_l12_l69_longwin/lm/tokens/sblimp_local_indices.pt batch_size=64 model.pretrained_model=outputs/12hz/checkpoint-4053
python cli/eval.py tokeniser=unit_hubert_25 tokeniser.feature_extractor.num_units=12800 metric=tsp metric.data_path=/home/smcintosh/ssl-resynth/metric_exp/12hz/tsp_local_indices.pt batch_size=1 model.pretrained_model=outputs/12hz/checkpoint-4053

python cli/eval.py tokeniser=unit_hubert_25 tokeniser.feature_extractor.num_units=12800 metric=swuggy_inter metric.data_path=/home/huangzj/ssl-resynth/exp/aug25/fsq12800_128d_vae6l_l12_l69_longwin_25hz/lm/tokens/swuggy_local_indices.pt batch_size=64 model.pretrained_model=outputs/25hz/checkpoint-3640 ;\
python cli/eval.py tokeniser=unit_hubert_25 tokeniser.feature_extractor.num_units=12800 metric=sblimp metric.data_path=/home/huangzj/ssl-resynth/exp/aug25/fsq12800_128d_vae6l_l12_l69_longwin_25hz/lm/tokens/sblimp_local_indices.pt batch_size=64 model.pretrained_model=outputs/25hz/checkpoint-3640 ;\
python cli/eval.py tokeniser=unit_hubert_25 tokeniser.feature_extractor.num_units=12800 metric=tsp metric.data_path=/home/smcintosh/ssl-resynth/metric_exp/25hz/tsp_local_indices.pt batch_size=1 model.pretrained_model=outputs/25hz/checkpoint-3640
```