import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="Convert .pt files to JSONL format")
    parser.add_argument("input_files", nargs="+", help="Input .pt files to convert")
    parser.add_argument("--output", required=True, help="Output directory for JSONL files")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for pt_file in args.input_files:
        pt_path = Path(pt_file)
        output_path = output_dir / f"{pt_path.stem}.jsonl"

        # Load the .pt file
        data = torch.load(pt_path, map_location="cpu")

        with open(output_path, "w") as f:
            for audio_id, tensor in data.items():
                # Ensure tensor is 1D or 2D with squeezable first dim
                if tensor.dim() == 2 and tensor.shape[0] == 1:
                    tensor = tensor.squeeze(0)
                elif tensor.dim() > 2 or (tensor.dim() == 2 and tensor.shape[0] != 1):
                    raise ValueError(f"Invalid tensor shape for {audio_id}: {tensor.shape}")

                # Convert to 1D integer array
                units = tensor.int().tolist()

                # Create JSONL entry
                entry = {"units": units, "filename": audio_id}
                f.write(json.dumps(entry) + "\n")

        print(f"Converted {pt_file} -> {output_path}")


if __name__ == "__main__":
    main()
