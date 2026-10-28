#!/usr/bin/env python3
"""Export the embedded Stage-3 manifest/config from a new checkpoint."""

import argparse
import json
import os

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Stage-3 model_<iteration>.pt")
    parser.add_argument("--output-dir", help="Defaults to the checkpoint directory")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.checkpoint))
    os.makedirs(output_dir, exist_ok=True)
    exported = []
    for checkpoint_key, filename in (
        ("stage3_manifest", "stage3_manifest.json"),
        ("stage3_config", "stage3_config.json"),
        ("stage3_curriculum_state", "stage3_curriculum_state.json"),
    ):
        if checkpoint_key not in checkpoint:
            raise KeyError(
                f"{args.checkpoint} has no {checkpoint_key}; it is not a complete Stage-3 checkpoint"
            )
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as output_file:
            json.dump(checkpoint[checkpoint_key], output_file, indent=2, sort_keys=True, default=str)
            output_file.write("\n")
        exported.append(path)
    print("Exported Stage-3 metadata:")
    for path in exported:
        print(f"  {path}")


if __name__ == "__main__":
    main()
