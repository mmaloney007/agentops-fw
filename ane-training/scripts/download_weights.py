#!/usr/bin/env python3
"""Download HuggingFace model weights and convert to CoreML for public path."""

import argparse
import os
import sys


def download_model(model_id: str, out_dir: str):
    """Download safetensors weights from HuggingFace."""
    from huggingface_hub import hf_hub_download, snapshot_download

    # Download tokenizer and model weights
    files = snapshot_download(
        repo_id=model_id,
        allow_patterns=["*.safetensors", "tokenizer.json", "config.json", "tokenizer_config.json"],
        local_dir=out_dir,
    )
    print(f"Downloaded {model_id} to {out_dir}")
    return out_dir


def convert_to_coreml(model_dir: str, out_path: str, model_name: str):
    """Convert safetensors to CoreML .mlmodelc (optional, for reference)."""
    try:
        import coremltools as ct
        import torch
        from transformers import AutoModelForCausalLM

        print(f"Loading {model_dir} for CoreML conversion...")
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16)

        # Trace for single-token decode
        dummy_input = torch.randint(0, 1000, (1, 1))
        traced = torch.jit.trace(model, dummy_input)

        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(name="input_ids", shape=(1, 1))],
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
        mlmodel.save(out_path)
        print(f"CoreML model saved to {out_path}")
    except Exception as e:
        print(f"CoreML conversion failed (not required for public path): {e}")
        print("The public path uses CPU forward/backward directly from safetensors.")


MODELS = {
    "stories110m": {
        "hf_id": "roneneldan/TinyStories-33M",  # or a stories110m checkpoint
        "description": "Stories110M (Llama2 architecture, 110M params)",
    },
    "qwen2.5-0.5b": {
        "hf_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "description": "Qwen2.5-0.5B-Instruct (GQA, 500M params)",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Download model weights for ANE GRPO experiments")
    parser.add_argument("--model", choices=list(MODELS.keys()), required=True,
                        help="Model to download")
    parser.add_argument("--out-dir", default="weights",
                        help="Output directory (default: weights/)")
    parser.add_argument("--convert-coreml", action="store_true",
                        help="Also convert to CoreML (optional)")
    args = parser.parse_args()

    model_info = MODELS[args.model]
    model_dir = os.path.join(args.out_dir, args.model)

    print(f"Downloading {model_info['description']}...")
    download_model(model_info["hf_id"], model_dir)

    if args.convert_coreml:
        coreml_path = os.path.join(model_dir, "model.mlpackage")
        convert_to_coreml(model_dir, coreml_path, args.model)

    print(f"\nReady. Use with:")
    print(f"  --weights {model_dir}/model.safetensors")
    print(f"  --tokenizer {model_dir}/tokenizer.json")


if __name__ == "__main__":
    main()
