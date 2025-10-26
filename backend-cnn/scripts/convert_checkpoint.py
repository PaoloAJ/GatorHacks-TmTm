#!/usr/bin/env python3
"""
Convert training checkpoint to final model format.
"""
import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).parent.parent))


def main():
    checkpoint_path = Path("models/checkpoint.pth")
    output_path = Path("models/finetuned_encoder.pth")

    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Convert to final model format
    final_model = {
        'projection_head': checkpoint['projection_head'],
        'embedding_dim': 512
    }

    # Save final model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(final_model, output_path)

    print(f"✅ Saved fine-tuned model to {output_path}")
    print(f"   Trained for {checkpoint['epoch']} epochs")
    print(f"   Final loss: {checkpoint['loss']:.4f}")
    print(f"\nNext step: Update encoder service to load this model")


if __name__ == "__main__":
    main()
