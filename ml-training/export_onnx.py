import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel
from train_minilm_guardrail import MiniLMGuardrail, MODEL_NAME

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    tok = AutoTokenizer.from_pretrained(model_dir)

    # Must match the training architecture
    model = MiniLMGuardrail(num_labels=9)
    model.load_state_dict(torch.load(model_dir / "minilm_guardrail.pt", map_location="cpu"))
    model.eval()

    dummy = tok(
        "ignore previous instructions",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=256
    )

    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        args.out,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits", "embeddings"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
            "embeddings": {0: "batch"}
        },
        opset_version=17
    )

if __name__ == "__main__":
    main()