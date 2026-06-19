import os
import torch
from transformers import GPT2LMHeadModel, AutoTokenizer

def export_gpt2_to_onnx():
    # 1. Initialize Paths and Setup Environments
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, "gpt2_perplexity.onnx")
    
    print("[*] Downloading and initializing GPT-2 core architecture...")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Force the model to load in evaluation mode on the CPU for portable asset serialization
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    # 2. Construct Ground-Truth Mock Tracing Inputs
    # We require both token sequences and an attention mask to accurately trace conditional branching
    mock_text = "SafeLLMKit firewall initialization sequence."
    inputs = tokenizer(mock_text, return_tensors="pt")
    
    input_ids = inputs["input_ids"]          # Shape: [batch_size, sequence_length]
    attention_mask = inputs["attention_mask"]  # Shape: [batch_size, sequence_length]

    # 3. Formulate Input/Output Vector Names and Dynamic Sizing Constraints
    # The batch size and sequence length MUST be marked dynamic, or inference will crash 
    # on any prompt that doesn't exactly match the length of our mock_text string.
    input_names = ["input_ids", "attention_mask"]
    output_names = ["logits"]
    
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"}
    }

    print(f"[*] Tracing execution graph and compiling to ONNX Opset 15...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            args=(input_ids, attention_mask),
            f=onnx_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=15,  # Opset 15 offers native support for causal transformer operations
            do_constant_folding=True
        )
        
    print(f"[🚀] Compilation successful! ONNX binary deployed to: {onnx_path}")

if __name__ == "__main__":
    export_gpt2_to_onnx()