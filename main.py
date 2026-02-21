#!/usr/bin/env python3
"""
This script loads the base Mistral, patches in the resized embedding layers,
applies the decoder_adapter LoRA via PEFT merge_and_unload(), converts to GGUF
via llama.cpp, quantizes to Q4_K_M, and registers with Ollama.

Usage:
    python tools/setup_clara.py
    python tools/setup_clara.py --output-dir /tmp/clara-build
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys


def find_tool(name, hint_dir=None):
    """Find a tool by name: check hint_dir, then PATH, then common locations."""
    if hint_dir:
        for sub in [name, os.path.join("build", "bin", name), os.path.join("bin", name)]:
            p = os.path.join(hint_dir, sub)
            if os.path.exists(p):
                return p
    found = shutil.which(name)
    if found:
        return found
    return None


def find_convert_script():
    """Find convert_hf_to_gguf.py from llama.cpp."""
    for candidate in [
        shutil.which("convert_hf_to_gguf.py"),
        os.path.expanduser("~/llama.cpp/convert_hf_to_gguf.py"),
        "/tmp/llama-cpp-src/convert_hf_to_gguf.py",
    ]:
        if candidate and os.path.exists(candidate):
            return candidate
    # Search nix store
    nix_matches = glob.glob("/nix/store/*/bin/convert_hf_to_gguf.py")
    if nix_matches:
        return nix_matches[0]
    return None


BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
CLARA_REPO = "apple/CLaRa-7B-Instruct"
CLARA_SUBDIR = "compression-16"
OLLAMA_MODEL_NAME = "clara-7b"

def run(cmd, **kwargs):
    print(f"  $ {' '.join(cmd)}")
    subprocess.check_call(cmd, **kwargs)

def main():
    parser = argparse.ArgumentParser(description="Merge CLaRa-7B decoder LoRA and register with Ollama")
    parser.add_argument("--output-dir", default="clara-out", help="Directory for all output files (default: clara-out)")
    parser.add_argument("--adapter", default="decoder_adapter",
                        help="Which adapter to merge (default: decoder_adapter)")
    parser.add_argument("--quantize", default="Q4_K_M",
                        help="GGUF quantization type (default: Q4_K_M)")
    parser.add_argument("--llama-cpp-dir", default=None,
                        help="Path to llama.cpp source (for convert_hf_to_gguf.py)")
    parser.add_argument("--skip-ollama", action="store_true",
                        help="Skip Ollama registration (just build the GGUF)")
    args = parser.parse_args()

    if not args.skip_ollama and not shutil.which("ollama"):
        sys.exit("ERROR: ollama not found in PATH. Install from https://ollama.com or use --skip-ollama")

    convert_script = find_convert_script() or (
        os.path.join(args.llama_cpp_dir, "convert_hf_to_gguf.py") if args.llama_cpp_dir else None
    )
    if not convert_script or not os.path.exists(convert_script):
        sys.exit(
            "ERROR: convert_hf_to_gguf.py not found.\n"
            "  Install llama.cpp source: git clone https://github.com/ggml-org/llama.cpp /tmp/llama-cpp-src\n"
            "  Then: pip install /tmp/llama-cpp-src/gguf-py\n"
            "  Or pass --llama-cpp-dir /path/to/llama.cpp"
        )

    quantize_bin = find_tool("llama-quantize", args.llama_cpp_dir)
    if not quantize_bin:
        sys.exit(
            "ERROR: llama-quantize not found.\n"
            "  Build llama.cpp or install via: nix-shell -p llama-cpp"
        )

    print(f"Tools found:")
    print(f"  convert_hf_to_gguf.py: {convert_script}")
    print(f"  llama-quantize: {quantize_bin}")

    work_dir = os.path.realpath(args.output_dir)
    os.makedirs(work_dir, exist_ok=True)
    merged_dir = os.path.join(work_dir, "merged_hf")
    gguf_f16 = os.path.join(work_dir, "clara-7b-f16.gguf")
    gguf_quant = os.path.join(work_dir, f"clara-7b-{args.quantize}.gguf")
    modelfile_path = os.path.join(work_dir, "Modelfile")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from huggingface_hub import hf_hub_download

    print(f"\n[1/5] Downloading CLaRa artifacts from {CLARA_REPO}/{CLARA_SUBDIR}...")
    adapters_path = hf_hub_download(CLARA_REPO, f"{CLARA_SUBDIR}/adapters.pth")
    first_last_path = hf_hub_download(CLARA_REPO, f"{CLARA_SUBDIR}/decoder_first_last_layers.pth")
    print(f"  adapters.pth: {adapters_path}")
    print(f"  decoder_first_last_layers.pth: {first_last_path}")

    print(f"\n[2/5] Loading base model {BASE_MODEL} and merging CLaRa decoder adapter...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

    n_mem_tokens = 256 // 16  # doc_max_length=256, compr_rate=16
    mem_tokens = [f'<MEM{i}>' for i in range(n_mem_tokens)]
    existing_special = tokenizer.special_tokens_map.get("additional_special_tokens", [])
    tokenizer.add_special_tokens({
        'additional_special_tokens': existing_special + mem_tokens + ['<AE>', '<ENC>', '<SEP>']
    })

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    base_model.resize_token_embeddings(len(tokenizer))

    print("  Loading decoder_first_last_layers.pth (resized embeddings)...")
    first_last_sd = torch.load(first_last_path, map_location="cpu", weights_only=True)
    base_model.load_state_dict(first_last_sd, strict=False)
    loaded_keys = list(first_last_sd.keys())
    print(f"  Loaded {len(loaded_keys)} tensors: {loaded_keys}")

    print(f"  Loading adapters.pth and extracting '{args.adapter}'...")
    adapters_sd = torch.load(adapters_path, map_location="cpu", weights_only=True)
    available_adapters = list(adapters_sd.keys())
    print(f"  Available adapters: {available_adapters}")

    if args.adapter not in adapters_sd:
        sys.exit(f"ERROR: adapter '{args.adapter}' not found. Available: {available_adapters}")

    adapter_state_dict = adapters_sd[args.adapter]

    # Apply LoRA via PEFT 
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=32,
        target_modules='all-linear',
        lora_dropout=0.0,
    )
    peft_model = get_peft_model(base_model, peft_config, adapter_name=args.adapter)
    peft_model.load_state_dict(
        {f"base_model.model.{k}": v for k, v in adapter_state_dict.items()},
        strict=False,
    )

    print("  Merging LoRA weights into base model...")
    base_model = peft_model.merge_and_unload()

    print(f"  Saving merged model to {merged_dir}...")
    base_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    del base_model, adapters_sd, first_last_sd
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n[3/5] Converting to GGUF f16...")
    run([sys.executable, convert_script, merged_dir,
         "--outfile", gguf_f16, "--outtype", "f16"])

    print(f"\n[4/5] Quantizing to {args.quantize}...")
    run([quantize_bin, gguf_f16, gguf_quant, args.quantize])

    modelfile_content = f"""FROM {gguf_quant}

PARAMETER temperature 0.0
PARAMETER num_predict 300
PARAMETER top_p 1.0
PARAMETER seed 1

SYSTEM \"\"\"You are a retrieval system. Given a user query and a set of numbered documents, select the most relevant document. Respond only in JSON.\"\"\"
"""
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)

    if args.skip_ollama:
        print(f"\n[5/5] Skipping Ollama registration (--skip-ollama).")
        print(f"  Modelfile written to: {modelfile_path}")
        print(f"  Register manually: ollama create {OLLAMA_MODEL_NAME} -f {modelfile_path}")
    else:
        print(f"\n[5/5] Registering model as '{OLLAMA_MODEL_NAME}' with Ollama...")
        run(["ollama", "create", OLLAMA_MODEL_NAME, "-f", modelfile_path])
        print(f"\nDone! Model registered as '{OLLAMA_MODEL_NAME}'.")
        print(f"Test with: ollama run {OLLAMA_MODEL_NAME} 'hello'")

    print(f"\nOutput files in: {work_dir}")
    print(f"  merged_hf/          — HF safetensors (can delete after GGUF is built)")
    print(f"  clara-7b-f16.gguf   — full-precision GGUF ({os.path.getsize(gguf_f16) / 1e9:.1f} GB)")
    print(f"  clara-7b-{args.quantize}.gguf — quantized GGUF ({os.path.getsize(gguf_quant) / 1e9:.1f} GB)")


if __name__ == "__main__":
    main()
