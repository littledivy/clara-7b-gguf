Provides the CLaRa-7B-Instruct model as GGUF

- `clara-7b-f16.gguf` (full precision)
- `clara-7b-Q4_K_M.gguf` (quantized)

This loads the base Mistral model and patches applies the decoder_adapter LoRa via PEFT, then converts to GGUF using llama.cpp:

```
python main.py --output-dir /tmp/clara-build
```
