# quant-studio

Drop-in replacement for `llama-quantize` that does the quantization 100% on GPU - only CUDA support for now.

Example usage:

```sh
python quant-studio.py in.gguf out.gguf q4_0 --mem 4G

python quant-studio.py in.gguf out.gguf iq2_xxs --mem 4G \
  --imatrix imatrix.gguf \
  --token-embedding-type q4_0

```

- `--mem` bounds the working set per chunk (`4G`, `512M`, ...)
- `--device` overrides device selection (`auto`/`cuda`/`mps`/`cpu`)
- `--imatrix` GGUF importance matrix, required for iq2_xxs
- `--token-embedding-type` override for token_embd.weight, same as llama-quantize

## Benchmark

Test input: `Qwen3.5-4B-BF16.gguf`

| | <b>CUDA RTX 5060 Ti<br/>(quant-studio)</b> | Mac CPU M-series<br/>(llama-quantize) | Intel CPU i7-12700KF 20t<br/>(llama-quantize) |
| --- | --- | --- | --- |
| q4_0 `--pure` | 2.4s | 4.6s | 8.9s |
| IQ2_XXS | ~15s | 43.9s | 93.6s |
