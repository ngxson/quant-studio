# quant-studio

Drop-in replacement for `llama-quantize` that does the quantization 100% on GPU - only CUDA support for now.

Example usage:

```sh
python quant-studio.py in.gguf out.gguf q4_0

python quant-studio.py in.gguf out.gguf q4_k --imatrix imatrix.gguf

python quant-studio.py in.gguf out.gguf iq2_xxs \
  --imatrix imatrix.gguf \
  --token-embedding-type q4_0

```

Most options are the same as `llama-quantize`

Extra options:
- `--mem` bounds the working set per chunk (`4G`, `512M`, ...)
- `--device` overrides device selection (`auto`/`cuda`/`mps`/`cpu`)

## Benchmark

Test input: `Qwen3.5-4B-BF16.gguf`

| Command | <b>CUDA RTX 5060 Ti<br/>(quant-studio)</b> | Mac CPU M5 Max<br/>(llama-quantize) | Intel CPU i7-12700KF 20t<br/>(llama-quantize) | Identical blocks vs<br/>llama-quantize |
| --- | --- | --- | --- | --- |
| IQ2_XXS | 14.8s | 47.6s | 82.5s | 99.49% † |
| IQ2_XS | 46.3s | 96.6s | 159.0s | 98.65% † |
| IQ2_S | 48.0s | 105.4s | 158.1s | 98.79% |
| IQ2_M | 43.4s | 35.5s | 50.7s | 98.27% |
| Q2_K_S | 6.0s | 20.5s | 37.5s | 99.65% |
| Q2_K | 6.2s | 18.2s | 32.0s | 99.71% |
| IQ3_XXS | 81.4s | 65.2s | 91.8s | 99.15% |
| IQ3_XS | 64.4s | 52.2s | 78.3s | 99.55% |
| IQ3_S | 51.7s | 44.3s | 68.4s | 99.72% |
| IQ3_M | 49.1s | 43.6s | 65.8s | 99.65% |
| Q3_K_S | 6.7s | 11.7s | 14.2s | 100.00% |
| Q3_K_M | 7.0s | 14.3s | 22.3s | 99.56% |
| Q3_K_L | 7.2s | 14.7s | 22.5s | 99.49% |
| IQ4_XS | 8.1s | 36.6s | 54.5s | 99.63% |
| Q4_0 * | 3.8s | 4.6s | 6.5s | 100.00% |
| Q4_K_S | 7.2s | 18.9s | 35.4s | 98.80% |
| Q4_K_M | 6.9s | 17.6s | 31.5s | 99.01% |
| Q5_K_S | 7.6s | 19.7s | 36.0s | 98.62% |
| Q5_K_M | 7.4s | 18.4s | 32.0s | 98.85% |
| Q6_K | 6.1s | 11.1s | 13.3s | 100.00% |

- All runs use the same imatrix, except (*) Q4_0 which runs without one: imatrix quantizes ffn_down to Q4_1, but quant-studio hasn't yet supported Q4_1.
- The non-identical blocks are ULP-level tie-breaks with equal or better weighted MSE.
- (†) measured before a tensor-ordering fix; 6 of 249 tensors got a different (equally valid) type than llama-quantize picks.

`--pure` results (used during development)

| | <b>CUDA RTX 5060 Ti<br/>(quant-studio)</b> | Mac CPU M5 Max<br/>(llama-quantize) | Intel CPU i7-12700KF 20t<br/>(llama-quantize) |
| --- | --- | --- | --- |
| Q4_0 `--pure` | 2.4s | 4.6s | 8.9s |
| Q4_K `--pure` (imatrix) | 5.1s | 25.1s | 37.9s |
| IQ2_XXS | ~15s | 43.9s | 93.6s |
