# Transformer Accounting

## GPT-2 XL-shaped model

Configuration:

- `vocab_size = 50,257`
- `context_length = 1,024`
- `num_layers = 48`
- `d_model = 1,600`
- `num_heads = 25`
- `d_ff = 4,288`

### Parameters

The trainable parameter count is:

```text
2 * 50257 * 1600
+ 48 * (2 * 1600 + 4 * 1600^2 + 3 * 1600 * 4288)
+ 1600
= 1,640,452,800 parameters
```

Using single-precision floating point, this requires:

```text
1,640,452,800 * 4 bytes = 6,561,811,200 bytes ~= 6.11 GiB
```

### Forward-pass FLOPs

For one sequence of length `context_length = 1024`, the matrix-multiply FLOPs are:

```text
48 * (8 * 1600^2 * 1024 + 4 * 1024^2 * 1600 + 6 * 1024 * 1600 * 4288)
+ 2 * 1024 * 1600 * 50257
= 3,516,769,894,400 FLOPs
~= 3.52e12 FLOPs
~= 3.52 TFLOPs
```

The block-level terms are attention projections/output projection, attention score/value matrix multiplies, and SwiGLU matrix multiplies; the final term is the LM head.

Note: this is the total work for one forward pass, not throughput. `TFLOPs` here means `10^12` floating point operations, while `TFLOP/s` would be a hardware speed.

### Largest FLOP Contributors

For the GPT-2 XL-shaped model at context length 1024, the SwiGLU/FFN matrix multiplies are the largest contributor. The approximate breakdown is:

```text
attention linear projections/output: 28.6%
attention QK/AV quadratic terms:      9.2%
SwiGLU/FFN:                          57.5%
LM head:                              4.7%
```

The FFN dominates because `d_ff = 4288` is much larger than `d_model = 1600`; the quadratic attention term is smaller at context length 1024, but grows as `context_length^2`.

## GPT-2 Small, Medium, and Large

Using `context_length = 1024` and `vocab_size = 50,257`:

### GPT-2 Small

Configuration: `num_layers = 12`, `d_model = 768`, `num_heads = 12`, `d_ff = 2048`.

```text
total FLOPs ~= 0.292 TFLOPs

attention linear projections/output: 19.9%
attention QK/AV quadratic terms:     13.3%
SwiGLU/FFN:                          39.8%
LM head:                             27.1%
```

### GPT-2 Medium

Configuration: `num_layers = 24`, `d_model = 1024`, `num_heads = 16`, `d_ff = 2752`.

```text
total FLOPs ~= 0.830 TFLOPs

attention linear projections/output: 24.8%
attention QK/AV quadratic terms:     12.4%
SwiGLU/FFN:                          50.1%
LM head:                             12.7%
```

### GPT-2 Large

Configuration: `num_layers = 36`, `d_model = 1280`, `num_heads = 20`, `d_ff = 3392`.

```text
total FLOPs ~= 1.77 TFLOPs

attention linear projections/output: 27.3%
attention QK/AV quadratic terms:     10.9%
SwiGLU/FFN:                          54.3%
LM head:                              7.4%
```

As model size increases with fixed context length, the `d_model^2`-like terms take up more of the total FLOPs, especially SwiGLU/FFN and attention projections. The LM head and quadratic attention terms become proportionally smaller because they scale only linearly in `d_model` when `context_length` and `vocab_size` are fixed.

## GPT-2 XL with Longer Context

Increasing GPT-2 XL from `context_length = 1024` to `context_length = 16384` changes the forward-pass FLOPs from about `3.52 TFLOPs` to about `133.6 TFLOPs`, roughly a `38x` increase.

At `context_length = 16384`, the breakdown is:

```text
attention linear projections/output: 12.1%
attention QK/AV quadratic terms:     61.7%
SwiGLU/FFN:                          24.2%
LM head:                              2.0%
```

The relative contribution changes because the attention QK/AV matrix multiplies scale as `context_length^2`, so they become dominant at long context lengths.
