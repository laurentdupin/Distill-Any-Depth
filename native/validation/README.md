# Native validation

## Distill AnyDepth Small

- Source revision: `6d8f415392eafb49c96a38cc4dedbd09a1607f50`
- Canonical checkpoint:
  `xingyang1/Distill-Any-Depth/small/model.safetensors`
- Canonical SHA-256:
  `56a173c0e1b5045bf6296a5c1fb16eace0bbde2eddc24b37532cb1774ac09caa`
- Native converter: `dad-export-safetensors-v1`
- Native format: `DAD1MOD` version 1
- Encoder: ViT-S/14
- Native tensor count: 239
- Derived size: 99,186,496 bytes

The CPU-reference comparison used all 22 images in the Depth Anything V2
assets directory at input size 140. PyTorch CPU used the pinned Distill
AnyDepth repository graph and canonical safetensors; the native graph ran on
the Radeon RX 9070.

| Metric | Result |
|---|---:|
| Images | 22 |
| Minimum relative L1 | 0.027759% |
| Median relative L1 | 0.118531% |
| Mean relative L1 | 0.120660% |
| Maximum relative L1 | 0.296132% |
| Maximum absolute error | 0.033489 |

All images pass the 1% correctness requirement. Detailed rows are stored in
`distill-small-140/vits_cpu.csv`.

The real D3D12/Vulkan full-graph canary also passed on the RX 9070. It covers
shared BGRA/RGBA texture input, direct full-graph inference, shared R32-float
texture output, source-frame correlation, explicit fences, three retained
leases, bounded exhaustion and stable reuse, cancellation, fence lifetime,
shutdown, and zero host pixel/depth transfer counters.

## Distill AnyDepth Base

- Canonical checkpoint:
  `xingyang1/Distill-Any-Depth/base/model.safetensors`
- Canonical SHA-256:
  `ee99258af6c40302e7046b510190e3e4624d22d03909fa34e543cd6f6ce567ac`
- Encoder: ViT-B/14
- Native tensor count: 239
- Derived size: 389,929,280 bytes

The same 22-image, input-size-140 PyTorch CPU comparison produced:

| Metric | Result |
|---|---:|
| Images | 22 |
| Minimum relative L1 | 0.017256% |
| Median relative L1 | 0.125860% |
| Mean relative L1 | 0.121030% |
| Maximum relative L1 | 0.238350% |
| Maximum absolute error | 0.079329 |

All images pass the 1% correctness requirement. Detailed rows are stored in
`distill-base-140/vitb_cpu.csv`.

## Distill AnyDepth Large

- Canonical checkpoint:
  `xingyang1/Distill-Any-Depth/large/model.safetensors`
- Canonical SHA-256:
  `c7d46f6f0048bc13e40691713294d0e8c6099231bf3130457123ce48aab425eb`
- Encoder: ViT-L/14
- Native tensor count: 407
- Derived size: 1,341,340,992 bytes

The converter flattens the DAM checkpoint's chunked `backbone.blocks.0.*`
names into the native DINOv2 namespace without changing tensor bytes. The
original validation exposed a 1.454475% outlier. Its cause was a semantic
activation mismatch: native used tanh-approximate GELU while the pinned
PyTorch graph uses exact-erf GELU.

After correcting GELU, the exact deployed InferBridge contract was compared
on all 22 assets at size 140:

| GPU | Raw relative L1 | Normalized relative L1 | Normalized max pixel |
|---|---:|---:|---:|
| Radeon RX 9070 | 0.000723% | 0.000834% | 0.004518% |
| GeForce GTX 1080 | 0.000857% | 0.001020% | 0.002939% |
| Radeon RX 6700 XT | 0.000654% | 0.000800% | 0.004482% |

All Small, Base, and Large variants now pass the less-than-1% requirement.
The additive `dad_get_inferbridge_shape` and
`dad_inferbridge_bgra8_f32` image ABI was exercised directly on the same
22 images. Worst normalized pixel error is 0.004518%.
