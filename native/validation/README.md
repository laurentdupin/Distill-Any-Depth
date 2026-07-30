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

## Distill AnyDepth Large diagnostic

- Canonical checkpoint:
  `xingyang1/Distill-Any-Depth/large/model.safetensors`
- Canonical SHA-256:
  `c7d46f6f0048bc13e40691713294d0e8c6099231bf3130457123ce48aab425eb`
- Encoder: ViT-L/14
- Native tensor count: 407
- Derived size: 1,341,340,992 bytes

The converter flattens the DAM checkpoint's chunked `backbone.blocks.0.*`
names into the native DINOv2 namespace without changing tensor bytes. Across
the same 22 images, median relative L1 is 0.237660% and mean is 0.342133%.
The worst image is 1.454475% (maximum absolute error 2.144165); 20 of 22
images remain below 1%. Detailed rows are in
`distill-large-140/vitl_cpu.csv`.

## Pending

- Large remains accuracy-pending under the earlier strict 1% per-image gate.
  The remaining deviation matches the cumulative FP32 transformer drift seen
  in other ViT-L ports rather than a missing operator or checkpoint mapping.
