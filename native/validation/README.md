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

## Pending

- Large is a distinct DAM ViT-L graph and is not accepted by the converter or
  runtime until that graph is ported and validated.
