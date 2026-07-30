# Distill AnyDepth native runtime

## Implemented graph

The current correctness-first runtime implements the pinned repository's
`DepthAnythingV2` Small and Base inference graphs:

- DINOv2 ViT-S/14 or ViT-B/14;
- feature captures at transformer blocks 2, 5, 8, and 11;
- the repository DPT head with `align_corners=True` internal resizes;
- ReLU depth output;
- source-size bilinear output resize with `align_corners=False`.

The Large checkpoint is constructed by a different Python `DAM` class, but
its selected configuration resolves to the same DINOv2 ViT-L/14 and DPT
operator topology. Its checkpoint nests transformer blocks under a chunked
`backbone.blocks.0` prefix; the trusted converter deterministically flattens
that prefix to the runtime's canonical tensor names. Large execution is
is validated alongside Small and Base. Matching PyTorch's exact-erf GELU
removed the earlier apparent ViT-L drift.

## Deployment boundary

Deployment consists of `distill_any_depth.dll`, the public
`include/distill_any_depth.h` C header, and a derived `.dad` model file. The
runtime links no tensor or inference framework. On Windows its only
non-system runtime API is the Vulkan loader supplied by the graphics driver.

Python, PyTorch, safetensors, OpenCV, training code, and visualization are
development-only conversion and validation tools.

The official `.safetensors` checkpoint remains canonical. `export_model.py`
accepts only safetensors (never pickle), validates every tensor, and writes a
bounded memory-mappable `DAD1MOD` file. Its derivation metadata records the
canonical SHA-256, converter identity, format version, and encoder so the file
can live in a hidden content-addressed cache rather than becoming a second
model identity.

## Accuracy contract

PyTorch CPU at the pinned source revision is authoritative. The initial
runtime keeps weights, attention scores, activations, and accumulators in
FP32 and uses deterministic correctness-first Vulkan kernels. Acceptance is
measured per image as:

`sum(abs(native - cpu)) / sum(abs(cpu))`

Every validation image must remain below 1%. Performance selection and packed
storage remain deferred until the graph has passed this gate.

## Input, output, and GPU interop

`dad_infer_bgr8` performs cubic aspect-preserving resize to multiples of 14,
BGR-to-RGB conversion, ImageNet normalization, inference, and source-size
depth resize. `dad_infer_tensor_f32` bypasses image processing for graph-only
validation.

The additive Windows GPU API preserves the proven three-slot lease model. It
can import shared D3D12 BGRA/RGBA buffers or textures and producer fences into
Vulkan, execute preprocessing and the full graph, and expose a shared D3D12
R32-float depth buffer or texture plus a signal fence. Jobs retain duplicated
handles and Vulkan resources through completion; output leases retain only
their slot. No pixel/depth readback, upload staging, or `vkQueueWaitIdle` is
used by this path. Capability bits are returned only when the complete path
is supported.

## ABI scope

The stable C ABI covers lifecycle, device selection, network shape, CPU-image
and normalized-tensor inference, truthful GPU capability probing, asynchronous
D3D12 submission, polling, cancellation, correlated frame metadata, output
leases, and stable status/error reporting. A context supports three live GPU
jobs/leases and otherwise returns `DAD_STATUS_INVALID_STATE` so a caller can
apply a latest-frame drop policy.

## InferBridge binary harness

The `distill_any_depth` library itself exports InferBridge harness ABI 1.0
through `ibrh_get_api`, so deployment needs no adapter DLL. Its vendored
contract declaration matches InferBridge's versioned public header.

The first integration slice truthfully advertises only host BGRA8 input and
host normalized float32 depth output. Submission is synchronous and serialized
per model, maximum in-flight jobs is one, and output memory is held by an
explicit lease that outlives the job handle. The neural graph still runs on
the selected Vulkan GPU, but the harness boundary stages input and output.

The existing external D3D12 path is not advertised through the harness until
its preprocessing, processing dimensions, min/max normalization, and output
texture exactly implement the deployed InferBridge contract without host
staging.
