"""Setup-only ONNX conversion from the canonical Distill Any Depth weights.

The graph returns raw inverse depth at inference dimensions. Image preprocessing
and depth normalization belong to the native GPU bridge, not this exporter.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--encoder", choices=("vits", "vitb"), required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.checkpoint.suffix.lower() != ".safetensors":
        parser.error("use the canonical .safetensors checkpoint")
    if min(args.width, args.height) < 14 or min(args.width, args.height) > 1000:
        parser.error("the shorter inference dimension must be between 14 and 1000")
    if args.width % 14 or args.height % 14:
        parser.error("inference dimensions must be multiples of 14")
    import torch
    from safetensors.torch import load_file
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from distillanydepth.depth_anything_v2.dpt import DepthAnythingV2
    configs = {"vits": (64, [48, 96, 192, 384]), "vitb": (128, [96, 192, 384, 768])}
    features, channels = configs[args.encoder]
    model = DepthAnythingV2(encoder=args.encoder, features=features, out_channels=channels)
    model.load_state_dict(load_file(str(args.checkpoint), device="cpu"), strict=True)
    model.eval()
    class DepthOnly(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, image):
            return self.model(image)[0]
    graph = DepthOnly(model)
    dtype = torch.float16 if args.precision == "fp16" else torch.float32
    graph.to(dtype=dtype)
    image = torch.zeros((1, 3, args.height, args.width), dtype=dtype)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial.onnx")
    with torch.inference_mode():
        torch.onnx.export(graph, image, str(temporary), input_names=["image"],
                          output_names=["depth"], opset_version=18, dynamo=False)
    import onnx
    onnx.checker.check_model(str(temporary))
    temporary.replace(args.output)
    digest = hashlib.sha256()
    with args.checkpoint.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    metadata = {"schema": 1, "converter": "dad-tensorrt-rtx-onnx-v1",
                "source_sha256": digest.hexdigest(), "encoder": args.encoder,
                "precision": args.precision, "input_shape": list(image.shape),
                "depth_convention": "raw_inverse_depth", "opset": 18}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
