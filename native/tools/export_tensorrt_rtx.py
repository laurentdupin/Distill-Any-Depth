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
import struct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("distill-any-depth", "depth-anything-v2", "depth-anything-3"), default="distill-any-depth")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--encoder", choices=("vits", "vitb", "vitl", "metric_hypersim_vits", "metric_hypersim_vitb", "metric_hypersim_vitl", "metric_vkitti_vits", "metric_vkitti_vitb", "metric_vkitti_vitl"), required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.family == "distill-any-depth" and args.checkpoint.suffix.lower() != ".safetensors":
        parser.error("use the canonical .safetensors checkpoint")
    sizes = (140, 182, 280, 420, 560, 700, 840, 980)
    if args.width != args.height or args.width not in sizes:
        parser.error("use a fixed square input at size 140, 182, 280, 420, 560, 700, 840, or 980")
    import torch
    from safetensors.torch import load_file
    source = Path(__file__).resolve().parents[2]
    families = source / 'families'
    if not families.exists():
        families = source.parent
    configs = {"vits": (64, [48, 96, 192, 384]), "vitb": (128, [96, 192, 384, 768]), "vitl": (256, [256, 512, 1024, 1024])}
    if args.family == 'depth-anything-3':
        if args.encoder != 'vits':
            parser.error('only the canonical DA3 Small snapshot is currently supported')
        sys.path.insert(0, str(families / 'depth-anything-v3' / 'src'))
        from omegaconf import OmegaConf
        from depth_anything_3.cfg import create_object
        configuration = json.loads((args.checkpoint / 'config.json').read_text(encoding='utf-8'))
        if configuration.get('model_name') != 'da3-small':
            parser.error('expected the canonical da3-small snapshot')
        model = create_object(OmegaConf.create(configuration['config']))
        checkpoint = args.checkpoint / 'model.safetensors'
        state = load_file(str(checkpoint), device='cpu')
        with checkpoint.open('rb') as stream:
            metadata = json.loads(stream.read(struct.unpack('<Q', stream.read(8))[0])).get('__metadata__', {})
        for alias, target in metadata.items():
            if target in state:
                state[alias] = state[target]
        state = {name.removeprefix('model.'): value for name, value in state.items()}
        model.load_state_dict(state, strict=True)
        # Camera outputs are unused for single-view depth. Keep backbone/head unchanged.
        model.cam_dec = None
        model.cam_enc = None
        import depth_anything_3.model.dinov2.layers.rope as rope
        def positions(self, batch_size, height, width, device):
            y, x = torch.meshgrid(torch.arange(height, device=device),
                                  torch.arange(width, device=device), indexing='ij')
            return torch.stack((y, x), dim=-1).reshape(1, height * width, 2).expand(batch_size, -1, -1).clone()
        rope.PositionGetter.__call__ = positions
    else:
        checkpoint = args.checkpoint
        if args.family == 'depth-anything-v2':
            metric = args.encoder.startswith('metric_')
            repo = families / 'depth-anything-v2'
            sys.path.insert(0, str(repo / 'metric_depth' if metric else repo))
            from depth_anything_v2.dpt import DepthAnythingV2
            state = torch.load(str(checkpoint), map_location='cpu', weights_only=True)
        else:
            sys.path.insert(0, str(source))
            from distillanydepth.depth_anything_v2.dpt import DepthAnythingV2
            state = load_file(str(checkpoint), device='cpu')
            if args.encoder == 'vitl':
                # Match the canonical Large checkpoint mapping in export_model.py.
                state = {
                    ('pretrained.blocks.' + name[len('backbone.blocks.0.'):]
                     if name.startswith('backbone.blocks.0.') else
                     'pretrained.' + name[len('backbone.'):]
                     if name.startswith('backbone.') else name): value
                    for name, value in state.items()}
        architecture = args.encoder.rsplit('_', 1)[-1]
        features, channels = configs[architecture]
        options = {'max_depth': 20.0 if 'hypersim' in args.encoder else 80.0} if args.family == 'depth-anything-v2' and args.encoder.startswith('metric_') else {}
        model = DepthAnythingV2(encoder=architecture, features=features, out_channels=channels, **options)
        model.load_state_dict(state, strict=True)
    model.eval()
    class DepthOnly(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, image):
            if args.family == 'depth-anything-3':
                depth = self.model(image.unsqueeze(1), export_feat_layers=[], infer_gs=False,
                                   use_ray_pose=False)['depth']
                # Preserve raw forward depth. The GPU publication step matches
                # the existing DA3 contract: 1 - minmax(depth), not minmax(1/depth).
                depth = depth.float()
            elif args.family == 'depth-anything-v2':
                depth = self.model(image)
                if args.encoder.startswith('metric_'):
                    depth = depth.float() / self.model.max_depth
            else:
                depth = self.model(image)[0]
            return depth.reshape(1, 1, args.height, args.width)
    graph = DepthOnly(model)
    dtype = torch.float16 if args.precision == "fp16" else torch.float32
    graph.to(dtype=dtype)
    if args.family == 'depth-anything-3':
        # DA3 explicitly converts backbone features to float for its depth head.
        # Preserve that precision boundary when exporting the half backbone.
        model.head.float()
        def float_features(value):
            if isinstance(value, torch.Tensor):
                return value.float()
            return type(value)(float_features(item) for item in value)
        model.head.register_forward_pre_hook(
            lambda module, inputs: (float_features(inputs[0]), *inputs[1:]))
    image = torch.zeros((1, 3, args.height, args.width), dtype=dtype)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial.onnx")
    with torch.inference_mode():
        torch.onnx.export(graph, image, str(temporary), input_names=["image"],
                          output_names=["depth"], opset_version=18, dynamo=False,
                          dynamic_axes=None)
    import onnx
    onnx.checker.check_model(str(temporary))
    temporary.replace(args.output)
    digest = hashlib.sha256()
    with checkpoint.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    metadata = {"schema": 1, "converter": args.family + "-tensorrt-rtx-onnx-v3",
                "family": args.family,
                "source_sha256": digest.hexdigest(), "encoder": args.encoder,
                "precision": args.precision, "input_shape": list(image.shape), "dynamic": False,
                "depth_convention": "raw_forward_depth" if args.family == "depth-anything-3" else
                    "normalized_forward_metric" if args.encoder.startswith("metric_") else "raw_inverse_depth", "opset": 18}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
