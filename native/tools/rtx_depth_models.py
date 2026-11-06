"""Setup-only adapters for the additional native TensorRT depth families.

Inputs use the bridge's ImageNet-normalized RGB contract. Adapters undo that
normalization where a reference model accepts RGB, and preserve the family's
publication convention. No model construction may download weights or code.
"""
from pathlib import Path
import sys

FAMILIES = {
    'depth-anything': ('vits', 'vitb', 'vitl'),
    'midas': ('midas_v21_small_256',),
    'moge-2': ('vits-normal', 'vitb-normal', 'vitl-normal'),
    'metricanything': ('metricanything-student-pointmap',),
    'depth-pro': ('default',),
    'zoedepth': ('ZoeN', 'ZoeK', 'ZoeNK'),
    'zipdepth': ('base_gpu', 'base_mobile'),
}
SUPPORTED_VARIANTS = {
    **FAMILIES,
    'distill-any-depth': ('vits', 'vitb', 'vitl'),
    'depth-anything-3': ('vits',),
    'depth-anything-v2': ('vits', 'vitb', 'vitl',
        'metric_hypersim_vits', 'metric_hypersim_vitb', 'metric_hypersim_vitl',
        'metric_vkitti_vits', 'metric_vkitti_vitb', 'metric_vkitti_vitl'),
}


def convention(family):
    if family in ('moge-2', 'metricanything'):
        return 'pointmap_metric'
    if family == 'depth-pro':
        return 'normalized_forward_metric'
    return 'normalized_inverse_depth'


def source_root(families, family):
    return families / ('depth-anything-v1' if family == 'depth-anything' else
                       'moge-2' if family == 'metricanything' else family)


def build(family, variant, checkpoint, families, width, height):
    import torch
    import torch.nn.functional as F
    if variant not in FAMILIES[family]:
        raise ValueError(f'Unsupported {family} weight: {variant}')
    repo = source_root(families, family)
    sys.path.insert(0, str(repo))
    checkpoint = Path(checkpoint)
    if family == 'depth-anything':
        from depth_anything.dpt import DPT_DINOv2
        features, channels = {'vits': (64, [48, 96, 192, 384]),
                              'vitb': (128, [96, 192, 384, 768]),
                              'vitl': (256, [256, 512, 1024, 1024])}[variant]
        # The official constructor uses a cwd-relative local torch.hub path.
        original_load = torch.hub.load
        def local_load(repository, *args, **kwargs):
            if repository == 'torchhub/facebookresearch_dinov2_main':
                repository = str(repo / repository)
            if kwargs.get('source') != 'local':
                raise RuntimeError('Network model construction is forbidden during setup')
            return original_load(repository, *args, **kwargs)
        torch.hub.load = local_load
        try:
            model = DPT_DINOv2(encoder=variant, features=features, out_channels=channels)
        finally:
            torch.hub.load = original_load
        checkpoint = checkpoint / 'pytorch_model.bin' if checkpoint.is_dir() else checkpoint
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    elif family == 'midas':
        from geffnet import tf_efficientnet_lite3
        import midas.blocks as blocks
        original = blocks._make_pretrained_efficientnet_lite3
        blocks._make_pretrained_efficientnet_lite3 = lambda *args, **kwargs: blocks._make_efficientnet_backbone(
            tf_efficientnet_lite3(pretrained=False, exportable=True))
        try:
            from midas.midas_net_custom import MidasNet_small
            model = MidasNet_small(str(checkpoint), features=64, backbone='efficientnet_lite3',
                                   exportable=True, non_negative=True, blocks={'expand': True})
        finally:
            blocks._make_pretrained_efficientnet_lite3 = original
    elif family in ('moge-2', 'metricanything'):
        from moge.model.v2 import MoGeModel
        model = MoGeModel.from_pretrained(str(checkpoint))
        model.onnx_compatible_mode = True
        if hasattr(model, 'normal_head'):
            del model.normal_head  # Surface normals are not consumed by depth rendering.
    elif family == 'zipdepth':
        from zipdepth.model.architecture import create_model
        from zipdepth.utils.model_utils import strip_state_dict_prefixes
        model = create_model(variant='base', upsample_unfold=variant != 'base_mobile')
        archive = torch.load(checkpoint, map_location='cpu', weights_only=True)
        state = strip_state_dict_prefixes(archive.get('model_state_dict', archive))
        model.load_state_dict(state, strict=True)
        model.eval().fuse_for_inference()
    elif family == 'depth-pro':
        from dataclasses import replace
        import types
        sys.path.insert(0, str(repo / 'src'))
        # Load the model package without the optional HEIF image-viewer helper
        # eagerly imported by its __init__.py. Inputs already come from D3D12.
        package = types.ModuleType('depth_pro')
        package.__path__ = [str(repo / 'src/depth_pro')]
        sys.modules['depth_pro'] = package
        from depth_pro.depth_pro import create_model_and_transforms, DEFAULT_MONODEPTH_CONFIG_DICT
        config = replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=str(checkpoint))
        model, _ = create_model_and_transforms(config=config, device=torch.device('cpu'))
    elif family == 'zoedepth':
        # Shared with the reference validation; constructs locally with no hub downloads.
        import importlib.util
        spec = importlib.util.spec_from_file_location('rtx_zoe_reference', repo / 'native/tools/dump_encoder_reference.py')
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        model = reference.build_model(repo, families / 'midas', checkpoint,
                                      max(32, round(width / 32) * 32),
                                      {'ZoeN': 'n', 'ZoeK': 'k', 'ZoeNK': 'nk'}[variant])
        class FixedPrep(torch.nn.Module):
            def forward(self, image):
                size = max(32, round(width / 32) * 32)
                return F.interpolate(image, (size, size), mode='bilinear', align_corners=True) * 2 - 1
        # The reference resizer calls NumPy on Python image dimensions, which
        # are tensors while tracing. This is its equivalent fixed-shape graph.
        model.core.prep = FixedPrep()
        network_size = max(32, round(width / 32) * 32)
        def fixed_transformer(pretrained, image):
            pretrained.model.forward_features(image)
            outputs = []
            for index in range(1, 5):
                postprocess = getattr(pretrained, 'act_postprocess' + str(index))
                feature = postprocess[:2](pretrained.activations[str(index)])
                feature = feature.reshape(1, feature.shape[1], network_size // 16, network_size // 16)
                outputs.append(postprocess[3:](feature))
            return tuple(outputs)
        model.core.core.forward_transformer = fixed_transformer
    else:
        raise ValueError(f'Unsupported family: {family}')

    class DepthGraph(torch.nn.Module):
        def __init__(self, reference):
            super().__init__()
            self.model = reference.eval().float()
            self.register_buffer('mean', torch.tensor([.485, .456, .406]).reshape(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([.229, .224, .225]).reshape(1, 3, 1, 1))

        def forward(self, image):
            rgb = image * self.std + self.mean
            if family in ('moge-2', 'metricanything'):
                result = self.model(rgb, num_tokens=(width // 14) * (height // 14))
                points = result['points'].permute(0, 3, 1, 2)
                mask = result['mask'].unsqueeze(1)
                scale = result.get('metric_scale', torch.ones_like(mask[:, 0, 0, 0]))
                # One packed GPU output: XYZ, foreground probability, metric scale.
                # The native GPU recovery matches the Vulkan focal/shift solver.
                return torch.cat([points, mask, scale.reshape(1, 1, 1, 1).expand_as(mask)], dim=1).float()
            if family == 'depth-pro':
                network = F.interpolate(rgb * 2 - 1, (1536, 1536), mode='bilinear', align_corners=False)
                inverse, _ = self.model(network)
                # Same default forced horizontal FoV=63 degrees as the Vulkan path.
                inverse = inverse * (2 * torch.tan(image.new_tensor(63.0 * 3.141592653589793 / 360)))
                result = (inverse.clamp(1e-4, 1e4).reciprocal() / 25).clamp(0, 1)
            elif family == 'zoedepth':
                result = zoe_nk(self.model, rgb) if variant == 'ZoeNK' else self.model(rgb)['metric_depth']
                result = result.clamp_min(1e-6).reciprocal()
            elif family == 'midas':
                network_size = max(32, round(width / 32) * 32)
                result = self.model(F.interpolate(image, (network_size, network_size), mode='bilinear', align_corners=False))
            elif family == 'zipdepth':
                network_size = max(32, round(width / 32) * 32)
                result = self.model(F.interpolate(rgb, (network_size, network_size), mode='bilinear', align_corners=False))
            else:
                result = self.model(image)
            result = result.reshape(1, 1, result.shape[-2], result.shape[-1]).float()
            return F.interpolate(result, (height, width), mode='bilinear', align_corners=False)

    return DepthGraph(model).eval(), checkpoint


def zoe_nk(model, image):
    """Export both domain heads and select on GPU; never freeze a Python .item()."""
    import torch
    import torch.nn.functional as F
    _, out = model.core(image, denorm=False, return_rel_depth=True)
    last, bottleneck, *blocks = out
    initial = model.conv2(bottleneck)
    logits = model.mlp_classifier(model.patch_transformer(initial)[0])
    outputs = []
    for name in ('nyu', 'kitti'):
        _, centers = model.seed_bin_regressors[name](initial)
        previous = model.seed_projector(initial)
        for projector, attractor, block in zip(model.projectors, model.attractors[name], blocks):
            embedding = projector(block)
            centers, physical_centers = attractor(embedding, centers, previous, interpolate=True)
            previous = embedding
        physical_centers = F.interpolate(physical_centers, last.shape[-2:], mode='bilinear', align_corners=True)
        embedding = F.interpolate(embedding, last.shape[-2:], mode='bilinear', align_corners=True)
        probabilities = model.conditional_log_binomial[name](last, embedding)
        outputs.append((probabilities * physical_centers).sum(dim=1, keepdim=True))
    return torch.where(logits.sum(dim=0)[0] >= logits.sum(dim=0)[1], outputs[0], outputs[1])
