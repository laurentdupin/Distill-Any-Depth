"""Compare the native DLL with the pinned PyTorch CPU reference.

This is a development validation tool. Python, PyTorch, OpenCV, and NumPy are
not used by or shipped with the native runtime.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
import torch


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, value):
        for transform in self.transforms:
            value = transform(value)
        return value


# dpt.py imports torchvision only for this trivial composition helper. Avoid
# requiring a functioning TorchVision installation in the validation setup.
torchvision = types.ModuleType("torchvision")
torchvision_transforms = types.ModuleType("torchvision.transforms")
torchvision_transforms.Compose = Compose
torchvision.transforms = torchvision_transforms
sys.modules["torchvision"] = torchvision
sys.modules["torchvision.transforms"] = torchvision_transforms

# The two diffusers mixins are only used by an unused projection helper in
# the pinned inference source. Stub that packaging-only dependency so the CPU
# authority uses the repository graph without installing the diffusion stack.
diffusers = types.ModuleType("diffusers")
diffusers_models = types.ModuleType("diffusers.models")
diffusers_modeling = types.ModuleType("diffusers.models.modeling_utils")
diffusers_configuration = types.ModuleType("diffusers.configuration_utils")


class ModelMixin:
    pass


class ConfigMixin:
    pass


def register_to_config(function):
    return function


diffusers_modeling.ModelMixin = ModelMixin
diffusers_configuration.ConfigMixin = ConfigMixin
diffusers_configuration.register_to_config = register_to_config
sys.modules["diffusers"] = diffusers
sys.modules["diffusers.models"] = diffusers_models
sys.modules["diffusers.models.modeling_utils"] = diffusers_modeling
sys.modules["diffusers.configuration_utils"] = diffusers_configuration

repository_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repository_root))

from distillanydepth.depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402
from distillanydepth.depth_anything_v2.util.transform import (  # noqa: E402
    NormalizeImage,
    PrepareForNet,
    Resize,
)


CONFIGS = {
    "vits": {
        "enum": 0,
        "model": {
            "encoder": "vits",
            "features": 64,
            "out_channels": [48, 96, 192, 384],
        },
    },
    "vitb": {
        "enum": 1,
        "model": {
            "encoder": "vitb",
            "features": 128,
            "out_channels": [96, 192, 384, 768],
        },
    },
}


def load_reference_state(checkpoint: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    if checkpoint.suffix.lower() != ".safetensors":
        raise ValueError("reference checkpoint must be .safetensors")
    return load_file(str(checkpoint), device="cpu")


class CreateOptions(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("encoder", ctypes.c_int),
        ("vulkan_device_index", ctypes.c_int32),
        ("flags", ctypes.c_uint32),
    ]


def configure_dll(path: Path):
    dll = ctypes.CDLL(str(path.resolve()))
    dll.dad_create.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(CreateOptions),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    dll.dad_create.restype = ctypes.c_int
    dll.dad_destroy.argtypes = [ctypes.c_void_p]
    dll.dad_infer_bgr8.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_ssize_t,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
    ]
    dll.dad_infer_bgr8.restype = ctypes.c_int
    dll.dad_last_error.restype = ctypes.c_char_p
    return dll


def python_input(
    image: np.ndarray, input_size: int, device: str
) -> torch.Tensor:
    transform = Compose(
        [
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method="lower_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            PrepareForNet(),
        ]
    )
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    prepared = transform({"image": rgb})["image"]
    return torch.from_numpy(prepared).unsqueeze(0).to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", choices=CONFIGS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=Path("assets"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument(
        "--reference-device",
        choices=("cpu", "vulkan"),
        default="cpu",
    )
    args = parser.parse_args()

    config = CONFIGS[args.encoder]
    start = time.perf_counter()
    model = DepthAnythingV2(**config["model"])
    state = load_reference_state(args.checkpoint)
    model.load_state_dict(state, strict=True)
    model.eval().to(args.reference_device)
    python_create_seconds = time.perf_counter() - start

    dll = configure_dll(args.dll)
    options = CreateOptions(
        ctypes.sizeof(CreateOptions), 1, config["enum"], 0, 0
    )
    context = ctypes.c_void_p()
    start = time.perf_counter()
    status = dll.dad_create(
        str(args.model.resolve()).encode(),
        ctypes.byref(options),
        ctypes.byref(context),
    )
    native_create_seconds = time.perf_counter() - start
    if status != 0:
        raise RuntimeError(dll.dad_last_error().decode())

    paths = sorted(
        path
        for path in args.assets.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    args.output.mkdir(parents=True, exist_ok=True)
    preview_directory = args.output / args.encoder
    preview_directory.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        for index, path in enumerate(paths, 1):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"cannot decode {path}")
            height, width = image.shape[:2]
            native = np.empty(width * height, dtype=np.float32)
            start = time.perf_counter()
            status = dll.dad_infer_bgr8(
                context,
                image.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                width,
                height,
                image.strides[0],
                args.input_size,
                native.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                native.size,
            )
            native_seconds = time.perf_counter() - start
            if status != 0:
                raise RuntimeError(
                    f"{path}: {dll.dad_last_error().decode()}"
                )

            start = time.perf_counter()
            tensor = python_input(
                image, args.input_size, args.reference_device
            )
            with torch.inference_mode():
                reference, _ = model(tensor)
                reference = torch.nn.functional.interpolate(
                    reference,
                    (height, width),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
            reference = reference.cpu().numpy().astype(np.float32, copy=False)
            python_seconds = time.perf_counter() - start

            difference = np.abs(native.reshape(height, width) - reference)
            differing = int(np.count_nonzero(native.reshape(height, width) != reference))
            reference_l1 = float(
                np.abs(reference, dtype=np.float64).sum()
            )
            row = {
                "encoder": args.encoder,
                "reference_device": args.reference_device,
                "image": path.as_posix(),
                "width": width,
                "height": height,
                "pixels": width * height,
                "differing": differing,
                "max_abs": float(difference.max()),
                "mean_abs": float(difference.mean()),
                "rmse": float(
                    np.sqrt(
                        np.square(difference, dtype=np.float64).mean()
                    )
                ),
                "relative_l1": (
                    float(
                        difference.astype(np.float64).sum() /
                        reference_l1
                    )
                    if reference_l1
                    else 0.0
                ),
                "reference_min": float(reference.min()),
                "reference_max": float(reference.max()),
                "native_seconds": native_seconds,
                "python_seconds": python_seconds,
            }
            rows.append(row)

            depth = native.reshape(height, width)
            span = float(depth.max() - depth.min())
            preview = (
                np.zeros_like(depth, dtype=np.uint8)
                if span == 0.0
                else np.clip((depth - depth.min()) * (255.0 / span), 0, 255)
                .astype(np.uint8)
            )
            relative = path.relative_to(args.assets)
            preview_name = "__".join(relative.parts)
            cv2.imwrite(str(preview_directory / f"{preview_name}.png"), preview)
            print(
                f"[{index:02d}/{len(paths)}] {path}: "
                f"diff={differing}, max={difference.max():.9g}, "
                f"C={native_seconds:.3f}s, Python={python_seconds:.3f}s",
                flush=True,
            )
    finally:
        dll.dad_destroy(context)

    csv_path = (
        args.output /
        f"{args.encoder}_{args.reference_device}.csv"
    )
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"{args.encoder}: create C={native_create_seconds:.3f}s, "
        f"Python={python_create_seconds:.3f}s; results={csv_path}"
    )


if __name__ == "__main__":
    main()
