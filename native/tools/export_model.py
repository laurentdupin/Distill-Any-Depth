"""Convert an official Distill AnyDepth checkpoint to the flat DAD format.

This development tool is not part of deployment. The resulting .dad file is
read directly by the dependency-free native DLL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import zlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


MAGIC = b"DAD1MOD\0"
FORMAT_VERSION = 1
ENDIAN_TAG = 0x01020304
DTYPE_FLOAT32 = 1
HEADER = struct.Struct("<8sIIIIQQQQQ")
RECORD = struct.Struct("<112sII4QQQQIIQ")
ALIGNMENT = 64
ENCODERS = {"vits": 0, "vitb": 1}
METADATA_MAGIC = b"DAD1META"
METADATA_VERSION = 1
CONVERTER_ID = "dad-export-safetensors-v1"
METADATA = struct.Struct("<8sIIIIII32s64s")


def align(value: int) -> int:
    return (value + ALIGNMENT - 1) & ~(ALIGNMENT - 1)


def tensor_bytes(tensor: Any) -> bytes:
    import torch

    if tensor.dtype != torch.float32:
        raise TypeError(f"unsupported tensor dtype: {tensor.dtype}")
    return tensor.detach().cpu().contiguous().numpy().tobytes(order="C")


def sha256_file(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.digest()


def derivation_metadata(
    canonical_sha256: bytes, encoder: str
) -> tuple[bytes, str]:
    if len(canonical_sha256) != 32:
        raise ValueError("canonical SHA-256 must contain 32 bytes")
    converter_id = CONVERTER_ID.encode("ascii")
    if len(converter_id) >= 64:
        raise RuntimeError("converter ID is too long")
    metadata = METADATA.pack(
        METADATA_MAGIC,
        METADATA_VERSION,
        METADATA.size,
        FORMAT_VERSION,
        ENCODERS[encoder],
        0,
        0,
        canonical_sha256,
        converter_id,
    )
    cache_key = (
        f"dad:{canonical_sha256.hex()}:"
        f"converter={CONVERTER_ID}:"
        f"format={FORMAT_VERSION}:encoder={encoder}"
    )
    return metadata, cache_key


def main() -> None:
    import torch
    from safetensors.torch import load_file

    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", choices=ENCODERS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")

    canonical_sha256 = sha256_file(args.checkpoint)
    if args.checkpoint.suffix.lower() != ".safetensors":
        parser.error(
            "only canonical .safetensors checkpoints are accepted; "
            "arbitrary pickle loading is intentionally unsupported"
        )
    state = load_file(str(args.checkpoint), device="cpu")
    if not isinstance(state, dict) or not state:
        raise TypeError("checkpoint is not a non-empty state dictionary")

    tensors: list[
        tuple[str, tuple[int, ...], torch.Tensor, int, int]
    ] = []
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"state entry is not a tensor: {name}")
        encoded_name = name.encode("utf-8")
        if len(encoded_name) >= 112:
            raise ValueError(f"tensor name is too long: {name}")
        if not 1 <= value.ndim <= 4:
            raise ValueError(f"unsupported tensor rank for {name}: {value.ndim}")
        payload = tensor_bytes(value)
        tensors.append(
            (
                name,
                tuple(value.shape),
                value,
                len(payload),
                zlib.crc32(payload),
            )
        )

    directory_offset = HEADER.size
    directory_bytes = len(tensors) * RECORD.size
    metadata_offset = directory_offset + directory_bytes
    data_offset = align(metadata_offset + METADATA.size)
    cursor = data_offset
    records = []
    for name, shape, _, payload_bytes, checksum in tensors:
        cursor = align(cursor)
        dimensions = list(shape) + [0] * (4 - len(shape))
        element_count = 1
        for dimension in shape:
            element_count *= dimension
        if element_count * 4 != payload_bytes:
            raise RuntimeError(f"tensor byte count mismatch: {name}")
        records.append(
            RECORD.pack(
                name.encode("utf-8"),
                DTYPE_FLOAT32,
                len(shape),
                *dimensions,
                cursor,
                payload_bytes,
                element_count,
                checksum,
                0,
                0,
            )
        )
        cursor += payload_bytes
    file_bytes = cursor
    header = HEADER.pack(
        MAGIC,
        FORMAT_VERSION,
        ENDIAN_TAG,
        ENCODERS[args.encoder],
        len(tensors),
        directory_offset,
        directory_bytes,
        data_offset,
        file_bytes,
        metadata_offset,
    )
    metadata, cache_key = derivation_metadata(
        canonical_sha256, args.encoder)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as output:
        output.write(header)
        for record in records:
            output.write(record)
        if output.tell() != metadata_offset:
            raise RuntimeError("metadata offset mismatch")
        output.write(metadata)
        output.write(b"\0" * (data_offset - output.tell()))
        for (_, _, tensor, payload_bytes, _), record in zip(tensors, records):
            record_values = RECORD.unpack(record)
            tensor_offset = record_values[7]
            output.write(b"\0" * (tensor_offset - output.tell()))
            payload = tensor_bytes(tensor)
            if len(payload) != payload_bytes:
                raise RuntimeError("tensor changed while exporting")
            output.write(payload)
        if output.tell() != file_bytes:
            raise RuntimeError(
                f"file length mismatch: {output.tell()} != {file_bytes}"
            )

    print(
        json.dumps(
            {
                "format": "DAD1MOD",
                "version": FORMAT_VERSION,
                "encoder": args.encoder,
                "tensor_count": len(tensors),
                "bytes": file_bytes,
                "output": str(args.output.resolve()),
                "derivation": {
                    "canonical_sha256": canonical_sha256.hex(),
                    "converter": CONVERTER_ID,
                    "format_version": FORMAT_VERSION,
                    "encoder": args.encoder,
                    "cache_key": cache_key,
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
