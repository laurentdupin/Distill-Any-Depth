"""Setup-only fixed-shape TensorRT RTX preparation. Never used during playback."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import shutil
import concurrent.futures
import ctypes
import os
import time

SIZES = (140, 182, 280, 420, 560, 700, 840, 980)
PIPELINE_VERSION = 'depth-pipeline-v3'


def available_physical_memory():
    class MemoryStatus(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                    ('total_physical', ctypes.c_ulonglong),
                    ('available_physical', ctypes.c_ulonglong),
                    ('total_page_file', ctypes.c_ulonglong),
                    ('available_page_file', ctypes.c_ulonglong),
                    ('total_virtual', ctypes.c_ulonglong),
                    ('available_virtual', ctypes.c_ulonglong),
                    ('available_extended_virtual', ctypes.c_ulonglong)]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    return status.available_physical if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)) else 0


def checkpoint_size(path):
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob('*') if item.is_file())


def build_worker_count(checkpoint, task_count):
    logical_cpus = os.cpu_count() or 1
    # Measurements on an 8-core/16-thread Ryzen 7 show four exporters with
    # four intra-op threads complete four shapes in about 10 seconds, versus
    # 18 seconds with two exporters and 31 seconds serially.
    cpu_workers = max(1, logical_cpus // 4)
    # Export temporarily holds the checkpoint, model, traced graph and ONNX
    # protobuf. Keep at least 2 GiB of available RAM per concurrent worker and
    # scale the estimate for larger Base/Large checkpoints.
    worker_memory = max(2 * 1024 ** 3, checkpoint_size(checkpoint) * 5)
    available = available_physical_memory()
    memory_workers = max(1, available // worker_memory) if available else 1
    return max(1, min(task_count, cpu_workers, memory_workers))

def main(family='distill-any-depth'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--encoder', choices=('vits', 'vitb', 'vitl', 'metric_hypersim_vits', 'metric_hypersim_vitb', 'metric_hypersim_vitl', 'metric_vkitti_vits', 'metric_vkitti_vitb', 'metric_vkitti_vitl'), required=True)
    parser.add_argument('--precision', choices=('fp16', 'fp32'), required=True)
    parser.add_argument('--engine-builder', type=Path, required=True)
    parser.add_argument('--cache-builder', type=Path, required=True)
    parser.add_argument('--cuda-device', type=int, default=-1)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--reuse-from', type=Path)
    parser.add_argument('--sizes', nargs='+', type=int, choices=SIZES, default=list(SIZES))
    args = parser.parse_args()
    if args.output.exists() and not args.resume:
        parser.error('output must be a fresh installation staging directory')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    pipeline_marker = args.output / 'pipeline-version.txt'
    if args.resume and (not pipeline_marker.is_file() or pipeline_marker.read_text(encoding='utf-8') != PIPELINE_VERSION):
        parser.error('cannot resume an obsolete pipeline; use a fresh output directory')
    pipeline_marker.write_text(PIPELINE_VERSION, encoding='utf-8')
    exporter = Path(__file__).with_name('export_tensorrt_rtx.py')
    devices = json.loads(subprocess.run([str(args.cache_builder), 'devices'], check=True, capture_output=True, text=True).stdout)
    if args.cuda_device >= 0:
        devices = [device for device in devices if device['index'] == args.cuda_device]
    if not devices:
        raise RuntimeError('No supported RTX adapter is available')
    # Incremental tier installation may reuse only the same complete adapter set;
    # cache-builder verification checks whether the current driver accepts each cache.
    # Copy into fresh staging, never modify the currently usable installation.
    if args.reuse_from and (args.reuse_from / 'shapes.json').is_file():
        previous = json.loads((args.reuse_from / 'shapes.json').read_text(encoding='utf-8'))
        if previous.get('pipeline_version') == PIPELINE_VERSION and previous.get('family') == family and previous.get('devices') == devices and previous.get('encoder') == args.encoder and previous.get('precision') == args.precision:
            for size in sorted(set(args.sizes)):
                name = f'{args.encoder}-{size}-{args.precision}'
                engine = args.reuse_from / (name + '.engine')
                caches = [(args.reuse_from / ('gpu-' + device['luid']) / (name + '.cache'), device) for device in devices]
                if engine.is_file() and all(cache.is_file() and subprocess.run(
                        [str(args.cache_builder), 'verify', str(engine), str(cache), str(device['index'])]).returncode == 0
                        for cache, device in caches):
                    shutil.copy2(engine, args.output / engine.name)
                    for cache, device in caches:
                        target = args.output / ('gpu-' + device['luid'])
                        target.mkdir(exist_ok=True)
                        shutil.copy2(cache, target / cache.name)
            args.resume = True
    selected_sizes = sorted(set(args.sizes))

    def build_engine(size):
        name = f'{args.encoder}-{size}-{args.precision}'
        onnx = args.output / (name + '.onnx')
        engine = args.output / (name + '.engine')
        if args.resume and engine.is_file():
            return
        started = time.monotonic()
        print(f'TensorRT CPU export started size={size} export_precision=fp32 '
              f'tensorrt_precision={args.precision}', flush=True)
        environment = os.environ.copy()
        environment['OMP_NUM_THREADS'] = str(max(1, (os.cpu_count() or 1) // workers))
        environment['MKL_NUM_THREADS'] = environment['OMP_NUM_THREADS']
        subprocess.run([sys.executable, '-X', 'utf8', str(exporter), '--checkpoint', str(args.checkpoint),
                        '--family', family, '--encoder', args.encoder, '--width', str(size), '--height', str(size),
                        '--precision', args.precision, '--output', str(onnx)], check=True, env=environment)
        exported = time.monotonic()
        print(f'TensorRT CPU export completed size={size} export_precision=fp32 '
              f'duration_ms={int((exported - started) * 1000)}', flush=True)
        subprocess.run([str(args.engine_builder), str(onnx), str(engine)], check=True)
        print(f'TensorRT engine completed size={size} duration_ms={int((time.monotonic() - exported) * 1000)}', flush=True)
        onnx.unlink(missing_ok=True)

    missing_engines = [size for size in selected_sizes if not (
        args.resume and (args.output / f'{args.encoder}-{size}-{args.precision}.engine').is_file())]
    workers = build_worker_count(args.checkpoint, len(missing_engines)) if missing_engines else 1
    print(f'TensorRT engine preparation workers={workers} shapes={len(missing_engines)}', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(build_engine, size) for size in missing_engines]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    entries = []
    for size in selected_sizes:
        name = f'{args.encoder}-{size}-{args.precision}'
        engine = args.output / (name + '.engine')
        cache = args.output / (name + '.cache')
        for device in devices:
            gpu_root = args.output / ('gpu-' + device['luid'])
            gpu_root.mkdir(exist_ok=True)
            gpu_cache = gpu_root / cache.name
            if args.resume and gpu_cache.is_file():
                verified = subprocess.run([str(args.cache_builder), 'verify', str(engine), str(gpu_cache), str(device['index'])])
                if verified.returncode == 0:
                    continue
                gpu_cache.unlink()
            started = time.monotonic()
            print(f'TensorRT cache started size={size} cuda_device={device["index"]}', flush=True)
            subprocess.run([str(args.cache_builder), 'setup', str(engine), str(gpu_cache), str(device['index'])], check=True)
            print(f'TensorRT cache completed size={size} cuda_device={device["index"]} duration_ms={int((time.monotonic() - started) * 1000)}', flush=True)
        entries.append({'size': size, 'input_shape': [1, 3, size, size],
                        'engine': engine.name, 'cache': cache.name})
    # Publish this marker only once every shape has finished conversion and GPU warmup.
    manifest = {'schema': 1, 'pipeline_version': PIPELINE_VERSION,
                'encoder': args.encoder, 'precision': args.precision,
                'depth_convention': 'raw_forward_depth' if family == 'depth-anything-3' else
                    'normalized_forward_metric' if args.encoder.startswith('metric_') else 'normalized_inverse_depth',
                'family': family, 'devices': devices, 'shapes': entries}
    (args.output / 'shapes.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')

if __name__ == '__main__':
    main()
