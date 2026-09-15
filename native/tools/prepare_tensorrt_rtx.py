"""Setup-only fixed-shape TensorRT RTX preparation. Never used during playback."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import shutil

SIZES = (140, 182, 280, 420, 560, 700, 840, 980)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--encoder', choices=('vits', 'vitb'), required=True)
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
    exporter = Path(__file__).with_name('export_tensorrt_rtx.py')
    devices = json.loads(subprocess.run([str(args.cache_builder), 'devices'], check=True, capture_output=True, text=True).stdout)
    if args.cuda_device >= 0:
        devices = [device for device in devices if device['index'] == args.cuda_device]
    if not devices:
        raise RuntimeError('No supported RTX adapter is available')
    # Incremental tier installation may reuse only the same complete adapter/driver set.
    # Copy into fresh staging, never modify the currently usable installation.
    if args.reuse_from and (args.reuse_from / 'shapes.json').is_file():
        previous = json.loads((args.reuse_from / 'shapes.json').read_text(encoding='utf-8'))
        if previous.get('devices') == devices and previous.get('encoder') == args.encoder and previous.get('precision') == args.precision:
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
    entries = []
    for size in sorted(set(args.sizes)):
        name = f'{args.encoder}-{size}-{args.precision}'
        onnx = args.output / (name + '.onnx')
        engine = args.output / (name + '.engine')
        cache = args.output / (name + '.cache')
        if not (args.resume and engine.is_file()):
            subprocess.run([sys.executable, '-X', 'utf8', str(exporter), '--checkpoint', str(args.checkpoint),
                            '--encoder', args.encoder, '--width', str(size), '--height', str(size),
                            '--precision', args.precision, '--output', str(onnx)], check=True)
            subprocess.run([str(args.engine_builder), str(onnx), str(engine)], check=True)
        for device in devices:
            gpu_root = args.output / ('gpu-' + device['luid'])
            gpu_root.mkdir(exist_ok=True)
            gpu_cache = gpu_root / cache.name
            if args.resume and gpu_cache.is_file():
                verified = subprocess.run([str(args.cache_builder), 'verify', str(engine), str(gpu_cache), str(device['index'])])
                if verified.returncode == 0:
                    continue
                gpu_cache.unlink()
            subprocess.run([str(args.cache_builder), 'setup', str(engine), str(gpu_cache), str(device['index'])], check=True)
        entries.append({'size': size, 'input_shape': [1, 3, size, size],
                        'engine': engine.name, 'cache': cache.name})
        onnx.unlink(missing_ok=True)
    # Publish this marker only once every shape has finished conversion and GPU warmup.
    manifest = {'schema': 1, 'encoder': args.encoder, 'precision': args.precision,
                'depth_convention': 'normalized_inverse_depth', 'devices': devices, 'shapes': entries}
    (args.output / 'shapes.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')

if __name__ == '__main__':
    main()
