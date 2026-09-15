"""Setup-only fixed-shape TensorRT RTX preparation. Never used during playback."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

SIZES = (140, 182, 280, 420, 560, 700, 840, 980)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--encoder', choices=('vits', 'vitb'), required=True)
    parser.add_argument('--precision', choices=('fp16', 'fp32'), required=True)
    parser.add_argument('--engine-builder', type=Path, required=True)
    parser.add_argument('--cache-builder', type=Path, required=True)
    parser.add_argument('--cuda-device', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sizes', nargs='+', type=int, choices=SIZES, default=list(SIZES))
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output must be a fresh installation staging directory')
    args.output.mkdir(parents=True)
    exporter = Path(__file__).with_name('export_tensorrt_rtx.py')
    entries = []
    for size in sorted(set(args.sizes)):
        name = f'{args.encoder}-{size}-{args.precision}'
        onnx = args.output / (name + '.onnx')
        engine = args.output / (name + '.engine')
        cache = args.output / (name + '.cache')
        subprocess.run([sys.executable, '-X', 'utf8', str(exporter), '--checkpoint', str(args.checkpoint),
                        '--encoder', args.encoder, '--width', str(size), '--height', str(size),
                        '--precision', args.precision, '--output', str(onnx)], check=True)
        subprocess.run([str(args.engine_builder), '--onnx=' + str(onnx), '--saveEngine=' + str(engine),
                        '--computeCapabilities=75,80,86,89,120', '--skipInference'], check=True)
        subprocess.run([str(args.cache_builder), 'setup', str(engine), str(cache), str(args.cuda_device)], check=True)
        entries.append({'size': size, 'input_shape': [1, 3, size, size],
                        'engine': engine.name, 'cache': cache.name})
        onnx.unlink()
    # Publish this marker only once every shape has finished conversion and GPU warmup.
    manifest = {'schema': 1, 'encoder': args.encoder, 'precision': args.precision,
                'depth_convention': 'normalized_inverse_depth', 'shapes': entries}
    (args.output / 'shapes.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')

if __name__ == '__main__':
    main()
