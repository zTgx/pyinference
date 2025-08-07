import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=8000, help="Which port to listen on for HTTP API requests")
parser.add_argument('--model', type=str, default='gpt2', help="Which model to load")
parser.add_argument(
    '--resume',
    action='store_true',
    help="Attempt to resume partial downloads if possible"
)
parser.add_argument(
    '--quant',
    type=str,
    choices=['int8', 'int4'],
    default=None,
    help='Quantization level for model loading (requires bitsandbytes and CUDA)'
)
args = parser.parse_args()