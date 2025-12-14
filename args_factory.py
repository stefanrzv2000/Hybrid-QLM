import argparse
import warnings
from typing import Iterable, Callable
import logging

logging.basicConfig(level=logging.INFO)

def float_or_none(value):
    if value.lower() == 'none':
        return None
    else:
        return float(value)

def int_or_none(value):
    if value.lower() == 'none':
        return None
    else:
        return int(value)
    
def str_or_none(value):
    if value.lower() == 'none':
        return None
    else:
        return value

def get_args():

    parser = argparse.ArgumentParser(description='Codebase for training QLMs on quantum hardware or simulators.')

    parser.add_argument('--model', type=str, choices=['qrnn', 'qcnn'], default='qrnn',
                        help='Type of quantum language model to use (default: qrnn)')
    parser.add_argument('--dataset', type=str, choices=['TS','TS-LM','MC','MC-LM','RP'], default='TS-LM',
                        help='Dataset to use (default: TS-LM)')
    parser.add_argument('--data_path', type=str, default='./data',
                        help='Path to the dataset (default: ./data)')
    parser.add_argument('--emb_size', type=int, default=3,
                        help='Embedding size (default: 3)')
    parser.add_argument('--seq_len', type=int, default=6,
                        help='Sequence length (default: 6)')
    parser.add_argument('--pred_type', type=str, choices=['full', 'nn'], default='full',
                        help='ZZ Observables to use for prediction (default: full)')
    parser.add_argument('--rev_emb', action='store_true',
                        help='Enable reverse embedding for QRNN (default: False)')
    parser.add_argument('--pred_reps', type=int, default=2,
                        help='Number of repetitions for QRNN prediction head (default: 2)')
    parser.add_argument('--qrnn_layers', type=int, default=1,
                        help='Number of layers for QRNN (default: 1)')
    parser.add_argument('--cnn_type', type=str_or_none, default=None,
                        help='Type of CNN architecture (e.g., "33", "23", "22") (default: None)')
    parser.add_argument('--no_pred_head', action='store_true',
                        help='Disable prediction head for QCNN (default: False)')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate (default: 0.1)')
    parser.add_argument('--popsize', type=int, default=8,
                        help='Population size for sample-based optimization algorithms (default: 8)')
    parser.add_argument('--reps', type=int, default=2,
                        help='Number of repetitions for quantum circuits (default: 2)')
    parser.add_argument('--sigma', type=float_or_none, default=0.05,
                        help='Sigma for sample-based optimization algorithms (default: 0.05)')
    parser.add_argument('--alg', type=str, choices=['PGPE', 'SPSA', 'GRAD'], default='PGPE',
                        help='Optimization algorithm (default: PGPE)')
    parser.add_argument('--backend', type=str, default='aer_simulator',
                        help='Backend to use (default: aer_simulator)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--shots', type=int_or_none, default=None,
                        help='Number of shots for quantum execution (default: None - use backend default: 4000 for real hardware and statevector for simulators)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--max_batches', type=int_or_none, default=None,
                        help='Maximum number of batches per epoch (default: None)')
    parser.add_argument('--pad_seq_len', type=int_or_none, default=None,
                        help='Pad sequences to this length (default: None)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--save_model', action='store_true',
                        help='Flag to save the trained model (default: False)')
    parser.add_argument('--save_every_epoch', action='store_true',
                        help='Flag to save the trained model every epoch (default: False)')
    parser.add_argument('--model_path', type=str, default='./models',
                        help='Path to save the trained model (default: ./models)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug print statements (default: False)')
    parser.add_argument('--load_model', type=str_or_none, default=None,
                        help='Path to a pre-trained model to load (default: None)')

    args = parser.parse_args()
    if args.save_every_epoch:
        args.save_model = True
    return args