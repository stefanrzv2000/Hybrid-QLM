import numpy as np
import pandas as pd
import os, json
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
import random
import torch
from scipy.special import logsumexp

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import RealAmplitudes
from qiskit.transpiler import generate_preset_pass_manager, PassManager
from qiskit.transpiler import InstructionProperties
from qiskit.visualization import plot_distribution
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.primitives import StatevectorEstimator
from qiskit.circuit import ParameterVector
from qiskit.providers import BackendV2 as Backend

from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import (
    QiskitRuntimeService, 
    EstimatorV2 as Estimator,
    SamplerV2 as Sampler,
    EstimatorOptions,
    Session,
)

from data import load_dataset
from qlms import Adam, HybridModel, HybridQRNNLanguageModel, HybridQCNNLanguageModel, train_hybrid_model, evaluate_perplexity_hybrid, get_model
from args_factory import get_args

def setup_qiskit_ibm_runtime():
    your_api_key = "<api_key_here>"
    CRN = "<crn_here>"

    from qiskit_ibm_runtime import QiskitRuntimeService

    QiskitRuntimeService.save_account(
        channel="ibm_cloud",
        token=your_api_key,
        instance=CRN,
        name="qlms",
        overwrite=True
    )

    service = QiskitRuntimeService(name="qlms")
    return service

def parse_model_path(args):
    # Example: ./models/TS-LM/qrnn/emb3_seq6_lr0.1_PGPE_pop8_sigma0.05_PGPE_aer_simulator_seed42/
    # Example: ./models/TS-LM/qcnn/type_33_emb3_seq6_reps2_lr0.1_PGPE_pop8_sigma0.05_aer_simulator_seed42/
    if args.model == 'qrnn':
        path = f"{args.model_path}/{args.dataset}/{args.model}/{args.backend}_shots{args.shots}/emb{args.emb_size}_seq{args.seq_len}/{args.alg}_lr{args.lr}_BS{args.batch_size}_EP{args.epochs}_pop{args.popsize}_sigma{args.sigma}_seed{args.seed}/"
    elif args.model == 'qcnn':
        path = f"{args.model_path}/{args.dataset}/{args.model}/{args.backend}_shots{args.shots}/type_{args.cnn_type}_emb{args.emb_size}_seq{args.seq_len}_reps{args.reps}/{args.alg}_lr{args.lr}_BS{args.batch_size}_EP{args.epochs}_pop{args.popsize}_sigma{args.sigma}_seed{args.seed}/"
    return path

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
def main():
    args = get_args()
    set_seed(args.seed)

    train_loader, test_loader, seq_len, vocab_size, task = load_dataset(args)
    args.seq_len = seq_len
    args.vocab_size = vocab_size
    args.task = task

    # service = setup_qiskit_ibm_runtime() # only needed once
    service = QiskitRuntimeService(name="fmc-eu")
    if args.backend == 'aer_simulator':
        backend = AerSimulator(seed_simulator=args.seed)
    else:
        try:
            backend = service.backend(args.backend)
        except:
            print(f"Backend {args.backend} not found. Using 'aer_simulator' instead.")
            raise
    print(f"Using backend: {backend.name}")

    estimator = Estimator(mode=backend)
    # estimator.options.default_shots = 1024
    if args.shots is not None:
        estimator.options.default_shots = args.shots
    pm = generate_preset_pass_manager(
        backend=backend,
        initial_layout=None,
        optimization_level=1
    )

    if args.save_model:
        save_path = parse_model_path(args)
        os.makedirs(save_path, exist_ok=True)
        print(f"Model and training history will be saved to {save_path}")
    model = get_model(args, backend, estimator, pm)

    if args.load_model is not None:
        model.load(args.load_model)
        print(f"Loaded model from {args.load_model}")

    start_time = time.time()
    print("Starting training...")
    history = {}
    history = train_hybrid_model(model,
                                 train_loader=train_loader, 
                                 test_loader=test_loader, 
                                 epochs=args.epochs, 
                                 max_batches=args.max_batches,
                                 history=history,
                                 save_path=save_path if args.save_model else None,
                                 save_every_epoch=args.save_every_epoch,
                                 )
    end_time = time.time()
    print(f"Training completed in {(end_time - start_time)/60:.2f} minutes.")
    history['training_time'] = end_time - start_time

    if args.save_model:
        # save_path = parse_model_path(args)
        model_path = os.path.join(save_path, "final_model.npz")
        model.save(model_path)
        print(f"Model saved to {model_path}")
        history_path = os.path.join(save_path, "training_history.json")
        best_params = history['best_params']
        best_model_path = os.path.join(save_path, "best_model.npz")
        np.savez_compressed(
            best_model_path,
            q_params=best_params['q_params'],
            W=best_params['W'],
            b=best_params['b']
        )
        print(f"Model saved to {best_model_path}")
        del history['best_params']
        with open(history_path, "w") as f:
            json.dump(history, f, indent=4)
        print(f"Training history saved to {history_path}")

if __name__ == "__main__":
    main()