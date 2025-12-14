import numpy as np
import pandas as pd
import os, json
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
from scipy.special import logsumexp, softmax

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

# ------- tiny Adam helper ---------
class Adam:
    def __init__(self, shape, lr=0.05):
        self.m  = np.zeros(shape)
        self.v  = np.zeros(shape)
        self.t  = 0
        self.lr = lr
        self.b1, self.b2, self.eps = 0.9, 0.999, 1e-8

    def update(self, w, grad):
        self.t += 1
        self.m  = self.b1 * self.m + (1 - self.b1) * grad
        self.v  = self.b2 * self.v + (1 - self.b2) * (grad ** 2)
        m_hat   = self.m / (1 - self.b1 ** self.t)
        v_hat   = self.v / (1 - self.b2 ** self.t)
        return w - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
# -----------------------------------

# Z and ZZ observables
def get_observables(num_qubits, indices_to_measure, pred='full'):
    obs = []
    for i in indices_to_measure:
        z = ['I'] * num_qubits
        z[i] = 'Z'
        obs.append(SparsePauliOp("".join(z)))
    if pred == 'full':
        for i in indices_to_measure:
            for j in indices_to_measure:
                if i >= j:
                    continue
                zz = ['I'] * num_qubits
                zz[i] = 'Z'
                zz[j] = 'Z'
                obs.append(SparsePauliOp("".join(zz)))
    else:
        for i in range(len(indices_to_measure) - 1):
            idx1 = indices_to_measure[i]
            idx2 = indices_to_measure[i + 1]
            zz = ['I'] * num_qubits
            zz[idx1] = 'Z'
            zz[idx2] = 'Z'
            obs.append(SparsePauliOp("".join(zz)))
    return obs

class HybridModel:
    """
    Generic hybrid quantum-classical model that:
      • stores a single transpiled, *parametric* circuit (self.qc_opt)
      • subclasses provide ._build_params_vector(tokens)
    """
    def __init__(self,
                 qc_template:  QuantumCircuit,
                 measure_indices: list[int],
                 q_params:     np.ndarray,
                 vocab_size:   int,
                 feature_dim:  int,
                 task:         str = "lm",   # NEW: "lm" or "binary"
                 pred:         str = 'full',
                 lr:           float = 0.05,
                 alg:          str = "SPSA",  # "SPSA" or "PGPE"
                 popsize:      int   = 8,
                 sigma:        float = 0.1,
                 backend:      Backend = None,
                 pass_manager: PassManager = None,
                 estimator:    Estimator = None,
                 debug:        bool = False):

        # --- transpile circuit ---
        backend  = backend or AerSimulator()
        self.qc_template = qc_template
        self.pm  = pass_manager or generate_preset_pass_manager(
                        backend=backend,
                        initial_layout=None,
                        optimization_level=3)
        self.qc_opt = self.pm.run(qc_template)

        # --- estimator ---
        self.estimator = estimator or Estimator(mode=backend)

        # --- parameters ---
        self.q_params = q_params
        self.vocab_size  = vocab_size
        self.feature_dim = feature_dim
        self.alg         = alg
        self.popsize     = popsize
        self.sigma       = sigma
        self.task        = task   # store task

        # Classical heads
        if task == "lm":
            self.W = np.random.randn(feature_dim, vocab_size) * 0.1
            self.b = np.zeros(vocab_size)
        elif task == "binary":
            self.W = np.random.randn(feature_dim, 1) * 0.1
            self.b = np.zeros(1)
        else:
            raise ValueError("task must be 'lm' or 'binary'")

        # optimizers
        self.opt_q  = Adam(self.q_params.shape, lr)
        self.opt_W  = Adam(self.W.shape,        lr)
        self.opt_b  = Adam(self.b.shape,        lr)

        # observables
        self.pred = pred
        self.debug = debug
        self.obs = self.get_observables(measure_indices)
        if isinstance(self.obs, list):
            self.obs = [op.apply_layout(self.qc_opt.layout) for op in self.obs]
        else:
            self.obs = self.obs.apply_layout(self.qc_opt.layout)

        self.global_start_time = 0
        self.batch_start_time = 0

    def save(self, path):
        np.savez_compressed(
            path,
            q_params=self.q_params,
            W=self.W,
            b=self.b
        )
        # print(f"Model saved to {path}")

    def load(self, path):
        data = np.load(path)
        assert self.q_params.shape == data["q_params"].shape, "Loaded q_params shape does not match model shape"
        assert self.W.shape == data["W"].shape, "Loaded W shape does not match model shape"
        assert self.b.shape == data["b"].shape, "Loaded b shape does not match model shape"
        self.q_params = data["q_params"]
        self.W = data["W"]
        self.b = data["b"]
        print(f"Model loaded from {path}")

    def get_observables(self, measure_indices) -> list[SparsePauliOp]:
        """Return list[SparsePauliOp] measured on *output* qubits."""
        return get_observables(len(self.qc_template.qubits), measure_indices, pred=self.pred)

    def build_params_vector(self, input_tokens, q_params=None):
        self.q_params = q_params if q_params is not None else self.q_params
        return self._build_params_vector(input_tokens)

    # ---------- to be overridden ----------
    def _build_params_vector(self, input_tokens):
        """
        Return a dict or ordered list mapping *all* circuit parameters
        to values for this (tokens, self.q_params) combination.
        """
        raise NotImplementedError
    # --------------------------------------

    # -------- quantum evaluation ----------
    def _features(self, tokens):
        param_binds = self._build_params_vector(tokens)
        pub = (self.qc_opt, self.obs, param_binds)
        job = self.estimator.run([pub])
        return np.asarray(job.result()[0].data.evs)      # 1‑D feature vector
    # ---------------------------------------

    # --------------- forward ---------------
    def forward(self, tokens):
        feats  = self._features(tokens)
        logits = feats @ self.W + self.b
        return logits, feats
    # ---------------------------------------

    # ---- task-specific loss ----
    def _loss_and_grads_classical(self, tokens, target, grads=True):
        logits, feats = self.forward(tokens)

        if self.task == "lm":
            # Language Modeling loss
            log_probs = logits - logsumexp(logits)
            loss = -log_probs[target]
            if not grads: 
                return loss, None, None, None

            probs = np.exp(log_probs)
            ptg = probs[target].copy()
            probs[target] -= 1
            dW = np.outer(feats, probs)
            db = probs

        elif self.task == "binary":
            # Binary Classification loss (BCE with sigmoid)
            logit = logits.squeeze()  # scalar
            prob = 1 / (1 + np.exp(-logit))
            loss = -(target * np.log(prob + 1e-8) + (1 - target) * np.log(1 - prob + 1e-8))
            if not grads:
                return loss, None, None, None

            dlogit = prob - target
            dW = np.outer(feats, dlogit)
            db = np.array([dlogit])
            ptg = (prob > 0.5) == target  # accuracy-like metric

        return loss, dW, db, ptg

    # ------------- finite‑difference grad -------------
    def _q_grad(self, tokens, target, eps=0.1):
        grad = np.zeros_like(self.q_params)
        for pidx in range(self.popsize):
            if self.debug: print(f"Computing finite difference gradient {pidx+1} / {self.popsize}", end='\r', flush=True)
            delta = np.random.randn(*self.q_params.shape) * eps
            old = self.q_params

            # forward with shifted params +
            self.q_params = old + delta
            lp, *_ = self._loss_and_grads_classical(tokens, target, grads=False)

            # forward with shifted params -
            self.q_params = old - delta
            lm, *_ = self._loss_and_grads_classical(tokens, target, grads=False)

            grad += (lp - lm) / (2 * eps) * delta
            self.q_params = old                       # restore
        return grad / self.popsize
    # --------------------------------------------------

    # --------------- one batch step -------------------
    def train_step_old(self, batch_inputs, batch_targets):
        tot_loss = 0
        tot_ptg = 0
        grad_W = np.zeros_like(self.W)
        grad_b = np.zeros_like(self.b)
        grad_q = np.zeros_like(self.q_params)

        for idx, (toks, tgt) in enumerate(zip(batch_inputs, batch_targets)):
            if self.debug: print(f"Processing batch item {idx+1}/{len(batch_inputs)}", end='\r', flush=True)
            loss, dW, db, ptg = self._loss_and_grads_classical(toks, tgt)
            if self.debug: print("Computed central loss",end='\r', flush=True)
            grad_W += dW
            grad_b += db
            grad_q += self._q_grad(toks, tgt)
            tot_loss += loss
            tot_ptg += ptg

        m = len(batch_inputs)
        self.W        = self.opt_W.update(self.W, grad_W / m)
        self.b        = self.opt_b.update(self.b, grad_b / m)
        self.q_params = self.opt_q.update(self.q_params, grad_q / m)
        return tot_loss / m, tot_ptg / m
    
    def _batched_features(self, tokens_list, deltas):
        """Evaluates all feature vectors in a single Estimator call."""
        old_params = self.q_params.copy()
        pubs = []
        for tokens, delta in zip(tokens_list, deltas):
            param_binds = self.build_params_vector(tokens, self.q_params + delta)
            pubs.append((self.qc_opt, self.obs, param_binds))
        if self.debug: print(f"Running {len(tokens_list)} feature evaluations in a single Estimator call, {time.time() - self.batch_start_time:.3f} seconds elapsed", flush=True)
        job = self.estimator.run(pubs)
        if self.debug: 
            start_time = time.time()
            print(f"Submitted Estimator job, job ID = {job.job_id()}, {time.time() - self.batch_start_time:.3f} seconds elapsed", flush=True)
            status = job.status()
            # I want this print to reload each time
            # print(f"Job status: {status} for {time.time() - start_time} seconds", flush=True)
            while status == 'QUEUED':
                print(f"Job status: {status} for {time.time() - start_time:.3f} seconds", end='\r', flush=True)
                time.sleep(1)
                status = job.status()
            print()
            print("Job is now RUNNING", flush=True)
            start_time = time.time()
            while status == 'RUNNING':
                print(f"Job status: {status} for {time.time() - start_time:.3f} seconds", end='\r', flush=True)
                time.sleep(1)
                status = job.status()
            print()
            print(f"Job is now {job.status()}, {time.time() - start_time:.3f} seconds elapsed", flush=True)

        results = job.result()
        if self.debug: print(f"Finished Running Estimator call", flush=True)
        self.q_params = old_params  # restore original params
        return [np.asarray(res.data.evs) for res in results]

    def train_step(self, batch_inputs, batch_targets):
        self.batch_start_time = time.time()
        tot_loss = 0
        tot_ptg = 0
        grad_W = np.zeros_like(self.W)
        grad_b = np.zeros_like(self.b)
        grad_q = np.zeros_like(self.q_params)

        # Collect param vectors and metadata for all forward passes
        deltas = []
        token_list = []
        meta = []  # for mapping results back

        for idx, (toks, tgt) in enumerate(zip(batch_inputs, batch_targets)):
            # Original (center) input
            token_list.append(toks)
            deltas.append(np.zeros(self.q_params.shape))  # no perturbation
            meta.append(('central', idx, None))  # kind, sample_idx, perturb_idx

            # Perturbed versions
            for pidx in range(self.popsize):
                if self.alg == "PGPE":
                    delta = np.random.randn(*self.q_params.shape) * self.sigma
                elif self.alg == "SPSA": # randomly choose +1/-1 for each param
                    signs = np.random.choice([-1, 1], size=self.q_params.shape)
                    delta = signs * self.sigma

                token_list.append(toks)
                deltas.append(delta)
                meta.append(('plus', idx, pidx))

                token_list.append(toks)
                deltas.append(-delta)
                meta.append(('minus', idx, pidx))

        # Batch Estimator call
        if self.debug: print(f"\nStarting batched Estimator call for {len(token_list)} evaluations, {time.time() - self.batch_start_time:.3f} seconds elapsed", flush=True)
        feature_vectors = self._batched_features(token_list, deltas)
        if self.debug: print(f"Completed batched Estimator call, {time.time() - self.batch_start_time:.3f} seconds elapsed", flush=True)

        # Group results by sample
        sample_data = {i: {'feats': None, 'tgt': tgt, 'perturbs': []}
                    for i, tgt in enumerate(batch_targets)}

        for i, (kind, sample_idx, pidx) in enumerate(meta):
            feats = feature_vectors[i]
            if kind == 'central':
                sample_data[sample_idx]['feats'] = feats
            else:
                sample_data[sample_idx]['perturbs'].append((kind, pidx, feats))

        for idx in range(len(batch_inputs)):
            feats = sample_data[idx]['feats']
            tgt   = sample_data[idx]['tgt']

            logits = feats @ self.W + self.b

            if self.task == "lm":
                # Language modeling
                log_probs = logits - logsumexp(logits)
                loss = -log_probs[tgt]
                probs = np.exp(log_probs)
                ptg = probs[tgt].copy()
                probs[tgt] -= 1
                dW = np.outer(feats, probs)
                db = probs

            elif self.task == "binary":
                # Binary classification
                logit = logits.squeeze()
                prob = 1 / (1 + np.exp(-logit))
                loss = -(tgt * np.log(prob + 1e-8) + (1 - tgt) * np.log(1 - prob + 1e-8))
                dlogit = prob - tgt
                dW = np.outer(feats, dlogit)
                db = np.array([dlogit])
                ptg = (prob > 0.5) == tgt  # accuracy metric

            grad_W += dW
            grad_b += db
            tot_loss += loss
            tot_ptg  += ptg

            # Quantum gradient estimate (finite diff via SPSA-like)
            eps = self.sigma
            qgrad = np.zeros_like(self.q_params)
            lp, lm = None, None
            for kind, pidx, pert_feats in sample_data[idx]['perturbs']:
                delta = deltas[meta.index(('plus', idx, pidx))]
                if kind == 'plus':
                    logits_p = pert_feats @ self.W + self.b
                    if self.task == "lm":
                        lp = - (logits_p - logsumexp(logits_p))[tgt]
                    else:
                        prob_p = 1 / (1 + np.exp(-logits_p.squeeze()))
                        lp = -(tgt * np.log(prob_p + 1e-8) + (1 - tgt) * np.log(1 - prob_p + 1e-8))
                else:  # minus
                    logits_m = pert_feats @ self.W + self.b
                    if self.task == "lm":
                        lm = - (logits_m - logsumexp(logits_m))[tgt]
                    else:
                        prob_m = 1 / (1 + np.exp(-logits_m.squeeze()))
                        lm = -(tgt * np.log(prob_m + 1e-8) + (1 - tgt) * np.log(1 - prob_m + 1e-8))

                    if self.alg == "SPSA":
                        qgrad += (lp - lm) / (2 * eps) * delta
                    elif self.alg == "PGPE":
                        qgrad += (lp - lm) / 2 * delta
                    lp, lm = None, None
            grad_q += qgrad / self.popsize

        m = len(batch_inputs)
        self.W        = self.opt_W.update(self.W, grad_W / m)
        self.b        = self.opt_b.update(self.b, grad_b / m)
        self.q_params = self.opt_q.update(self.q_params, grad_q / m)
        return tot_loss / m, tot_ptg / m

# --------------------------------------------------------------------
# QRNN architecture
# --------------------------------------------------------------------

def _embed_angles(qc: QuantumCircuit, regs, params):
    """RY-angle embedding on a register."""
    for q, p in zip(regs, params):
        qc.ry(p, q)

def _recurrent_block(qc: QuantumCircuit, emb, hid, theta):
    es = len(emb)
    for i in range(es):
        qc.cx(emb[i], hid[i])
        qc.ry(theta[i], hid[i])
    for i in range(0, es - 1, 2):
        qc.cx(hid[i], hid[i + 1])
    for i in range(1, es - 1, 2):
        qc.cx(hid[i], hid[i + 1])
    for i in range(es):
        qc.rz(theta[es + i], hid[i])

def _pred_head(qc: QuantumCircuit, hid, phi):
    es = len(hid)
    for i in range(0, es - 1, 2):
        qc.cx(hid[i], hid[i + 1])
    for i in range(1, es - 1, 2):
        qc.cx(hid[i], hid[i + 1])
    for i in range(es):
        qc.ry(phi[i], hid[i])

def _recurrent_block_mapping(qc: QuantumCircuit, emb, hid, mapping, theta):
    es = len(emb)
    hs = len(hid)
    for i in range(es):
        qc.cx(emb[i], hid[mapping[i]])
    for i in range(hs):
        qc.ry(theta[i], hid[i])
    for i in range(0, hs - 1, 2):
        qc.cx(hid[i], hid[i + 1])
    for i in range(1, hs - 1, 2):
        qc.cx(hid[i], hid[i + 1])
    for i in range(hs):
        qc.rz(theta[hs + i], hid[i])

def _recurrent_block_repeat(qc: QuantumCircuit, emb, hid, mapping, theta):
    es = len(emb)
    hs = len(hid)
    for i in range(es):
        qc.ry(theta[i], emb[i])
    for i in range(es):
        qc.cx(emb[i], hid[mapping[i]])
    for i in range(hs):
        qc.ry(theta[es + i], hid[i])
    for i in range(0, hs - 1, 2):
        qc.cx(hid[i], hid[i + 1])
    for i in range(1, hs - 1, 2):
        qc.cx(hid[i], hid[i + 1])
    for i in range(hs):
        qc.rz(theta[es + hs + i], hid[i])

def _pred_head_param(qc: QuantumCircuit, hid, params, reps=2):
    """Parametric 1-token prediction head with (Ry -> CNOT -> Rz)."""
    qlist = hid[:]
    N = len(qlist)

    for rep in range(reps):
        # Ry rotation bank
        base = rep * (2 * N)
        for i, q in enumerate(qlist):
            qc.ry(params[base + i], q)

        # entangling CNOT layer
        for i in range(0, N - 1, 2):
            if reps % 2 == 0:
                qc.cx(qlist[i], qlist[i + 1])
            else:
                qc.cx(qlist[i + 1], qlist[i])
        for i in range(1, N - 1, 2):
            if reps % 2 == 0:
                qc.cx(qlist[i], qlist[i + 1])
            else:
                qc.cx(qlist[i + 1], qlist[i])

        # Rz rotation bank
        base = N + rep * (2 * N)
        for i, q in enumerate(qlist):
            qc.rz(params[base + i], q)

# --------------------------------------------------------------------
class HybridQRNNLanguageModel(HybridModel):
    """
    Quantum-Recurrent Hybrid model that reuses one transpiled, parametric circuit.
    """

    # ---------------- constructor ----------------
    def __init__(self, emb_size: int, vocab_size: int,
                 layers: int = 1, seq_len: int = 6, repeats: int = 1, reverse_emb: bool = True, pred:str = 'full',
                 lr: float = 0.05, popsize: int = 8, sigma: float = 0.1, task: str = 'lm', alg: str = 'SPSA', pred_reps: int = 2,
                 backend=None, estimator=None, pm=None, debug=False):

        self.emb_size   = emb_size
        self.hidden_size = 2 * emb_size - 3  # same as embedding size
        # self.mapping = [0, 1, 3, 5, ... , self.emb_size-4, self.emb_size-2, self.emb_size-1]
        self.mapping = [0] + list(range(1, 2*emb_size - 3, 2)) + [2*emb_size - 4]
        if self.emb_size == 2:
            self.mapping = [0, 1]
            self.hidden_size = 2
        self.vocab_size = vocab_size
        assert layers == 1, "Only one recurrent layer supported in this model."
        self.layers     = layers
        self.repeats    = repeats                # number of times to repeat the recurrent block
        self.seq_len    = seq_len                # max sentence length processed
        self.num_qubits = self.emb_size + self.hidden_size
        self.reverse_emb = reverse_emb
        self.pred_reps = pred_reps

        # ---- param counts ---------------------------------------------------
        n_pred = 2 * self.pred_reps * self.hidden_size
        n_recur_layer = 2 * self.hidden_size
        n_recur_repeat = 2 * self.hidden_size + self.emb_size
        n_recur_total = layers * n_recur_layer + (repeats-1) * n_recur_repeat
        n_emb_table   = vocab_size * emb_size

        # flat trainable parameter vector (pred + all recurrent + embedding table)
        q_params = np.random.randn(n_pred + n_recur_total + n_emb_table)

        # build parametric circuit template once
        qc_template, param_meta, measure_indices = self.build_parametric_circuit()
        self._meta = param_meta      # store Parameter references / slices

        # feature dimension (Z + ZZ on final hidden register)
        if pred == 'full':
            feat_dim = self.hidden_size + self.hidden_size * (self.hidden_size - 1) // 2
        else:
            feat_dim = self.hidden_size + (self.hidden_size - 1)

        super().__init__(qc_template,       # parametric circuit
                         measure_indices=measure_indices,  # qubits to measure
                         q_params=q_params,
                         vocab_size=vocab_size,
                         feature_dim=feat_dim,
                         pred=pred,
                         lr=lr,
                         alg=alg,
                         popsize=popsize,
                         backend=backend,
                         pass_manager=pm,
                         estimator=estimator,
                         debug=debug,
                         task=task,
                         )

    # ---------------- parametric template ----------------
    def build_parametric_circuit(self):
        """
        Returns:
            qc_template : QuantumCircuit with Parameter objects
            meta        : dict holding ParameterVectors for easy binding
        """
        es = self.emb_size
        hs = self.hidden_size
        layers = self.layers
        seq_len = self.seq_len

        # registers : emb + hidden_1 ... hidden_L
        reg_emb = QuantumRegister(es, 'emb')
        reg_hid = [QuantumRegister(hs, f'HL{i+1}') for i in range(layers)]
        qc = QuantumCircuit(reg_emb, *reg_hid)

        # ------ ParameterVectors ------
        p_pred = ParameterVector('pred', 2 * self.pred_reps * hs)
        p_recs = [[ParameterVector(f'rec{i+1}_{t}', 2 * hs) for i in range(layers)] for t in range(seq_len)]
        p_recs_rep = []
        if self.repeats > 1:
            # additional parameters for repeated recurrent blocks
            p_recs_rep = [[[ParameterVector(f'rec{i+1}_{t}_r{r}', 2 * hs + es) for i in range(layers)] for t in range(seq_len)] for r in range(1, self.repeats)]
        # embedding for each time‑step (seq_len‑1) because last token is target
        p_emb_steps = [ParameterVector(f'e{t}', es) for t in range(seq_len)]
        if self.reverse_emb:
            p_emb_reverse = [ParameterVector(f'e{t}_r', es) for t in range(seq_len)]

        # ------ build template ------
        for t in range(seq_len):
            _embed_angles(qc, reg_emb, p_emb_steps[t])
            _recurrent_block_mapping(qc, reg_emb, reg_hid[0], self.mapping, p_recs[t][0])
            if self.repeats > 1:
                for r in range(1, self.repeats):
                    _recurrent_block_repeat(qc, reg_emb, reg_hid[0], self.mapping, p_recs_rep[r-1][t][0])
            if self.reverse_emb:
                _embed_angles(qc, reg_emb, p_emb_reverse[t])  # uncompute
            qc.barrier()

        # prediction head on last hidden layer
        _pred_head_param(qc, reg_hid[-1], p_pred, reps=self.pred_reps)
        # _pred_head(qc, reg_hid[-1], p_pred[:hs])
        # _pred_head(qc, reg_hid[-1][::-1], p_pred[hs:])

        meta = {
            'pred':   p_pred,
            'recur':  p_recs,          # list of ParameterVectors
            'emb_ts': p_emb_steps,     # per‑step embedding parameters
        }
        if self.reverse_emb:
            meta['emb_rev'] = p_emb_reverse  # per‑step embedding parameters
        if self.repeats > 1:
            meta['recur_rep'] = p_recs_rep
        indices_to_measure = [qc.qubits.index(q) for q in reg_hid[-1]]  # final hidden layer
        return qc, meta, indices_to_measure

    # ---------------- parameter‑binding map ----------------
    def _build_params_vector(self, input_tokens):
        """
        Map template Parameters → numerical values.
        Embedding table (trainable) is at the tail of self.q_params, slice into it.
        """
        es = self.emb_size
        hs = self.hidden_size
        layers = self.layers

        # split trainable vector
        n_pred = 2 * self.pred_reps * hs
        n_recur = 2 * hs * layers
        n_recur_rep = (2 * hs + es) * (self.repeats - 1) * layers
        pred_vals   = self.q_params[:n_pred]
        recur_vals  = self.q_params[n_pred:n_pred + n_recur].reshape(layers, hs * 2)
        if self.repeats > 1:
            recur_vals_rep = self.q_params[n_pred + n_recur:n_pred + n_recur + n_recur_rep].reshape(self.repeats - 1, layers, hs * 2 + es)
        emb_table   = self.q_params[n_pred + n_recur + n_recur_rep:].reshape(self.vocab_size, es)

        # build dictionary for qiskit binding
        bind_dict = {}
        # prediction head
        par = self._meta['pred']
        bind_dict[par] = pred_vals

        # recurrent layers
        for t, vec in enumerate(self._meta['recur']):
            for l, lay in enumerate(vec):                
                bind_dict[lay] = recur_vals[l]    

        if self.repeats > 1:
            for r, vecs in enumerate(self._meta['recur_rep']):
                for t, vec in enumerate(vecs):
                    for l, lay in enumerate(vec):                
                        bind_dict[lay] = recur_vals_rep[r][l]
        
        # per‑time‑step embeddings
        for t, vec in enumerate(self._meta['emb_ts']):
            tok = input_tokens[t]     # token id
            bind_dict[vec] = emb_table[tok]

        if self.reverse_emb:
            # print(self._meta['emb_rev'])
            # print(self._meta['emb_rev'][0])
            # per‑time‑step reverse embeddings
            for t, vec in enumerate(self._meta['emb_rev']):
                # print(vec)
                tok = input_tokens[t]     # token id
                bind_dict[vec] = -emb_table[tok]

        return bind_dict

# ------------------------------------------------------------------
# QCNN architecture
# ------------------------------------------------------------------

def _cnn2_layer_param_new(qc: QuantumCircuit, tk1, tk2, params, reps=2):
    """Parametric 2-token CNN layer (tk1, tk2) with (Ry -> CNOT -> Rz)."""
    qlist = tk1[:] + tk2[:]
    N = len(qlist)

    # # initial Ry rotation bank
    # for i, q in enumerate(qlist):
    #     qc.ry(params[i], q)

    for rep in range(reps):
        # Ry rotation bank
        base = rep * (2 * N)
        for i, q in enumerate(qlist):
            qc.ry(params[base + i], q)

        # entangling CNOT ladder
        for i in range(0, N - 1, 2):
            qc.cx(qlist[i], qlist[i + 1])
        for i in range(1, N - 1, 2):
            qc.cx(qlist[i], qlist[i + 1])

        # Rz rotation bank
        base = N + rep * (2 * N)
        for i, q in enumerate(qlist):
            qc.rz(params[base + i], q)

    return tk2

def _pred_head_param(qc: QuantumCircuit, tk, params, reps=2):
    """Parametric 1-token prediction head with (Ry -> CNOT -> Rz)."""
    qlist = tk[:]
    N = len(qlist)

    for rep in range(reps):
        # Ry rotation bank
        base = rep * (2 * N)
        for i, q in enumerate(qlist):
            qc.ry(params[base + i], q)

        # entangling CNOT ladder
        for i in range(0, N - 1, 2):
            qc.cx(qlist[i], qlist[i + 1])
        for i in range(1, N - 1, 2):
            qc.cx(qlist[i], qlist[i + 1])

        # Rz rotation bank
        base = N + rep * (2 * N)
        for i, q in enumerate(qlist):
            qc.rz(params[base + i], q)

    return tk

def _cnn3_layer_param_new(qc: QuantumCircuit, tk1, tk2, tk3, params, reps=2):
    """Parametric 3-token CNN layer with tk2 as middle (Ry -> CNOT -> Rz)."""
    qlist = tk1[:] + tk2[:] + tk3[:]
    N = len(qlist)
    mid = N // 2

    # # initial Ry rotation bank
    # for i, q in enumerate(qlist):
    #     qc.ry(params[i], q)

    for rep in range(reps):
        # Ry rotation bank
        base = rep * (2 * N)
        for i, q in enumerate(qlist):
            qc.ry(params[base + i], q)

        # entangling CNOT ladder
        for i in range(0, N - 1, 2):
            qc.cx(qlist[i], qlist[i + 1]) if i < mid else qc.cx(qlist[i + 1], qlist[i])
        for i in range(1, N - 1, 2):
            qc.cx(qlist[i], qlist[i + 1]) if i < mid else qc.cx(qlist[i + 1], qlist[i])

        # Rz rotation bank
        base = N + rep * (2 * N)
        for i, q in enumerate(qlist):
            qc.rz(params[base + i], q)


    if len(tk2) >= len(tk1):
        return tk2
    else:
        delta = len(tk1) - len(tk2)
        st = len(tk1) - delta // 2
        end = st + len(tk1)
        return qlist[st:end]

# ------------------------------------------------------------------

class HybridQCNNLanguageModel(HybridModel):
    """
    Hybrid quantum-classical CNN language model
    (embedding → 2-token CNN → 3-token CNN, Z/ZZ read-out),
    using (Ry → CNOT → Rz) repeated reps times as ansatz.
    """

    def __init__(self, emb_size: int, vocab_size: int,
                 seq_len: int = 6, cnn_type='23', reps: int = 2, pred_head: bool = True, pred: str = 'full', task: str = 'lm',
                 lr: float = 0.05, popsize: int = 8, alg: str = 'SPSA', sigma: float = 0.1,
                 backend=None, estimator=None, pm=None, debug: bool = False):

        # store hyper-params
        self.emb_q   = emb_size
        self.vocab   = vocab_size
        self.type    = cnn_type
        self.seq_len = seq_len
        self.reps    = reps
        self.pred    = pred
        self.pred_head = pred_head

        # parameter counts
        n_emb_table = vocab_size * emb_size  # embedding angles
        if self.type == '23':
            n_l1 = emb_size * 2 * 2 * reps   # Ry and Rz for 2-token convs
            n_l2 = emb_size * 3 * 2 * reps   # Ry and Rz for 3-token conv
        elif self.type == '33':
            n_l1 = emb_size * 3 * 2 * reps
            n_l2 = (2 * emb_size + 1) * 2 * reps
        elif self.type == '22':
            n_l1 = emb_size * 2 * 2 * reps
            n_l2 = 2 * emb_size * 2 * reps
        n_ph = emb_size * 2 * reps if self.pred_head else 0

        q_params = np.random.randn(n_emb_table + n_l1 + n_l2 + n_ph)

        # build parametric circuit template once
        qc_templ, meta, out_idx = self.build_parametric_circuit()
        self._meta = meta

        # feature dimension (single Z + ZZ on final register)
        es = emb_size
        feat_dim = es + es * (es - 1) // 2

        super().__init__(qc_template=qc_templ,
                         measure_indices=out_idx,
                         q_params=q_params,
                         vocab_size=vocab_size,
                         feature_dim=feat_dim,
                         task=task,
                         lr=lr,
                         popsize=popsize,
                         alg=alg,
                         sigma=sigma,
                         backend=backend,
                         estimator=estimator,
                         pass_manager=pm,
                         debug=debug)

    # ------------------------------------------------------------
    # build parametric QCNN template
    # ------------------------------------------------------------
    def build_parametric_circuit(self):
        es   = self.emb_q
        L    = self.seq_len
        reps = self.reps

        # token registers
        emb_regs = [QuantumRegister(es, f"Tk{i}") for i in range(L)]
        if self.type == '23' or self.type == '22':
            regs = emb_regs
        elif self.type == '33':
            anc = QuantumRegister(1, 'anc')
            regs = emb_regs[:3] + [anc] + emb_regs[3:]
        qc = QuantumCircuit(*regs)

        # Parameter vectors
        emb_params = [ParameterVector(f"e{i}", es) for i in range(L)]
        if self.type == '23':
            # three pairwise layers in first stage
            l1_params = [ParameterVector(f"L1_{i}", es * 2 * 2 * reps) for i in range(3)]
            l2_params = ParameterVector("L2", es * 3 * 2 * reps)
        elif self.type == '33':
            l1_params = [ParameterVector(f"L1_{i}", es * 3 * 2 * reps) for i in range(2)]
            l2_params = ParameterVector("L2", (2 * es + 1) * 2 * reps)
        elif self.type == '22':
            l1_params = [ParameterVector(f"L1_{i}", es * 2 * 2 * reps) for i in range(2)]
            l2_params = ParameterVector("L2", 2 * es * 2 * reps)

        if self.pred_head:
            pred_params = ParameterVector("PH", es * 2 * reps)

        # embeddings
        for i in range(L):
            for q, p in zip(emb_regs[i], emb_params[i]):
                qc.ry(p, q)
        qc.barrier()

        # first CNN layer(s)
        outs = []
        if self.type == '23' or self.type == '22':
            for i, j in enumerate(range(0, L - 1, 2)):
                outs.append(_cnn2_layer_param_new(qc, regs[j], regs[j + 1], l1_params[i], reps))
        elif self.type == '33':
            for i, j in enumerate(range(0, L - 2, 3)):
                outs.append(_cnn3_layer_param_new(qc, emb_regs[j], emb_regs[j + 2],
                                                  emb_regs[j + 1], l1_params[i], reps))
            outs.append([regs[3][0]])  # ancilla output
        qc.barrier()

        # final CNN layer
        if self.type == '22':
            final_out = _cnn2_layer_param_new(qc, outs[0], outs[1], l2_params, reps)
        else:  # '23' or '33'
            final_out = _cnn3_layer_param_new(qc, outs[0], outs[2], outs[1][::-1], l2_params, reps)

        # optional prediction head
        if self.pred_head:
            qc.barrier()
            final_out = _pred_head_param(qc, final_out, pred_params, reps)

        # indices for measurement (all qubits of final_out)
        out_indices = [qc.qubits.index(q) for q in final_out]

        meta = dict(emb=emb_params,
                    l1=l1_params,
                    l2=l2_params)
        if self.pred_head:
            meta['ph'] = pred_params
        return qc, meta, out_indices

    # ------------------------------------------------------------
    # parameter-binding dict
    # ------------------------------------------------------------
    def _build_params_vector(self, tokens):
        es   = self.emb_q
        reps = self.reps
        L    = self.seq_len

        n_emb = self.vocab * es
        if self.type == '23' or self.type == '22':
            n_l1 = es * 2 * 2 * reps
        else:
            n_l1 = es * 3 * 2 * reps
        n_l2 = (es * 3 if self.type == '23' else (2 * es + 1) if self.type == '33' else 2 * es) * 2 * reps
        n_ph = es * 2 * reps if self.pred_head else 0

        emb_table = self.q_params[:n_emb].reshape(self.vocab, es)
        l1_vals   = self.q_params[n_emb:n_emb + n_l1]
        l2_vals   = self.q_params[n_emb + n_l1:n_emb + n_l1 + n_l2]
        if self.pred_head:
            ph_vals   = self.q_params[n_emb + n_l1 + n_l2:]
            
        bind = {}
        # embeddings
        for i, vec in enumerate(self._meta['emb']):
            bind[vec] = emb_table[tokens[i]]

        # conv layers
        for vec in self._meta['l1']:
            bind[vec] = l1_vals
        bind[self._meta['l2']] = l2_vals
        if self.pred_head:
            bind[self._meta['ph']] = ph_vals

        return bind

# --------------------------------------------------------------
# Training and evaluation utils
# --------------------------------------------------------------

def submit_job_evaluate_perplexity_real(model: HybridModel, dataloader, max_batches=10):
    total_loss, total_tokens = 0, 0

    total = len(dataloader)
    if max_batches is not None and max_batches > 0:
        total = min(total, max_batches)
    pubs = []
    for bid, (x_batch, y_batch) in enumerate(dataloader):
        # print(f"Evaluating batch {bid+1}/{total}", end='\r', flush=True)
        for x, y in zip(x_batch, y_batch):
            input_tokens = x.tolist()
            param_binds = model.build_params_vector(input_tokens)
            pubs.append((model.qc_opt, model.obs, param_binds))
            total_tokens += 1
        if bid + 1 >= total:
            break

    job = model.estimator.run(pubs)
    print(f"Submitted job to evaluate {len(pubs)} feature vectors", end='\n', flush=True)
    return job

def evaluate_job_perplexity_real(model: HybridModel, job, dataloader, max_batches=10):
    total_loss, total_p_target, total_tokens = 0, 0, 0

    total = len(dataloader)
    if max_batches is not None and max_batches > 0:
        total = min(total, max_batches)
    results = [res.data.evs for res in job.result()]        
    idx = 0
    for bid, (x_batch, y_batch) in enumerate(dataloader):
        # print(f"Evaluating batch {bid+1}/{total}", end='\r', flush=True)
        for x, y in zip(x_batch, y_batch):
            res = results[idx]
            idx += 1
            logits = res @ model.W + model.b
            total_tokens += 1
            target = int(y)
            if model.task == "lm":
                probs = softmax(logits)
                probs = np.clip(probs, 1e-8, 1.0)     # avoid log(0)
                total_loss += -np.log(probs[target])
                total_p_target += probs[target]
            elif model.task == "binary":
                logit = logits.squeeze()
                prob = 1 / (1 + np.exp(-logit))
                prob = np.clip(prob, 1e-8, 1 - 1e-8)
                loss = -(y * np.log(prob) + (1 - y) * np.log(1 - prob))
                total_loss += loss
                total_p_target += (prob > 0.5) == y
        if bid + 1 >= total:
            break

    return np.exp(total_loss / total_tokens), total_p_target / total_tokens

def evaluate_perplexity_hybrid(model: HybridModel, dataloader, max_batches=10):
    job = submit_job_evaluate_perplexity_real(model, dataloader, max_batches)
    return evaluate_job_perplexity_real(model, job, dataloader, max_batches)

def train_hybrid_model(model: HybridModel, train_loader, test_loader, epochs=100, max_batches=-1, history={}, start_epoch=0, save_every_epoch=False, save_path=None):

    tr_len = len(train_loader)
    if max_batches <= 0 or max_batches > tr_len:
        max_batches = tr_len

    history['train_loss'] = []
    history['train_ppl'] = []
    history['train_acc'] = []
    history['test_loss'] = []
    history['test_ppl'] = []
    history['test_acc'] = []
    history['best_test_ppl'] = 1e6
    history['best_test_acc'] = 0
    history['best_test_epoch'] = 0
    history['best_params'] = None

    for epoch in range(start_epoch, epochs):
        total_loss = 0
        total_p_target = 0
        total_samples = 0

        for bid, tb in enumerate(train_loader):
            inputs, targets = tb
            batch_inputs = [x.tolist() for x in inputs]
            batch_targets = targets.tolist()
            # print(batch_targets)
            loss, p_target = model.train_step(batch_inputs, batch_targets)
            total_loss += loss * len(batch_inputs)
            total_p_target += p_target * len(batch_inputs)
            total_samples += len(batch_inputs)
            # print(f"Epoch {epoch+1:03d} | Batch {bid+1:03d}/{tr_len} | Batch size: {len(batch_inputs)} | Loss = {loss.item():.4f}", end='\r', flush=True)
            print(f"Epoch {epoch+1:03d} | Batch {bid+1:02d}/{max_batches} | Batch size: {len(batch_inputs)} | Loss = {loss.item():.4f}", end='\r', flush=True)
            if bid >= max_batches - 1:
                break

        avg_loss = total_loss / total_samples
        avg_p_target = total_p_target / total_samples
        test_ppl, test_acc = evaluate_perplexity_hybrid(model, test_loader)
        avg_loss = float(avg_loss)
        train_ppl = float(np.exp(avg_loss))
        avg_p_target = float(avg_p_target)
        test_ppl = float(test_ppl)
        test_acc = float(test_acc)
        test_loss = float(np.log(test_ppl))
        print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.4f} | PPL: {np.exp(avg_loss):.4f} | Acc: {avg_p_target:.4f} | Test Loss: {test_loss:.4f} | Test PPL: {test_ppl:.4f} | Test Acc: {test_acc:.4f}")
        history['train_loss'].append(avg_loss)
        history['train_ppl'].append(train_ppl)
        history['train_acc'].append(avg_p_target)
        history['test_ppl'].append(test_ppl)
        history['test_acc'].append(test_acc)
        history['test_loss'].append(test_loss)
        if model.task == 'lm' and test_ppl < history['best_test_ppl']:
            history['best_test_ppl'] = test_ppl
            history['best_test_acc'] = test_acc
            history['best_test_epoch'] = epoch
            history['best_params'] = {
                'q_params': model.q_params.copy(),
                'W': model.W.copy(),
                'b': model.b.copy(),
            }
        if model.task == 'binary' and test_acc > history.get('best_test_acc', 0):
            history['best_test_acc'] = test_acc
            history['best_test_epoch'] = epoch
            history['best_test_ppl'] = test_ppl
            
            history['best_params'] = {
                'q_params': model.q_params.copy(),
                'W': model.W.copy(),
                'b': model.b.copy(),
            }
        if save_path is not None:
            if save_every_epoch:
                os.makedirs(os.path.join(save_path, 'epoch_models'), exist_ok=True)
                epoch_save_path = os.path.join(save_path, 'epoch_models', f'epoch_{epoch+1:03d}.npz')
                model.save(epoch_save_path)
            else:
                model.save(os.path.join(save_path, 'last_model.npz'))
            best_params = history['best_params']
            best_model_path = os.path.join(save_path, "best_model.npz")
            np.savez_compressed(
                best_model_path,
                q_params=best_params['q_params'],
                W=best_params['W'],
                b=best_params['b']
            )
            del history['best_params']
            # print(history)
            with open(os.path.join(save_path, 'training_history.json'), 'w') as f:
                json.dump(history, f, indent=4)
            history['best_params'] = best_params
        
        if avg_loss < 1e-3 or (avg_p_target == 1 and test_acc == 1):
            print("Training converged.",flush=True)
            break

    return history

def count_parameters(model: HybridModel):
    """
    Count the number of trainable parameters in the model.
    """
    return (model.q_params.size, 
            model.W.size, 
            model.b.size)

def get_model(args, backend, estimator, pm):
    if args.model == 'qrnn':
        model = HybridQRNNLanguageModel(emb_size=args.emb_size,
                                        vocab_size=args.vocab_size,
                                        seq_len=args.seq_len,
                                        layers=args.qrnn_layers,
                                        reverse_emb=args.rev_emb,
                                        pred_reps=args.pred_reps,
                                        pred=args.pred_type,
                                        alg=args.alg,
                                        lr=args.lr,
                                        popsize=args.popsize,
                                        sigma=args.sigma,
                                        task=args.task,
                                        backend=backend,
                                        estimator=estimator,
                                        pm=pm,
                                        debug=args.debug,
                                        )
    elif args.model == 'qcnn':
        model = HybridQCNNLanguageModel(emb_size=args.emb_size,
                                        vocab_size=args.vocab_size,
                                        seq_len=args.seq_len,
                                        cnn_type=args.cnn_type,
                                        reps=args.reps,
                                        pred_head=not args.no_pred_head,
                                        pred=args.pred_type,
                                        task=args.task,
                                        alg=args.alg,
                                        popsize=args.popsize,
                                        sigma=args.sigma,
                                        lr=args.lr,
                                        backend=backend,
                                        estimator=estimator,
                                        pm=pm,
                                        debug=args.debug,
                                        )
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    return model