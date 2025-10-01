# --- Helper functions: reusable VQE runner and plotting
from typing import Callable, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import time
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import VQE
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_algorithms.optimizers import COBYLA
    
def run_vqe(ansatz: QuantumCircuit,
            hamiltonian: SparsePauliOp,
            optimizer=None,
            estimator=None,
            initial_point: Optional[List[float]] = None,
            maxiter: int = 200,
            callback: Optional[Callable] = None):
    """Run VQE on a Hamiltonian and return result and history.

    Returns: (result, history, runtime_seconds)
    - result: VQE result object
    - history: list of (eval_count, energy)
    - runtime_seconds: float
    """

    if optimizer is None:
        optimizer = COBYLA(maxiter=maxiter)
    if estimator is None:
        estimator = Estimator()

    # build params list from ansatz parameters if not provided
    params = []
    for p in ansatz.parameters:
        params.append(p)

    history = []
    def _store_history(eval_count, params_vals, value, meta=None):
        history.append((eval_count, float(value)))
        if callback is not None:
            callback(eval_count, params_vals, value, meta)

    init_pt = initial_point if initial_point is not None else [0.0] * len(params)

    vqe = VQE(
        ansatz=ansatz,
        optimizer=optimizer,
        estimator=estimator,
        initial_point=init_pt,
        callback=_store_history,
    )

    t0 = time.perf_counter()
    res = vqe.compute_minimum_eigenvalue(hamiltonian)
    t1 = time.perf_counter()

    return res, history, (t1 - t0)

def plot_vqe_results(result,
                    history: List[Tuple[int, float]],
                    hamiltonian: SparsePauliOp,
                    ansatz: QuantumCircuit,
                    runtime_seconds: Optional[float] = None,
                    show_circuit: bool = True):
    """Plot VQE convergence and print a short summary."""
    iters = [h[0] for h in history]
    energies = [h[1] for h in history]

    E_exact = float(np.linalg.eigvalsh(hamiltonian.to_matrix()).min().real)

    plt.figure(figsize=(6,4))
    plt.plot(iters, energies, marker="o", label="VQE")
    plt.axhline(E_exact, linestyle="--", label="GS energy", color="red", linewidth=1)
    plt.xlim(0, max(iters) if iters else 1)
    plt.ylim(min(min(energies, default=E_exact)-0.1, E_exact-0.1), max(energies, default=E_exact)+0.1)
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.title("VQE Energy Convergence")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

    print("Final result")
    print("-------------------------------------------")
    print("E_answer   :", E_exact)
    print("E_min      :", float(result.eigenvalue.real))
    print("Error      :", float(result.eigenvalue.real)-E_exact)
    try:
        print("Optimal parameters  :", [float(np.round(x,2)) for x in result.optimal_point])
    except Exception:
        print("Optimal parameters  : (not available)")
    nq = hamiltonian.num_qubits
    print("n_qubits:", nq)
    print("n_params:", len(ansatz.parameters))
    print("depth   :", ansatz.depth())
    if runtime_seconds is not None:
        print(f"VQE runtime: {runtime_seconds:.2f} seconds")
    print(f"Iteration steps: {len(iters)}")
    n_ent = sum(1 for ci in ansatz.data if len(ci[1]) > 1)
    print("Number of entangling gates:", n_ent)
    print("-------------------------------------------")
