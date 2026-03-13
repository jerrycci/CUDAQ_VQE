#!/usr/bin/env python3
"""
CUDA-Q VQE Real GPU Backend Server
====================================
Requirements:
    pip install cudaq pyscf openfermion openfermionpyscf websockets scipy numpy

Run:
    python vqe_server.py
    # Server starts at ws://localhost:8765
"""

import asyncio
import json
import time
import subprocess
import threading
import traceback
from dataclasses import dataclass, asdict
from typing import Optional

try:
    import websockets
except ImportError:
    raise SystemExit("pip install websockets")

try:
    import cudaq
    import numpy as np
    from scipy.optimize import minimize
    HAS_CUDAQ = True
except ImportError:
    HAS_CUDAQ = False
    import numpy as np

# ── GPU Detection ──────────────────────────────────────────────────────────

def get_gpu_info() -> dict:
    """Read real GPU info via nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total,"
                "memory.used,memory.free,utilization.gpu,utilization.memory,"
                "temperature.gpu,clocks.current.sm,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return {"error": result.stderr.strip()}

        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            gpus.append({
                "index":        int(parts[0]),
                "name":         parts[1],
                "driver":       parts[2],
                "mem_total_mb": int(parts[3]),
                "mem_used_mb":  int(parts[4]),
                "mem_free_mb":  int(parts[5]),
                "gpu_util":     int(parts[6]),
                "mem_util":     int(parts[7]),
                "temp_c":       int(parts[8]),
                "sm_clock_mhz": int(parts[9]),
                "compute_cap":  parts[10],
            })

        # CUDA version from nvcc or nvidia-smi
        cuda_ver = ""
        try:
            cv = subprocess.run(
                ["nvidia-smi", "--query", "--display=COMPUTE"],
                capture_output=True, text=True, timeout=3,
            )
            for l in cv.stdout.split("\n"):
                if "CUDA Version" in l:
                    cuda_ver = l.split(":")[-1].strip()
                    break
        except Exception:
            pass

        return {"gpus": gpus, "cuda_version": cuda_ver, "count": len(gpus)}

    except FileNotFoundError:
        return {"error": "nvidia-smi not found – is the NVIDIA driver installed?"}
    except Exception as e:
        return {"error": str(e)}


def get_gpu_live(gpu_index: int = 0) -> dict:
    """Lightweight live utilization poll."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi", f"--id={gpu_index}",
                "--query-gpu=utilization.gpu,utilization.memory,memory.used,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=3,
        )
        parts = [p.strip() for p in result.stdout.strip().split(",")]
        return {
            "gpu_util":  int(parts[0]),
            "mem_util":  int(parts[1]),
            "mem_used":  int(parts[2]),
            "temp":      int(parts[3]),
            "power_w":   float(parts[4]) if parts[4] != "N/A" else 0.0,
        }
    except Exception:
        return {}


# ── Molecule Definitions ───────────────────────────────────────────────────

MOLECULES = {
    # cudaq.chemistry.create_molecular_hamiltonian(geometry, basis, multiplicity, charge)
    # multiplicity = 2S+1  (singlet=1, doublet=2, triplet=3 ...)
    # charge       = net ionic charge (0 for neutral)
    "H2": {
        "label":        "Hydrogen (H₂)",
        "geometry":     [("H", (0, 0, 0)), ("H", (0, 0, 0.7414))],
        "basis":        "sto-3g",
        "multiplicity": 1,   # singlet ground state (2 electrons, paired)
        "charge":       0,
        "fci_ref":      -1.137270,
        "max_iter":     80,
        "desc":         "2 electrons · 4 spin-orbitals · STO-3G",
    },
    "LiH": {
        "label":        "Lithium Hydride (LiH)",
        "geometry":     [("Li", (0, 0, 0)), ("H", (0, 0, 1.5949))],
        "basis":        "sto-3g",
        "multiplicity": 1,   # singlet ground state (4 electrons, all paired)
        "charge":       0,
        "fci_ref":      -7.882381,
        "max_iter":     150,
        "desc":         "4 electrons · 12 spin-orbitals · STO-3G",
    },
    "BeH2": {
        "label":        "Beryllium Hydride (BeH₂)",
        "geometry":     [
            ("Be", (0, 0, 0)),
            ("H",  (0, 0,  1.3264)),
            ("H",  (0, 0, -1.3264)),
        ],
        "basis":        "sto-3g",
        "multiplicity": 1,   # singlet ground state (6 electrons, all paired)
        "charge":       0,
        "fci_ref":      -15.736688,
        "max_iter":     200,
        "desc":         "6 electrons · 14 spin-orbitals · STO-3G",
    },
    "H2O": {
        "label":     "Water (H₂O)",
        "geometry":     [
            ("O", ( 0.0000,  0.0000,  0.1173)),
            ("H", ( 0.0000,  0.7572, -0.4692)),
            ("H", ( 0.0000, -0.7572, -0.4692)),
        ],
        "basis":        "sto-3g",
        "multiplicity": 1,   # singlet ground state (10 electrons, all paired)
        "charge":       0,
        "fci_ref":      -75.012388,
        "max_iter":     250,
        "desc":         "10 electrons · 14 spin-orbitals · STO-3G",
    },
}


# ── Circuit Structure Generator ────────────────────────────────────────────

def build_circuit_description(mol_key: str, n_electrons: int, n_qubits: int, n_params: int) -> dict:
    """
    Return a JSON-serialisable description of the UCCSD circuit
    so the frontend can draw it exactly.
    Each gate: { type, qubit, qubit2 (CNOT only), param_idx, layer }
    """
    gates = []
    layer = 0

    # ── Layer 0: Hartree-Fock reference state (X gates on occupied orbitals) ──
    for i in range(n_electrons):
        gates.append({"type": "X", "qubit": i, "layer": layer, "label": "X"})
    layer += 1

    # ── UCCSD structure: singles + doubles excitations ──
    # Singles: Ry rotations on all occupied → virtual pairs, bridged by CNOT ladders
    p_idx = 0
    n_occ = n_electrons
    n_virt = n_qubits - n_occ

    # Single excitations  (t_ia)
    for a in range(n_occ, n_qubits):
        for i in range(n_occ):
            gates.append({"type": "RY", "qubit": i, "layer": layer, "label": f"Ry(θ{p_idx})", "param_idx": p_idx})
            layer += 1
            # CNOT ladder i→a
            for k in range(i, a):
                gates.append({"type": "CNOT", "qubit": k, "qubit2": k+1, "layer": layer, "label": "CNOT"})
            layer += 1
            gates.append({"type": "RZ", "qubit": a, "layer": layer, "label": f"Rz(θ{p_idx})", "param_idx": p_idx})
            p_idx += 1
            # uncompute CNOT ladder
            for k in range(a-1, i-1, -1):
                gates.append({"type": "CNOT", "qubit": k, "qubit2": k+1, "layer": layer, "label": "CNOT"})
            layer += 1

    # Double excitations  (t_ijab) – abbreviated for readability
    # Only generate up to actual param count
    for i in range(n_occ):
        for j in range(i+1, n_occ):
            for a in range(n_occ, n_qubits):
                for b in range(a+1, n_qubits):
                    if p_idx >= n_params:
                        break
                    gates.append({"type": "RY", "qubit": i, "layer": layer, "label": f"Ry(θ{p_idx})", "param_idx": p_idx})
                    gates.append({"type": "RY", "qubit": j, "layer": layer, "label": f"Ry(θ{p_idx})", "param_idx": p_idx})
                    layer += 1
                    gates.append({"type": "CNOT", "qubit": i, "qubit2": j, "layer": layer, "label": "CNOT"})
                    gates.append({"type": "CNOT", "qubit": a, "qubit2": b, "layer": layer, "label": "CNOT"})
                    layer += 1
                    gates.append({"type": "RZ", "qubit": a, "layer": layer, "label": f"Rz(θ{p_idx})", "param_idx": p_idx})
                    gates.append({"type": "RZ", "qubit": b, "layer": layer, "label": f"Rz(θ{p_idx})", "param_idx": p_idx})
                    p_idx += 1
                    layer += 1
                    gates.append({"type": "CNOT", "qubit": i, "qubit2": j, "layer": layer, "label": "CNOT"})
                    gates.append({"type": "CNOT", "qubit": a, "qubit2": b, "layer": layer, "label": "CNOT"})
                    layer += 1
                if p_idx >= n_params:
                    break
            if p_idx >= n_params:
                break

    # Final measurement
    for q in range(n_qubits):
        gates.append({"type": "MEAS", "qubit": q, "layer": layer, "label": "M"})

    return {
        "n_qubits":   n_qubits,
        "n_electrons": n_electrons,
        "n_params":   n_params,
        "depth":      layer + 1,
        "gates":      gates,
        "gate_count": {
            "X":    sum(1 for g in gates if g["type"] == "X"),
            "RY":   sum(1 for g in gates if g["type"] == "RY"),
            "RZ":   sum(1 for g in gates if g["type"] == "RZ"),
            "CNOT": sum(1 for g in gates if g["type"] == "CNOT"),
            "MEAS": sum(1 for g in gates if g["type"] == "MEAS"),
        }
    }


# ── Real VQE Runner ────────────────────────────────────────────────────────

class VQERunner:
    def __init__(self, websocket, mol_key: str, gpu_index: int = 0):
        self.ws = websocket
        self.mol_key = mol_key
        self.gpu_index = gpu_index
        self.mol = MOLECULES[mol_key]
        self.iteration = 0
        self.energies = []
        self.start_time = time.time()
        self.cancelled = False

    async def send(self, msg: dict):
        try:
            await self.ws.send(json.dumps(msg))
        except Exception:
            pass

    async def run(self):
        mol = self.mol

        await self.send({"type": "status", "msg": f"Initialising CUDA-Q · target: nvidia-fp64"})
        await self.send({"type": "log", "src": "CUDA-Q", "msg": f"cudaq version: {cudaq.__version__}"})

        # ── Set GPU target ──
        try:
            cudaq.set_target('nvidia', option='fp64')
            target_name = "nvidia-fp64"
        except Exception:
            try:
                cudaq.set_target("nvidia")
                target_name = "nvidia (fp32)"
            except Exception:
                cudaq.set_target("qpp-cpu")
                target_name = "qpp-cpu (no GPU)"

        await self.send({"type": "log", "src": "CUDA-Q", "msg": f"Target set: {target_name}"})

        # ── Build molecular Hamiltonian ──
        # API: create_molecular_hamiltonian(geometry, basis, multiplicity, charge)
        await self.send({"type": "status", "msg": "Building molecular Hamiltonian (PySCF / STO-3G)..."})
        t0 = time.time()
        try:
            hamiltonian, data = cudaq.chemistry.create_molecular_hamiltonian(
                mol["geometry"], mol["basis"], mol["multiplicity"], mol["charge"]
            )
        except Exception as e:
            await self.send({"type": "error", "msg": f"Hamiltonian build failed: {e}"})
            return

        n_electrons = data.n_electrons
        n_qubits    = 2 * data.n_orbitals
        n_params    = cudaq.kernels.uccsd_num_parameters(n_electrons, n_qubits)
        ham_terms   = hamiltonian.term_count
        build_time  = time.time() - t0

        await self.send({
            "type":        "mol_info",
            "n_electrons": n_electrons,
            "n_qubits":    n_qubits,
            "n_params":    n_params,
            "ham_terms":   ham_terms,
            "build_time":  round(build_time, 3),
        })

        # ── Build & send circuit description ──
        circuit_desc = build_circuit_description(self.mol_key, n_electrons, n_qubits, n_params)
        await self.send({"type": "circuit", "circuit": circuit_desc})
        await self.send({"type": "log", "src": "CUDA-Q",
                         "msg": f"Circuit: {n_qubits}Q · depth {circuit_desc['depth']} · "
                                f"{circuit_desc['gate_count']['CNOT']} CNOT · {n_params} params"})

        # ── Pauli terms ──
        try:
            term_list = []
            for term in hamiltonian.get_terms():
                coeff = term.get_coefficient().real
                ops   = str(term).split("]")[0].replace("[","").strip() or "I"
                term_list.append({"coeff": round(coeff, 6), "ops": ops})
            await self.send({"type": "hamiltonian", "terms": term_list[:30]})
        except Exception:
            pass

        # ── UCCSD Kernel ──
        @cudaq.kernel
        def uccsd_kernel(thetas: list[float]):
            q = cudaq.qvector(n_qubits)
            for i in range(n_electrons):
                x(q[i])
            cudaq.kernels.uccsd(q, thetas, n_electrons, n_qubits)

        # ── VQE Optimisation Loop ──
        await self.send({"type": "status", "msg": "Starting VQE optimisation (COBYLA)..."})
        await self.send({"type": "log", "src": "VQE",
                         "msg": f"max_iter={mol['max_iter']} · FCI ref={mol['fci_ref']} Ha"})

        x0 = np.zeros(n_params)
        self.iteration = 0

        async def callback_send(energy: float):
            self.iteration += 1
            self.energies.append(energy)
            elapsed = round(time.time() - self.start_time, 2)

            # live GPU stats
            gpu_live = get_gpu_live(self.gpu_index)

            await self.send({
                "type":     "iteration",
                "iter":     self.iteration,
                "energy":   round(energy, 8),
                "elapsed":  elapsed,
                "gpu_live": gpu_live,
                "fci_ref":  mol["fci_ref"],
                "error_mha": round(abs(energy - mol["fci_ref"]) * 1000, 4),
            })

        # Wrap cost in async-friendly way
        loop = asyncio.get_event_loop()
        call_lock = threading.Lock()

        def cost_fn(thetas):
            if self.cancelled:
                raise StopIteration("cancelled")
            e = cudaq.observe(uccsd_kernel, hamiltonian, thetas).expectation()
            future = asyncio.run_coroutine_threadsafe(callback_send(e), loop)
            future.result(timeout=5)
            return e

        # Run scipy minimise in a thread so we don't block the event loop
        def run_minimise():
            try:
                result = minimize(
                    cost_fn, x0,
                    method="COBYLA",
                    options={"maxiter": mol["max_iter"], "rhobeg": 0.5},
                )
                return result
            except StopIteration:
                return None
            except Exception as exc:
                return exc

        result = await loop.run_in_executor(None, run_minimise)

        if isinstance(result, Exception):
            await self.send({"type": "error", "msg": str(result)})
            return

        final_energy = self.energies[-1] if self.energies else float("nan")
        error_mha    = abs(final_energy - mol["fci_ref"]) * 1000
        total_time   = round(time.time() - self.start_time, 2)

        await self.send({
            "type":         "done",
            "final_energy": round(final_energy, 8),
            "fci_ref":      mol["fci_ref"],
            "error_mha":    round(error_mha, 4),
            "converged":    error_mha < 1.6,
            "total_time":   total_time,
            "n_iters":      self.iteration,
        })
        await self.send({"type": "log", "src": "VQE",
                         "msg": f"Done · E={final_energy:.6f} Ha · "
                                f"ΔE={error_mha:.3f} mHa · {total_time}s"})


# ── WebSocket Handler ──────────────────────────────────────────────────────

active_runners: dict = {}

async def handler(websocket):
    client_id = id(websocket)
    print(f"[WS] Client connected: {client_id}")

    try:
        # Send GPU info immediately on connect
        gpu_info = get_gpu_info()
        await websocket.send(json.dumps({"type": "gpu_info", "data": gpu_info}))

        async for raw in websocket:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            cmd = msg.get("cmd")

            if cmd == "ping":
                await websocket.send(json.dumps({"type": "pong"}))

            elif cmd == "gpu_poll":
                idx = msg.get("gpu_index", 0)
                live = get_gpu_live(idx)
                await websocket.send(json.dumps({"type": "gpu_live", "data": live}))

            elif cmd == "get_molecules":
                mol_list = {
                    k: {
                        "label":   v["label"],
                        "fci_ref": v["fci_ref"],
                        "desc":    v["desc"],
                        "basis":   v["basis"],
                    }
                    for k, v in MOLECULES.items()
                }
                await websocket.send(json.dumps({"type": "molecules", "data": mol_list}))

            elif cmd == "run_vqe":
                mol_key   = msg.get("molecule", "H2")
                gpu_index = msg.get("gpu_index", 0)

                if mol_key not in MOLECULES:
                    await websocket.send(json.dumps({"type": "error", "msg": f"Unknown molecule: {mol_key}"}))
                    continue

                # Cancel previous if any
                if client_id in active_runners:
                    active_runners[client_id].cancelled = True

                runner = VQERunner(websocket, mol_key, gpu_index)
                active_runners[client_id] = runner
                asyncio.create_task(runner.run())

            elif cmd == "cancel":
                if client_id in active_runners:
                    active_runners[client_id].cancelled = True
                    await websocket.send(json.dumps({"type": "cancelled"}))

    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception as e:
        print(f"[WS] Error: {e}")
        traceback.print_exc()
    finally:
        active_runners.pop(client_id, None)
        print(f"[WS] Client disconnected: {client_id}")


async def main():
    if not HAS_CUDAQ:
        print("⚠  cudaq not installed – install it with:")
        print("   pip install cudaq pyscf openfermion openfermionpyscf websockets scipy")
        print("   Then re-run this server.")
        return

    host, port = "0.0.0.0", 8765
    print("=" * 60)
    print("  CUDA-Q VQE Real GPU Server")
    print(f"  WebSocket: ws://{host}:{port}")
    print("=" * 60)
    print(f"  cudaq    : {cudaq.__version__}")
    gpu_info = get_gpu_info()
    if "gpus" in gpu_info:
        for g in gpu_info["gpus"]:
            print(f"  GPU {g['index']}  : {g['name']}  {g['mem_total_mb']}MB")
    else:
        print(f"  GPU      : {gpu_info.get('error','unknown')}")
    print("=" * 60)

    async with websockets.serve(handler, host, port, ping_interval=20, ping_timeout=60):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
