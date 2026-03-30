import pytest
import numpy as np
import matplotlib.pyplot as plt
from rqcopt_mpo.circuit.circuit_dataclasses import Circuit, GateLayer, Gate
from rqcopt_mpo.utils_graphics.circuit_plotter import plot_circuit_heatmap

def test_plot_circuit_heatmap_runs():
    """
    Smoke test to ensure plot_circuit_heatmap runs without errors on a dummy circuit.
    """
    n_sites = 4
    
    # Layer 0: Odd (0,1), (2,3)
    l0 = GateLayer(
        layer_index=0,
        is_odd=True,
        gates=[
            Gate(matrix=np.eye(4), qubits=(0, 1), layer_index=0, name="G1"),
            Gate(matrix=np.eye(4), qubits=(2, 3), layer_index=0, name="G2")
        ]
    )
    
    # Layer 1: Even (1,2)
    l1 = GateLayer(
        layer_index=1,
        is_odd=False,
        gates=[
            Gate(matrix=np.eye(4), qubits=(1, 2), layer_index=1, name="G3"),
            Gate(matrix=np.eye(2), qubits=(0,), layer_index=1, name="G4"),
            Gate(matrix=np.eye(2), qubits=(3,), layer_index=1, name="G5")
        ]
    )
    
    circuit = Circuit(n_sites=n_sites, layers=[l0, l1])
    
    # Scalars: 2 gates in L0, 3 gates in L1 = 5 values
    scalars = [0.1, 0.9, 0.5, 0.2, 0.8]
    
    # Test with scalars
    fig, ax = plot_circuit_heatmap(circuit, gate_values=scalars, title="Test Plot")
    assert fig is not None
    assert ax is not None
    plt.close(fig) # Cleanup
    
    # Test without scalars
    fig2, ax2 = plot_circuit_heatmap(circuit, gate_values=None, title="Test Plot No Scalars")
    assert fig2 is not None
    plt.close(fig2)

def test_plot_circuit_heatmap_mismatched_lengths():
    """
    Test that an error is raised if the number of scalars doesn't match the number of gates.
    """
    l0 = GateLayer(layer_index=0, is_odd=True, gates=[Gate(matrix=np.eye(2), qubits=(0,), layer_index=0)])
    circuit = Circuit(n_sites=2, layers=[l0])
    
    with pytest.raises(ValueError, match="Length of gate_values"):
        plot_circuit_heatmap(circuit, gate_values=[0.1, 0.2]) # 2 values for 1 gate
