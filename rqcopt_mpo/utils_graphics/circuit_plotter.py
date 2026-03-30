import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import List, Optional, Union
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from rqcopt_mpo.circuit.circuit_dataclasses import Circuit, Gate

def plot_circuit_heatmap(
    circuit: Circuit,
    gate_values: Optional[List[float]] = None,
    title: str = "Circuit Heatmap",
    cmap: str = "viridis",
    ax: Optional[plt.Axes] = None,
    show_colorbar: bool = True,
    gate_spacing: float = 0.8,
    layer_spacing: float = 0.8,
    log_scale: bool = False
):
    """
    Plots the quantum circuit, coloring each gate according to the corresponding value in `gate_values`.

    Args:
        circuit: The Circuit object to plot.
        gate_values: A list of scalar values, one for each gate in the circuit.
                     The order must match the order of gates in `[g for l in circuit.layers for g in l.gates]`.
                     If None, gates are colored uniformly.
        title: Title of the plot.
        cmap: Matplotlib colormap name.
        ax: Matplotlib Axes to plot on. If None, a new figure is created.
        show_colorbar: Whether to display a colorbar for the values.
        gate_spacing: Vertical size of the gate relative to qubit spacing (default 1.0).
        layer_spacing: Horizontal size of the gate relative to layer spacing (default 1.0).
        log_scale: If True, use a logarithmic scale for the colormap.
    """
    
    # Flatten the gates to match the expected structure of gate_values
    all_gates = [gate for layer in circuit.layers for gate in layer.gates]
    
    if gate_values is not None:
        if len(gate_values) != len(all_gates):
            raise ValueError(
                f"Length of gate_values ({len(gate_values)}) does not match "
                f"number of gates in circuit ({len(all_gates)})."
            )
        values = np.array(gate_values)
    else:
        values = np.zeros(len(all_gates)) # Default value

    # Create figure if ax is not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(len(circuit.layers) * 0.8, 8), max(circuit.n_sites * 0.5, 4)))
    else:
        fig = ax.get_figure()

    # Normalize values for color mapping
    if gate_values is not None:
        # Handle NaNs and Infs for range calculation
        finite_values = values[np.isfinite(values)]
        
        if log_scale:
            # Filter strictly positive values for LogNorm
            positive_values = finite_values[finite_values > 0]
            
            if len(positive_values) > 0:
                min_val = np.min(positive_values)
                max_val = np.max(finite_values) # Use max of all finite to catch large 0-like noise if any? No, max of positive.
                # Actually, if there are large negative values, LogNorm will mask them.
                # Let's stick to positive range.
                max_val = np.max(positive_values)
                
                if min_val >= max_val:
                    # Avoid singular range
                    max_val = min_val * 10.0 if min_val != 0 else 1.0
                    
                norm = mcolors.LogNorm(vmin=min_val, vmax=max_val)
            else:
                # Fallback if no positive data found (e.g. all 0 or all NaN)
                norm = mcolors.Normalize(vmin=0, vmax=1)
        else:
            if len(finite_values) > 0:
                min_val = np.min(finite_values)
                max_val = np.max(finite_values)
                if min_val >= max_val:
                    max_val = min_val + 1.0 # arbitrary shift
                norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
            else:
                norm = mcolors.Normalize(vmin=0, vmax=1)
                
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        # Handle NaN/Inf in the plot by setting them to a distinct color or transparent
        # By default to_rgba handles them, but let's be explicit if needed.
    else:
        # If no values provided, use a fixed color (e.g., light blue)
        norm = mcolors.Normalize(vmin=0, vmax=1)
        m = None # Handle manually
        default_color = "skyblue"

    gate_idx = 0
    
    # Grid settings
    # X axis: Layer index
    # Y axis: Qubit index (inverted usually looks better: qubit 0 at top)
    # Let's put qubit 0 at y=0, qubit N at y=N, but invert y-axis at the end.
    
    for layer in circuit.layers:
        # x center for this layer
        x_center = layer.layer_index
        
        for gate in layer.gates:
            val = values[gate_idx]
            color = m.to_rgba(val) if m is not None else default_color
            
            # Determine gate bounds
            qubits = gate.qubits
            q_min = min(qubits)
            q_max = max(qubits)
            
            # We draw a rectangle.
            # Width: layer_spacing
            # Height: (q_max - q_min) + gate_spacing (to cover the qubits)
            # Center it at x_center, and vertically around the qubits.
            
            # If single qubit: q_min == q_max. Height should be gate_spacing.
            # If two qubit: Height covers the distance plus some padding.
            
            # Let's say qubit lines are at integers 0, 1, 2...
            # A gate on qubit q should span [q - w/2, q + w/2] roughly.
            
            width = layer_spacing
            height = (q_max - q_min) + gate_spacing
            
            # Bottom-left corner calculation
            # x_start = x_center - width / 2
            # y_start = q_min - gate_spacing / 2
            
            # Rect(xy, width, height)
            rect = patches.Rectangle(
                (x_center - width / 2, q_min - gate_spacing / 2),
                width,
                height,
                linewidth=1,
                edgecolor='black',
                facecolor=color,
                zorder=2
            )
            ax.add_patch(rect)
            
            # Optional: Add text for gate name or value?
            # Keeping it simple for now as requested.
            
            gate_idx += 1

    # Draw qubit lines (horizontal)
    # From x = -0.5 to x = num_layers - 0.5
    x_min = -0.5
    x_max = (circuit.layers[-1].layer_index if circuit.layers else 0) + 0.5
    
    for q in range(circuit.n_sites):
        ax.hlines(q, x_min, x_max, color='gray', linestyle='--', linewidth=0.5, zorder=1)

    # Set limits and labels
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-1, circuit.n_sites) # Inverted later, so -1 is top visual margin
    
    # Invert y axis so qubit 0 is at the top
    ax.invert_yaxis()
    
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Qubit Index")
    ax.set_yticks(range(circuit.n_sites))
    ax.set_title(title)
    
    # Add colorbar if requested and values were provided
    if show_colorbar and m is not None:
        plt.colorbar(m, ax=ax, label="Value")

    return fig, ax
