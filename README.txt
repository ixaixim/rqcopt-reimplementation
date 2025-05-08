03.04.2025
About the cons of current approach to handle circuit gates:
Lack of Metadata: A raw tensor doesn't inherently carry information about which qubits it acts on, what kind of gate it is (e.g., RZZ, CNOT, Identity), or any associated parameters (like rotation angles). This information is implicit in how and where the tensor was generated and placed in the list.

Readability/Maintainability: Code that uses these tensors needs comments or relies on the developer remembering the context (e.g., "the tensor at layer[i][j] acts on qubits (k, k+1)"). This makes debugging and modification harder.

Manipulation Complexity: Operations like "remove all identity gates" or "find all gates acting on qubit 5" require iterating and potentially complex logic based on list indices and implicit conventions.


Add dataclasses (python 3.7+) clear structure, type hints, mutable by default, easy to add methods, minimal boilerplate, good readability. Overhead is very low. 

__init__.py is used to turn a folder of .py into an importable package (e.g. for simpler user experience, e.g. 'from circuit import Gate')
Ideal structure of the folder:
RQCOPT/
├── main.py                  # Main script to run experiments
├── config.yaml              # Configuration file (parameters, paths, etc.)
│
├── circuit/
│   ├── __init__.py
│   ├── circuit_dataclasses.py # Defines Gate, GateLayer, Circuit classes
│   ├── circuit_builder.py     # Functions to create circuits (get_initial_gates, etc.)
│   ├── circuit_ops.py         # Functions operating on circuits (absorb_single_qubit_gates - or make it a Circuit method)
│   └── weyl_decomposition.py  # Weyl decomposition logic
│
├── mpo/
│   ├── __init__.py
│   ├── mpo_dataclass.py       # Defines MPO class (including canonicalization methods)
│   └── mpo_builder.py         # Functions to construct MPOs (e.g., from Trotter formulas)
│
├── optimization/
│   ├── __init__.py
│   ├── optimizer.py           # Contains optimize_swap_network_circuit_RieADAM
│   ├── gradient.py            # Contains get_riemannian_gradient_and_cost_function
│   ├── riemannian_ad.py       # RieADAM implementation (or import from library)
│   └── geometry.py            # Retractions, projections, inner products (retract_unitary, etc.)
│
├── tensor_network/
│   ├── __init__.py
│   └── core_ops.py            # Low-level TN contractions (e.g., compute_full_gradient, apply_mpo_to_mps if needed)
│
└── utils/
    ├── __init__.py
    ├── plotting.py            # plot_loss function
    ├── saving.py              # save_optimized_model function
    └── timing.py              # get_duration function


# questions for Isabel (18.05.2025):
# in tn_brickwall_methods, in get_riemannian_gradient_and_cost_function, why do you compute cost_F in two different ways depending on whether the reference MPO is normalized or not?
# Note: When calculating the cost, Isabel computes the normalized Frobenius norm difference (also: should we divide by two? to normalize it). On the other hand, the Gibbs paper computes the Hilbert-Schmidt-Test (HST), where the cost is naturally in (0,1). What is the difference?
# Frobenius norm of the difference: represents a squared euclidean distance measure between the two operators in matrix space. Has the term Re(Tr(U^dagW))
# HST: insensitive to global phases. Has the term |Tr(U^dagW)|^2. Relates to the average fidelity when the unitaries act on random quantum states. 

# What I need to understand: 
 # Get Frobenius norm and Hilbert-Schmidt test from overlap -> normalization holds for normalized reference!
    if reference_is_normalized: const_F=2**int(n_sites/2)
    else: const_F=2**n_sites
    cost_F = 2 - 2*overlap.real/const_F  # Frobenius norm

I.e. why does each site contribute sqrt(2) instead of 2 to the Hilbert space scaling, for a normalized MPO? does it have to do with the norm of the identity, which is sqrt(2^n)


Why using dataclass:
Instead of writing boilerplate code for methods like __init__, __repr__, and __eq__, you simply annotate your class with @dataclass, and these methods are automatically generated based on the class’s annotations.
Note: plain dataclasses are mutable by default, whereas JAX workflows are immutable. So if we want to use dataclasses as containers for JAX arrays, consider @dataclass(frozen=True). 
The frozen=True flag ensures that instances of the class are immutable after creation, which can help maintain the functional purity that JAX favors.

Some libraries in the JAX ecosystem (for example, Flax) offer their own dataclass decorators (e.g., flax.struct.dataclass). These variants might offer additional functionality or optimizations that are tailored to the kinds of immutable, functionally pure data containers that JAX transformations require.


Question for Isabel: (01.05.2025)
- Since the loss is Tr(U_circuit U_MPO^dagger), do we need to take the Dagger of the Ref MPO? Or is this already provided in the correct form? 
- Clarification on the update step process (vertical and horizontal sweep and caching system). 
- Possible clarification on the environment tensor ordering (do we take the transpose?) -> verify by doing finite differences. 
- Environment: do we have to take the conjugate after "cutting out the gate"?