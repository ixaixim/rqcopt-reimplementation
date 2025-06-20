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


Update: Added deepcopies of Gate, GateLayer, Circuit object.
Questions: 
during contractions, how would you treat None gates (i.e. no gates at circuit)? 
Two alternatives:
- more performing code (but longer code): add a case that handles None gates.
- more clean code (but longer run time): have an identity tensor in place of the None gate. 
It boils down to whether this slows down the overall computation or not. Are identity contractions less expensive than regular contractions?

Unit tests added: 
test_identity: target: id mpo, init circuit: identical circuit. Results:
[-2.3841858e-07 -2.3841858e-07  0.0000000e+00 -2.3841858e-07
  0.0000000e+00 -2.3841858e-07 -2.3841858e-07 -2.3841858e-07
  0.0000000e+00 -2.3841858e-07  0.0000000e+00 -2.3841858e-07]
  Why does it not stay zero? 

Canonicalize MPO:
how to treat the norm? There is no guarantee that the last tensor (resulting from upper triangular matrix) will be real. 
Isabel simply assumes it is real: if nrm<0: nrm=-nrm; R,Q=-R,-Q  # Flip sign of last tensor and dummy (norm) tensor to ensure positive norm
In case Norm is not real: absorb phase of R into Q and keep the 

30.05.2025
NOTE for Isabel: Review QR decomposition when canonicalizing tensors. It is not guaranteed that R diagonal entries are real, according to numpy/jax.
                  Therefore we need to manually enforce positiveness (https://www.geeksforgeeks.org/qr-decomposition-in-machine-learning/?utm_source=chatgpt.com).
                  Note: this will not affect the use case because we always absorb R. 


Added: test merge MPO with layer (check matrix result, check left/right isometry conserved on all sites.)
Result: every TN operation in core_ops.py is correct.
Added: test trace with MPO.
Result: Found out that before contracting init_circuit with ref_mpo to compute Tr(ref_mpo^dag init_circuit) we need to "dagger" the ref_mpo (or the circuit, depends on the order).

11.06.2025
Adding jax.config.update("jax_enable_x64", True) everywhere fixes the problem of single-to-double precision.
NOTE: add before jax.numpy import. Made a jax_config file.

When testing accuracy of a contraction, how much should I set the tolerance?
Example at the moment:         assert jnp.allclose(trace, matrix_trace, rtol=1e-6, atol=1e-8), f"{trace} != {matrix_trace}"


NOTE: jax numpy DOES NOT DO AUTOMATIC UPCASTING, you have to make sure every matrix you deal with is in the highest precision.

Added: test for holomorphic function and of gradient up to machine precision.
Added: right to left sweep. Test for shallow circuits with same layout, but different gates.

Q for isabel:
top-bottom sweep after the bottom-top sweep: it makes sense to store the recently updated bottom envs from the bottom-top sweep. 
We use these layers in the top-bottom. Right? I.e. no need for a new computation from zero.

Q for Isabel: 
overview of how you do trotterization? E.g. for Heisenberg terms at site i,j do you have a gate for (omitting exp) X_iX_j + Y_iY_j + Z_iZ_j or do you have each component on a differnt layer (X_iX_j layer 0, Y_iY_j layer 1, etc.)?

Q: Weyl decomposition as we are optimizing. Should we do it as soon as we update the gate? Or after we update all gates? 
Note: Weyl decomposition expands a N-layer brickwall circuit to a 2N+1 layer circuit. When considering the benefits of compression, we need to account for this negative aspect.

Q: I am doing bottom-up then top-down sweep. The Gibbs paper does it opposite. SHould I consider imitating?
Q: should I report the loss only after the sweep?

02.07
Q: tests for compression are failing for deeper circuits. Possible reason: error accumulation. error diverges for deeper (depth 30 layers) circuits because of splitting tensors when merging layers.
Possible solution: maybe normalization?
A: it was a conjugate issue in the optimizer. Environment has been redefined such that everything is consistent: E^* = d/dG Tr(U_ref^dag U_QC) = d/dG Tr(E^dag G). E is defined s.t. SVD gate update is U@Vh.

Q: can we normalize the QC as well as the target?

commit:
Added Heisenberg trotterization for a circuit (with error plot test) in trotter/ folder. Heisenberg model build hamiltonian (heisenberg/ folder).
Important fix to gradient function (a conjugate was missing) so that local SVD update compression leads to monotonic loss. All test work, even for long circuits (this was an issue before the fix).
Updated tests that were relying on old gradient function.
Add 'normalize' method in MPO.
Fix optimizer (pass left_to_right is now coherent with pass right_to_left). 
Add Weyl decomposition circuit from brickwall. (weyl_circuit_builder.py) 
Add archived. 