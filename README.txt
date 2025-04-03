03.04.2025
About the cons of current approach to handle circuit gates:
Lack of Metadata: A raw tensor doesn't inherently carry information about which qubits it acts on, what kind of gate it is (e.g., RZZ, CNOT, Identity), or any associated parameters (like rotation angles). This information is implicit in how and where the tensor was generated and placed in the list.

Readability/Maintainability: Code that uses these tensors needs comments or relies on the developer remembering the context (e.g., "the tensor at layer[i][j] acts on qubits (k, k+1)"). This makes debugging and modification harder.

Manipulation Complexity: Operations like "remove all identity gates" or "find all gates acting on qubit 5" require iterating and potentially complex logic based on list indices and implicit conventions.


Add dataclasses (python 3.7+) clear structure, type hints, mutable by default, easy to add methods, minimal boilerplate, good readability. Overhead is very low. 