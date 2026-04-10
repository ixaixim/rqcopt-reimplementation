from pathlib import Path
import sys

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.utils import load_experiment_json
from rqcopt_mpo.circuit.circuit_dataclasses import Circuit

def test_read_all_circuits():
    # Use the folder provided by the user
    data_dir = REPO_ROOT / "experiments" / "rzz_ising_experiment" / "data"
    
    if not data_dir.exists():
        print(f"Error: Folder not found at {data_dir}")
        return

    json_files = list(data_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {data_dir}")
        return

    print(f"Found {len(json_files)} JSON files in {data_dir}")
    for file_path in json_files:
        print(f"\n--- Loading experiment from {file_path.name} ---")
        try:
            data = load_experiment_json(file_path)
            
            print("Experiment Metadata:")
            print(f"  Method: {data.get('method')}")
            print(f"  Final Loss: {data.get('final_loss')}")
            
            circuit = data.get("circuit")
            if isinstance(circuit, Circuit):
                print("Circuit loaded successfully!")
                print(f"  Sites: {circuit.n_sites}, Layers: {circuit.num_layers}, 2Q Layers: {circuit.num_2q_layers}")
            else:
                print("Error: Circuit not found or failed to reconstruct.")
        except Exception as e:
            print(f"Failed to load {file_path.name}: {e}")

if __name__ == "__main__":
    test_read_all_circuits()
