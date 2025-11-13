from pathlib import Path
from experiments.utils import load_all_npz, plot_all_losses

here = Path(__file__).resolve().parent # dir that contains this file
base_dir = here.parent  # go up one dir
data_dir = base_dir / "data"  

all_data = load_all_npz(Path(data_dir))
plot_all_losses(all_data, out_name="loss_plot", out_dir=here)
