import numpy as np

# callbacks:
def make_early_stop(patience: int = 10, min_delta: float = 1e-8):
    best_loss = [np.inf]
    stalled_steps = [0]
    def early_stop(patience=10, min_delta=1e-8, *, step: int, loss: float, **_):
        if loss < best_loss[0] - min_delta:
            best_loss[0] = loss
            stalled_steps[0] = 0
        else:
            stalled_steps[0] += 1

        if stalled_steps[0] >= patience:
            print(f"Early stopping at step {step}")
            return True
        return False
    return early_stop