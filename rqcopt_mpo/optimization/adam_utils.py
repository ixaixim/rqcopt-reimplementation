import numpy as np

# callbacks:
def make_early_stop(patience: int = 10, min_delta: float = 1e-8, target_loss: float = None):
    best_loss = [np.inf]
    stalled_steps = [0]
    def early_stop(*, step: int, loss: float, **_):
        if target_loss is not None and loss < target_loss:
            print(f"Target loss {target_loss} reached at step {step}")
            return True

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