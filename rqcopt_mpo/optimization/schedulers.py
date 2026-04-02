from typing import Optional

class ReduceLROnPlateau:
    """
    Learning rate scheduler that reduces the learning rate when a metric has 
    stopped improving.
    """
    def __init__(
        self, 
        factor: float = 0.5, 
        patience: int = 10, 
        min_lr: float = 1e-7, 
        min_delta: float = 1e-9, 
        verbose: bool = True
    ):
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.verbose = verbose
        
        self.best_loss = float('inf')
        self.stalled_steps = 0
        self.lr = None

    def step(self, loss: float, optimizer) -> float:
        """
        Updates the learning rate of the optimizer based on the current loss.
        """
        if self.lr is None:
            self.lr = optimizer.lr

        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.stalled_steps = 0
        else:
            self.stalled_steps += 1

        if self.stalled_steps >= self.patience:
            old_lr = self.lr
            new_lr = max(self.lr * self.factor, self.min_lr)
            
            if new_lr < old_lr:
                self.lr = new_lr
                optimizer.lr = self.lr
                if self.verbose:
                    print(f"Learning rate reduced from {old_lr:.2e} to {self.lr:.2e}")
            
            self.stalled_steps = 0
        
        return self.lr
