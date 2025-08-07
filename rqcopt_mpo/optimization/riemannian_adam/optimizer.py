
from .riemannian_adam import RiemannianAdam
from .geometry import project_to_tangent
from rqcopt_mpo.optimization.gradient import cost_and_euclidean_grad

# NOTE: consider making an object optimizer later on (for checkpointing, schedulers, multiple objectives, multi-optimizer, etc.)
def optimize(
    circuit,
    reference_mpo,
    *,
    lr: float = 1e-3,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    clip_grad_norm: float = None,
    max_steps: int = 1000,
    callback: callable = None
):
    opt = RiemannianAdam(
        lr=lr,
        betas=betas,
        eps=eps,
        clip_grad_norm=clip_grad_norm
    )

    state = opt.init(U)  # U: (G, d, d) stacked 2-qubit gates
    # TODO: set init_vertical_sweep and alternate it.
    init_vertical_sweep = 'bottom-up'

    def dir_for_step(l: int) -> str:
        if init_vertical_sweep not in ('left_to_right', 'right_to_left'):
            raise ValueError("vertical_sweep must be one of: 'bottom-up', 'top-down'")
        if l % 2 == 0:
            return init_vertical_sweep
        return 'bottom-up' if init_vertical_sweep == 'top-down' else 'bottom-up'

    for it in range(max_steps):
        loss, grad_e = cost_and_euclidean_grad(circuit, reference_mpo, vertical_sweep=dir_for_step(it))
        U, state, stats = opt.step(U, grad_e, state)
        # write U back into your circuit object here
        # write inplace
        # TODO.
        # log loss, stats["grad_norm"], etc.
        vertical_sweep = 'bottom-up' if vertical_sweep is 'top-down' else 'bottom-up'
        # NOTE: you can have also an alternating sweep based on iteration parity.
        return loss # for now only loss