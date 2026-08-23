import numpy as np

def adadelta_step(w: list, grad: list, E_grad_sq: list, E_update_sq: list, rho: float = 0.9, eps: float = 1e-6) -> dict:
    """Return updated parameters and AdaDelta state."""
    # Write code here
    w = np.array(w, dtype=float)
    grad = np.array(grad, dtype=float)
    E_grad_sq = np.array(E_grad_sq, dtype=float)
    E_update_sq = np.array(E_update_sq, dtype=float)

    # 1. Update squared-gradient average
    new_E_grad_sq = (
        rho * E_grad_sq
        + (1 - rho) * grad**2
    )

    # 2. Compute parameter change
    update = -(
        np.sqrt(E_update_sq + eps)
        / np.sqrt(new_E_grad_sq + eps)
    ) * grad

    # 3. Update squared-update average
    new_E_update_sq = (
        rho * E_update_sq
        + (1 - rho) * update**2
    )

    # 4. Update parameters
    new_w = w + update

    return {
        "new_w": new_w,
        "new_E_grad_sq": new_E_grad_sq,
        "new_E_update_sq": new_E_update_sq
    }