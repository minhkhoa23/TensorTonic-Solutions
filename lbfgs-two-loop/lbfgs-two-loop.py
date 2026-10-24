def lbfgs_direction(grad: list, s_list: list, y_list: list) -> list:
    """
    Returns the L-BFGS descent direction from the stored history.
    """
    # Write code here
    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    n = len(grad)
    m = len(s_list)

    # No history: steepest descent
    if m == 0:
        return [-g for g in grad]

    # Step 1: Compute rho
    rho = [
        1.0 / dot(y_list[i], s_list[i])
        for i in range(m)
    ]

    # Step 2: Backward pass
    q = grad.copy()
    alpha = [0.0] * m

    for i in range(m - 1, -1, -1):
        alpha[i] = rho[i] * dot(s_list[i], q)

        for j in range(n):
            q[j] -= alpha[i] * y_list[i][j]

    # Step 3: Initial Hessian scaling
    s = s_list[-1]
    y = y_list[-1]

    gamma = dot(s, y) / dot(y, y)

    r = [gamma * value for value in q]

    # Step 4: Forward pass
    for i in range(m):
        beta = rho[i] * dot(y_list[i], r)

        for j in range(n):
            r[j] += s_list[i][j] * (alpha[i] - beta)

    # Step 5: Return descent direction
    return [-value for value in r]