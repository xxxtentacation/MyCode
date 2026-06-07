def solve_s_batch_forward_dp(p, w, d, s):

    n = len(p)
    if n == 0:
        return 0

    # Sort jobs by due date (non-decreasing)
    jobs = sorted(range(n), key=lambda i: d[i])
    p_sorted = [p[i] for i in jobs]
    w_sorted = [w[i] for i in jobs]
    d_sorted = [d[i] for i in jobs]


    max_time = sum(p_sorted) + n * s
    unique_dues = sorted(set(d_sorted))


    from collections import defaultdict
    F = [defaultdict(lambda: float('inf')) for _ in range(n + 1)]


    F[0][(0, -float('inf'))] = 0

    for j in range(n):
        p_j = p_sorted[j]
        w_j = w_sorted[j]
        d_j = d_sorted[j]

        for (t_prev, dd_prev), cost_prev in F[j].items():
            if cost_prev == float('inf'):
                continue

            t_new_tardy = t_prev + s + p_j
            cost_new_tardy = cost_prev + w_j
            F[j + 1][(t_new_tardy, dd_prev)] = min(F[j + 1][(t_new_tardy, dd_prev)], cost_new_tardy)

            new_completion_time = t_prev + p_j
            if new_completion_time <= min(d_j, dd_prev) if dd_prev != -float('inf') else new_completion_time <= d_j:
                F[j + 1][(new_completion_time, min(d_j, dd_prev) if dd_prev != -float('inf') else d_j)] = min(
                    F[j + 1][(new_completion_time, min(d_j, dd_prev) if dd_prev != -float('inf') else d_j)],
                    cost_prev
                )

            new_t_early = t_prev + s + p_j
            if new_t_early <= d_j:
                F[j + 1][(new_t_early, d_j)] = min(
                    F[j + 1][(new_t_early, d_j)],
                    cost_prev
                )

    min_cost = float('inf')
    for cost in F[n].values():
        if cost < min_cost:
            min_cost = cost

    if min_cost == float('inf'):
        return float('inf')

    return min_cost
# Example input
p = [3, 2, 4]       # processing times
w = [5, 3, 4]       # weights
d = [6, 7, 9]       # due dates
s = 1               # setup time

result = solve_s_batch_forward_dp(p, w, d, s)
print("Optimal cost (Forward DP):", result)