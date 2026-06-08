def solve_knapsack(weights, values, capacity):

    n = len(weights)
    dp = [0] * (capacity + 1)
    
    for i in range(n):
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    
    return dp[capacity]

if __name__ == "__main__":
    weights = [2, 3, 4, 5]
    values = [3, 4, 5, 6]
    capacity = 10
    
    max_val = solve_knapsack(weights, values, capacity)
    print(f"物品重量: {weights}")
    print(f"物品价值: {values}")
    print(f"背包容量: {capacity}")
    print(f"最大价值: {max_val}")