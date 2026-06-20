def solve_lcs(text1, text2):

    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs_list = []
    i, j = m, n
    while i > 0 and j > 0:
        if text1[i-1] == text2[j-1]:
            lcs_list.append(text1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    
    lcs_str = ''.join(reversed(lcs_list))
    return dp[m][n], lcs_str

if __name__ == "__main__":
    str1 = "ABCBDAB"
    str2 = "CDCABA"
    length, sequence = solve_lcs(str1, str2)
    print(f"字符串1: {str1}")
    print(f"字符串2: {str2}")
    print(f"最长公共子序列长度: {length}")
    print(f"最长公共子序列内容: {sequence}")