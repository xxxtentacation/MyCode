def solve_max_subarray(nums):

    if not nums:
        return 0, -1, -1
        
    current_max = global_max = nums[0]
    start = end = temp_start = 0
    
    for i in range(1, len(nums)):
        num = nums[i]
        if num > current_max + num:
            current_max = num
            temp_start = i
        else:
            current_max = current_max + num
        
        if current_max > global_max:
            global_max = current_max
            start = temp_start
            end = i
            
    return global_max, start, end

if __name__ == "__main__":
    nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    max_sum, start_idx, end_idx = solve_max_subarray(nums)
    
    print(f"原始数组: {nums}")
    print(f"最大子段和: {max_sum}")
    print(f"子段起始索引: {start_idx}, 结束索引: {end_idx}")
    print(f"对应的子段: {nums[start_idx:end_idx+1]}")