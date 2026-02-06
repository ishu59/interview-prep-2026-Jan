def left_bin_search(nums, target):

    lo = 0
    hi = len(nums) - 1

    while lo < hi:
        mid = lo + (hi - lo) // 2

        if nums[mid] >= target:
            mid = hi
        else:
            mi = lo + 1
    
    return lo if nums[lo] == target  else -1 