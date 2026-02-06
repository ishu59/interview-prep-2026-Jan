def binary_search(nums, target):
    lo = 0
    hi = len(nums) - 1

    while lo < hi:
        mid = lo + (hi - lo) // 2

        if check_condition(nums, mid, target):
            hi = mid
        else:
            lo = mid + 1

        if target == nums[lo]:
            return lo
        
        return -1
    
    def check_condition(nums, mid, target):
        return nums[mid] >= target
    


def binary_search(nums, target):
    lo = 0
    hi = len(nums) - 1

    while (lo < hi):
        mid = lo + (hi - lo)//2 # (hi+lo)//2
        if (nums[mid] >= target): # if (check_condition()):
            hi = mid
        else:
            lo = mid + 1

    if lo < len(nums) and nums[lo] == target:
        return lo
    return -1


def check_condition():
    return False
