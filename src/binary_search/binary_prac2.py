

def left_biased_bin_search(nums, condition):
    lo = 0
    hi = len(nums) - 1

    while lo < hi:
        mid =  lo + (hi - lo) // 2
        if condition.isValid(nums, hi, lo, mid):
            hi = mid
        else:
            lo = mid + 1
            
    return lo


def right_biased_bin_search(nums, condition):
    lo = 0 
    hi = len(nums) - 1

    while lo < hi:
        mid = lo + (hi - lo + 1) // 2

        if condition.isValid(nums, hi, lo, mid):
            lo = mid
        else:
            hi = mid - 1