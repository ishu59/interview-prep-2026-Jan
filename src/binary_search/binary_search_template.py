
# Ref: https://leetcode.com/discuss/general-discussion/786126/Python-Powerful-Ultimate-Binary-Search-Template.-Solved-many-problems

'''
Correctly initialize the boundary variables left and right to specify search space. Only one rule: set up the boundary to include all possible elements;
Decide return value. Is it return left or return left - 1? Remember this: after exiting the while loop, left is the minimal k​ satisfying the condition function;
Design the condition function. This is the most difficult and most beautiful part. Needs lots of practice.
'''
def binary_search(array) -> int:
    def condition(value) -> bool:
        pass

    left, right = min(search_space), max(search_space) # could be [0, n], [1, n] etc. Depends on problem
    while left < right:
        mid = left + (right - left) // 2
        if condition(mid):
            right = mid
        else:
            left = mid + 1
    return left


def binary_search(arrya):
    def condition(value):
        pass

    left, right = min(saerch_sapce), max(search_space)
    while left < right:
        mid = left + ((right - left) // 2)
        if condition(mid):
            right = mid
        else:
            left = mid + 1

    return left


def bin_search(array):
    def condition(value):
        pass

    left, right = min(search_space), max(search_space)
    while left < right:
        mid = left + (right -left) // 2
        if condition(mid):
            mid = right
        else:
            mid  =  left + 1

    return left