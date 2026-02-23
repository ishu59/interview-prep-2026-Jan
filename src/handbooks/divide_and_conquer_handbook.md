# The Complete Divide and Conquer Handbook
> A template-based approach for mastering divide and conquer in coding interviews

**Philosophy:** Divide and Conquer is not about splitting arrays in half for fun. It's about reducing a problem of size N to smaller independent subproblems, solving them recursively, and combining results — achieving better-than-brute-force time by avoiding redundant work.

---

## Table of Contents
1. [Understanding the Core Philosophy](#core-philosophy)
2. [The 4 Master Templates](#master-templates)
3. [Pattern Classification Guide](#pattern-guide)
4. [Complete Pattern Library](#patterns)
5. [Post-Processing Reference](#post-processing)
6. [Common Pitfalls & Solutions](#pitfalls)
7. [Problem Recognition Framework](#recognition)
8. [Interview Preparation Checklist](#checklist)

---

<a name="core-philosophy"></a>
## 1. Understanding the Core Philosophy

### First Principles

- **The Tournament Bracket**: To find the champion among 64 players, you don't compare everyone against everyone (2016 matches). You split into two halves, find each half's champion, then play one final. That's O(n) vs O(n^2).
- **The Sorted Stack of Papers**: To sort 1000 papers, split into two piles of 500, sort each, then merge. Merging two sorted piles is easy — always pick the smaller top sheet. That's merge sort.

### No-Jargon Translation

- **Divide**: split the problem into smaller pieces (usually halves)
- **Conquer**: solve each piece recursively (base case: problem is trivially small)
- **Combine**: merge the sub-solutions into the final answer
- **Recurrence relation**: a formula describing runtime in terms of smaller inputs (e.g., T(n) = 2T(n/2) + O(n))
- **Master theorem**: a shortcut to solve common recurrence relations

### Mental Model

> "Divide and Conquer is like a manager who breaks a big project into independent tasks, delegates each to a team, and then combines their deliverables — the magic is that combining is cheaper than doing the whole project at once."

---

### Why Divide and Conquer?

The brute-force approach to many problems checks all possible configurations:

```python
# O(n^2) - Compare every pair
for i in range(n):
    for j in range(i + 1, n):
        process(arr[i], arr[j])
```

Divide and Conquer reduces this by **splitting, solving independently, and combining**:

```python
# O(n log n) - Divide, solve halves, combine
def solve(arr, lo, hi):
    # Why >= and not just ==? A single element (lo == hi) is already
    # sorted/solved, and lo > hi means an empty range (no work to do).
    # Both are base cases that need no further division.
    if lo >= hi:
        return base_case
    # Why not (lo + hi) // 2? If lo and hi are both large,
    # lo + hi can overflow in languages like Java/C++.
    # lo + (hi - lo) // 2 avoids this by never exceeding hi.
    # In Python, integers don't overflow, but this is a good habit.
    mid = (lo + hi) // 2
    left_result = solve(arr, lo, mid)
    right_result = solve(arr, mid + 1, hi)
    return combine(left_result, right_result)
```

### The Key Insight: Independence of Subproblems

D&C works when:
1. The problem can be broken into **independent** subproblems (no overlap)
2. Subproblems have **identical structure** to the original
3. The **combine step** is efficient (cheaper than re-solving from scratch)

If subproblems overlap, you need Dynamic Programming instead.

### Visual Understanding of D&C

```
Original Problem (size n)
        |
   ┌────┴────┐
   |         |
 Left      Right
(size n/2) (size n/2)
   |         |
  ┌┴┐      ┌┴┐
  | |      | |
 n/4 n/4  n/4 n/4      ← log(n) levels
  ...      ...
  |  |    |  |
  1  1    1  1          ← n base cases

Combine step happens at each level going back up.
Total work per level: O(n)
Number of levels: O(log n)
Total: O(n log n)
```

### The Master Theorem Reference

For recurrences of the form **T(n) = aT(n/b) + O(n^d)**:

| Condition | Result | Intuition |
|-----------|--------|-----------|
| d > log_b(a) | O(n^d) | Combine dominates |
| d = log_b(a) | O(n^d log n) | Equal work at each level |
| d < log_b(a) | O(n^(log_b(a))) | Subproblems dominate |

**Common examples:**

| Recurrence | a | b | d | Result |
|-----------|---|---|---|--------|
| T(n) = 2T(n/2) + O(n) | 2 | 2 | 1 | O(n log n) — merge sort |
| T(n) = 2T(n/2) + O(1) | 2 | 2 | 0 | O(n) — binary tree traversal |
| T(n) = T(n/2) + O(1) | 1 | 2 | 0 | O(log n) — binary search |
| T(n) = T(n/2) + O(n) | 1 | 2 | 1 | O(n) — quick select (avg) |
| T(n) = 2T(n/2) + O(n^2) | 2 | 2 | 2 | O(n^2) — bad combine step |

### The Three Steps in Detail

```
Step 1: DIVIDE
  - Split input into smaller chunks
  - Usually halves, but not always
  - Choose split point that creates balanced subproblems

Step 2: CONQUER
  - Recursively solve each subproblem
  - Base case: trivially small problem (size 0 or 1)
  - Each recursive call is independent

Step 3: COMBINE
  - Merge sub-solutions into solution for original problem
  - This is where the cleverness lies
  - Must be efficient — typically O(n) or O(n log n)
```

---

<a name="master-templates"></a>
## 2. The 4 Master Templates

### Template A: Merge Sort Skeleton (Divide, Recurse, Merge)

**Use when:** You need to sort, count inversions, or count cross-boundary relationships.

```python
def merge_sort_template(arr: list[int]) -> list[int]:
    """
    Classic merge sort: divide in half, sort halves, merge.
    T(n) = 2T(n/2) + O(n) = O(n log n)
    """
    # Why <= 1? An array of 0 or 1 elements is already sorted.
    # This is the base case that stops recursion.
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort_template(arr[:mid])
    right = merge_sort_template(arr[mid:])

    return merge(left, right)

def merge(left: list[int], right: list[int]) -> list[int]:
    """Merge two sorted arrays into one sorted array."""
    result = []
    i = j = 0

    # Why both conditions with `and`? We can only compare when both
    # halves still have elements. Once one is exhausted, we drain the other.
    while i < len(left) and j < len(right):
        # Why `<=` and not `<`? Using <= makes the sort STABLE:
        # when left[i] == right[j], we take from the LEFT side first,
        # preserving the original relative order of equal elements.
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    # Why extend with remaining elements? When the while loop ends,
    # one half is exhausted but the other may still have elements.
    # Those remaining elements are already sorted and all greater than
    # everything placed so far, so we append them directly.
    result.extend(left[i:])
    result.extend(right[j:])
    return result
```

**When to modify the merge step:** To count inversions or other cross-boundary statistics, add counting logic inside the merge when `right[j] < left[i]`.

---

### Template B: Quick Select Skeleton (Partition, Recurse on One Side)

**Use when:** You need the kth smallest/largest element or top-k elements.

```python
import random

def quick_select_template(arr: list[int], k: int) -> int:
    """
    Find kth smallest element. Only recurse on the side containing k.
    Average T(n) = T(n/2) + O(n) = O(n).
    Worst case O(n^2) but random pivot makes it unlikely.
    """
    # Why == 1? Only one element left -- it must be the answer.
    if len(arr) == 1:
        return arr[0]

    pivot = random.choice(arr)

    lows = [x for x in arr if x < pivot]
    highs = [x for x in arr if x > pivot]
    pivots = [x for x in arr if x == pivot]

    # Why k < len(lows)? The kth smallest (0-indexed) falls among
    # elements strictly less than the pivot, so recurse only there.
    if k < len(lows):
        return quick_select_template(lows, k)
    # Why k < len(lows) + len(pivots)? k falls in the pivot group.
    # All pivots have the same value, so the answer is the pivot itself.
    elif k < len(lows) + len(pivots):
        return pivot
    # Otherwise, k is in the highs group. Subtract lows and pivots
    # counts to get k's position relative to the highs subarray.
    else:
        return quick_select_template(highs, k - len(lows) - len(pivots))
```

**Key advantage over sorting:** We only recurse on ONE side, giving O(n) average instead of O(n log n).

---

### Template C: Tree Build Skeleton (Find Root, Recurse Left/Right)

**Use when:** You need to construct a tree from traversal data or sorted arrays.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def tree_build_template(arr: list[int], lo: int, hi: int) -> TreeNode:
    """
    Build a balanced BST from sorted array.
    Pick middle as root, recurse on left and right halves.
    T(n) = 2T(n/2) + O(1) = O(n)
    """
    # Why lo > hi (not >=)? When lo == hi, there is exactly one element
    # to place as a leaf node. Only when lo > hi is the range truly empty,
    # meaning no node should be created (return None).
    if lo > hi:
        return None

    mid = (lo + hi) // 2
    node = TreeNode(arr[mid])
    node.left = tree_build_template(arr, lo, mid - 1)
    node.right = tree_build_template(arr, mid + 1, hi)

    return node
```

**Variants:** For preorder+inorder reconstruction, find root position in inorder traversal instead of using the midpoint.

---

### Template D: Generic D&C with Combine Step

**Use when:** Splitting a problem into subproblems that require a non-trivial combine step (e.g., expression evaluation, geometric problems).

```python
def generic_dc_template(problem, lo: int, hi: int):
    """
    Generic D&C: split at every possible point, combine results.
    Used for expression evaluation, matrix search, etc.
    """
    # Base case
    # Why == and not >=? In the generic template, lo == hi means we have
    # a single indivisible unit (e.g., one number or one operand).
    # lo should never exceed hi if called correctly.
    if lo == hi:
        return base_solution(problem, lo)

    results = []
    # Why range(lo, hi) and not range(lo, hi+1)? Each split point
    # divides into [lo..split] and [split+1..hi]. If split == hi,
    # the right side would be empty. So we stop at hi-1.
    for split in range(lo, hi):
        left_results = generic_dc_template(problem, lo, split)
        right_results = generic_dc_template(problem, split + 1, hi)

        # Combine every left result with every right result
        for l in left_results:
            for r in right_results:
                results.append(combine(l, r))

    return results
```

**Note:** This template can be exponential without memoization. Often combined with caching for overlapping subproblems.

---

### Decision Matrix: Which Template to Use?

| Signal | Template | Typical Complexity |
|--------|----------|-------------------|
| "Sort the array" / count inversions / count cross-pairs | A: Merge Sort | O(n log n) |
| "Kth largest" / "top k" / selection | B: Quick Select | O(n) average |
| "Build a tree" / "convert sorted array to BST" | C: Tree Build | O(n) |
| "All possible results" / expression evaluation / geometric | D: Generic D&C | Varies |
| Merge k sorted structures | A variant: k-way merge | O(N log k) |
| Search in sorted 2D matrix | D variant: quadrant elimination | O(n) or O(n log n) |

---

<a name="pattern-guide"></a>
## 3. Pattern Classification Guide

### Category 1: Merge Sort Variants
Count cross-boundary relationships during merge. The merge step is where the magic happens.
- Sort an Array (LC 912)
- Reverse Pairs (LC 493)
- Count of Smaller Numbers After Self (LC 315)

### Category 2: Quick Select / Partition
Only recurse on the side that matters. Partition around a pivot.
- Kth Largest Element (LC 215)
- Top K Frequent Elements (LC 347)

### Category 3: Binary Search as D&C
Eliminate half the search space each time. The simplest form of D&C.
- Median of Two Sorted Arrays (LC 4)

### Category 4: Tree Construction
Identify the root, then recursively build left and right subtrees.
- Convert Sorted Array to BST (LC 108)
- Build from Preorder and Inorder (LC 105)
- Build from Inorder and Postorder (LC 106)

### Category 5: Expression / Matrix D&C
Split at every operator or boundary; combine results from all splits.
- Search a 2D Matrix II (LC 240)
- Different Ways to Add Parentheses (LC 241)

### Category 6: Geometric / K-Way D&C
Divide spatial or multi-source problems, combine with merge or selection.
- K Closest Points to Origin (LC 973)
- Merge k Sorted Lists (LC 23)

---

<a name="patterns"></a>
## 4. Complete Pattern Library

---

### PATTERN 1: Merge Sort & Counting Inversions

---

#### Pattern 1A: Sort an Array

**Problem:** LeetCode 912 — Given an array of integers `nums`, sort the array in ascending order and return it.

**Example:**
```
Input: nums = [5, 2, 3, 1]
Output: [1, 2, 3, 5]
```

**Key Insight:** Merge sort is the prototypical D&C algorithm. Split the array in half, recursively sort each half, then merge the two sorted halves. The merge step takes O(n) because both halves are already sorted — just pick the smaller front element.

**Visual Trace:**
```
[5, 2, 3, 1]
       |
  ┌────┴────┐
[5, 2]    [3, 1]
  |          |
┌─┴─┐    ┌──┴──┐
[5] [2]  [3]  [1]     ← base cases (size 1, already sorted)
 └─┬─┘    └──┬──┘
[2, 5]    [1, 3]      ← merge: compare fronts, pick smaller
   └────┬────┘
  [1, 2, 3, 5]        ← merge: 1<2, 2<3, 3<5, done
```

**Python Solution:**
```python
def sortArray(nums: list[int]) -> list[int]:
    # Why `len(nums) <= 1`?
    # An array of size 0 (empty) or size 1 is trivially sorted.
    # Using <= 1 (not == 0) catches both: an empty call would produce
    # infinite recursion without this guard.
    if len(nums) <= 1:
        return nums

    mid = len(nums) // 2
    left = sortArray(nums[:mid])
    right = sortArray(nums[mid:])

    # Merge two sorted halves
    result = []
    i = j = 0
    # Why `i < len(left) and j < len(right)` (both conditions)?
    # We can only safely compare left[i] and right[j] when BOTH indices
    # are in bounds. If either half is exhausted, there's nothing to compare
    # against — we simply drain the remainder directly.
    while i < len(left) and j < len(right):
        # Why `<=` and not `<`?
        # Using <= makes the sort STABLE: when left[i] == right[j], we take
        # from the LEFT side first, preserving the original relative order
        # of equal elements. Changing to < would make the sort unstable.
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    # Drain whichever half has remaining elements (already sorted).
    result.extend(left[i:])
    result.extend(right[j:])
    return result
```

**Complexity:**
- Time: O(n log n) — recurrence T(n) = 2T(n/2) + O(n)
- Space: O(n) — for the merged arrays at each level

---

#### Pattern 1B: Reverse Pairs

**Problem:** LeetCode 493 — Given an integer array `nums`, return the number of reverse pairs. A reverse pair is a pair (i, j) where i < j and nums[i] > 2 * nums[j].

**Example:**
```
Input: nums = [1, 3, 2, 3, 1]
Output: 2
Explanation: (3, 1) at indices (1,4) and (3, 1) at indices (3,4)
```

**Key Insight:** Brute force checks all pairs in O(n^2). With merge sort, after sorting both halves we can count cross-boundary reverse pairs in O(n) using two pointers — because both halves are sorted, once nums[i] > 2*nums[j] for some j, it holds for all earlier i values too.

**Visual Trace:**
```
[1, 3, 2, 3, 1]
       |
  ┌────┴─────┐
[1, 3]     [2, 3, 1]
  |            |
┌─┴─┐    ┌────┴────┐
[1] [3]  [2]     [3, 1]
              ┌────┴────┐
             [3]       [1]

Counting happens during merge:
Left=[1,3], Right=[1,2,3] (after right is sorted)
  i=0 (val=1): find j where 1 > 2*right[j] → none → count += 0
  i=1 (val=3): find j where 3 > 2*right[j] → 3 > 2*1=2 → count += 1
Total reverse pairs from all levels: 2
```

**Python Solution:**
```python
def reversePairs(nums: list[int]) -> int:
    # Why `if not nums`?
    # An empty list has no pairs at all, so we return 0 immediately.
    # Without this guard, len(arr) // 2 would be 0 and slices would create
    # infinite empty recursive calls.
    if not nums:
        return 0

    count = [0]

    def merge_sort(arr):
        # Why `len(arr) <= 1`?
        # A single element cannot form any reverse pair. This is the base
        # case that terminates recursion and returns an already-sorted array.
        if len(arr) <= 1:
            return arr

        mid = len(arr) // 2
        left = merge_sort(arr[:mid])
        right = merge_sort(arr[mid:])

        # Count reverse pairs across left and right.
        # Both halves are sorted, so we use two pointers.
        j = 0
        for i in range(len(left)):
            # Why `left[i] > 2 * right[j]`? This is the reverse pair
            # condition. Because left is sorted, once left[i] > 2*right[j],
            # all later left elements (i+1, i+2, ...) also satisfy it.
            # And because right is sorted, j only moves forward --
            # we never re-check earlier right elements.
            # Why `j < len(right)` guard? j advances but never resets;
            # once we've exhausted right, no more pairs are possible.
            while j < len(right) and left[i] > 2 * right[j]:
                j += 1
            # j is the count of right elements that form reverse pairs
            # with left[i]. Because j never resets (left is sorted,
            # so left[i+1] >= left[i] still beats all those right[j]s),
            # the total counting across all i is O(n).
            count[0] += j

        # Standard merge
        merged = []
        i = j = 0
        # Why `i < len(left) and j < len(right)`?
        # Both pointers must be in bounds before we can compare. Once one
        # side runs out, the other is appended directly (already sorted).
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                j += 1
        merged.extend(left[i:])
        merged.extend(right[j:])
        return merged

    merge_sort(nums)
    return count[0]
```

**Complexity:**
- Time: O(n log n) — counting step is O(n) per level, merge is O(n) per level, log n levels
- Space: O(n) — for temporary merged arrays

---

#### Pattern 1C: Count of Smaller Numbers After Self

**Problem:** LeetCode 315 — Given an integer array `nums`, return an array `counts` where `counts[i]` is the number of smaller elements to the right of `nums[i]`.

**Example:**
```
Input: nums = [5, 2, 6, 1]
Output: [2, 1, 1, 0]
Explanation:
  5 has 2 smaller to its right: [2, 1]
  2 has 1 smaller to its right: [1]
  6 has 1 smaller to its right: [1]
  1 has 0 smaller to its right: []
```

**Key Insight:** We need to track the original indices through the sort. Pair each element with its index, then during merge, when a left element is placed after right elements, those right elements are smaller and originally to its right. Count them.

**Visual Trace:**
```
[(5,0), (2,1), (6,2), (1,3)]
           |
    ┌──────┴──────┐
[(5,0),(2,1)]  [(6,2),(1,3)]
    |               |
 ┌──┴──┐       ┌───┴───┐
(5,0) (2,1)   (6,2)  (1,3)

Merge (5,0),(2,1) → sorted: (2,1),(5,0)
  When (5,0) is placed, j=1 right elements already placed → counts[0] += 1

Merge (6,2),(1,3) → sorted: (1,3),(6,2)
  When (6,2) is placed, j=1 right elements already placed → counts[2] += 1

Merge (2,1),(5,0) with (1,3),(6,2):
  (1,3) placed first (j moves), then (2,1) placed → counts[1] += 1 (one right elem before it)
  Then (5,0) placed → counts[0] += 1 (one more right elem)
  Then (6,2) placed → counts[2] += 0

Final counts = [2, 1, 1, 0]
```

**Python Solution:**
```python
def countSmaller(nums: list[int]) -> list[int]:
    counts = [0] * len(nums)

    # Pair each number with its original index
    indexed = list(enumerate(nums))  # [(index, value), ...]

    def merge_sort(arr):
        # Why `len(arr) <= 1`?
        # A single element has no right-side neighbors. This base case stops
        # recursion and returns the element (paired with its original index)
        # as a trivially sorted subarray.
        if len(arr) <= 1:
            return arr

        mid = len(arr) // 2
        left = merge_sort(arr[:mid])
        right = merge_sort(arr[mid:])

        merged = []
        i = j = 0
        right_count = 0  # How many from right have been placed

        # Why `i < len(left) and j < len(right)`?
        # We compare values from both halves simultaneously. Once either
        # half is exhausted we break out and handle each side separately.
        while i < len(left) and j < len(right):
            # Why compare [1] (the value) and not [0] (the index)?
            # We sort by value to determine ordering, but track original
            # indices to know WHERE to store each count.
            # Why `<=`? When values are equal, taking from the left side
            # first avoids counting equal elements as "smaller."
            if left[i][1] <= right[j][1]:
                # left[i] is placed; all right elements placed so far
                # (right_count of them) came from the right half and
                # have smaller values -- they are "smaller numbers after self."
                counts[left[i][0]] += right_count
                merged.append(left[i])
                i += 1
            else:
                # right[j] is smaller than left[i], so it contributes
                # to the count for every left element still unplaced.
                right_count += 1
                merged.append(right[j])
                j += 1

        # Why `while i < len(left)` and add right_count?
        # All remaining left elements haven't been placed yet. Every one of
        # them was preceded by `right_count` smaller right elements, so we
        # credit each of them with the full accumulated right_count.
        while i < len(left):
            counts[left[i][0]] += right_count
            merged.append(left[i])
            i += 1

        # Remaining right elements: no left elements care about these
        # (all left elements are already placed).
        while j < len(right):
            merged.append(right[j])
            j += 1

        return merged

    merge_sort(indexed)
    return counts
```

**Complexity:**
- Time: O(n log n) — merge sort with O(n) work per level
- Space: O(n) — for indexed pairs and merge buffers

---

### PATTERN 2: Quick Select (Kth Element)

---

#### Pattern 2A: Kth Largest Element in an Array

**Problem:** LeetCode 215 — Given an integer array `nums` and an integer `k`, return the kth largest element. Note that it is the kth largest element in sorted order, not the kth distinct element.

**Example:**
```
Input: nums = [3, 2, 1, 5, 6, 4], k = 2
Output: 5
```

**Key Insight:** Sorting gives O(n log n), but we only need ONE element, not the full sorted order. Quick select partitions around a random pivot: if the pivot lands at the kth position, we're done. Otherwise, recurse only on the side containing k. Average O(n).

**Visual Trace:**
```
nums = [3, 2, 1, 5, 6, 4], k=2 → want 2nd largest = 5th smallest (0-indexed: k=4)

Round 1: pivot = 3
  lows  = [2, 1]       (< 3)
  pivots = [3]          (== 3)
  highs = [5, 6, 4]    (> 3)
  len(lows)=2, len(pivots)=1 → k=4 >= 3
  Recurse on highs with k = 4 - 2 - 1 = 1

Round 2: arr = [5, 6, 4], k=1
  pivot = 5
  lows  = [4]          (< 5)
  pivots = [5]          (== 5)
  highs = [6]           (> 5)
  len(lows)=1, k=1 falls in pivots range → return 5
```

**Python Solution:**
```python
import random

def findKthLargest(nums: list[int], k: int) -> int:
    # kth largest = (n-k)th smallest (0-indexed)
    target = len(nums) - k

    def quick_select(arr, k):
        # Why `len(arr) == 1`?
        # Only one candidate remains — it must be the element at position k.
        # This is the base case that terminates the partition recursion.
        if len(arr) == 1:
            return arr[0]

        pivot = random.choice(arr)
        lows = [x for x in arr if x < pivot]
        highs = [x for x in arr if x > pivot]
        pivots = [x for x in arr if x == pivot]

        # Why `k < len(lows)` (strict less-than)?
        # lows contains exactly the elements at positions 0..len(lows)-1.
        # If k is within that range, the answer lives in lows, not in pivots.
        # Using < (not <=) is correct because lows occupies indices 0 to
        # len(lows)-1 — k == len(lows) would fall in the pivot group.
        if k < len(lows):
            return quick_select(lows, k)
        # Why `k < len(lows) + len(pivots)`?
        # All pivot copies occupy the contiguous block [len(lows),
        # len(lows)+len(pivots)-1]. If k falls in that range, every pivot
        # has the same value, so the answer is just the pivot itself.
        elif k < len(lows) + len(pivots):
            return pivot
        # Otherwise k is in the highs region. Subtract the sizes of lows
        # and pivots to get k's 0-based position within the highs subarray.
        else:
            return quick_select(highs, k - len(lows) - len(pivots))

    return quick_select(nums, target)
```

**Complexity:**
- Time: O(n) average — T(n) = T(n/2) + O(n) by Master Theorem. Worst case O(n^2) with bad pivots.
- Space: O(n) — for the lows/highs/pivots arrays. Can be O(1) with in-place partitioning.

---

#### Pattern 2B: Top K Frequent Elements

**Problem:** LeetCode 347 — Given an integer array `nums` and an integer `k`, return the `k` most frequent elements. You may return the answer in any order.

**Example:**
```
Input: nums = [1, 1, 1, 2, 2, 3], k = 2
Output: [1, 2]
```

**Key Insight:** First count frequencies. Then finding the top k frequent elements is a selection problem — use quick select on the frequency values. We don't need a full sort.

**Python Solution:**
```python
import random
from collections import Counter

def topKFrequent(nums: list[int], k: int) -> list[int]:
    freq = Counter(nums)
    unique = list(freq.keys())

    def quick_select(lo, hi, k_smallest):
        """Partition unique[lo..hi] so that the k_smallest most frequent are on the right."""
        # Why `lo == hi`?
        # Only one element remains in the range — it's already in its
        # correct position. No further partitioning is possible or needed.
        if lo == hi:
            return

        pivot_idx = random.randint(lo, hi)
        pivot_freq = freq[unique[pivot_idx]]

        # Move pivot to end
        unique[pivot_idx], unique[hi] = unique[hi], unique[pivot_idx]

        # Partition: elements with freq < pivot_freq go left
        store = lo
        # Why `freq[unique[i]] < pivot_freq`?
        # We want the k_smallest most-frequent elements on the RIGHT side.
        # Elements with frequency less than the pivot's belong to the left
        # (less frequent) region. The pivot and higher-frequency elements
        # accumulate on the right via the store pointer.
        for i in range(lo, hi):
            if freq[unique[i]] < pivot_freq:
                unique[store], unique[i] = unique[i], unique[store]
                store += 1

        # Move pivot to final position
        unique[store], unique[hi] = unique[hi], unique[store]

        # Why `store == k_smallest`?
        # The pivot has landed exactly at the boundary index. Everything to
        # its right (indices store+1..hi) is at least as frequent, which
        # are exactly the top-k elements we want. We're done.
        if store == k_smallest:
            return
        # Why `store < k_smallest`?
        # The pivot settled too far left — k_smallest falls in the right
        # partition, so we recurse right to find the true boundary.
        elif store < k_smallest:
            quick_select(store + 1, hi, k_smallest)
        else:
            quick_select(lo, store - 1, k_smallest)

    n = len(unique)
    # We want the top k frequent, i.e., the k elements with highest frequency
    # After partitioning, elements at indices [n-k, n-1] should be the top k
    quick_select(0, n - 1, n - k)
    return unique[n - k:]
```

**Complexity:**
- Time: O(n) average — O(n) for counting, O(m) average for quick select where m = number of unique elements
- Space: O(n) — for the frequency map

---

### PATTERN 3: Binary Search as D&C

---

#### Pattern 3A: Median of Two Sorted Arrays

**Problem:** LeetCode 4 — Given two sorted arrays `nums1` and `nums2` of size `m` and `n`, return the median of the two sorted arrays. The overall run time complexity should be O(log(m+n)).

**Example:**
```
Input: nums1 = [1, 3], nums2 = [2]
Output: 2.0
Explanation: merged = [1, 2, 3], median = 2
```

```
Input: nums1 = [1, 2], nums2 = [3, 4]
Output: 2.5
Explanation: merged = [1, 2, 3, 4], median = (2 + 3) / 2 = 2.5
```

**Key Insight:** We binary search on the shorter array for a partition point. If we put `i` elements from nums1 and `j` elements from nums2 on the left side (where i + j = half of total), we need `max(left side) <= min(right side)`. Binary search on `i` finds the correct partition in O(log(min(m,n))).

**Visual Trace:**
```
nums1 = [1, 3, 8, 9, 15]
nums2 = [7, 11, 18, 19, 21, 25]

Total = 11, half = 5 (we want 5 elements on the left side)

Binary search on nums1 (shorter array):
  lo=0, hi=5

  i=2, j=5-2=3:
    Left1: [1,3]     Right1: [8,9,15]
    Left2: [7,11,18] Right2: [19,21,25]
    maxLeft1=3, minRight1=8
    maxLeft2=18, minRight2=19
    3 <= 19? Yes. 18 <= 8? No → 18 > 8, so i is too small, move right.

  i=4, j=5-4=1:
    Left1: [1,3,8,9]  Right1: [15]
    Left2: [7]         Right2: [11,18,19,21,25]
    maxLeft1=9, minRight1=15
    maxLeft2=7, minRight2=11
    9 <= 11? Yes. 7 <= 15? Yes → Found!
    Median = max(9, 7) = 9
```

**Python Solution:**
```python
def findMedianSortedArrays(nums1: list[int], nums2: list[int]) -> float:
    # Why `if len(nums1) > len(nums2): swap`?
    # We binary search on nums1. Binary search on the shorter array gives
    # O(log(min(m,n))) time. If nums1 were longer, we'd do O(log(max(m,n)))
    # work unnecessarily. Swapping ensures we always search the smaller space.
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1

    m, n = len(nums1), len(nums2)
    half = (m + n + 1) // 2

    lo, hi = 0, m

    # Why `lo <= hi`?
    # We are binary searching over valid partition counts [0..m]. Using <=
    # ensures we check every possible partition including lo==hi (where only
    # 0 or all elements from nums1 go to the left side).
    while lo <= hi:
        i = (lo + hi) // 2  # Partition index in nums1
        j = half - i         # Partition index in nums2

        # Edge cases: if partition is at boundary, use -inf or inf
        left1 = nums1[i - 1] if i > 0 else float('-inf')
        right1 = nums1[i] if i < m else float('inf')
        left2 = nums2[j - 1] if j > 0 else float('-inf')
        right2 = nums2[j] if j < n else float('inf')

        # Why `left1 <= right2 and left2 <= right1`?
        # This is the partition correctness invariant: every element on the
        # left side must be <= every element on the right side. We need both
        # cross-comparisons because elements can come from either array.
        # If this holds, we have the correct left/right split for the median.
        if left1 <= right2 and left2 <= right1:
            # Found correct partition
            if (m + n) % 2 == 1:
                return max(left1, left2)
            else:
                return (max(left1, left2) + min(right1, right2)) / 2
        # Why `hi = i - 1` when `left1 > right2`?
        # left1 is too large — we put too many elements from nums1 on the
        # left side. Shrink i by moving hi left to try a smaller partition.
        elif left1 > right2:
            hi = i - 1  # Too many from nums1
        # Otherwise left2 > right1: too few from nums1, expand i rightward.
        else:
            lo = i + 1  # Too few from nums1

    return 0.0  # Should never reach here for valid input
```

**Complexity:**
- Time: O(log(min(m, n))) — binary search on the shorter array
- Space: O(1)
- Recurrence: T(n) = T(n/2) + O(1) by Master Theorem gives O(log n)

---

### PATTERN 4: Tree-Based D&C (Build / Reconstruct Trees)

---

#### Pattern 4A: Convert Sorted Array to BST

**Problem:** LeetCode 108 — Given an integer array `nums` where the elements are sorted in ascending order, convert it to a height-balanced binary search tree.

**Example:**
```
Input: nums = [-10, -3, 0, 5, 9]
Output: [0, -3, 9, -10, null, 5] (one valid BST)
       0
      / \
    -3   9
    /   /
  -10  5
```

**Key Insight:** The middle element becomes the root (ensuring balance). Everything to the left of the middle forms the left subtree, everything to the right forms the right subtree. Recurse.

**Visual Trace:**
```
nums = [-10, -3, 0, 5, 9]

build(0, 4):
  mid = 2 → root = 0
  left = build(0, 1)
    mid = 0 → root = -10
    left = build(0, -1) → None
    right = build(1, 1)
      mid = 1 → root = -3
      left = build(1, 0) → None
      right = build(2, 1) → None
      return -3
    return -10 with right child -3
  right = build(3, 4)
    mid = 3 → root = 5
    left = build(3, 2) → None
    right = build(4, 4)
      mid = 4 → root = 9
      return 9
    return 5 with right child 9
  return 0 with left=-10(-3), right=5(9)

Result:
       0
      / \
    -10   5
      \    \
      -3    9
```

**Python Solution:**
```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def sortedArrayToBST(nums: list[int]) -> TreeNode:
    def build(lo, hi):
        # Why `lo > hi` (not `lo >= hi`)?
        # When lo == hi, exactly one element remains — it becomes a leaf node.
        # Only when lo > hi is the range empty (e.g., after mid-1 < lo),
        # meaning no node should be created. Using >= would skip single-element
        # leaves and produce an incorrect or incomplete tree.
        if lo > hi:
            return None

        mid = (lo + hi) // 2
        node = TreeNode(nums[mid])
        node.left = build(lo, mid - 1)
        node.right = build(mid + 1, hi)
        return node

    return build(0, len(nums) - 1)
```

**Complexity:**
- Time: O(n) — visit each element once. T(n) = 2T(n/2) + O(1) = O(n).
- Space: O(log n) — recursion depth for a balanced tree

---

#### Pattern 4B: Construct Binary Tree from Preorder and Inorder Traversal

**Problem:** LeetCode 105 — Given two integer arrays `preorder` and `inorder` where `preorder` is the preorder traversal and `inorder` is the inorder traversal of the same tree, construct and return the binary tree.

**Example:**
```
Input: preorder = [3, 9, 20, 15, 7], inorder = [9, 3, 15, 20, 7]
Output:
    3
   / \
  9  20
    /  \
   15   7
```

**Key Insight:** The first element in preorder is always the root. Find that root in inorder — everything to its left is the left subtree, everything to its right is the right subtree. Use a hashmap for O(1) lookups in inorder.

**Visual Trace:**
```
preorder = [3, 9, 20, 15, 7]
inorder  = [9, 3, 15, 20, 7]

Step 1: root = preorder[0] = 3
  Find 3 in inorder at index 1
  Left inorder: [9]       → left subtree has 1 node
  Right inorder: [15,20,7] → right subtree has 3 nodes
  Left preorder: [9]       (next 1 element)
  Right preorder: [20,15,7] (remaining 3 elements)

Step 2: Left subtree: pre=[9], in=[9]
  root = 9, no children → leaf node

Step 3: Right subtree: pre=[20,15,7], in=[15,20,7]
  root = 20, find in inorder at index 1
  Left: in=[15], pre=[15] → leaf 15
  Right: in=[7], pre=[7]  → leaf 7

Result:
    3
   / \
  9  20
    /  \
   15   7
```

**Python Solution:**
```python
def buildTree(preorder: list[int], inorder: list[int]) -> TreeNode:
    # Map value -> index in inorder for O(1) lookup
    inorder_map = {val: idx for idx, val in enumerate(inorder)}
    pre_idx = [0]  # Use list to allow mutation in nested function

    def build(in_lo, in_hi):
        # Why `in_lo > in_hi`?
        # The inorder range is empty — this subtree has no nodes. Returning
        # None correctly represents an absent child. Without this check, the
        # next preorder value would be consumed as a phantom root node,
        # corrupting the entire tree structure.
        if in_lo > in_hi:
            return None

        root_val = preorder[pre_idx[0]]
        pre_idx[0] += 1

        root = TreeNode(root_val)
        # Why `mid = inorder_map[root_val]`?
        # In inorder traversal, the root splits the array: everything to its
        # LEFT is the left subtree, everything to its RIGHT is the right
        # subtree. The hashmap gives us this split index in O(1) instead of
        # scanning the array (which would degrade to O(n^2) overall).
        mid = inorder_map[root_val]

        # Build left subtree first (preorder: root, LEFT, right)
        root.left = build(in_lo, mid - 1)
        root.right = build(mid + 1, in_hi)

        return root

    return build(0, len(inorder) - 1)
```

**Complexity:**
- Time: O(n) — each node visited once, O(1) lookup in hashmap
- Space: O(n) — hashmap + O(h) recursion stack where h is tree height

---

#### Pattern 4C: Construct Binary Tree from Inorder and Postorder Traversal

**Problem:** LeetCode 106 — Given two integer arrays `inorder` and `postorder` where `inorder` is the inorder traversal and `postorder` is the postorder traversal of the same tree, construct and return the binary tree.

**Example:**
```
Input: inorder = [9, 3, 15, 20, 7], postorder = [9, 15, 7, 20, 3]
Output:
    3
   / \
  9  20
    /  \
   15   7
```

**Key Insight:** Identical to Pattern 4B but reversed. The LAST element in postorder is the root. Build right subtree first (postorder: left, right, ROOT — so we consume from the end, processing right before left).

**Visual Trace:**
```
inorder  = [9, 3, 15, 20, 7]
postorder = [9, 15, 7, 20, 3]

Step 1: root = postorder[-1] = 3
  Find 3 in inorder at index 1
  Left inorder: [9], Right inorder: [15,20,7]

Step 2: Build RIGHT first (consume postorder from end)
  postorder next from end: 20
  root = 20, in inorder at index 3
  Right of 20: in=[7], post consumes 7 → leaf
  Left of 20: in=[15], post consumes 15 → leaf

Step 3: Build LEFT
  postorder next: 9
  root = 9, leaf node

Result:
    3
   / \
  9  20
    /  \
   15   7
```

**Python Solution:**
```python
def buildTree(inorder: list[int], postorder: list[int]) -> TreeNode:
    inorder_map = {val: idx for idx, val in enumerate(inorder)}
    post_idx = [len(postorder) - 1]

    def build(in_lo, in_hi):
        # Why `in_lo > in_hi`?
        # The inorder range is empty — this subtree has no nodes.
        # Without this guard, we'd consume the next postorder element as a
        # phantom root for a nonexistent subtree, corrupting the entire tree.
        if in_lo > in_hi:
            return None

        root_val = postorder[post_idx[0]]
        post_idx[0] -= 1

        root = TreeNode(root_val)
        # Why `inorder_map[root_val]` gives the split?
        # Inorder traversal is: [LEFT subtree] ROOT [RIGHT subtree].
        # Everything at indices < mid belongs to the left subtree;
        # everything > mid belongs to the right. The hashmap gives this
        # split in O(1) — linear search would make the whole algorithm O(n^2).
        mid = inorder_map[root_val]

        # Why build RIGHT before LEFT here?
        # Postorder is: left, right, ROOT. We're consuming from the END
        # of postorder (post_idx decrements). So the element just before
        # the current root is the root of the RIGHT subtree, not the left.
        # If we built left first, we'd consume the right subtree's root
        # for the left subtree — the tree would be completely wrong.
        # Analogy: reading a stack of papers placed face-down (left, right, root)
        # — the first paper you pick up IS the right subtree's root.
        root.right = build(mid + 1, in_hi)
        root.left = build(in_lo, mid - 1)

        return root

    return build(0, len(inorder) - 1)
```

**Complexity:**
- Time: O(n) — each node visited once
- Space: O(n) — hashmap + recursion stack

---

### PATTERN 5: Expression / Matrix D&C

---

#### Pattern 5A: Search a 2D Matrix II

**Problem:** LeetCode 240 — Write an efficient algorithm that searches for a value `target` in an `m x n` integer matrix. This matrix has the following properties: integers in each row are sorted left to right, integers in each column are sorted top to bottom.

**Example:**
```
Input: matrix = [
  [1,  4,  7, 11, 15],
  [2,  5,  8, 12, 19],
  [3,  6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
], target = 5
Output: True
```

**Key Insight:** Start from the top-right corner. If the current value equals target, found it. If the current value is greater than target, move left (eliminate column). If less, move down (eliminate row). Each step eliminates a row or column — at most m+n steps.

This is D&C in spirit: each comparison eliminates a portion of the search space (a row or column), dividing the problem.

**Visual Trace:**
```
Start at top-right: matrix[0][4] = 15
  15 > 5 → move left
matrix[0][3] = 11
  11 > 5 → move left
matrix[0][2] = 7
  7 > 5 → move left
matrix[0][1] = 4
  4 < 5 → move down
matrix[1][1] = 5
  5 == 5 → FOUND!
```

**Python Solution:**
```python
def searchMatrix(matrix: list[list[int]], target: int) -> bool:
    if not matrix or not matrix[0]:
        return False

    rows, cols = len(matrix), len(matrix[0])
    # Why start at the TOP-RIGHT corner (not top-left or bottom-right)?
    # Top-right is the unique corner where ONE comparison can eliminate
    # an entire row OR an entire column — never both, never neither:
    #   - current > target: every cell BELOW in this column is also > target
    #     (columns are sorted top-to-bottom). Eliminate column → move left.
    #   - current < target: every cell to the LEFT in this row is also < target
    #     (rows are sorted left-to-right). Eliminate row → move down.
    # Top-left has two "increase" directions (right, down) — no clean elimination.
    # Bottom-right has two "decrease" directions (left, up) — same problem.
    row, col = 0, cols - 1

    # Why `row < rows AND col >= 0`?
    # row == rows means we've fallen off the bottom (exhausted all rows).
    # col < 0 means we've fallen off the left (exhausted all columns).
    # Either exit means target is not in the matrix — we've searched everything
    # reachable from the top-right corner.
    while row < rows and col >= 0:
        val = matrix[row][col]
        if val == target:
            return True
        elif val > target:
            # Why `col -= 1`? Current value is too large.
            # Every cell in this column below us is even larger (sorted).
            # So the entire column is ruled out. Step left to a smaller column.
            col -= 1
        else:
            # Why `row += 1`? Current value is too small.
            # Every cell in this row to our left is even smaller (sorted).
            # So the entire row is ruled out. Step down to a larger row.
            row += 1

    return False
```

**Complexity:**
- Time: O(m + n) — at most m + n steps
- Space: O(1)

---

#### Pattern 5B: Different Ways to Add Parentheses

**Problem:** LeetCode 241 — Given a string `expression` of numbers and operators (`+`, `-`, `*`), return all possible results from computing all the different possible ways to group numbers and operators using parentheses.

**Example:**
```
Input: expression = "2*3-4*5"
Output: [-34, -14, -10, -10, 10]
Explanation:
  (2*(3-(4*5))) = -34
  ((2*3)-(4*5)) = -14
  ((2*(3-4))*5) = -10
  (2*((3-4)*5)) = -10
  (((2*3)-4)*5) = 10
```

**Key Insight:** For each operator in the expression, split into left and right subexpressions. Recursively compute all possible results for each side, then combine every left result with every right result using the operator. Base case: the expression is a single number.

**Visual Trace:**
```
"2*3-4*5"

Split at '*' (index 1): left="2", right="3-4*5"
  left results: [2]
  right "3-4*5":
    Split at '-': left="3", right="4*5"
      left: [3], right: [20]
      combine: 3-20 = [-17]
    Split at '*': left="3-4", right="5"
      left "3-4": [3-4] = [-1]
      right: [5]
      combine: -1*5 = [-5]
    right results: [-17, -5]
  combine: 2*(-17)=-34, 2*(-5)=-10

Split at '-' (index 3): left="2*3", right="4*5"
  left: [6], right: [20]
  combine: 6-20 = [-14]

Split at '*' (index 5): left="2*3-4", right="5"
  left "2*3-4":
    Split at '*': left="2", right="3-4"
      left: [2], right: [-1]
      combine: 2*(-1) = [-2]
    Split at '-': left="2*3", right="4"
      left: [6], right: [4]
      combine: 6-4 = [2]
    left results: [-2, 2]
  right: [5]
  combine: -2*5=-10, 2*5=10

All results: [-34, -10, -14, -10, 10]
```

**Python Solution:**
```python
def diffWaysToCompute(expression: str) -> list[int]:
    # Memoization to avoid recomputing same subexpressions
    memo = {}

    def compute(expr):
        # Why check memo first?
        # The same subexpression (e.g., "3-4") appears many times as you split
        # "2*3-4*5" at different operators. Without memoization, you recompute
        # "3-4" independently for each split point that produces it — the total
        # work grows as the Catalan number (exponential). With memo: compute once,
        # reuse everywhere. This is D&C + memoization = essentially top-down DP.
        if expr in memo:
            return memo[expr]

        results = []

        for i, ch in enumerate(expr):
            # Why `ch in '+-*'`?
            # Operators are the only valid split points. Splitting inside a
            # multi-digit number (e.g., splitting "23" into "2" and "3") would
            # create two separate numbers that were meant to be one — wrong.
            # Each operator is a "parenthesization boundary": (left_expr) op (right_expr).
            if ch in '+-*':
                left_results = compute(expr[:i])
                right_results = compute(expr[i + 1:])

                # Combine every left result with every right result.
                # Why every pair? Because each is a DIFFERENT way to parenthesize
                # the sub-expression. All combinations are valid distinct results.
                for l in left_results:
                    for r in right_results:
                        if ch == '+':
                            results.append(l + r)
                        elif ch == '-':
                            results.append(l - r)
                        else:
                            results.append(l * r)

        # Why `if not results`?
        # If we never entered the loop (no operators found), the expression is a
        # single number like "23" or "5". There's no split point — it's the base
        # case. Convert the string to an integer and return it as the only result.
        if not results:
            results.append(int(expr))

        memo[expr] = results
        return results

    return compute(expression)
```

**Complexity:**
- Time: O(C(n)) where C(n) is the nth Catalan number — the number of ways to fully parenthesize n operators. Roughly O(4^n / n^(3/2)).
- Space: O(C(n)) — for storing all possible results

---

### PATTERN 6: Geometric / K-Way D&C

---

#### Pattern 6A: K Closest Points to Origin

**Problem:** LeetCode 973 — Given an array of points on the X-Y plane and an integer `k`, return the `k` closest points to the origin (0, 0).

**Example:**
```
Input: points = [[1, 3], [-2, 2]], k = 1
Output: [[-2, 2]]
Explanation: dist(1,3) = sqrt(10), dist(-2,2) = sqrt(8). Closer: [-2,2]
```

**Key Insight:** This is a selection problem in disguise. We don't need sorted order, just the k smallest distances. Quick select on distances gives O(n) average.

**Visual Trace:**
```
points = [[3,3],[5,-1],[-2,4],[0,1],[1,1]], k=2
distances: [18, 26, 20, 1, 2]

Quick select to find 2 smallest:
  pivot dist = 20 (random, say [-2,4])
  lows (dist < 20): [[0,1],[1,1],[3,3]]  dists=[1,2,18]
  pivots (dist == 20): [[-2,4]]
  highs (dist > 20): [[5,-1]]  dist=[26]

  k=2 < len(lows)=3 → recurse on lows

  lows: [[0,1],[1,1],[3,3]], k=2
  pivot dist = 2 (say [1,1])
  lows: [[0,1]]  dist=[1]
  pivots: [[1,1]]
  highs: [[3,3]] dist=[18]

  k=2 >= len(lows)+len(pivots)=2? k=2 == 2 → exactly at boundary
  Return lows + pivots = [[0,1],[1,1]]
```

**Python Solution:**
```python
import random

def kClosest(points: list[list[int]], k: int) -> list[list[int]]:
    def dist(point):
        return point[0] ** 2 + point[1] ** 2

    def quick_select(pts, k):
        # Why `len(pts) <= k`?
        # If we have FEWER points than k, every point is among the k closest
        # by definition — return them all. This also handles the base case
        # where recursion has narrowed pts to exactly 1 element.
        # Without this guard, we'd compute a pivot from a list smaller than k
        # and index into lows/mids/highs incorrectly.
        if len(pts) <= k:
            return pts

        pivot = dist(random.choice(pts))
        lows = [p for p in pts if dist(p) < pivot]
        mids = [p for p in pts if dist(p) == pivot]
        highs = [p for p in pts if dist(p) > pivot]

        # Why `k <= len(lows)` (using `<=` not `<`)?
        # lows contains points at positions 0..len(lows)-1 by distance rank.
        # If k <= len(lows), all k closest points live inside lows — recurse there.
        # We want the k closest (indices 0..k-1), and lows has at least k of them.
        if k <= len(lows):
            return quick_select(lows, k)
        # Why `k <= len(lows) + len(mids)`?
        # All points in lows ARE closer than the pivot. All points in mids ARE
        # the pivot distance. If k falls within lows + mids, we've already
        # found all points closer than the pivot — take all of lows plus
        # however many mids we need to reach exactly k.
        elif k <= len(lows) + len(mids):
            return lows + mids[:k - len(lows)]
        # k exceeds lows + mids: we must also include some highs.
        # Subtract the sizes of lows and mids so k is relative to highs.
        else:
            return lows + mids + quick_select(highs, k - len(lows) - len(mids))

    return quick_select(points, k)
```

**Complexity:**
- Time: O(n) average — quick select. Worst case O(n^2).
- Space: O(n) — for partition arrays

---

#### Pattern 6B: Merge k Sorted Lists

**Problem:** LeetCode 23 — You are given an array of `k` linked-lists `lists`, each linked-list is sorted in ascending order. Merge all the linked-lists into one sorted linked-list and return it.

**Example:**
```
Input: lists = [[1,4,5],[1,3,4],[2,6]]
Output: [1,1,2,3,4,4,5,6]
```

**Key Insight:** This is D&C applied to merging. Instead of merging all k lists at once, merge them pairwise: merge lists[0] with lists[1], lists[2] with lists[3], etc. This reduces k lists to k/2, then k/4, ... until one list remains. Each level does O(N) total work across all merges, and there are O(log k) levels.

**Visual Trace:**
```
lists = [[1,4,5], [1,3,4], [2,6]]

Level 1 (pair up):
  Merge [1,4,5] + [1,3,4] → [1,1,3,4,4,5]
  [2,6] has no pair → stays

Level 2:
  Merge [1,1,3,4,4,5] + [2,6] → [1,1,2,3,4,4,5,6]

Done! 2 levels of merging instead of k-1 sequential merges.
```

**Python Solution:**
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def mergeKLists(lists: list[ListNode]) -> ListNode:
    if not lists:
        return None

    def merge_two(l1, l2):
        """Merge two sorted linked lists."""
        dummy = ListNode(0)
        curr = dummy
        # Why `while l1 and l2` (both, not one)?
        # We compare l1.val and l2.val each step. If either is None (exhausted),
        # there's nothing to compare — we break out and drain the remainder.
        while l1 and l2:
            # Why `l1.val <= l2.val` (using `<=`, not `<`)?
            # When values are equal, taking from l1 first keeps the merge STABLE
            # (preserves original ordering of equal elements). Changing to `<`
            # would take from l2 when equal — still correct but unstable.
            if l1.val <= l2.val:
                curr.next = l1
                l1 = l1.next
            else:
                curr.next = l2
                l2 = l2.next
            curr = curr.next
        # Why `curr.next = l1 if l1 else l2`?
        # When the while loop ends, exactly one list still has elements remaining.
        # Those elements are already sorted AND are all greater than everything
        # placed so far (since both input lists were sorted). Attach the remainder
        # directly — no need to traverse it node by node.
        curr.next = l1 if l1 else l2
        return dummy.next

    def merge_lists(lists, lo, hi):
        """D&C: merge lists[lo..hi] by splitting in half."""
        # Why `lo == hi`?
        # One list remains in the range — it's already "merged" (nothing to pair
        # it with). Return it directly as the base case.
        # If we didn't check this, we'd compute mid == lo and recurse infinitely.
        if lo == hi:
            return lists[lo]
        mid = (lo + hi) // 2
        left = merge_lists(lists, lo, mid)
        right = merge_lists(lists, mid + 1, hi)
        return merge_two(left, right)

    # Filter out empty lists
    lists = [l for l in lists if l]
    if not lists:
        return None

    return merge_lists(lists, 0, len(lists) - 1)
```

**Complexity:**
- Time: O(N log k) — N = total elements across all lists, log k levels of merging
- Space: O(log k) — recursion depth
- Recurrence: T(k) = 2T(k/2) + O(N) per level, log k levels, so O(N log k)

---

#### Pattern 6C: (Bonus) Closest Pair of Points — Classic D&C

This is not a direct LeetCode problem but is the classic D&C geometric algorithm worth knowing.

**Problem:** Given n points in 2D, find the pair with the smallest Euclidean distance.

**Key Insight:** Sort by x-coordinate. Split in half at the median x. Recursively find closest pair in each half. The tricky part: the closest pair might straddle the dividing line. But we only need to check points within distance `d` (current best) of the line, and for each such point, only check at most 7 neighbors in y-sorted order.

**Python Solution:**
```python
import math

def closest_pair(points: list[tuple[int, int]]) -> float:
    def dist(p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def solve(pts_x, pts_y):
        n = len(pts_x)
        # Why brute force when `n <= 3`?
        # With 3 or fewer points, there are at most 3 pairs to check — no benefit
        # to dividing further. More importantly, dividing 3 points gives halves of
        # size 1 and 2 — the size-1 side returns immediately with no pair found.
        # Brute force here keeps the recursion clean and avoids degenerate splits.
        if n <= 3:
            min_d = float('inf')
            for i in range(n):
                for j in range(i + 1, n):
                    min_d = min(min_d, dist(pts_x[i], pts_x[j]))
            return min_d

        mid = n // 2
        mid_x = pts_x[mid][0]

        left_x = pts_x[:mid]
        right_x = pts_x[mid:]
        left_y = [p for p in pts_y if p[0] <= mid_x]
        right_y = [p for p in pts_y if p[0] > mid_x]

        # Why correct for duplicate x-coordinates?
        # If multiple points share mid_x, they all go to left_y (because of <=).
        # This can make left_y larger than left_x, breaking the size invariant.
        # We move the excess to right_y so both halves have exactly the right count.
        if len(left_y) > len(left_x):
            excess = len(left_y) - len(left_x)
            right_y = left_y[-excess:] + right_y
            left_y = left_y[:-excess]

        d_left = solve(left_x, left_y)
        d_right = solve(right_x, right_y)
        d = min(d_left, d_right)

        # Why only check points within distance `d` of the dividing line?
        # Any pair with one point more than d away from the line has x-distance > d,
        # so their total distance > d — they can't improve our best answer.
        # Only points in this "strip" can possibly form a closer pair.
        strip = [p for p in pts_y if abs(p[0] - mid_x) < d]

        for i in range(len(strip)):
            j = i + 1
            # Why `strip[j][1] - strip[i][1]) < d` (y-distance only)?
            # The strip is sorted by y-coordinate (pts_y was pre-sorted by y).
            # Once a strip point's y-coordinate exceeds strip[i]'s y by d,
            # their Euclidean distance is already > d — stop checking further.
            # Key theorem: at most 8 points can fit in any d×2d strip rectangle,
            # so this inner while loop runs at most 7 iterations per i → O(n) total.
            while j < len(strip) and (strip[j][1] - strip[i][1]) < d:
                d = min(d, dist(strip[i], strip[j]))
                j += 1

        return d

    pts_x = sorted(points, key=lambda p: p[0])
    pts_y = sorted(points, key=lambda p: p[1])
    return solve(pts_x, pts_y)
```

**Complexity:**
- Time: O(n log n) — T(n) = 2T(n/2) + O(n) for the strip check
- Space: O(n) — for sorted copies and strip

---

<a name="post-processing"></a>
## 5. Post-Processing Reference

After solving a D&C problem, common post-processing steps:

| Situation | Post-Processing | Example |
|-----------|----------------|---------|
| Need sorted output from merge sort | Already sorted — no extra work | LC 912 |
| Need count from modified merge | Extract count variable from closure | LC 315, 493 |
| Need kth element from quick select | Single value returned directly | LC 215 |
| Need top-k list from quick select | Collect elements on correct side of partition | LC 347, 973 |
| Need tree from build | Return root node; structure is already correct | LC 105, 106, 108 |
| Need all possible results | Collect into list during recursion | LC 241 |
| Need median from binary search | Compute from partition boundary values | LC 4 |
| Need boolean (found/not found) | Return True/False from search | LC 240 |
| Merge k lists into one | Iterative/recursive pairwise merge returns head | LC 23 |
| Result needs deduplication | Use set or sort + deduplicate at end | Varies |

### Common Combine Functions

```python
# Merge two sorted arrays
def merge(left, right):
    result, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    return result + left[i:] + right[j:]

# Merge two sorted linked lists
def merge_lists(l1, l2):
    dummy = ListNode(0)
    curr = dummy
    while l1 and l2:
        if l1.val <= l2.val:
            curr.next = l1; l1 = l1.next
        else:
            curr.next = l2; l2 = l2.next
        curr = curr.next
    curr.next = l1 or l2
    return dummy.next

# Combine expression results
def combine(lefts, rights, op):
    ops = {'+': lambda a,b: a+b, '-': lambda a,b: a-b, '*': lambda a,b: a*b}
    return [ops[op](l, r) for l in lefts for r in rights]
```

---

<a name="pitfalls"></a>
## 6. Common Pitfalls & Solutions

### Pitfall 1: Forgetting the Base Case

**Wrong:**
```python
def merge_sort(arr):
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])    # Infinite recursion when len=1!
    right = merge_sort(arr[mid:])
    return merge(left, right)
```

**Right:**
```python
def merge_sort(arr):
    if len(arr) <= 1:               # BASE CASE: single element is sorted
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)
```

---

### Pitfall 2: Off-by-One in Partition Indices

**Wrong:**
```python
def build(lo, hi):
    mid = (lo + hi) // 2
    node = TreeNode(nums[mid])
    node.left = build(lo, mid)      # Includes mid! Infinite recursion when lo==mid
    node.right = build(mid, hi)     # Includes mid again!
    return node
```

**Right:**
```python
def build(lo, hi):
    if lo > hi:
        return None
    mid = (lo + hi) // 2
    node = TreeNode(nums[mid])
    node.left = build(lo, mid - 1)  # Exclude mid
    node.right = build(mid + 1, hi) # Exclude mid
    return node
```

---

### Pitfall 3: Quick Select Worst-Case with Bad Pivots

**Wrong:**
```python
def quick_select(arr, k):
    pivot = arr[0]                  # Always first element — worst case O(n^2) on sorted input
    lows = [x for x in arr if x < pivot]
    highs = [x for x in arr if x > pivot]
    pivots = [x for x in arr if x == pivot]
    # ...
```

**Right:**
```python
import random

def quick_select(arr, k):
    pivot = random.choice(arr)      # Random pivot — expected O(n)
    lows = [x for x in arr if x < pivot]
    highs = [x for x in arr if x > pivot]
    pivots = [x for x in arr if x == pivot]
    # ...
```

---

### Pitfall 4: Not Handling Duplicates in Quick Select

**Wrong:**
```python
def quick_select(arr, k):
    pivot = random.choice(arr)
    lows = [x for x in arr if x < pivot]
    highs = [x for x in arr if x >= pivot]  # Pivot goes into highs — infinite loop if all equal!
    if k < len(lows):
        return quick_select(lows, k)
    else:
        return quick_select(highs, k - len(lows))  # highs never shrinks!
```

**Right:**
```python
def quick_select(arr, k):
    pivot = random.choice(arr)
    lows = [x for x in arr if x < pivot]
    highs = [x for x in arr if x > pivot]
    pivots = [x for x in arr if x == pivot]   # Separate bucket for duplicates!
    if k < len(lows):
        return quick_select(lows, k)
    elif k < len(lows) + len(pivots):
        return pivot                            # k falls among equal elements
    else:
        return quick_select(highs, k - len(lows) - len(pivots))
```

---

### Pitfall 5: Building Tree from Postorder — Wrong Subtree Order

**Wrong:**
```python
def build(in_lo, in_hi):
    root_val = postorder[post_idx[0]]
    post_idx[0] -= 1
    root = TreeNode(root_val)
    mid = inorder_map[root_val]
    root.left = build(in_lo, mid - 1)    # LEFT first — wrong!
    root.right = build(mid + 1, in_hi)
    return root
```

**Right:**
```python
def build(in_lo, in_hi):
    root_val = postorder[post_idx[0]]
    post_idx[0] -= 1
    root = TreeNode(root_val)
    mid = inorder_map[root_val]
    root.right = build(mid + 1, in_hi)   # RIGHT first! Postorder = L,R,Root
    root.left = build(in_lo, mid - 1)    # So consuming from end: Root, R, L
    return root
```

---

### Pitfall 6: Counting Inversions — Merging Before Counting

**Wrong:**
```python
def merge_sort(arr):
    # ...
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    merged = merge(left, right)     # Merged first!
    count_inversions(left, right)   # Too late — left and right are now sorted and merged
    return merged
```

**Right:**
```python
def merge_sort(arr):
    # ...
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    # Count BEFORE or DURING merge — both halves are sorted, count cross-pairs
    count_cross_pairs(left, right)  # Count first!
    merged = merge(left, right)
    return merged
```

---

### Pitfall 7: Forgetting to Use a HashMap for Inorder Lookups

**Wrong:**
```python
def build(pre_lo, pre_hi, in_lo, in_hi):
    root_val = preorder[pre_lo]
    mid = inorder.index(root_val)   # O(n) search EVERY time! Total O(n^2)
    # ...
```

**Right:**
```python
inorder_map = {val: idx for idx, val in enumerate(inorder)}  # O(n) once

def build(in_lo, in_hi):
    root_val = preorder[pre_idx[0]]
    mid = inorder_map[root_val]     # O(1) lookup!
    # ...
```

---

<a name="recognition"></a>
## 7. Problem Recognition Framework

### Decision Tree

```
START: Can the problem be broken into independent subproblems?
  │
  ├── NO → Consider DP (overlapping subproblems), greedy, or other approach
  │
  └── YES
       │
       ├── Does it involve sorting or counting cross-boundary pairs?
       │    │
       │    ├── YES → PATTERN 1: Merge Sort Variant
       │    │         "Sort an array" → basic merge sort
       │    │         "Count inversions/pairs" → modified merge step
       │    │         "Count smaller after self" → merge with index tracking
       │    │
       │    └── NO
       │         │
       │         ├── Does it ask for kth element or top-k?
       │         │    │
       │         │    ├── YES → PATTERN 2: Quick Select
       │         │    │         "Kth largest" → partition around pivot
       │         │    │         "Top k frequent" → count + select
       │         │    │         "K closest points" → distance + select
       │         │    │
       │         │    └── NO
       │         │         │
       │         │         ├── Does it involve two sorted arrays / binary search?
       │         │         │    │
       │         │         │    ├── YES → PATTERN 3: Binary Search as D&C
       │         │         │    │         "Median of two sorted arrays" → partition search
       │         │         │    │
       │         │         │    └── NO
       │         │         │         │
       │         │         │         ├── Does it involve building a tree?
       │         │         │         │    │
       │         │         │         │    ├── YES → PATTERN 4: Tree Build D&C
       │         │         │         │    │         "Sorted array to BST" → mid as root
       │         │         │         │    │         "From traversals" → find root, split
       │         │         │         │    │
       │         │         │         │    └── NO
       │         │         │         │         │
       │         │         │         │         ├── Does it involve expressions or all combos?
       │         │         │         │         │    │
       │         │         │         │         │    ├── YES → PATTERN 5: Expression D&C
       │         │         │         │         │    │         "All ways to parenthesize"
       │         │         │         │         │    │         "Search sorted 2D matrix"
       │         │         │         │         │    │
       │         │         │         │         │    └── NO
       │         │         │         │         │         │
       │         │         │         │         │         └── Does it merge k sorted things?
       │         │         │         │         │              │
       │         │         │         │         │              ├── YES → PATTERN 6: K-Way Merge
       │         │         │         │         │              │         "Merge k sorted lists"
       │         │         │         │         │              │
       │         │         │         │         │              └── NO → May not be D&C.
       │         │         │         │         │                       Consider other approaches.
```

### Quick Recognition Signals

| Signal in Problem | Likely Pattern | Template |
|-------------------|---------------|----------|
| "Sort" | Merge sort | A |
| "Count pairs with condition across array" | Modified merge sort | A |
| "Count smaller/larger after self" | Merge sort with index tracking | A |
| "Kth largest/smallest" | Quick select | B |
| "Top k" / "k closest" | Quick select or heap | B |
| "Median of sorted arrays" | Binary search partition | Special |
| "Convert sorted array to BST" | Tree build | C |
| "Construct tree from traversals" | Tree build | C |
| "All possible results of expression" | Expression D&C | D |
| "Search sorted 2D matrix" | Staircase / quadrant elimination | Special |
| "Merge k sorted lists" | Pairwise D&C merge | A variant |

---

<a name="checklist"></a>
## 8. Interview Preparation Checklist

### Before the Interview

**Master the fundamentals:**
- [ ] Can write merge sort from scratch in under 5 minutes
- [ ] Can write quick select with proper duplicate handling
- [ ] Understand the Master Theorem for common recurrences
- [ ] Know when D&C is better than brute force (and when it's not)
- [ ] Can explain the three steps: Divide, Conquer, Combine

**Practice pattern recognition:**
- [ ] Can identify merge sort variant problems (counting during merge)
- [ ] Can identify quick select problems (kth element, top-k)
- [ ] Can identify tree construction problems from traversals
- [ ] Know the difference between D&C and DP (independent vs overlapping subproblems)

**Know the patterns:**
- [ ] Merge sort with counting (Template A)
- [ ] Quick select with 3-way partition (Template B)
- [ ] Tree build from sorted data or traversals (Template C)
- [ ] Expression/combination D&C (Template D)

**Common problems solved:**
- [ ] LC 912: Sort an Array
- [ ] LC 493: Reverse Pairs
- [ ] LC 315: Count of Smaller Numbers After Self
- [ ] LC 215: Kth Largest Element
- [ ] LC 4: Median of Two Sorted Arrays
- [ ] LC 108: Convert Sorted Array to BST
- [ ] LC 105: Construct from Preorder and Inorder
- [ ] LC 241: Different Ways to Add Parentheses
- [ ] LC 23: Merge k Sorted Lists

### During the Interview

**1. Clarify (30 seconds)**
- What are the constraints on n? (Determines if O(n log n) is needed vs O(n))
- Are there duplicates?
- Do I need sorted output or just the answer?
- Can I modify the input array?

**2. Identify pattern (30 seconds)**
- Is this a counting problem during sort? → Merge sort
- Is this a selection problem? → Quick select
- Is this a tree construction problem? → Tree build
- Is this an "all possible results" problem? → Expression D&C

**3. Code (5-7 minutes)**
- Write the base case FIRST
- Write the divide step (find mid or split point)
- Write the recursive calls
- Write the combine/merge step
- Handle edge cases (empty input, single element)

**4. Test (1-2 minutes)**
- Empty array / single element
- Already sorted / reverse sorted
- All duplicates
- Odd/even length arrays
- Walk through one small example

**5. Analyze (30 seconds)**
- State the recurrence relation: T(n) = aT(n/b) + O(n^d)
- Apply Master Theorem to get Big-O
- State space complexity (recursion depth + auxiliary space)

---

## 9. Quick Reference Cards

### Merge Sort
```python
def merge_sort(arr):
    if len(arr) <= 1: return arr
    mid = len(arr) // 2
    L, R = merge_sort(arr[:mid]), merge_sort(arr[mid:])
    res, i, j = [], 0, 0
    while i < len(L) and j < len(R):
        if L[i] <= R[j]: res.append(L[i]); i += 1
        else: res.append(R[j]); j += 1
    return res + L[i:] + R[j:]
# T(n) = 2T(n/2) + O(n) = O(n log n), Space O(n)
```

### Quick Select (Kth Smallest)
```python
import random
def quick_select(arr, k):
    pivot = random.choice(arr)
    lo = [x for x in arr if x < pivot]
    hi = [x for x in arr if x > pivot]
    eq = [x for x in arr if x == pivot]
    if k < len(lo): return quick_select(lo, k)
    elif k < len(lo) + len(eq): return pivot
    else: return quick_select(hi, k - len(lo) - len(eq))
# Average O(n), Worst O(n^2), Space O(n)
```

### Tree Build (Sorted Array to BST)
```python
def build(nums, lo, hi):
    if lo > hi: return None
    mid = (lo + hi) // 2
    node = TreeNode(nums[mid])
    node.left = build(nums, lo, mid - 1)
    node.right = build(nums, mid + 1, hi)
    return node
# O(n) time, O(log n) space
```

### Build from Preorder + Inorder
```python
def build(preorder, inorder):
    idx_map = {v: i for i, v in enumerate(inorder)}
    pre_i = [0]
    def helper(lo, hi):
        if lo > hi: return None
        val = preorder[pre_i[0]]; pre_i[0] += 1
        node = TreeNode(val)
        mid = idx_map[val]
        node.left = helper(lo, mid - 1)
        node.right = helper(mid + 1, hi)
        return node
    return helper(0, len(inorder) - 1)
# O(n) time, O(n) space
```

### Merge Two Sorted Linked Lists
```python
def merge(l1, l2):
    dummy = curr = ListNode(0)
    while l1 and l2:
        if l1.val <= l2.val: curr.next = l1; l1 = l1.next
        else: curr.next = l2; l2 = l2.next
        curr = curr.next
    curr.next = l1 or l2
    return dummy.next
```

### Master Theorem Cheat Sheet
```
T(n) = aT(n/b) + O(n^d)

Case 1: d > log_b(a) → O(n^d)
Case 2: d = log_b(a) → O(n^d * log n)
Case 3: d < log_b(a) → O(n^(log_b(a)))
```

---

## 10. Complexity Reference

| Algorithm / Problem | Time Complexity | Space Complexity | Recurrence |
|---------------------|----------------|-----------------|------------|
| Merge Sort | O(n log n) | O(n) | T(n) = 2T(n/2) + O(n) |
| Quick Select (avg) | O(n) | O(n) | T(n) = T(n/2) + O(n) |
| Quick Select (worst) | O(n^2) | O(n) | T(n) = T(n-1) + O(n) |
| Quick Sort (avg) | O(n log n) | O(log n) | T(n) = 2T(n/2) + O(n) |
| Binary Search | O(log n) | O(1) | T(n) = T(n/2) + O(1) |
| Median of Two Sorted Arrays | O(log min(m,n)) | O(1) | T(n) = T(n/2) + O(1) |
| Sorted Array to BST | O(n) | O(log n) | T(n) = 2T(n/2) + O(1) |
| Build Tree from Traversals | O(n) | O(n) | T(n) = 2T(n/2) + O(1) |
| Merge k Sorted Lists | O(N log k) | O(log k) | T(k) = 2T(k/2) + O(N) |
| Count Inversions | O(n log n) | O(n) | T(n) = 2T(n/2) + O(n) |
| Count Smaller After Self | O(n log n) | O(n) | T(n) = 2T(n/2) + O(n) |
| Reverse Pairs | O(n log n) | O(n) | T(n) = 2T(n/2) + O(n) |
| Different Ways to Parenthesize | O(C_n) Catalan | O(C_n) | Catalan recurrence |
| Search 2D Matrix II | O(m + n) | O(1) | Eliminates 1 row or col per step |
| K Closest Points (avg) | O(n) | O(n) | T(n) = T(n/2) + O(n) |
| Closest Pair of Points | O(n log n) | O(n) | T(n) = 2T(n/2) + O(n) |

### Master Theorem Quick Reference

| Recurrence | a | b | d | log_b(a) | Case | Result |
|-----------|---|---|---|----------|------|--------|
| T(n) = 2T(n/2) + O(n) | 2 | 2 | 1 | 1 | 2 (d = log_b a) | O(n log n) |
| T(n) = 2T(n/2) + O(1) | 2 | 2 | 0 | 1 | 3 (d < log_b a) | O(n) |
| T(n) = T(n/2) + O(1) | 1 | 2 | 0 | 0 | 2 (d = log_b a) | O(log n) |
| T(n) = T(n/2) + O(n) | 1 | 2 | 1 | 0 | 1 (d > log_b a) | O(n) |
| T(n) = 4T(n/2) + O(n) | 4 | 2 | 1 | 2 | 3 (d < log_b a) | O(n^2) |
| T(n) = 3T(n/2) + O(n) | 3 | 2 | 1 | 1.58 | 3 (d < log_b a) | O(n^1.58) |
| T(n) = 2T(n/2) + O(n^2) | 2 | 2 | 2 | 1 | 1 (d > log_b a) | O(n^2) |
| T(n) = 7T(n/2) + O(n^2) | 7 | 2 | 2 | 2.81 | 3 (d < log_b a) | O(n^2.81) Strassen |

---

## Final Thoughts

**Remember:**
1. D&C requires **independent** subproblems — if they overlap, use DP
2. The **combine step** is where the algorithm's cleverness lives — get this right
3. **Merge sort** is the workhorse for counting cross-boundary relationships
4. **Quick select** gives O(n) average for selection problems — always use random pivots
5. **Tree construction** from traversals always follows the same pattern: find root, split, recurse
6. Always state the **recurrence relation** and apply the **Master Theorem** in interviews

**When stuck:**
1. Ask: "Can I split this problem into two independent halves?"
2. Ask: "What information do I gain by solving each half?"
3. Ask: "How do I combine the half-solutions? Is combining cheaper than brute force?"
4. If combining is not cheap, consider whether a different split strategy helps
5. If subproblems overlap, pivot to Dynamic Programming
6. Remember: D&C problems often have elegant O(n log n) solutions hiding behind O(n^2) brute force

---

## Appendix: Conditional Quick Reference

This table lists every key condition used in this handbook, its plain-English meaning, and the intuition behind it.

### A. Base Case & Termination Conditions

| Condition | Plain English | Why it works |
|-----------|---------------|--------------|
| `if len(arr) <= 1` | "0 or 1 elements — already solved" | Single element is trivially sorted/selected. Using `<= 1` catches both empty and singleton; `== 0` would miss the singleton and loop infinitely on a 1-element array |
| `if lo > hi` (tree build) | "Index range is empty — no node here" | `lo == hi` means exactly one element (a leaf). `lo > hi` means the range has collapsed past that leaf — return None. Using `>= hi` would skip valid leaf nodes |
| `if lo == hi` (quick select / merge k lists) | "Only one candidate / one list left" | In quick select, one element must be the answer. In merge k lists, one list has nothing to merge with — return it directly |
| `if in_lo > in_hi` (tree from traversal) | "Inorder range is empty — no subtree here" | If empty, don't consume the next traversal element as a phantom root. Consuming it would shift all subsequent roots one position left, corrupting the entire tree |
| `if n <= 3` (closest pair) | "Brute force the tiny base case" | Dividing 3 points gives halves of 1 and 2 — no efficiency gain. With ≤3 points there are ≤3 pairs; checking all is O(1). Going smaller would degenerate the strip-check logic |
| `if len(pts) <= k` (k closest) | "Fewer points than k — all qualify" | If we have fewer candidates than k, every one is among the k closest. Also catches the single-element base case implicitly |

### B. Partition & Selection Conditions

| Condition | Plain English | Why it works |
|-----------|---------------|--------------|
| `if k < len(lows)` (quick select) | "kth position lives entirely within the smaller elements" | lows occupies positions 0..len(lows)-1. Strict `<` means k falls inside that range (not on the boundary). k == len(lows) would land in pivots |
| `elif k < len(lows) + len(pivots)` | "kth position lands on a pivot value" | All pivots are identical, so any of them is the answer. The range [len(lows), len(lows)+len(pivots)-1] belongs to the pivot group |
| `if k <= len(lows)` (k closest, `<=` not `<`) | "k closest all live among the smaller distances" | We want positions 0..k-1. If k ≤ len(lows), lows contains at least k points — recurse there. Using `<` would miss the case k == len(lows) |
| `elif k <= len(lows) + len(mids)` | "k lands in the equal-distance group" | All mids are the same distance. Take all of lows plus exactly (k - len(lows)) from mids to reach k total |
| `if store == k_smallest` (in-place quick select) | "Pivot landed exactly at the kth boundary" | Everything right of `store` is at least as frequent; exactly k elements are on the right. Done — no further recursion needed |

### C. Merge & Combine Conditions

| Condition | Plain English | Why it works |
|-----------|---------------|--------------|
| `while i < len(left) and j < len(right)` | "Both sides still have elements to compare" | Comparison requires two operands. Once either side is exhausted, the other is already sorted and greater than everything placed — append it directly |
| `if left[i] <= right[j]` (stable sort) | "Left wins ties" | `<=` takes from left when equal, preserving original relative order of equal elements (stability). Changing to `<` makes the sort unstable |
| `root.right = build(...); root.left = build(...)` (postorder) | "Build right subtree before left" | Postorder is left→right→ROOT. Consuming from the END gives ROOT, then right root, then left root. Building right first matches this consumption order |
| `if ch in '+-*'` (expression D&C) | "Only split at operators, never inside numbers" | Splitting inside "23" would create phantom numbers "2" and "3". Operators are the only valid parenthesization boundaries |
| `if not results` (expression base case) | "No operators found — this IS a number" | A sub-expression with no operators is a raw integer literal. `int(expr)` converts it. Without this, expressions like "5" would return an empty list |
| `curr.next = l1 if l1 else l2` (linked list merge) | "Attach the non-exhausted remainder directly" | The remaining nodes are already sorted and all larger than everything placed. No need to traverse — O(1) tail attachment instead of O(n) node-by-node copy |

### D. Binary Search & Elimination Conditions

| Condition | Plain English | Why it works |
|-----------|---------------|--------------|
| `if len(nums1) > len(nums2): swap` (median) | "Binary search on the shorter array" | Binary searching m gives O(log m). Always searching the shorter array gives O(log(min(m,n))). Without the swap, we'd do unnecessary O(log(max(m,n))) work |
| `while lo <= hi` (median binary search) | "Search range is non-empty" | `lo == hi` still represents one valid partition count to check. `lo > hi` means we've exhausted all partitions — target not found |
| `if left1 <= right2 and left2 <= right1` | "Every left-side element ≤ every right-side element" | This is the partition correctness invariant. Both cross-comparisons are needed because elements interleave from two different arrays. If both hold, the median is at this partition |
| `elif left1 > right2: hi = i - 1` | "Too many elements from nums1 on the left" | left1 is too large — we included too much of nums1. Shrink the nums1 partition by moving hi left |
| `while row < rows and col >= 0` (2D matrix) | "Still inside the search zone" | row == rows: fallen off the bottom. col < 0: fallen off the left. Either exit means target is absent — the top-right staircase walk has covered all reachable cells |
| `while j < len(strip) and (strip[j][1] - strip[i][1]) < d` | "Only check strip points within y-distance d" | Strip is y-sorted. Once y-gap ≥ d, Euclidean distance > d regardless of x — skip. Proven at most 8 points fit in a d×2d box, so inner loop runs ≤7 times → O(n) strip total |

**D&C vs Other Paradigms:**
- **D&C vs DP:** D&C has independent subproblems; DP has overlapping ones. If you find yourself recomputing the same subproblem, add memoization (turning D&C into DP).
- **D&C vs Greedy:** Greedy makes one local choice and never looks back. D&C explores both halves and combines. Use D&C when greedy can't guarantee optimality.
- **D&C vs Two Pointers:** Two pointers works on sorted data with a single pass. D&C creates the sorted data (merge sort) or uses partitioning (quick select).

---

## Appendix: Practice Problem Set

### Easy
- 108. Convert Sorted Array to Binary Search Tree
- 912. Sort an Array (implement merge sort)
- 169. Majority Element (D&C approach)

### Medium
- 215. Kth Largest Element in an Array
- 347. Top K Frequent Elements
- 973. K Closest Points to Origin
- 105. Construct Binary Tree from Preorder and Inorder Traversal
- 106. Construct Binary Tree from Inorder and Postorder Traversal
- 240. Search a 2D Matrix II
- 241. Different Ways to Add Parentheses
- 148. Sort List (merge sort on linked list)
- 395. Longest Substring with At Least K Repeating Characters

### Hard
- 4. Median of Two Sorted Arrays
- 23. Merge k Sorted Lists
- 315. Count of Smaller Numbers After Self
- 493. Reverse Pairs
- 327. Count of Range Sum
- 312. Burst Balloons (D&C with memoization)
- 932. Beautiful Array

**Recommended Practice Order:**
1. Start with LC 912, 108 (basic merge sort and tree build)
2. Practice LC 215, 347 (quick select)
3. Master LC 315, 493 (modified merge sort with counting)
4. Do LC 105, 106 (tree reconstruction from traversals)
5. Tackle LC 4 (median of two sorted arrays — hardest binary search D&C)
6. Attempt LC 241 (expression D&C)
7. Finish with LC 23 (merge k sorted lists — combines merge with D&C)

Good luck with your interview preparation!
