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
    if lo >= hi:
        return base_case
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

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

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
    if len(arr) == 1:
        return arr[0]

    pivot = random.choice(arr)

    lows = [x for x in arr if x < pivot]
    highs = [x for x in arr if x > pivot]
    pivots = [x for x in arr if x == pivot]

    if k < len(lows):
        return quick_select_template(lows, k)
    elif k < len(lows) + len(pivots):
        return pivot
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
    if lo == hi:
        return base_solution(problem, lo)

    results = []
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
    if len(nums) <= 1:
        return nums

    mid = len(nums) // 2
    left = sortArray(nums[:mid])
    right = sortArray(nums[mid:])

    # Merge two sorted halves
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
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
    if not nums:
        return 0

    count = [0]

    def merge_sort(arr):
        if len(arr) <= 1:
            return arr

        mid = len(arr) // 2
        left = merge_sort(arr[:mid])
        right = merge_sort(arr[mid:])

        # Count reverse pairs across left and right
        j = 0
        for i in range(len(left)):
            while j < len(right) and left[i] > 2 * right[j]:
                j += 1
            count[0] += j

        # Standard merge
        merged = []
        i = j = 0
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
        if len(arr) <= 1:
            return arr

        mid = len(arr) // 2
        left = merge_sort(arr[:mid])
        right = merge_sort(arr[mid:])

        merged = []
        i = j = 0
        right_count = 0  # How many from right have been placed

        while i < len(left) and j < len(right):
            if left[i][1] <= right[j][1]:
                # left[i] is placed; all right elements placed so far are smaller
                counts[left[i][0]] += right_count
                merged.append(left[i])
                i += 1
            else:
                # right[j] is smaller, place it
                right_count += 1
                merged.append(right[j])
                j += 1

        # Remaining left elements
        while i < len(left):
            counts[left[i][0]] += right_count
            merged.append(left[i])
            i += 1

        # Remaining right elements
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
        if len(arr) == 1:
            return arr[0]

        pivot = random.choice(arr)
        lows = [x for x in arr if x < pivot]
        highs = [x for x in arr if x > pivot]
        pivots = [x for x in arr if x == pivot]

        if k < len(lows):
            return quick_select(lows, k)
        elif k < len(lows) + len(pivots):
            return pivot
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
        if lo == hi:
            return

        pivot_idx = random.randint(lo, hi)
        pivot_freq = freq[unique[pivot_idx]]

        # Move pivot to end
        unique[pivot_idx], unique[hi] = unique[hi], unique[pivot_idx]

        # Partition: elements with freq < pivot_freq go left
        store = lo
        for i in range(lo, hi):
            if freq[unique[i]] < pivot_freq:
                unique[store], unique[i] = unique[i], unique[store]
                store += 1

        # Move pivot to final position
        unique[store], unique[hi] = unique[hi], unique[store]

        # store is the final index of the pivot
        if store == k_smallest:
            return
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
    # Ensure nums1 is the shorter array
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1

    m, n = len(nums1), len(nums2)
    half = (m + n + 1) // 2

    lo, hi = 0, m

    while lo <= hi:
        i = (lo + hi) // 2  # Partition index in nums1
        j = half - i         # Partition index in nums2

        # Edge cases: if partition is at boundary, use -inf or inf
        left1 = nums1[i - 1] if i > 0 else float('-inf')
        right1 = nums1[i] if i < m else float('inf')
        left2 = nums2[j - 1] if j > 0 else float('-inf')
        right2 = nums2[j] if j < n else float('inf')

        if left1 <= right2 and left2 <= right1:
            # Found correct partition
            if (m + n) % 2 == 1:
                return max(left1, left2)
            else:
                return (max(left1, left2) + min(right1, right2)) / 2
        elif left1 > right2:
            hi = i - 1  # Too many from nums1
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
        if in_lo > in_hi:
            return None

        root_val = preorder[pre_idx[0]]
        pre_idx[0] += 1

        root = TreeNode(root_val)
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
        if in_lo > in_hi:
            return None

        root_val = postorder[post_idx[0]]
        post_idx[0] -= 1

        root = TreeNode(root_val)
        mid = inorder_map[root_val]

        # Build RIGHT subtree first (postorder: left, right, ROOT)
        # We consume from the end, so right comes before left
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
    row, col = 0, cols - 1  # Start at top-right corner

    while row < rows and col >= 0:
        val = matrix[row][col]
        if val == target:
            return True
        elif val > target:
            col -= 1  # Eliminate this column
        else:
            row += 1  # Eliminate this row

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
        if expr in memo:
            return memo[expr]

        results = []

        for i, ch in enumerate(expr):
            if ch in '+-*':
                # Split at operator
                left_results = compute(expr[:i])
                right_results = compute(expr[i + 1:])

                # Combine all left results with all right results
                for l in left_results:
                    for r in right_results:
                        if ch == '+':
                            results.append(l + r)
                        elif ch == '-':
                            results.append(l - r)
                        else:
                            results.append(l * r)

        # Base case: no operators found, it's a number
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
        if len(pts) <= k:
            return pts

        pivot = dist(random.choice(pts))
        lows = [p for p in pts if dist(p) < pivot]
        mids = [p for p in pts if dist(p) == pivot]
        highs = [p for p in pts if dist(p) > pivot]

        if k <= len(lows):
            return quick_select(lows, k)
        elif k <= len(lows) + len(mids):
            return lows + mids[:k - len(lows)]
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
        while l1 and l2:
            if l1.val <= l2.val:
                curr.next = l1
                l1 = l1.next
            else:
                curr.next = l2
                l2 = l2.next
            curr = curr.next
        curr.next = l1 if l1 else l2
        return dummy.next

    def merge_lists(lists, lo, hi):
        """D&C: merge lists[lo..hi] by splitting in half."""
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
        if n <= 3:
            # Brute force for small inputs
            min_d = float('inf')
            for i in range(n):
                for j in range(i + 1, n):
                    min_d = min(min_d, dist(pts_x[i], pts_x[j]))
            return min_d

        mid = n // 2
        mid_x = pts_x[mid][0]

        # Split points sorted by y into left and right
        left_x = pts_x[:mid]
        right_x = pts_x[mid:]
        left_y = [p for p in pts_y if p[0] <= mid_x]
        right_y = [p for p in pts_y if p[0] > mid_x]

        # Fix: ensure split is correct even with duplicate x-coords
        if len(left_y) > len(left_x):
            excess = len(left_y) - len(left_x)
            right_y = left_y[-excess:] + right_y
            left_y = left_y[:-excess]

        d_left = solve(left_x, left_y)
        d_right = solve(right_x, right_y)
        d = min(d_left, d_right)

        # Check strip: points within d of the dividing line
        strip = [p for p in pts_y if abs(p[0] - mid_x) < d]

        for i in range(len(strip)):
            j = i + 1
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
