# Ultimate Coding Interview Patterns Cheat Sheet
## Community-Curated, Condensed, Memorizable Reference (2025-2026)

Compiled from the highest-rated GitHub repos, LeetCode Discuss posts, Reddit recommendations, Tech Interview Handbook, AlgoMaster, NeetCode, AlgoMonster, Design Gurus, and other top community resources.

---

# TABLE OF CONTENTS

1. [Master Decision Framework: "If You See X, Use Y"](#1-master-decision-framework)
2. [The 15 Core LeetCode Patterns](#2-the-15-core-leetcode-patterns)
3. [Pattern 1: Prefix Sum](#3-prefix-sum)
4. [Pattern 2: Two Pointers](#4-two-pointers)
5. [Pattern 3: Sliding Window](#5-sliding-window)
6. [Pattern 4: Fast & Slow Pointers](#6-fast--slow-pointers)
7. [Pattern 5: Linked List In-Place Reversal](#7-linked-list-in-place-reversal)
8. [Pattern 6: Monotonic Stack](#8-monotonic-stack)
9. [Pattern 7: Top K Elements (Heap)](#9-top-k-elements)
10. [Pattern 8: Overlapping Intervals](#10-overlapping-intervals)
11. [Pattern 9: Modified Binary Search](#11-modified-binary-search)
12. [Pattern 10: Binary Tree Traversal](#12-binary-tree-traversal)
13. [Pattern 11: DFS (Depth-First Search)](#13-dfs)
14. [Pattern 12: BFS (Breadth-First Search)](#14-bfs)
15. [Pattern 13: Matrix/Grid Traversal](#15-matrix-grid-traversal)
16. [Pattern 14: Backtracking](#16-backtracking)
17. [Pattern 15: Dynamic Programming](#17-dynamic-programming)
18. [Greedy Algorithm Patterns](#18-greedy-algorithm-patterns)
19. [Divide and Conquer Patterns](#19-divide-and-conquer)
20. [Union-Find / Disjoint Set](#20-union-find)
21. [Topological Sort](#21-topological-sort)
22. [Trie Patterns](#22-trie-patterns)
23. [20 Dynamic Programming Sub-Patterns](#23-dp-sub-patterns)
24. [Big-O Complexity Cheat Sheet](#24-big-o-cheat-sheet)
25. [Source URLs & References](#25-sources)

---

# 1. MASTER DECISION FRAMEWORK

## "If You See X, Use Y" -- Quick Pattern Recognition Rules

Source: Community consensus from AlgoMaster, AlgoMonster, LeetCode Discuss, NeetCode, Design Gurus

```
IF problem says...                          THEN use...
-----------------------------------------   ----------------------------------------
"subarray" or "substring" (contiguous)   --> Sliding Window
"subarray sum equals k"                  --> Prefix Sum + HashMap
"sorted array" + "find pair/target"      --> Two Pointers (from both ends)
"linked list" + "cycle"                  --> Fast & Slow Pointers
"linked list" + "reverse"               --> In-place Reversal
"top K" / "K largest" / "K smallest"     --> Heap (min-heap or max-heap)
"K closest" / "K frequent"              --> Heap
"merge intervals" / "overlapping"        --> Sort by start + Merge Intervals
"insert interval"                        --> Merge Intervals
"find position" / "sorted" / "O(log n)" --> Binary Search
"minimum/maximum in rotated sorted"      --> Modified Binary Search
"all permutations / combinations"        --> Backtracking
"all subsets / power set"                --> Backtracking
"word search in grid"                    --> Backtracking + DFS
"shortest path" (unweighted)             --> BFS
"number of islands" / "connected comp."  --> DFS or BFS on grid
"level order traversal"                  --> BFS with queue
"tree diameter / height / path"          --> DFS (recursive)
"course schedule" / "dependencies"       --> Topological Sort
"detect cycle in directed graph"         --> Topological Sort (Kahn's)
"union/merge components"                 --> Union-Find
"redundant connection"                   --> Union-Find
"next greater/smaller element"           --> Monotonic Stack
"largest rectangle in histogram"         --> Monotonic Stack
"prefix/word search/autocomplete"        --> Trie
"optimize overlapping subproblems"       --> Dynamic Programming
"min cost / max profit / count ways"     --> Dynamic Programming
"0/1 choices per item"                   --> 0/1 Knapsack DP
"unlimited choices per item"             --> Unbounded Knapsack DP
"edit distance / string matching"        --> String DP (2D table)
"longest increasing subsequence"         --> LIS DP (patience sorting)
"palindrome partitioning"               --> Interval DP
"matrix path min/max cost"              --> Grid DP
"best time to buy/sell stock"           --> State Machine DP
"partitioning into groups"              --> Bitmask DP
"divide array into two equal halves"    --> Meet in the Middle
"stream of data / median"              --> Two Heaps (min + max)
```

## The 3-Question Rapid Classifier (from Reddit/Blind community)

```
Q1: Is the input SORTED or can it be sorted?
    YES -> Two Pointers, Binary Search, or Merge Intervals
    NO  -> Continue to Q2

Q2: Is the problem about a SUBARRAY, SUBSTRING, or WINDOW?
    YES -> Sliding Window or Prefix Sum
    NO  -> Continue to Q3

Q3: Does the problem involve a TREE, GRAPH, or GRID?
    YES -> DFS, BFS, Topological Sort, or Union-Find
    NO  -> Consider Heap, Stack, DP, or Backtracking
```

---

# 2. THE 15 CORE LEETCODE PATTERNS

Source: AlgoMaster (blog.algomaster.io) -- "87% of FAANG interview questions are built around 10-12 core patterns"

```
 #  Pattern                     Key Data Structure    Time         Space
--- --------------------------  --------------------  -----------  --------
 1  Prefix Sum                  Array + HashMap       O(n)         O(n)
 2  Two Pointers                Array/String          O(n)         O(1)
 3  Sliding Window              Array/String          O(n)         O(k)
 4  Fast & Slow Pointers        Linked List           O(n)         O(1)
 5  Linked List In-place Rev.   Linked List           O(n)         O(1)
 6  Monotonic Stack             Stack                 O(n)         O(n)
 7  Top 'K' Elements            Heap                  O(n log k)   O(k)
 8  Overlapping Intervals       Array (sorted)        O(n log n)   O(n)
 9  Modified Binary Search      Array                 O(log n)     O(1)
10  Binary Tree Traversal       Tree                  O(n)         O(h)
11  DFS                         Graph/Tree/Grid       O(V+E)       O(V)
12  BFS                         Graph/Tree/Grid       O(V+E)       O(V)
13  Matrix/Grid Traversal       2D Array              O(m*n)       O(m*n)
14  Backtracking                Recursion Tree        O(2^n/n!)    O(n)
15  Dynamic Programming         Table/Memo            Varies       Varies
```

---

# 3. PREFIX SUM

## When to Use
- "Subarray sum equals K"
- Range sum queries
- "Count subarrays with sum..."
- Cumulative frequency problems

## Template (Python)
```python
# Build prefix sum
prefix = [0] * (len(nums) + 1)
for i in range(len(nums)):
    prefix[i + 1] = prefix[i] + nums[i]

# Range sum query [i, j] inclusive
range_sum = prefix[j + 1] - prefix[i]
```

## Subarray Sum Equals K (with HashMap)
```python
def subarraySum(nums, k):
    count = 0
    curr_sum = 0
    prefix_counts = {0: 1}  # sum -> frequency
    for num in nums:
        curr_sum += num
        # If (curr_sum - k) was seen before, those subarrays sum to k
        count += prefix_counts.get(curr_sum - k, 0)
        prefix_counts[curr_sum] = prefix_counts.get(curr_sum, 0) + 1
    return count
```

## Key Problems
- LC 560: Subarray Sum Equals K
- LC 523: Continuous Subarray Sum
- LC 974: Subarray Sums Divisible by K
- LC 238: Product of Array Except Self

---

# 4. TWO POINTERS

## When to Use
- Sorted array + find pair with target sum
- Remove duplicates in-place
- Comparing strings/arrays from both ends
- Container with most water / trapping rain water
- Partition problems

## Decision: Two Pointers vs Sliding Window
```
"Pair/Triplet/Specific elements"  --> Two Pointers (90%)
"Subarray/Substring (contiguous)" --> Sliding Window (90%)
```

## Template: Opposite Direction
```python
def two_pointer_opposite(arr, target):
    left, right = 0, len(arr) - 1
    while left < right:
        curr = arr[left] + arr[right]
        if curr == target:
            return [left, right]
        elif curr < target:
            left += 1
        else:
            right -= 1
```

## Template: Same Direction
```python
def two_pointer_same(arr):
    slow = 0
    for fast in range(len(arr)):
        if condition(arr[fast]):
            arr[slow] = arr[fast]
            slow += 1
    return slow  # new length
```

## Key Problems
- LC 1: Two Sum (use HashMap for unsorted)
- LC 15: 3Sum
- LC 11: Container With Most Water
- LC 42: Trapping Rain Water
- LC 26: Remove Duplicates from Sorted Array
- LC 125: Valid Palindrome

---

# 5. SLIDING WINDOW

## When to Use
- "Longest/shortest subarray/substring with condition"
- "Maximum/minimum sum subarray of size K"
- Contiguous sequence optimization

## Rule of Thumb
```
Fixed window size given?    --> Fixed Sliding Window
"Longest/shortest" + condition --> Variable Sliding Window
```

## Template: Variable-Size Window
```python
def sliding_window_variable(s, condition):
    left = 0
    window = {}  # or any state tracker
    result = 0

    for right in range(len(s)):
        # EXPAND: add s[right] to window state
        window[s[right]] = window.get(s[right], 0) + 1

        # SHRINK: while window is invalid
        while not is_valid(window, condition):
            window[s[left]] -= 1
            if window[s[left]] == 0:
                del window[s[left]]
            left += 1

        # UPDATE result with current valid window
        result = max(result, right - left + 1)

    return result
```

## Template: Fixed-Size Window
```python
def sliding_window_fixed(nums, k):
    window_sum = sum(nums[:k])
    best = window_sum

    for right in range(k, len(nums)):
        window_sum += nums[right]       # add new element
        window_sum -= nums[right - k]   # remove old element
        best = max(best, window_sum)

    return best
```

## Key Problems
- LC 3: Longest Substring Without Repeating Characters
- LC 76: Minimum Window Substring
- LC 239: Sliding Window Maximum
- LC 209: Minimum Size Subarray Sum
- LC 424: Longest Repeating Character Replacement
- LC 567: Permutation in String

---

# 6. FAST & SLOW POINTERS (Floyd's Tortoise and Hare)

## When to Use
- Cycle detection in linked list
- Finding the middle of a linked list
- Finding the start of a cycle
- Happy number problem

## Template: Cycle Detection
```python
def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False
```

## Template: Find Cycle Start
```python
def find_cycle_start(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            # Reset one pointer to head
            slow = head
            while slow != fast:
                slow = slow.next
                fast = fast.next
            return slow  # cycle start
    return None
```

## Template: Find Middle
```python
def find_middle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow  # middle node
```

## Key Problems
- LC 141: Linked List Cycle
- LC 142: Linked List Cycle II
- LC 876: Middle of the Linked List
- LC 202: Happy Number
- LC 287: Find the Duplicate Number

---

# 7. LINKED LIST IN-PLACE REVERSAL

## When to Use
- "Reverse a linked list"
- "Reverse between positions m and n"
- "Reverse in groups of K"

## Template: Full Reversal
```python
def reverse_list(head):
    prev = None
    curr = head
    while curr:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node
    return prev  # new head
```

## Template: Reverse Between m and n
```python
def reverse_between(head, m, n):
    dummy = ListNode(0, head)
    prev = dummy
    for _ in range(m - 1):
        prev = prev.next

    curr = prev.next
    for _ in range(n - m):
        next_node = curr.next
        curr.next = next_node.next
        next_node.next = prev.next
        prev.next = next_node

    return dummy.next
```

## Key Problems
- LC 206: Reverse Linked List
- LC 92: Reverse Linked List II
- LC 25: Reverse Nodes in k-Group
- LC 24: Swap Nodes in Pairs

---

# 8. MONOTONIC STACK

## When to Use
- "Next greater/smaller element"
- "Previous greater/smaller element"
- "Largest rectangle in histogram"
- "Stock span" problems
- Any problem needing nearest larger/smaller values

## Core Rule
```
Looking for NEXT/PREVIOUS GREATER  --> Maintain DECREASING stack
Looking for NEXT/PREVIOUS SMALLER  --> Maintain INCREASING stack

Mnemonic: "Decreasing finds Greater, Increasing finds Smaller" (DG-IS)
```

## Template: Next Greater Element (Decreasing Stack)
```python
def next_greater(nums):
    n = len(nums)
    result = [-1] * n
    stack = []  # stores indices

    for i in range(n):
        # Pop all elements smaller than current
        while stack and nums[stack[-1]] < nums[i]:
            idx = stack.pop()
            result[idx] = nums[i]
        stack.append(i)

    return result
```

## Template: Next Smaller Element (Increasing Stack)
```python
def next_smaller(nums):
    n = len(nums)
    result = [-1] * n
    stack = []  # stores indices

    for i in range(n):
        while stack and nums[stack[-1]] > nums[i]:
            idx = stack.pop()
            result[idx] = nums[i]
        stack.append(i)

    return result
```

## Template: Previous Greater Element
```python
def previous_greater(nums):
    n = len(nums)
    result = [-1] * n
    stack = []  # stores indices

    for i in range(n):
        while stack and nums[stack[-1]] <= nums[i]:
            stack.pop()
        if stack:
            result[i] = nums[stack[-1]]
        stack.append(i)

    return result
```

## Template: Previous Smaller Element
```python
def previous_smaller(nums):
    n = len(nums)
    result = [-1] * n
    stack = []

    for i in range(n):
        while stack and nums[stack[-1]] >= nums[i]:
            stack.pop()
        if stack:
            result[i] = nums[stack[-1]]
        stack.append(i)

    return result
```

## All-in-One: Get Both Previous AND Next in Single Pass
```python
def all_nearest(nums):
    """Get previous_smaller and next_smaller in one pass."""
    n = len(nums)
    prev_smaller = [-1] * n
    next_smaller = [-1] * n
    stack = []

    for i in range(n):
        while stack and nums[stack[-1]] >= nums[i]:
            idx = stack.pop()
            next_smaller[idx] = nums[i]  # current is next smaller for popped
        if stack:
            prev_smaller[i] = nums[stack[-1]]  # top is previous smaller for current
        stack.append(i)

    return prev_smaller, next_smaller
```

## Complexity: O(n) time, O(n) space -- each element pushed/popped at most once

## Key Problems
- LC 496: Next Greater Element I
- LC 503: Next Greater Element II (circular)
- LC 84: Largest Rectangle in Histogram
- LC 85: Maximal Rectangle
- LC 739: Daily Temperatures
- LC 901: Online Stock Span
- LC 42: Trapping Rain Water (monotonic stack approach)

---

# 9. TOP K ELEMENTS (HEAP)

## When to Use
- "Find K largest / smallest"
- "K closest points"
- "K most frequent"
- "Median from data stream"

## Decision Rule
```
K SMALLEST --> Use MAX-heap of size K (pop when size > K)
K LARGEST  --> Use MIN-heap of size K (pop when size > K)
Alternatively: negate values to flip min-heap to max-heap in Python
```

## Template: K Largest Elements
```python
import heapq

def k_largest(nums, k):
    # Min-heap of size k -- smallest of the k largest is at top
    heap = []
    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)  # remove smallest
    return heap  # contains k largest elements
```

## Template: K Most Frequent
```python
from collections import Counter
import heapq

def top_k_frequent(nums, k):
    counts = Counter(nums)
    return heapq.nlargest(k, counts.keys(), key=counts.get)
```

## Template: Two Heaps (Median of Stream)
```python
import heapq

class MedianFinder:
    def __init__(self):
        self.small = []  # max-heap (negate values)
        self.large = []  # min-heap

    def addNum(self, num):
        heapq.heappush(self.small, -num)
        # Ensure max of small <= min of large
        if self.small and self.large and (-self.small[0] > self.large[0]):
            val = -heapq.heappop(self.small)
            heapq.heappush(self.large, val)
        # Balance sizes
        if len(self.small) > len(self.large) + 1:
            val = -heapq.heappop(self.small)
            heapq.heappush(self.large, val)
        elif len(self.large) > len(self.small):
            val = heapq.heappop(self.large)
            heapq.heappush(self.small, -val)

    def findMedian(self):
        if len(self.small) > len(self.large):
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2
```

## Key Problems
- LC 215: Kth Largest Element
- LC 347: Top K Frequent Elements
- LC 295: Find Median from Data Stream
- LC 973: K Closest Points to Origin
- LC 703: Kth Largest Element in a Stream
- LC 23: Merge K Sorted Lists

---

# 10. OVERLAPPING INTERVALS

## When to Use
- "Merge intervals"
- "Insert interval"
- "Non-overlapping intervals"
- "Meeting rooms"
- Any problem with time ranges, schedules, or ranges

## Core Rule
```
Step 1: SORT by start time (almost always)
Step 2: Compare current.end with next.start
  - If current.end >= next.start --> OVERLAP --> merge
  - If current.end < next.start  --> NO OVERLAP --> move on
```

## Template: Merge Intervals
```python
def merge(intervals):
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]

    for start, end in intervals[1:]:
        if start <= merged[-1][1]:  # overlapping
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return merged
```

## Template: Find Minimum Meeting Rooms (Sweep Line)
```python
import heapq

def minMeetingRooms(intervals):
    intervals.sort(key=lambda x: x[0])
    heap = []  # end times of ongoing meetings

    for start, end in intervals:
        if heap and heap[0] <= start:
            heapq.heappop(heap)  # room freed
        heapq.heappush(heap, end)

    return len(heap)
```

## Key Problems
- LC 56: Merge Intervals
- LC 57: Insert Interval
- LC 435: Non-overlapping Intervals
- LC 252: Meeting Rooms
- LC 253: Meeting Rooms II
- LC 986: Interval List Intersections

---

# 11. MODIFIED BINARY SEARCH

## When to Use
- "Find target in sorted array"
- "Find first/last occurrence"
- "Search in rotated sorted array"
- "Find minimum in rotated array"
- "Find peak element"
- Any problem with O(log n) requirement
- "Find boundary / threshold"

## Core Rule
```
Binary Search works when you can define a CONDITION that partitions
the search space into two halves: one where condition is TRUE,
one where it's FALSE.
```

## Ultimate Binary Search Template
```python
def binary_search(arr, target):
    """Find first index where condition is True."""
    left, right = 0, len(arr)  # or len(arr) - 1 depending on variant

    while left < right:
        mid = left + (right - left) // 2  # avoids overflow
        if condition(mid):
            right = mid      # answer is at mid or left of mid
        else:
            left = mid + 1   # answer is right of mid

    return left  # first index where condition holds
```

## Template: Standard Search
```python
def search(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

## Template: Search in Rotated Sorted Array
```python
def search_rotated(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        # Left half is sorted
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # Right half is sorted
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1
```

## Template: Binary Search on Answer (Minimize/Maximize)
```python
def binary_search_on_answer(lo, hi):
    """Find minimum value that satisfies feasible()."""
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if feasible(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo
```

## Key Problems
- LC 704: Binary Search
- LC 33: Search in Rotated Sorted Array
- LC 153: Find Minimum in Rotated Sorted Array
- LC 162: Find Peak Element
- LC 34: Find First and Last Position
- LC 875: Koko Eating Bananas
- LC 1011: Capacity to Ship Packages

---

# 12. BINARY TREE TRAVERSAL

## When to Use
- "Inorder / preorder / postorder traversal"
- "Level order traversal"
- "Tree serialization"
- Construct tree from traversals

## Traversal Order Quick Reference
```
In-order:    LEFT -> ROOT -> RIGHT   (gives sorted order for BST)
Pre-order:   ROOT -> LEFT -> RIGHT   (good for serialization / copying)
Post-order:  LEFT -> RIGHT -> ROOT   (good for deletion / bottom-up)
Level-order: Level by level (BFS)    (good for width / level problems)
```

## Template: Recursive DFS Traversals
```python
def inorder(root, result):
    if not root: return
    inorder(root.left, result)
    result.append(root.val)
    inorder(root.right, result)

def preorder(root, result):
    if not root: return
    result.append(root.val)
    preorder(root.left, result)
    preorder(root.right, result)

def postorder(root, result):
    if not root: return
    postorder(root.left, result)
    postorder(root.right, result)
    result.append(root.val)
```

## Template: Iterative Inorder (using Stack)
```python
def inorder_iterative(root):
    result, stack = [], []
    curr = root
    while curr or stack:
        while curr:
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()
        result.append(curr.val)
        curr = curr.right
    return result
```

## Template: Level Order (BFS)
```python
from collections import deque

def level_order(root):
    if not root: return []
    result = []
    queue = deque([root])
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
        result.append(level)
    return result
```

## Key Problems
- LC 94: Binary Tree Inorder Traversal
- LC 102: Binary Tree Level Order Traversal
- LC 104: Maximum Depth of Binary Tree
- LC 543: Diameter of Binary Tree
- LC 236: Lowest Common Ancestor
- LC 297: Serialize and Deserialize Binary Tree

---

# 13. DFS (DEPTH-FIRST SEARCH)

## When to Use
- Tree traversal (depth-first)
- Graph exploration
- Path finding (all paths, any path)
- Connected components
- Cycle detection (undirected graph)
- Topological ordering

## Template: Recursive DFS on Graph
```python
def dfs(graph, node, visited):
    visited.add(node)
    # Process node here

    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
```

## Template: Iterative DFS on Graph
```python
def dfs_iterative(graph, start):
    visited = set()
    stack = [start]

    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        # Process node here

        for neighbor in graph[node]:
            if neighbor not in visited:
                stack.append(neighbor)
```

## Key Problems
- LC 200: Number of Islands
- LC 133: Clone Graph
- LC 417: Pacific Atlantic Water Flow
- LC 79: Word Search
- LC 130: Surrounded Regions
- LC 695: Max Area of Island

---

# 14. BFS (BREADTH-FIRST SEARCH)

## When to Use
- Shortest path in unweighted graph
- Level-order traversal
- "Minimum number of steps/moves"
- Multi-source BFS ("rotten oranges" type)

## Core Rule
```
SHORTEST PATH in unweighted graph? --> ALWAYS BFS
"Minimum moves/steps"?             --> ALWAYS BFS
"Level by level"?                  --> ALWAYS BFS
```

## Template: Standard BFS
```python
from collections import deque

def bfs(graph, start):
    visited = {start}
    queue = deque([start])
    level = 0

    while queue:
        for _ in range(len(queue)):
            node = queue.popleft()
            # Process node

            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        level += 1

    return level
```

## Template: Multi-Source BFS
```python
from collections import deque

def multi_source_bfs(grid, sources):
    """BFS from multiple starting points simultaneously."""
    m, n = len(grid), len(grid[0])
    visited = set(sources)
    queue = deque(sources)
    steps = 0

    while queue:
        for _ in range(len(queue)):
            r, c = queue.popleft()
            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        steps += 1

    return steps
```

## Key Problems
- LC 994: Rotting Oranges
- LC 127: Word Ladder
- LC 752: Open the Lock
- LC 1091: Shortest Path in Binary Matrix
- LC 286: Walls and Gates
- LC 317: Shortest Distance from All Buildings

---

# 15. MATRIX / GRID TRAVERSAL

## When to Use
- "Number of islands"
- "Shortest path in grid"
- "Flood fill"
- "Surrounded regions"
- Any 2D grid problem with connectivity

## 4-Direction and 8-Direction Movement
```python
# 4 directions (up, down, left, right)
DIRS_4 = [(0, 1), (0, -1), (1, 0), (-1, 0)]

# 8 directions (including diagonals)
DIRS_8 = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
```

## Template: Grid DFS
```python
def grid_dfs(grid, r, c, visited):
    m, n = len(grid), len(grid[0])
    if r < 0 or r >= m or c < 0 or c >= n:
        return
    if (r, c) in visited or grid[r][c] == 0:
        return

    visited.add((r, c))
    for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
        grid_dfs(grid, r + dr, c + dc, visited)
```

## Template: Grid BFS (Shortest Path)
```python
from collections import deque

def grid_bfs(grid, start_r, start_c):
    m, n = len(grid), len(grid[0])
    visited = {(start_r, start_c)}
    queue = deque([(start_r, start_c, 0)])  # (row, col, distance)

    while queue:
        r, c, dist = queue.popleft()

        if is_target(r, c):  # define your target condition
            return dist

        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n and (nr, nc) not in visited and grid[nr][nc] != 0:
                visited.add((nr, nc))
                queue.append((nr, nc, dist + 1))

    return -1  # target not reachable
```

## Template: Grid DFS (Iterative -- avoids Python recursion limit)
```python
def grid_dfs_iterative(grid, start_r, start_c):
    m, n = len(grid), len(grid[0])
    visited = set()
    stack = [(start_r, start_c)]

    while stack:
        r, c = stack.pop()
        if (r, c) in visited:
            continue
        visited.add((r, c))
        # Process (r, c)

        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n and (nr, nc) not in visited and grid[nr][nc] != 0:
                stack.append((nr, nc))
```

## Key Rule: BFS vs DFS on Grids
```
Shortest path / minimum steps  --> BFS (always)
Explore all connected / count  --> DFS or BFS (either works)
All paths to target            --> DFS with backtracking
Large grid, deep recursion     --> Iterative DFS (avoid stack overflow)
```

## Key Problems
- LC 200: Number of Islands
- LC 695: Max Area of Island
- LC 733: Flood Fill
- LC 994: Rotting Oranges
- LC 1091: Shortest Path in Binary Matrix
- LC 417: Pacific Atlantic Water Flow
- LC 130: Surrounded Regions
- LC 286: Walls and Gates
- LC 542: 01 Matrix

---

# 16. BACKTRACKING

## When to Use
- "Generate all permutations"
- "Generate all combinations"
- "Generate all subsets"
- "N-Queens problem"
- "Sudoku solver"
- "Word search"
- Any problem that asks for ALL valid configurations

## Core Framework (Labuladong's Template)
```python
result = []

def backtrack(path, choices):
    if is_solution(path):
        result.append(path[:])  # COPY the path
        return

    for choice in choices:
        if is_valid(choice):
            path.append(choice)          # MAKE choice
            backtrack(path, new_choices)  # RECURSE
            path.pop()                   # UNDO choice (backtrack)
```

## Template: Subsets
```python
def subsets(nums):
    result = []
    def backtrack(start, path):
        result.append(path[:])
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()
    backtrack(0, [])
    return result
```

## Template: Permutations
```python
def permutations(nums):
    result = []
    def backtrack(path, remaining):
        if not remaining:
            result.append(path[:])
            return
        for i in range(len(remaining)):
            path.append(remaining[i])
            backtrack(path, remaining[:i] + remaining[i+1:])
            path.pop()
    backtrack([], nums)
    return result
```

## Template: Combinations (n choose k)
```python
def combinations(n, k):
    result = []
    def backtrack(start, path):
        if len(path) == k:
            result.append(path[:])
            return
        for i in range(start, n + 1):
            path.append(i)
            backtrack(i + 1, path)
            path.pop()
    backtrack(1, [])
    return result
```

## Handling Duplicates (Critical!)
```python
def subsets_with_dup(nums):
    nums.sort()  # MUST sort first
    result = []
    def backtrack(start, path):
        result.append(path[:])
        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i-1]:  # skip duplicates
                continue
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()
    backtrack(0, [])
    return result
```

## Quick Reference: Subsets vs Permutations vs Combinations
```
Subsets:       Order doesn't matter, each element used 0 or 1 times
               Loop from `start`, increment start
Permutations:  Order matters, each element used exactly once
               Loop from 0, track used elements
Combinations:  Order doesn't matter, choose exactly k elements
               Loop from `start`, stop when path length == k
```

## Key Problems
- LC 78: Subsets
- LC 90: Subsets II (with duplicates)
- LC 46: Permutations
- LC 47: Permutations II (with duplicates)
- LC 77: Combinations
- LC 39: Combination Sum
- LC 40: Combination Sum II
- LC 51: N-Queens
- LC 79: Word Search
- LC 131: Palindrome Partitioning

---

# 17. DYNAMIC PROGRAMMING

## When to Use
- "Find optimal (min/max) value"
- "Count number of ways"
- "Is it possible to..."
- Problem has OVERLAPPING SUBPROBLEMS + OPTIMAL SUBSTRUCTURE
- Brute force would be exponential

## 5-Step DP Framework
```
1. Define STATE:   What variables describe a subproblem?  dp[i], dp[i][j]
2. Define TRANSITION: How does dp[i] relate to smaller subproblems?
3. Define BASE CASE: What are the trivially known values?
4. Define ANSWER:  Which dp cell contains the final answer?
5. Determine ORDER: In what order to fill the table?
```

## Template: Top-Down (Memoization)
```python
from functools import lru_cache

def solve(nums):
    @lru_cache(maxsize=None)
    def dp(i, state):
        if i == base_case:
            return base_value
        # Recurrence relation
        option1 = dp(i - 1, state)         # skip
        option2 = dp(i - 1, new_state) + cost  # take
        return min(option1, option2)  # or max

    return dp(n, initial_state)
```

## Template: Bottom-Up (Tabulation)
```python
def solve(nums):
    n = len(nums)
    dp = [0] * (n + 1)

    # Base cases
    dp[0] = base_value

    # Fill table
    for i in range(1, n + 1):
        dp[i] = transition(dp[i-1], dp[i-2], ...)

    return dp[n]
```

## Key DP Sub-Patterns (see full list in Section 23)
```
Pattern                  Signature Problem          Recurrence Sketch
-----------------------  -------------------------  --------------------------
Fibonacci               Climbing Stairs             dp[i] = dp[i-1] + dp[i-2]
Kadane's Algorithm      Max Subarray                dp[i] = max(nums[i], dp[i-1]+nums[i])
0/1 Knapsack            Partition Equal Subset Sum   dp[i][w] = dp[i-1][w] || dp[i-1][w-wt[i]]
Unbounded Knapsack      Coin Change                 dp[i] = min(dp[i-c]+1) for c in coins
LCS                     Longest Common Subsequence  if match: dp[i][j]=dp[i-1][j-1]+1
LIS                     Longest Increasing Subseq.  dp[i] = max(dp[j]+1) for j < i
Edit Distance           Edit Distance               dp[i][j] based on insert/delete/replace
Grid DP                 Unique Paths                dp[i][j] = dp[i-1][j] + dp[i][j-1]
State Machine DP        Buy/Sell Stock w/ cooldown  hold[i], sold[i], rest[i]
Interval DP             Burst Balloons              dp[i][j] = max(dp[i][k]+dp[k][j]+cost)
```

## Key Problems
- LC 70: Climbing Stairs
- LC 53: Maximum Subarray (Kadane's)
- LC 198: House Robber
- LC 300: Longest Increasing Subsequence
- LC 1143: Longest Common Subsequence
- LC 72: Edit Distance
- LC 322: Coin Change
- LC 416: Partition Equal Subset Sum
- LC 62: Unique Paths
- LC 309: Best Time to Buy/Sell Stock w/ Cooldown

---

# 18. GREEDY ALGORITHM PATTERNS

## When to Use
- You can prove LOCAL optimum leads to GLOBAL optimum
- Sorting + one-pass decision making
- Interval scheduling / activity selection
- Problems where you never need to "undo" a choice

## How to Verify Greedy Works (Proof Template)
```
1. GREEDY CHOICE PROPERTY: Show that making the locally optimal
   choice at each step is safe (doesn't eliminate global optima).

2. OPTIMAL SUBSTRUCTURE: Show the remaining problem after the
   greedy choice is also an optimization problem of the same type.

3. PROOF BY CONTRADICTION: Assume greedy doesn't give optimal.
   Show you can swap a non-greedy choice with the greedy one
   without worsening the solution. Contradiction.

Shortcut for interviews: "Exchange argument" -- take any optimal
solution, show you can transform it into the greedy solution
step by step without losing optimality.
```

## 15 Core Greedy Patterns (from LeetCode Discuss)

```
 #  Pattern                              Example Problem
--- -----------------------------------  -----------------------------------
 1  Sort by end time, pick non-overlap   Activity Selection / LC 435
 2  Sort by ratio/metric                 Fractional Knapsack
 3  Greedy digit-by-digit selection      LC 402: Remove K Digits
 4  Greedy character selection            LC 316: Remove Duplicate Letters
 5  Partition into optimal segments      LC 763: Partition Labels
 6  Maximize/minimize contribution       LC 1029: Two City Scheduling
 7  Capacity/balance constraints         LC 134: Gas Station
 8  Jump game / reachability             LC 55: Jump Game, LC 45: Jump Game II
 9  Huffman-style merging                LC 1167: Min Cost to Connect Sticks
10  Task scheduling with cooldown        LC 621: Task Scheduler
11  Assign cookies / match greedily      LC 455: Assign Cookies
12  Sweep line / event processing        LC 253: Meeting Rooms II
13  Stock buy/sell (collect all profits)  LC 122: Buy/Sell Stock II
14  Monotonic greedy (stack-based)        LC 84: Largest Rectangle
15  Mathematical greedy                   LC 343: Integer Break
```

## Interval Scheduling Template
```python
def interval_scheduling(intervals):
    """Maximum number of non-overlapping intervals."""
    intervals.sort(key=lambda x: x[1])  # Sort by END time
    count = 0
    end = float('-inf')

    for start, finish in intervals:
        if start >= end:  # non-overlapping
            count += 1
            end = finish

    return count
```

## Jump Game Template
```python
def can_jump(nums):
    max_reach = 0
    for i in range(len(nums)):
        if i > max_reach:
            return False
        max_reach = max(max_reach, i + nums[i])
    return True
```

## Key Problems
- LC 55: Jump Game
- LC 45: Jump Game II
- LC 134: Gas Station
- LC 435: Non-overlapping Intervals
- LC 763: Partition Labels
- LC 402: Remove K Digits
- LC 621: Task Scheduler
- LC 455: Assign Cookies
- LC 122: Best Time to Buy and Sell Stock II
- LC 1029: Two City Scheduling

---

# 19. DIVIDE AND CONQUER

## When to Use
- Problem can be split into independent subproblems
- Merge step combines subproblem results
- "Find kth largest/smallest" (Quick Select)
- Merge sort, Quick sort
- "Count inversions"
- "Closest pair of points"

## Generic Template
```python
def divide_and_conquer(problem):
    # BASE CASE
    if is_base_case(problem):
        return solve_directly(problem)

    # DIVIDE
    left_sub, right_sub = split(problem)

    # CONQUER (recursive)
    left_result = divide_and_conquer(left_sub)
    right_result = divide_and_conquer(right_sub)

    # COMBINE
    return merge(left_result, right_result)
```

## Template: Merge Sort
```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return merge(left, right)

def merge(left, right):
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

## Template: Quick Select (Kth Smallest, Average O(n))
```python
import random

def quick_select(nums, k):
    """Find kth smallest element (0-indexed)."""
    if len(nums) == 1:
        return nums[0]

    pivot = random.choice(nums)
    left = [x for x in nums if x < pivot]
    mid = [x for x in nums if x == pivot]
    right = [x for x in nums if x > pivot]

    if k < len(left):
        return quick_select(left, k)
    elif k < len(left) + len(mid):
        return pivot
    else:
        return quick_select(right, k - len(left) - len(mid))
```

## Master Theorem Quick Reference
```
T(n) = aT(n/b) + O(n^d)

If d < log_b(a):  T(n) = O(n^(log_b(a)))     -- recursion dominates
If d = log_b(a):  T(n) = O(n^d * log n)       -- balanced
If d > log_b(a):  T(n) = O(n^d)               -- work at top dominates

Examples:
  Merge Sort:  T(n) = 2T(n/2) + O(n)    --> a=2, b=2, d=1 --> O(n log n)
  Binary Srch: T(n) = T(n/2) + O(1)     --> a=1, b=2, d=0 --> O(log n)
  Quick Select: T(n) = T(n/2) + O(n)    --> a=1, b=2, d=1 --> O(n) average
```

## Key Problems
- LC 912: Sort an Array (Merge Sort)
- LC 215: Kth Largest Element (Quick Select)
- LC 23: Merge K Sorted Lists
- LC 4: Median of Two Sorted Arrays
- LC 315: Count of Smaller Numbers After Self
- LC 493: Reverse Pairs

---

# 20. UNION-FIND (DISJOINT SET)

## When to Use
- "Are nodes A and B connected?"
- "Count number of connected components"
- "Detect cycle in undirected graph"
- "Redundant connection"
- Dynamic connectivity (edges added over time)
- MST (Kruskal's algorithm)

## Template: Union-Find with Path Compression + Union by Rank
```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = n  # number of components

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, x, y):
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y:
            return False  # already connected

        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x
        self.parent[root_y] = root_x
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1

        self.count -= 1
        return True

    def connected(self, x, y):
        return self.find(x) == self.find(y)
```

## Complexity
```
With path compression + union by rank:
  find():  O(alpha(n)) ~= O(1) amortized (inverse Ackermann)
  union(): O(alpha(n)) ~= O(1) amortized
```

## Key Problems
- LC 200: Number of Islands (alternative to DFS/BFS)
- LC 547: Number of Provinces
- LC 684: Redundant Connection
- LC 721: Accounts Merge
- LC 323: Number of Connected Components
- LC 1135: Connecting Cities with Minimum Cost (Kruskal's)
- LC 128: Longest Consecutive Sequence

---

# 21. TOPOLOGICAL SORT

## When to Use
- "Course schedule" / task dependencies
- "Build order"
- Detect cycle in DIRECTED graph
- Any DAG ordering problem

## Template: Kahn's Algorithm (BFS -- Preferred)
```python
from collections import deque, defaultdict

def topological_sort_bfs(num_nodes, edges):
    graph = defaultdict(list)
    in_degree = [0] * num_nodes

    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1

    # Start with all nodes having in-degree 0
    queue = deque([i for i in range(num_nodes) if in_degree[i] == 0])
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # If order doesn't contain all nodes, there's a cycle
    if len(order) != num_nodes:
        return []  # cycle detected!

    return order
```

## Template: DFS-Based Topological Sort
```python
def topological_sort_dfs(num_nodes, edges):
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)

    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * num_nodes
    order = []
    has_cycle = False

    def dfs(node):
        nonlocal has_cycle
        if has_cycle: return
        color[node] = GRAY
        for neighbor in graph[node]:
            if color[neighbor] == GRAY:
                has_cycle = True
                return
            if color[neighbor] == WHITE:
                dfs(neighbor)
        color[node] = BLACK
        order.append(node)

    for i in range(num_nodes):
        if color[i] == WHITE:
            dfs(i)

    if has_cycle:
        return []
    return order[::-1]
```

## Cycle Detection Quick Rule
```
Kahn's: If len(order) < num_nodes --> cycle exists
DFS:    If you visit a GRAY node  --> cycle exists (back edge)
```

## Key Problems
- LC 207: Course Schedule
- LC 210: Course Schedule II
- LC 269: Alien Dictionary
- LC 310: Minimum Height Trees
- LC 802: Find Eventual Safe States

---

# 22. TRIE PATTERNS

## When to Use
- "Prefix search / autocomplete"
- "Word search in dictionary"
- "Word search II (multiple words in grid)"
- "Spell checker"
- String matching with shared prefixes

## Template: Trie Implementation
```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    def search(self, word):
        node = self._find(word)
        return node is not None and node.is_end

    def starts_with(self, prefix):
        return self._find(prefix) is not None

    def _find(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node
```

## Key Problems
- LC 208: Implement Trie
- LC 211: Design Add and Search Words
- LC 212: Word Search II
- LC 648: Replace Words
- LC 677: Map Sum Pairs

---

# 23. 20 DYNAMIC PROGRAMMING SUB-PATTERNS

Source: AlgoMaster (blog.algomaster.io/p/20-patterns-to-master-dynamic-programming)

```
 #  Pattern                   Key Idea                              Canonical Problem
--- -----------------------   -------------------------------------  --------------------------
 1  Fibonacci Sequence        dp[i] = dp[i-1] + dp[i-2]            Climbing Stairs (LC 70)
 2  Kadane's Algorithm        dp[i] = max(nums[i], dp[i-1]+nums[i]) Max Subarray (LC 53)
 3  0/1 Knapsack              Take or skip each item once            Partition Equal Subset Sum (LC 416)
 4  Unbounded Knapsack        Take each item unlimited times         Coin Change (LC 322)
 5  Longest Common Subseq.    Match/skip on two sequences            LCS (LC 1143)
 6  Longest Increasing Subseq Binary search + patience sorting       LIS (LC 300)
 7  Palindromic Subsequence   Expand from center / interval DP       Longest Palindromic Subseq (LC 516)
 8  Edit Distance             Insert/Delete/Replace costs            Edit Distance (LC 72)
 9  Subset Sum                Can we reach target sum?               Subset Sum (LC 416)
10  String Partition           Partition string into valid parts      Word Break (LC 139)
11  Catalan Numbers           Count structurally unique BSTs         Unique BSTs (LC 96)
12  Matrix Chain Multiply     Optimal parenthesization               Burst Balloons (LC 312)
13  Count Distinct Ways       Sum paths / ways to reach target       Decode Ways (LC 91)
14  DP on Grids               2D table with directional transitions  Unique Paths (LC 62)
15  DP on Trees               DFS + memo on tree structure           House Robber III (LC 337)
16  DP on Graphs              Shortest paths, DAG DP                 Cheapest Flights (LC 787)
17  Digit DP                  Count numbers with digit constraints   Numbers At Most N Given Digits (LC 902)
18  Bitmasking DP             State = bitmask of used items          Partition to K Equal Sum (LC 698)
19  Probability DP            Expected value / probability states    Knight Probability (LC 688)
20  State Machine DP          States + transitions (buy/sell/rest)   Buy Sell Stock w/ Cooldown (LC 309)
```

## DP Pattern Recognition Rules
```
"Minimum cost to reach end"       --> Grid DP or Shortest Path DP
"Number of ways to..."            --> Count Distinct Ways
"Can we partition into..."        --> Subset Sum / Knapsack
"Longest common/increasing..."    --> LCS / LIS
"Best time to buy/sell"           --> State Machine DP
"Count numbers less than N..."    --> Digit DP
"Assign items to K groups"        --> Bitmasking DP
"Probability of event"            --> Probability DP
"Optimal split point"             --> Matrix Chain / Interval DP
```

---

# 24. BIG-O COMPLEXITY CHEAT SHEET

Source: bigocheatsheet.com, Interview Cake

## Common Data Structure Operations

```
Data Structure       Access    Search    Insert    Delete    Space
-------------------  --------  --------  --------  --------  ------
Array                O(1)      O(n)      O(n)      O(n)      O(n)
Stack                O(n)      O(n)      O(1)      O(1)      O(n)
Queue                O(n)      O(n)      O(1)      O(1)      O(n)
Singly Linked List   O(n)      O(n)      O(1)      O(1)      O(n)
Doubly Linked List   O(n)      O(n)      O(1)      O(1)      O(n)
Hash Table           N/A       O(1)*     O(1)*     O(1)*     O(n)
BST (balanced)       O(log n)  O(log n)  O(log n)  O(log n)  O(n)
BST (worst)          O(n)      O(n)      O(n)      O(n)      O(n)
Heap                 O(1)top   O(n)      O(log n)  O(log n)  O(n)
Trie                 N/A       O(m)      O(m)      O(m)      O(n*m)

* = amortized; m = key length; n = number of elements
```

## Sorting Algorithms

```
Algorithm        Best        Average     Worst       Space    Stable?
--------------   ---------   ---------   ---------   ------   -------
Bubble Sort      O(n)        O(n^2)      O(n^2)      O(1)     Yes
Selection Sort   O(n^2)      O(n^2)      O(n^2)      O(1)     No
Insertion Sort   O(n)        O(n^2)      O(n^2)      O(1)     Yes
Merge Sort       O(n log n)  O(n log n)  O(n log n)  O(n)     Yes
Quick Sort       O(n log n)  O(n log n)  O(n^2)      O(log n) No
Heap Sort        O(n log n)  O(n log n)  O(n log n)  O(1)     No
Counting Sort    O(n+k)      O(n+k)      O(n+k)      O(k)     Yes
Radix Sort       O(nk)       O(nk)       O(nk)       O(n+k)   Yes
Tim Sort         O(n)        O(n log n)  O(n log n)  O(n)     Yes
```

## Graph Algorithm Complexities

```
Algorithm            Time             Space       Notes
-------------------  ---------------  ----------  -------------------------
BFS                  O(V + E)         O(V)        Shortest path (unweighted)
DFS                  O(V + E)         O(V)        Path finding, cycle detect
Dijkstra             O((V+E) log V)   O(V)        Shortest path (weighted, no neg)
Bellman-Ford         O(V * E)         O(V)        Handles negative weights
Floyd-Warshall       O(V^3)           O(V^2)      All pairs shortest path
Topological Sort     O(V + E)         O(V)        DAG ordering
Kruskal's MST        O(E log E)       O(V)        MST with Union-Find
Prim's MST           O((V+E) log V)   O(V)        MST with priority queue
Union-Find           O(alpha(n))      O(n)        ~O(1) amortized per op
```

## Common Time Complexity Ranking (Fastest to Slowest)
```
O(1) < O(log n) < O(sqrt(n)) < O(n) < O(n log n) < O(n^2) < O(n^3) < O(2^n) < O(n!) < O(n^n)
```

## What n Can Your Algorithm Handle?
```
n <= 10       --> O(n!) or O(n * 2^n)  -- brute force / backtracking OK
n <= 20       --> O(2^n)               -- bitmask DP
n <= 100      --> O(n^3)               -- Floyd-Warshall, cubic DP
n <= 1,000    --> O(n^2)               -- simple DP, nested loops
n <= 100,000  --> O(n log n)           -- sort + scan, binary search
n <= 1M       --> O(n)                 -- linear scan, two pointers
n <= 1B       --> O(log n) or O(1)     -- binary search, math
```

---

# 25. SOURCES & REFERENCES

## Top GitHub Repositories
- [Coding Interview University (315k+ stars)](https://github.com/jwasham/coding-interview-university) -- Complete CS study plan
- [Tech Interview Handbook (125k+ stars)](https://github.com/yangshun/tech-interview-handbook) -- Curated prep materials by Blind 75 author
- [Awesome LeetCode Resources](https://github.com/ashishps1/awesome-leetcode-resources) -- Curated links
- [ByteByteGo Coding Interview Patterns](https://github.com/ByteByteGoHq/coding-interview-patterns) -- 19 chapters, 1000+ diagrams
- [Grokking the Coding Interview (Patterns)](https://github.com/dipjul/Grokking-the-Coding-Interview-Patterns-for-Coding-Questions)
- [Grokking Patterns with LeetCode Mapping](https://github.com/navidre/new_grokking_to_leetcode)
- [Python CP Cheatsheet](https://github.com/peterlamar/python-cp-cheatsheet)
- [LeetCode CheatSheet (Markdown Notes)](https://github.com/lichenma/LeetCode-CheatSheet)
- [NeetCode 150 + Blind 75 (Anki Flashcards)](https://github.com/envico801/Neetcode-150-and-Blind-75)
- [Big-O Complexity Cheat Sheet](https://github.com/ReaVNaiL/Big-O-Complexity-Cheat-Sheet)
- [Sean Prashad's LeetCode Patterns](https://github.com/seanprashad/leetcode-patterns)
- [Reddit Programming Wiki Resources](https://github.com/antariksh17/Reddit-wiki-programming)

## Top Blog Posts & Articles
- [15 LeetCode Patterns (AlgoMaster)](https://blog.algomaster.io/p/15-leetcode-patterns)
- [20 Patterns to Master Dynamic Programming (AlgoMaster)](https://blog.algomaster.io/p/20-patterns-to-master-dynamic-programming)
- [20 DSA Patterns (AlgoMaster)](https://blog.algomaster.io/p/20-dsa-patterns)
- [17 Patterns to Rule Them All (Medium)](https://samiyadev.medium.com/ace-your-tech-interview-17-patterns-to-rule-them-all-leetcode-cheat-sheet-a067324c54f3)
- [14 Patterns to Solve Any Question (CodeInMotion)](https://www.blog.codeinmotion.io/p/leetcode-patterns)
- [20 Essential Coding Patterns (DEV Community)](https://dev.to/arslan_ah/20-essential-coding-patterns-to-ace-your-next-coding-interview-32a3)
- [16 DSA Patterns That Did What 3000 Problems Couldn't (Medium)](https://medium.com/@himanshusingour7/these-16-dsa-patterns-did-what-3000-leetcode-problems-couldnt-47420b507564)
- [10+ Top LeetCode Patterns 2026 (Educative)](https://www.educative.io/blog/coding-interview-leetcode-patterns)
- [Coding Interviews for Dummies (freeCodecamp)](https://www.freecodecamp.org/news/coding-interviews-for-dummies-5e048933b82b/)

## Top LeetCode Discuss Posts
- [Monotonic Stack Comprehensive Guide & Template](https://leetcode.com/discuss/post/2347639/a-comprehensive-guide-and-template-for-m-irii/)
- [Monotonic Stack Templates (with solved problems)](https://leetcode.com/discuss/post/5085517/templates-for-monotonic-stacks-and-queue-2lfq/)
- [Monotonic Stack Guide + Problem List](https://leetcode.com/discuss/study-guide/5148505/Monotonic-Stack-Guide-+-List-of-Problems/)
- [15 Core Greedy Patterns](https://leetcode.com/discuss/post/7344979/15-core-greedy-patterns-for-coding-inter-a1wp/)
- [ABCs of Greedy](https://leetcode.com/discuss/post/1061059/abcs-of-greedy-by-sapphire_skies-hbo3/)
- [Dynamic Programming Patterns (aatalyk)](https://leetcode.com/discuss/post/458695/dynamic-programming-patterns-by-aatalyk-pmgr/)
- [14 Patterns to Ace Any Coding Interview](https://leetcode.com/discuss/post/4039411/14-Patterns-to-Ace-Any-Coding-Interview-Question/)
- [DSA Patterns You Need to Know](https://leetcode.com/discuss/post/5886397/dsa-patterns-you-need-to-know-by-anubhav-x7og/)
- [Famous Posts with Templates and Suggested Patterns](https://leetcode.com/discuss/study-guide/2007535/some-famous-posts-with-templates-and-suggested-patterns/)
- [Two Pointers Cheat Sheet (25+ Problems)](https://leetcode.com/discuss/post/7350824/)
- [Python Ultimate Binary Search Template](https://leetcode.com/discuss/general-discussion/786126/Python-Powerful-Ultimate-Binary-Search-Template.-Solved-many-problems)
- [DP for Beginners](https://leetcode.com/discuss/general-discussion/662866/Dynamic-Programming-for-Practice-Problems-Patterns-and-Sample-Solutions/)
- [LeetCode 14 Patterns: 130 Questions with Company Tags](https://leetcode.com/discuss/post/6853134/leetcode-14-patterns-mastery-130-questio-bx4e/)

## Reference Sites
- [Tech Interview Handbook -- Algorithm Study Cheatsheets](https://www.techinterviewhandbook.org/algorithms/study-cheatsheet/)
- [Tech Interview Handbook -- Graph Cheatsheet](https://www.techinterviewhandbook.org/algorithms/graph/)
- [Tech Interview Handbook -- DP Cheatsheet](https://www.techinterviewhandbook.org/algorithms/dynamic-programming/)
- [AlgoMonster Decision Flowchart](https://algo.monster/flowchart)
- [AlgoMonster Algorithm Templates](https://algo.monster/templates)
- [Sean Prashad's LeetCode Patterns (Web)](https://seanprashad.com/leetcode-patterns/)
- [LeetCode Cheatsheet (JWL-7)](https://jwl-7.github.io/leetcode-cheatsheet/)
- [LeetCode Wizard Ultimate Cheat Sheet](https://leetcodewizard.io/blog/ultimate-leetcode-coding-interview-cheat-sheet-faang-prep-for-grads)
- [Pirate King LeetCode Cheat Sheet](https://www.piratekingdom.com/leetcode/cheat-sheet)
- [15 Essential LeetCode Patterns (Cheat Sheets)](https://cheat.aktagon.com/sheets/leetcode/)
- [Big-O Cheat Sheet](https://www.bigocheatsheet.com/)
- [Design Gurus -- Top LC Patterns](https://www.designgurus.io/blog/top-lc-patterns)
- [Design Gurus -- Coding Interview Cheatsheet](https://www.designgurus.io/blog/coding-interview-cheatsheet)
- [Labuladong Monotonic Stack Template](https://labuladong.online/en/algo/data-structure/monotonic-stack/)
- [Labuladong Backtracking Framework](https://labuladong.online/en/algo/essential-technique/backtrack-framework/)
- [Labuladong Sliding Window Framework](https://labuladong.online/en/algo/essential-technique/sliding-window-framework/)
- [NeetCode Blind 75](https://neetcode.io/practice/practice/blind75)
- [NeetCode 150](https://neetcode.io/practice/practice/neetcode150)
- [LeetCopilot Monotonic Stack Guide](https://leetcopilot.dev/leetcode-pattern/monotonic-stack-queue/guide)
- [LeetCopilot Sliding Window Guide](https://leetcopilot.dev/leetcode-pattern/sliding-window/guide)
- [LeetCopilot BFS vs DFS Decision Guide](https://leetcopilot.dev/blog/how-to-choose-between-bfs-and-dfs-leetcode)
- [Blind -- LeetCode Patterns Cheatsheet Discussion](https://www.teamblind.com/post/leetcode-patterns-cheatsheet-tvfq4hmy)

## Problem Lists (Curated)
- [Blind 75](https://neetcode.io/practice/practice/blind75) -- The original 75 essential problems
- [NeetCode 150](https://neetcode.io/practice/practice/neetcode150) -- Blind 75 + 75 more
- [Grind 75](https://www.techinterviewhandbook.org/grind75) -- Customizable version of Blind 75
- [Sean Prashad Pattern-Based List](https://seanprashad.com/leetcode-patterns/)

---

*This cheat sheet was compiled from the highest-rated community resources, GitHub repositories (300k+ combined stars), top LeetCode Discuss posts, and the most recommended guides from Reddit r/leetcode, r/cscareerquestions, and Blind communities.*
