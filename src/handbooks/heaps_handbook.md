# The Complete Heaps and Priority Queues Handbook
> A template-based approach for mastering heap problems in coding interviews

**Philosophy:** Heaps are not just about sorting. They're about **efficiently maintaining the extreme value** (min or max) in a dynamic collection — enabling O(log n) insertions and O(1) access to the best element.

---

## Table of Contents
1. [Understanding the Core Philosophy](#core-philosophy)
2. [The Master Templates](#master-templates)
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

- **The Bouncer Line**: Imagine a nightclub where the bouncer always lets in (or kicks out) the smallest/largest person. You don't need the whole line sorted -- you just need to always know who's at the front. That's a heap: it only guarantees the extreme element is accessible in O(1).
- **The Gatekeeper for Top-K**: To find the K tallest people in a crowd, keep a "shortest-of-the-tall" gatekeeper. Anyone shorter gets rejected. Anyone taller replaces the current shortest. At the end, everyone still in line is top-K.

### No-Jargon Translation

- **Heap**: a partially sorted structure where the smallest or largest element is always on top
- **Min-heap**: smallest on top
- **Max-heap**: largest on top
- **Heapify**: dumping a pile of items into heap order -- like shuffling a deck into a specific arrangement
- **Push/pop**: adding or removing the top element
- **Priority queue**: a fancy name for a heap where "priority" decides who's on top

### Mental Model

> "A heap is an impatient waiting room where the most urgent patient always sits in the front chair, and you never bother sorting the rest."

---

### What is a Heap?

A heap is a **complete binary tree** where every parent node satisfies the **heap property**:
- **Min-heap:** Parent ≤ children (root is minimum)
- **Max-heap:** Parent ≥ children (root is maximum)

```
Min-Heap:          Max-Heap:
    1                  9
   / \                / \
  3   2              7   8
 / \                / \
5   4              5   6
```

### Key Operations and Complexity

| Operation | Time Complexity | Description |
|-----------|----------------|-------------|
| `heappush` | O(log n) | Insert element |
| `heappop` | O(log n) | Remove and return root |
| `heap[0]` | O(1) | Peek at root |
| `heapify` | O(n) | Convert list to heap |
| `nlargest/nsmallest` | O(n log k) | Find k extreme elements |

### Python's `heapq` Module

Python only has **min-heap**. For max-heap, negate values.

```python
import heapq

# Min-heap operations
heap = []
heapq.heappush(heap, 5)
heapq.heappush(heap, 3)
heapq.heappush(heap, 7)
print(heap[0])  # 3 (minimum)
min_val = heapq.heappop(heap)  # 3

# Max-heap: negate values
max_heap = []
heapq.heappush(max_heap, -5)  # Store -5
heapq.heappush(max_heap, -3)
heapq.heappush(max_heap, -7)
max_val = -heapq.heappop(max_heap)  # 7 (negate back)

# Heapify existing list
arr = [5, 3, 7, 1, 9]
heapq.heapify(arr)  # O(n), arr is now a heap

# K largest/smallest
heapq.nlargest(3, arr)   # [9, 7, 5]
heapq.nsmallest(3, arr)  # [1, 3, 5]
```

### Why Heaps Over Sorting?

| Approach | Time | When to Use |
|----------|------|-------------|
| Sort once | O(n log n) | Static data, need all sorted |
| Heap | O(n log k) | Dynamic data, need k extremes |

**Key Insight:** When you only need the top k elements from n items, heap is more efficient than sorting when k << n.

### The Mental Model

Think of a heap as a **competition**:
- Min-heap: Competitors fight to be the smallest (winner at top)
- Max-heap: Competitors fight to be the largest (winner at top)
- New element enters: climbs up if it can beat parents
- Remove winner: second-best takes over, then restructures

---

<a name="master-templates"></a>
## 2. The Master Templates

### Template A: Top K Elements (Keep K Smallest/Largest)

**Use when:** Find k largest elements → use min-heap of size k

```python
import heapq

def top_k_largest(nums: list[int], k: int) -> list[int]:
    """
    Find k largest elements using min-heap.
    Min-heap of size k: smallest of the k largest is at top.
    """
    heap = []

    for num in nums:
        # Why `< k` and not `<= k`?
        # We want exactly k items. When len == k, the heap is full.
        # Adding one more would make it k+1, so we switch to replace mode.
        if len(heap) < k:
            heapq.heappush(heap, num)
        elif num > heap[0]:  # Larger than smallest in top-k
            # heapreplace = pop + push in ONE sift (faster than separate calls).
            # Safe here because we already checked num > heap[0],
            # so the new element belongs in the top-k.
            heapq.heapreplace(heap, num)

    return heap  # Contains k largest (not sorted)
```

**Why min-heap for k largest?**
- We maintain the k largest seen so far
- The smallest of these k (at heap top) is the "gatekeeper"
- New elements must beat the gatekeeper to enter

**For k smallest, use max-heap (negate values):**
```python
def top_k_smallest(nums: list[int], k: int) -> list[int]:
    heap = []  # Max-heap using negation

    for num in nums:
        # Same size-k gatekeeper logic, but inverted with negation.
        # -heap[0] is the largest value currently in the "k smallest" set.
        if len(heap) < k:
            heapq.heappush(heap, -num)
        elif num < -heap[0]:  # Smaller than largest in top-k
            heapq.heapreplace(heap, -num)

    return [-x for x in heap]
```

---

### Template B: Merge K Sorted Lists/Arrays

**Use when:** Combining multiple sorted sequences

```python
import heapq

def merge_k_sorted(lists: list[list[int]]) -> list[int]:
    """
    Merge k sorted lists using min-heap.
    Heap contains (value, list_index, element_index).
    """
    result = []
    heap = []

    # Initialize with first element from each list
    for i, lst in enumerate(lists):
        # Skip empty lists to avoid index errors and empty heap entries.
        if lst:
            heapq.heappush(heap, (lst[0], i, 0))

    # Why `while heap`?
    # The heap shrinks as lists are exhausted. When heap is empty,
    # every element from every list has been processed.
    while heap:
        val, list_idx, elem_idx = heapq.heappop(heap)
        result.append(val)

        # Push next element from same list.
        # Why `elem_idx + 1 < len(...)`? Bounds check -- only push
        # if this list still has more elements to contribute.
        if elem_idx + 1 < len(lists[list_idx]):
            next_val = lists[list_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))

    return result
```

**Complexity:** O(N log k) where N = total elements, k = number of lists

---

### Template C: Running Median / Two Heaps

**Use when:** Need to track median in a stream of numbers

```python
import heapq

class MedianFinder:
    """
    Two heaps: max-heap for lower half, min-heap for upper half.
    Median is derived from the heap tops.
    """
    def __init__(self):
        self.small = []  # Max-heap (negate values) for lower half
        self.large = []  # Min-heap for upper half

    def addNum(self, num: int) -> None:
        # Step 1: Always add to small (lower half) first.
        # Why always small? This is a convention -- we arbitrarily
        # let small hold the extra element when count is odd.
        heapq.heappush(self.small, -num)

        # Step 2: Move the largest of small to large.
        # This GUARANTEES max(small) <= min(large) after every insert,
        # because we always pass the biggest "small" value to "large".
        heapq.heappush(self.large, -heapq.heappop(self.small))

        # Step 3: Balance sizes -- small can have at most 1 more than large.
        # Why `len(large) > len(small)` and not `>=`?
        # We allow small to be 1 bigger (odd count), but large must
        # never be bigger than small. If it is, move one back.
        if len(self.large) > len(self.small):
            heapq.heappush(self.small, -heapq.heappop(self.large))

    def findMedian(self) -> float:
        # Why check `len(small) > len(large)`?
        # If odd count, small has the extra element -- that's the median.
        # If even count, average the two heap tops.
        if len(self.small) > len(self.large):
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2
```

**Visual:**
```
Numbers: [5, 2, 3, 4, 1]

After 5: small=[-5], large=[]
         median = 5

After 2: small=[-2], large=[5]
         median = (2+5)/2 = 3.5

After 3: small=[-3,-2], large=[5]
         median = 3

After 4: small=[-3,-2], large=[4,5]
         median = (3+4)/2 = 3.5

After 1: small=[-3,-2,-1], large=[4,5]
         median = 3
```

---

### Template D: Lazy Deletion

**Use when:** Need to "remove" elements but can't efficiently find them in heap

```python
import heapq
from collections import defaultdict

class LazyHeap:
    """
    Heap with lazy deletion - mark elements as deleted,
    clean up when they reach the top.
    """
    def __init__(self):
        self.heap = []
        self.deleted = defaultdict(int)  # Count of deleted elements

    def push(self, val):
        heapq.heappush(self.heap, val)

    def remove(self, val):
        self.deleted[val] += 1

    def pop(self):
        self._clean()
        return heapq.heappop(self.heap)

    def top(self):
        self._clean()
        return self.heap[0]

    def _clean(self):
        """Remove deleted elements from top of heap."""
        # Why `while` and not `if`? Multiple consecutive deleted elements
        # may stack up at the top. We must drain ALL of them before the
        # real top is exposed.
        # Why `self.heap and ...`? Guard against popping from an empty heap.
        # Why `> 0`? We track deletion counts (not booleans) to handle
        # duplicate values -- each deletion cancels one occurrence.
        while self.heap and self.deleted[self.heap[0]] > 0:
            self.deleted[self.heap[0]] -= 1
            heapq.heappop(self.heap)
```

---

### Template E: K-way Processing

**Use when:** Processing elements from multiple sources in order

```python
import heapq

def k_way_process(sources: list, process_func):
    """
    Process elements from k sources in sorted order.
    Sources can be iterators, generators, or any iterable.
    """
    heap = []

    # Initialize with first element from each source
    for i, source in enumerate(sources):
        iterator = iter(source)
        first = next(iterator, None)
        # Skip exhausted/empty sources -- they contribute nothing.
        if first is not None:
            heapq.heappush(heap, (first, i, iterator))

    # Heap always holds at most one element per source.
    # When a source is exhausted, its slot disappears. When all
    # sources are exhausted, the heap is empty and we stop.
    while heap:
        val, source_idx, iterator = heapq.heappop(heap)

        # Process current element
        process_func(val, source_idx)

        # Get next from same source.
        # `next(iterator, None)` returns None when exhausted --
        # we only re-push if the source has more data.
        next_val = next(iterator, None)
        if next_val is not None:
            heapq.heappush(heap, (next_val, source_idx, iterator))
```

---

### Quick Decision Matrix

| Problem Type | Template | Heap Type |
|--------------|----------|-----------|
| K largest | A | Min-heap of size k |
| K smallest | A | Max-heap of size k |
| Merge K sorted | B | Min-heap |
| Running median | C | Two heaps |
| Sliding window max/min | D (lazy) or deque | Max/Min-heap |
| Task scheduling | E | Max-heap by frequency (negate for heapq) |
| Dijkstra's algorithm | E | Min-heap by distance |

---

<a name="pattern-guide"></a>
## 3. Pattern Classification Guide

### Category 1: Top K Elements
- Find k largest/smallest
- Kth largest element
- K most frequent
- **Template A**

### Category 2: K-Way Merge
- Merge k sorted lists
- K sorted arrays
- External sorting
- **Template B**

### Category 3: Two Heaps (Median)
- Running median
- Sliding window median
- Balance two groups
- **Template C**

### Category 4: Scheduling/Simulation
- Task scheduling
- Meeting rooms
- Event processing
- **Template E**

### Category 5: Modified Heap Problems
- Custom comparators
- Lazy deletion
- Heap with updates
- **Template D + custom**

---

<a name="patterns"></a>
## 4. Complete Pattern Library

### PATTERN 1: Top K Elements

---

#### Pattern 1A: Kth Largest Element

**Problem:** LeetCode 215 - Find kth largest element in array

**Example:** `nums = [3,2,1,5,6,4]`, k = `2` → `5`

```python
import heapq

def findKthLargest(nums: list[int], k: int) -> int:
    """
    Min-heap of size k: root is kth largest.
    """
    heap = []

    for num in nums:
        heapq.heappush(heap, num)
        # Why `> k` and not `>= k`?
        # After pushing, the heap has one too many elements.
        # We pop when size is k+1, keeping exactly k elements.
        # Using `>= k` would keep only k-1 elements (off by one!).
        if len(heap) > k:
            heapq.heappop(heap)

    # After processing all nums, the heap holds the k largest.
    # The root (heap[0]) is the smallest of those k -- the kth largest.
    return heap[0]
```

**Alternative using heapq.nlargest:**
```python
def findKthLargest_builtin(nums: list[int], k: int) -> int:
    return heapq.nlargest(k, nums)[-1]
```

**Alternative using QuickSelect (O(n) average):**
```python
import random

def findKthLargest_quickselect(nums: list[int], k: int) -> int:
    k = len(nums) - k  # Convert to kth smallest index

    def quickselect(left, right):
        pivot = nums[right]
        p = left
        for i in range(left, right):
            if nums[i] <= pivot:
                nums[p], nums[i] = nums[i], nums[p]
                p += 1
        nums[p], nums[right] = nums[right], nums[p]

        if p == k:
            return nums[p]
        elif p < k:
            return quickselect(p + 1, right)
        else:
            return quickselect(left, p - 1)

    return quickselect(0, len(nums) - 1)
```

**Complexity:** Heap: O(n log k), QuickSelect: O(n) average, O(n²) worst

---

#### Pattern 1B: Top K Frequent Elements

**Problem:** LeetCode 347 - Find k most frequent elements

**Example:** `nums = [1,1,1,2,2,3]`, k = `2` → `[1,2]`

```python
import heapq
from collections import Counter

def topKFrequent(nums: list[int], k: int) -> list[int]:
    count = Counter(nums)

    # Min-heap by frequency, size k.
    # The heap tuple is (freq, num): freq is compared first,
    # so the LEAST frequent element sits at the root as gatekeeper.
    heap = []
    for num, freq in count.items():
        heapq.heappush(heap, (freq, num))
        # Evict the least frequent whenever heap exceeds k.
        # This guarantees only the k most frequent survive.
        if len(heap) > k:
            heapq.heappop(heap)

    return [num for freq, num in heap]
```

**Alternative using nlargest:**
```python
def topKFrequent_nlargest(nums: list[int], k: int) -> list[int]:
    count = Counter(nums)
    return heapq.nlargest(k, count.keys(), key=count.get)
```

**Alternative using Bucket Sort (O(n)):**
```python
def topKFrequent_bucket(nums: list[int], k: int) -> list[int]:
    count = Counter(nums)

    # Bucket: index = frequency, value = list of numbers
    buckets = [[] for _ in range(len(nums) + 1)]
    for num, freq in count.items():
        buckets[freq].append(num)

    result = []
    # Walk from highest frequency down to 0.
    for freq in range(len(buckets) - 1, -1, -1):
        result.extend(buckets[freq])
        # Why `>= k` and not `== k`? Multiple numbers can share the
        # same frequency, so we may overshoot. Slice to exactly k.
        if len(result) >= k:
            return result[:k]

    return result
```

---

#### Pattern 1C: K Closest Points to Origin

**Problem:** LeetCode 973 - Find k closest points to origin

```python
import heapq

def kClosest(points: list[list[int]], k: int) -> list[list[int]]:
    """
    Max-heap of size k: farthest of k closest is at top.
    Store negative distance for max-heap behavior.
    """
    heap = []

    for x, y in points:
        dist = x*x + y*y  # No need for sqrt (preserves ordering)
        if len(heap) < k:
            # Negate distance: Python's min-heap becomes a max-heap.
            # The FARTHEST of the k closest sits at the root as gatekeeper.
            heapq.heappush(heap, (-dist, [x, y]))
        elif dist < -heap[0][0]:
            # Why `dist < -heap[0][0]`?
            # -heap[0][0] is the farthest distance in our k-closest set.
            # If this point is closer, it replaces the farthest.
            heapq.heapreplace(heap, (-dist, [x, y]))

    return [point for dist, point in heap]
```

**Using nsmallest:**
```python
def kClosest_nsmallest(points: list[list[int]], k: int) -> list[list[int]]:
    return heapq.nsmallest(k, points, key=lambda p: p[0]**2 + p[1]**2)
```

---

#### Pattern 1D: Sort Characters By Frequency

**Problem:** LeetCode 451 - Sort characters by frequency

```python
import heapq
from collections import Counter

def frequencySort(s: str) -> str:
    count = Counter(s)

    # Max-heap by frequency (negate so most frequent pops first).
    heap = [(-freq, char) for char, freq in count.items()]
    heapq.heapify(heap)  # O(n) -- faster than n individual pushes

    result = []
    # Pop characters in frequency order (highest first).
    while heap:
        freq, char = heapq.heappop(heap)
        # -freq converts back to the positive count for repetition.
        result.append(char * (-freq))

    return ''.join(result)
```

---

### PATTERN 2: K-Way Merge

---

#### Pattern 2A: Merge K Sorted Lists

**Problem:** LeetCode 23 - Merge k sorted linked lists

```python
import heapq

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    # For heap comparison (when values are equal)
    def __lt__(self, other):
        return self.val < other.val

def mergeKLists(lists: list[ListNode]) -> ListNode:
    heap = []

    # Initialize with heads. Skip None lists (empty linked lists).
    # We include index `i` as a tiebreaker: when two nodes have the
    # same val, Python compares the next tuple element. Without `i`,
    # it would try to compare ListNode objects, which crashes.
    for i, node in enumerate(lists):
        if node:
            heapq.heappush(heap, (node.val, i, node))

    dummy = ListNode(0)
    current = dummy

    while heap:
        val, idx, node = heapq.heappop(heap)
        current.next = node
        current = current.next

        # Only push the next node if this list isn't exhausted.
        if node.next:
            heapq.heappush(heap, (node.next.val, idx, node.next))

    return dummy.next
```

**Why include index in tuple?**
When two nodes have equal values, Python can't compare `ListNode` objects. Including index breaks ties.

**Complexity:** O(N log k) where N = total nodes, k = number of lists

---

#### Pattern 2B: Kth Smallest Element in Sorted Matrix

**Problem:** LeetCode 378 - Kth smallest in row-column sorted matrix

```python
import heapq

def kthSmallest(matrix: list[list[int]], k: int) -> int:
    n = len(matrix)
    heap = []

    # Initialize with first column
    for i in range(min(n, k)):
        heapq.heappush(heap, (matrix[i][0], i, 0))

    # Pop k-1 times, push next element from same row
    for _ in range(k - 1):
        val, row, col = heapq.heappop(heap)
        if col + 1 < n:
            heapq.heappush(heap, (matrix[row][col + 1], row, col + 1))

    return heap[0][0]
```

**Alternative: Binary Search (often faster)**
```python
def kthSmallest_bs(matrix: list[list[int]], k: int) -> int:
    n = len(matrix)

    def count_less_equal(target):
        count = 0
        row, col = n - 1, 0
        while row >= 0 and col < n:
            if matrix[row][col] <= target:
                count += row + 1
                col += 1
            else:
                row -= 1
        return count

    lo, hi = matrix[0][0], matrix[n-1][n-1]
    while lo < hi:
        mid = (lo + hi) // 2
        if count_less_equal(mid) < k:
            lo = mid + 1
        else:
            hi = mid

    return lo
```

---

#### Pattern 2C: Find K Pairs with Smallest Sums

**Problem:** LeetCode 373 - K pairs from two arrays with smallest sums

```python
import heapq

def kSmallestPairs(nums1: list[int], nums2: list[int], k: int) -> list[list[int]]:
    if not nums1 or not nums2:
        return []

    heap = []
    result = []

    # Initialize with (nums1[i], nums2[0]) for all i
    for i in range(min(len(nums1), k)):
        heapq.heappush(heap, (nums1[i] + nums2[0], i, 0))

    while heap and len(result) < k:
        sum_val, i, j = heapq.heappop(heap)
        result.append([nums1[i], nums2[j]])

        # Push next pair from same nums1[i]
        if j + 1 < len(nums2):
            heapq.heappush(heap, (nums1[i] + nums2[j + 1], i, j + 1))

    return result
```

---

#### Pattern 2D: Smallest Range Covering Elements from K Lists

**Problem:** LeetCode 632 - Find smallest range containing at least one element from each list

```python
import heapq

def smallestRange(nums: list[list[int]]) -> list[int]:
    heap = []
    max_val = float('-inf')

    # Initialize with first element from each list
    for i, lst in enumerate(nums):
        heapq.heappush(heap, (lst[0], i, 0))
        max_val = max(max_val, lst[0])

    result = [float('-inf'), float('inf')]

    while len(heap) == len(nums):
        min_val, list_idx, elem_idx = heapq.heappop(heap)

        # Update result if current range is smaller
        if max_val - min_val < result[1] - result[0]:
            result = [min_val, max_val]

        # Push next element from same list
        if elem_idx + 1 < len(nums[list_idx]):
            next_val = nums[list_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))
            max_val = max(max_val, next_val)

    return result
```

---

### PATTERN 3: Two Heaps (Median Problems)

---

#### Pattern 3A: Find Median from Data Stream

**Problem:** LeetCode 295 - Design data structure supporting addNum and findMedian

```python
import heapq

class MedianFinder:
    def __init__(self):
        self.small = []  # Max-heap (lower half), store negated
        self.large = []  # Min-heap (upper half)

    def addNum(self, num: int) -> None:
        # Add to small (max-heap)
        heapq.heappush(self.small, -num)

        # Balance: move max of small to large
        heapq.heappush(self.large, -heapq.heappop(self.small))

        # Ensure small has same or one more element
        if len(self.large) > len(self.small):
            heapq.heappush(self.small, -heapq.heappop(self.large))

    def findMedian(self) -> float:
        if len(self.small) > len(self.large):
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2.0
```

**Invariants maintained:**
1. `small` contains the smaller half
2. `large` contains the larger half
3. `len(small) == len(large)` or `len(small) == len(large) + 1`
4. `max(small) <= min(large)`

---

#### Pattern 3B: Sliding Window Median

**Problem:** LeetCode 480 - Median of each window of size k

```python
import heapq
from collections import defaultdict

def medianSlidingWindow(nums: list[int], k: int) -> list[float]:
    small = []  # Max-heap (negated)
    large = []  # Min-heap
    removed = defaultdict(int)  # Lazy removal counts

    def add(num):
        if not small or num <= -small[0]:
            heapq.heappush(small, -num)
        else:
            heapq.heappush(large, num)

    def remove(num):
        removed[num] += 1

    def balance():
        # Balance sizes
        while len(small) > len(large) + 1:
            heapq.heappush(large, -heapq.heappop(small))
            prune(small, -1)
        while len(large) > len(small):
            heapq.heappush(small, -heapq.heappop(large))
            prune(large, 1)

    def prune(heap, sign):
        # Remove deleted elements from top
        while heap:
            val = -heap[0] if sign == -1 else heap[0]
            if removed[val] > 0:
                removed[val] -= 1
                heapq.heappop(heap)
            else:
                break

    def get_median():
        if k % 2 == 1:
            return float(-small[0])
        return (-small[0] + large[0]) / 2.0

    # Initialize first window
    for i in range(k):
        add(nums[i])
        balance()
        prune(small, -1)
        prune(large, 1)

    result = [get_median()]

    # Slide window
    for i in range(k, len(nums)):
        add(nums[i])
        remove(nums[i - k])
        balance()
        prune(small, -1)
        prune(large, 1)
        result.append(get_median())

    return result
```

---

#### Pattern 3C: IPO (Maximize Capital)

**Problem:** LeetCode 502 - Select at most k projects to maximize capital

```python
import heapq

def findMaximizedCapital(k: int, w: int, profits: list[int], capital: list[int]) -> int:
    # Min-heap of (capital_required, profit)
    available = list(zip(capital, profits))
    heapq.heapify(available)

    # Max-heap of profits we can afford
    affordable = []

    for _ in range(k):
        # Move all affordable projects to affordable heap
        while available and available[0][0] <= w:
            cap, prof = heapq.heappop(available)
            heapq.heappush(affordable, -prof)

        if not affordable:
            break

        # Select most profitable affordable project
        w += -heapq.heappop(affordable)

    return w
```

---

### PATTERN 4: Scheduling and Simulation

---

#### Pattern 4A: Task Scheduler

**Problem:** LeetCode 621 - Minimum intervals to complete tasks with cooldown

```python
import heapq
from collections import Counter

def leastInterval(tasks: list[str], n: int) -> int:
    count = Counter(tasks)

    # Max-heap of task frequencies
    heap = [-freq for freq in count.values()]
    heapq.heapify(heap)

    time = 0
    while heap:
        cycle = []
        for _ in range(n + 1):  # Each cycle is n+1 slots
            if heap:
                freq = heapq.heappop(heap)
                if freq < -1:  # More occurrences remaining
                    cycle.append(freq + 1)
            time += 1
            if not heap and not cycle:
                break

        # Push remaining tasks back
        for freq in cycle:
            heapq.heappush(heap, freq)

    return time
```

**Alternative formula approach:**
```python
def leastInterval_formula(tasks: list[str], n: int) -> int:
    count = Counter(tasks)
    max_freq = max(count.values())
    max_count = sum(1 for freq in count.values() if freq == max_freq)

    # (max_freq - 1) full cycles + tasks with max frequency
    result = (max_freq - 1) * (n + 1) + max_count

    return max(result, len(tasks))
```

---

#### Pattern 4B: Meeting Rooms II

**Problem:** LeetCode 253 - Minimum meeting rooms needed

```python
import heapq

def minMeetingRooms(intervals: list[list[int]]) -> int:
    if not intervals:
        return 0

    # Sort by start time
    intervals.sort(key=lambda x: x[0])

    # Min-heap of end times (rooms in use)
    rooms = []
    heapq.heappush(rooms, intervals[0][1])

    for start, end in intervals[1:]:
        # If earliest ending room is free, reuse it
        if start >= rooms[0]:
            heapq.heappop(rooms)

        # Add current meeting's end time
        heapq.heappush(rooms, end)

    return len(rooms)
```

---

#### Pattern 4C: Reorganize String

**Problem:** LeetCode 767 - Rearrange so no adjacent characters are same

```python
import heapq
from collections import Counter

def reorganizeString(s: str) -> str:
    count = Counter(s)

    # Check if possible
    max_freq = max(count.values())
    if max_freq > (len(s) + 1) // 2:
        return ""

    # Max-heap of (frequency, char)
    heap = [(-freq, char) for char, freq in count.items()]
    heapq.heapify(heap)

    result = []
    prev_freq, prev_char = 0, ''

    while heap:
        freq, char = heapq.heappop(heap)
        result.append(char)

        # Push back previous character if it has remaining count
        if prev_freq < 0:
            heapq.heappush(heap, (prev_freq, prev_char))

        # Update previous
        prev_freq = freq + 1  # Used one occurrence
        prev_char = char

    return ''.join(result)
```

---

#### Pattern 4D: Ugly Number II

**Problem:** LeetCode 264 - Find nth ugly number (factors only 2, 3, 5)

```python
import heapq

def nthUglyNumber(n: int) -> int:
    heap = [1]
    seen = {1}

    for _ in range(n - 1):
        ugly = heapq.heappop(heap)

        for factor in [2, 3, 5]:
            new_ugly = ugly * factor
            if new_ugly not in seen:
                seen.add(new_ugly)
                heapq.heappush(heap, new_ugly)

    return heap[0]
```

**DP approach (more efficient):**
```python
def nthUglyNumber_dp(n: int) -> int:
    ugly = [1] * n
    p2 = p3 = p5 = 0

    for i in range(1, n):
        next2 = ugly[p2] * 2
        next3 = ugly[p3] * 3
        next5 = ugly[p5] * 5

        ugly[i] = min(next2, next3, next5)

        if ugly[i] == next2:
            p2 += 1
        if ugly[i] == next3:
            p3 += 1
        if ugly[i] == next5:
            p5 += 1

    return ugly[n - 1]
```

---

### PATTERN 5: Advanced Heap Problems

---

#### Pattern 5A: Trapping Rain Water II (3D)

**Problem:** LeetCode 407 - Calculate trapped water in 3D heightmap

```python
import heapq

def trapRainWater(heightMap: list[list[int]]) -> int:
    if not heightMap or not heightMap[0]:
        return 0

    m, n = len(heightMap), len(heightMap[0])
    visited = [[False] * n for _ in range(m)]
    heap = []

    # Add boundary cells to heap
    for i in range(m):
        for j in [0, n - 1]:
            heapq.heappush(heap, (heightMap[i][j], i, j))
            visited[i][j] = True
    for j in range(n):
        for i in [0, m - 1]:
            if not visited[i][j]:
                heapq.heappush(heap, (heightMap[i][j], i, j))
                visited[i][j] = True

    water = 0
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    while heap:
        height, i, j = heapq.heappop(heap)

        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < m and 0 <= nj < n and not visited[ni][nj]:
                visited[ni][nj] = True
                water += max(0, height - heightMap[ni][nj])
                heapq.heappush(heap, (max(height, heightMap[ni][nj]), ni, nj))

    return water
```

---

#### Pattern 5B: Minimum Cost to Connect Sticks

**Problem:** LeetCode 1167 - Minimum cost to connect all sticks

```python
import heapq

def connectSticks(sticks: list[int]) -> int:
    heapq.heapify(sticks)
    cost = 0

    while len(sticks) > 1:
        first = heapq.heappop(sticks)
        second = heapq.heappop(sticks)
        combined = first + second
        cost += combined
        heapq.heappush(sticks, combined)

    return cost
```

---

#### Pattern 5C: Maximum Performance of a Team

**Problem:** LeetCode 1383 - Choose at most k engineers to maximize performance

```python
import heapq

def maxPerformance(n: int, speed: list[int], efficiency: list[int], k: int) -> int:
    MOD = 10**9 + 7

    # Sort by efficiency descending
    engineers = sorted(zip(efficiency, speed), reverse=True)

    heap = []  # Min-heap of speeds
    speed_sum = 0
    max_perf = 0

    for eff, spd in engineers:
        heapq.heappush(heap, spd)
        speed_sum += spd

        if len(heap) > k:
            speed_sum -= heapq.heappop(heap)

        # Current efficiency is minimum (we sorted descending)
        max_perf = max(max_perf, speed_sum * eff)

    return max_perf % MOD
```

---

<a name="post-processing"></a>
## 5. Post-Processing Reference

| Problem Type | Return Value | Notes |
|--------------|--------------|-------|
| **Kth element** | Single value | `heap[0]` after k pops or pushes |
| **Top K elements** | List | Contents of heap (may need to negate) |
| **Merge K sorted** | Merged list/linked list | Build result incrementally |
| **Median** | Float | Depends on odd/even count |
| **Schedule** | Count or time | Track during simulation |

---

<a name="pitfalls"></a>
## 6. Common Pitfalls & Solutions

### Pitfall 1: Forgetting Python Has Only Min-Heap

```python
# WRONG: This gives smallest, not largest
max_val = heapq.heappop(heap)
```

**Solution:** Negate for max-heap
```python
heapq.heappush(heap, -val)  # Store negated
max_val = -heapq.heappop(heap)  # Negate back
```

---

### Pitfall 2: Comparing Non-Comparable Objects

```python
# WRONG: Crashes when values are equal
heapq.heappush(heap, (priority, custom_object))
```

**Solution:** Add tie-breaker or implement `__lt__`
```python
# Option 1: Add unique index
heapq.heappush(heap, (priority, index, custom_object))

# Option 2: Implement __lt__ in class
class Task:
    def __lt__(self, other):
        return self.priority < other.priority
```

---

### Pitfall 3: Modifying Heap During Iteration

```python
# WRONG: Heap property violated
for i in range(len(heap)):
    if some_condition(heap[i]):
        heap[i] = new_value
```

**Solution:** Use lazy deletion or rebuild heap
```python
# Lazy deletion
removed.add(item_to_remove)

# Or rebuild
heap = [x for x in heap if not some_condition(x)]
heapq.heapify(heap)
```

---

### Pitfall 4: Using heappop on Empty Heap

```python
# WRONG: IndexError on empty heap
val = heapq.heappop(heap)
```

**Solution:** Check before popping
```python
if heap:
    val = heapq.heappop(heap)
```

---

### Pitfall 5: Off-by-One in K Elements

**Problem:** Returning k+1 or k-1 elements

**Solution:** Track size explicitly
```python
while len(heap) > k:
    heapq.heappop(heap)
# Now heap has exactly k elements
```

---

### Pitfall 6: Heap vs Sorted for One-Time Query

```python
# Inefficient for one-time k-largest
heap = []
for num in nums:
    heapq.heappush(heap, num)
    if len(heap) > k:
        heapq.heappop(heap)  # O(n log k)
```

**Better for one-time:**
```python
heapq.nlargest(k, nums)  # Optimized internally
# Or sort if k is close to n
```

---

<a name="recognition"></a>
## 7. Problem Recognition Framework

### Step 1: Is Heap Appropriate?

**Use heap when:**
1. Need **k largest/smallest** elements
2. Need to **repeatedly get min/max** from dynamic data
3. **Merging multiple sorted** sequences
4. **Scheduling** with priorities
5. Need **running median** or percentile

**Don't use heap when:**
1. Need all elements sorted (use sort)
2. Need random access to elements
3. K is very close to n (sort might be simpler)
4. Need to update arbitrary elements frequently

### Step 2: Min-Heap or Max-Heap?

| Finding | Heap Type | Why |
|---------|-----------|-----|
| K largest | Min-heap of size k | Keep k largest, eject smallest |
| K smallest | Max-heap of size k | Keep k smallest, eject largest |
| Always need min | Min-heap | Direct access |
| Always need max | Max-heap (negate) | Direct access |
| Median | Both | Split at median |

### Step 3: Additional Data Structures Needed?

| Scenario | Additional Structure |
|----------|---------------------|
| Track if element seen | Set |
| Count elements | Dictionary |
| Lazy deletion | Dictionary of counts |
| Two-way split | Two heaps |

### Decision Tree

```
                  Need extreme values?
                         ↓
                  ┌──────┴──────┐
                 Yes            No
                  ↓              ↓
          Dynamic or static?   Other approach
                  ↓
           ┌──────┴──────┐
        Dynamic        Static
           ↓              ↓
      Need k or 1?      Sort/Select
           ↓
    ┌──────┴──────┐
    K             1
    ↓             ↓
Template A    Simple heap
(size k)
```

---

<a name="checklist"></a>
## 8. Interview Preparation Checklist

### Before the Interview

**Master the fundamentals:**
- [ ] Know heapq operations: heappush, heappop, heapify, heapreplace
- [ ] Understand min-heap → max-heap conversion (negation)
- [ ] Can implement heap from scratch (rare but possible)
- [ ] Know time complexity of all operations

**Practice pattern recognition:**
- [ ] Can identify top-k problems quickly
- [ ] Know when to use heap vs sort
- [ ] Understand two-heap pattern for median

**Know the patterns:**
- [ ] Top K elements (Template A)
- [ ] K-way merge (Template B)
- [ ] Running median (Template C)
- [ ] Task scheduling
- [ ] Lazy deletion

**Common problems solved:**
- [ ] LC 215: Kth Largest Element
- [ ] LC 347: Top K Frequent
- [ ] LC 23: Merge K Sorted Lists
- [ ] LC 295: Find Median from Data Stream
- [ ] LC 378: Kth Smallest in Sorted Matrix
- [ ] LC 253: Meeting Rooms II
- [ ] LC 621: Task Scheduler

### During the Interview

**1. Clarify (30 seconds)**
- What's k relative to n?
- Static or dynamic data?
- Need top k or kth element?

**2. Identify pattern (30 seconds)**
- Top K → min-heap of size k
- Running median → two heaps
- Merge sorted → min-heap with indices

**3. Code (3-4 minutes)**
- Import heapq
- Initialize heap
- Main loop: push, pop, process
- Return result

**4. Test (1-2 minutes)**
- Empty input
- k = 1, k = n
- Duplicates
- Negative numbers

**5. Analyze (30 seconds)**
- Time: O(n log k) for top-k
- Space: O(k) for heap size

---

## 9. Quick Reference Cards

### Min-Heap Operations
```python
import heapq
heap = []
heapq.heappush(heap, val)      # O(log n)
min_val = heapq.heappop(heap)   # O(log n)
peek = heap[0]                  # O(1)
heapq.heapify(list)            # O(n)
heapq.heapreplace(heap, val)   # Pop + push O(log n)
```

### Max-Heap (via negation)
```python
heapq.heappush(heap, -val)
max_val = -heapq.heappop(heap)
peek = -heap[0]
```

### Top K Largest
```python
def topK(nums, k):
    heap = []
    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)
    return heap
```

### Two Heaps Median
```python
small = []  # max-heap (negated)
large = []  # min-heap
# Add: push to small, balance
# Median: -small[0] or average
```

---

## 10. Complexity Reference

| Operation | Time | Space |
|-----------|------|-------|
| heappush | O(log n) | O(1) |
| heappop | O(log n) | O(1) |
| peek (heap[0]) | O(1) | O(1) |
| heapify | O(n) | O(1) |
| Top K from n | O(n log k) | O(k) |
| Merge K lists (N total) | O(N log k) | O(k) |
| Running median | O(log n) per add | O(n) |

---

## Final Thoughts

**Remember:**
1. Python heapq is **min-heap only** — negate for max-heap
2. For **k largest**, use **min-heap of size k**
3. For **running median**, use **two heaps**
4. Include **tie-breaker** when comparing tuples with objects
5. Consider **lazy deletion** when you can't efficiently find elements

**When stuck:**
1. Ask: "Do I need the minimum or maximum repeatedly?"
2. Ask: "Is k much smaller than n?"
3. Consider: "Can I process in sorted order?"
4. Remember: heap excels at dynamic extremes, sort excels at static order

---

## Appendix: Practice Problem Set

### Easy
- 703. Kth Largest Element in a Stream
- 1046. Last Stone Weight
- 1337. The K Weakest Rows in a Matrix

### Medium
- 215. Kth Largest Element in an Array
- 347. Top K Frequent Elements
- 373. Find K Pairs with Smallest Sums
- 378. Kth Smallest Element in a Sorted Matrix
- 451. Sort Characters By Frequency
- 621. Task Scheduler
- 692. Top K Frequent Words
- 767. Reorganize String
- 973. K Closest Points to Origin
- 1167. Minimum Cost to Connect Sticks

### Hard
- 23. Merge k Sorted Lists
- 253. Meeting Rooms II
- 295. Find Median from Data Stream
- 407. Trapping Rain Water II
- 480. Sliding Window Median
- 502. IPO
- 632. Smallest Range Covering Elements from K Lists

**Recommended Practice Order:**
1. Start with LC 215, 347 (basic top-k)
2. Practice LC 23 (k-way merge)
3. Master LC 295 (two heaps)
4. Do LC 621, 253 (scheduling)
5. Attempt LC 480 (combines patterns)

Good luck with your interview preparation!
