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
        # Why `num > heap[0]`?
        # heap[0] is the smallest of the k largest seen so far (the "gatekeeper").
        # If the new number is larger, it deserves to be in the top-k and the
        # current gatekeeper gets evicted. If it's smaller or equal, ignore it.
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
        # Why `num < -heap[0]`?
        # -heap[0] is the largest element in our current k-smallest set (max-heap top).
        # If the new number is smaller than that, it belongs in k-smallest and
        # the current largest gets evicted. Without this check we'd never shrink.
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
        # Why `if lst`?
        # An empty list has no first element, so indexing lst[0] would crash.
        # Skipping it simply means that source contributes zero elements.
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
        # Why `-num`? Python only has min-heap. Storing negated values
        # makes the min-heap behave as a max-heap (largest negated = smallest stored).
        heapq.heappush(self.small, -num)

        # Step 2: Move the largest of small to large.
        # This GUARANTEES max(small) <= min(large) after every insert,
        # because we always pass the biggest "small" value to "large".
        heapq.heappush(self.large, -heapq.heappop(self.small))

        # Step 3: Balance sizes -- small can have at most 1 more than large.
        # Why `len(large) > len(small)` and not `>=`?
        # We allow small to be 1 bigger (odd count), but large must
        # never be bigger than small. If it is, move one back.
        # Example: after inserting 5 into [2|3, 4] the large side gains
        # an extra element. This rebalance restores the invariant.
        if len(self.large) > len(self.small):
            heapq.heappush(self.small, -heapq.heappop(self.large))

    def findMedian(self) -> float:
        # Why check `len(small) > len(large)`?
        # If odd count, small has the extra element -- that's the median.
        # If even count, average the two heap tops.
        # Without this check we'd always average, giving a wrong answer
        # for odd-length streams (e.g., reporting 3.5 instead of 3).
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
        # Why increment a counter instead of actually removing from the heap?
        # Heaps don't support arbitrary removal efficiently (it would require
        # a full O(n) scan). Lazy deletion defers the work: we mark the value
        # as deleted and only discard it when it surfaces to the top via _clean.
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
        # Why `if first is not None`?
        # `next(iterator, None)` returns None when the source is empty.
        # Pushing None into the heap would corrupt tuple comparisons.
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
        # Why `if next_val is not None`?
        # When a source runs out, we simply don't re-add its slot to the heap.
        # The heap naturally shrinks until all sources are exhausted.
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
    # Why `(freq, num)` and not `(num, freq)`?
    # Python compares tuples left-to-right, so the FIRST element determines
    # heap order. Putting freq first means the least-frequent element floats
    # to the top -- making it the correct gatekeeper to evict.
    # If we put num first, ordering would be alphabetical, not by frequency.
    heap = []
    for num, freq in count.items():
        heapq.heappush(heap, (freq, num))
        # Why `len(heap) > k`?
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
        # Why `if len(heap) < k`?
        # Fill the heap to exactly k before switching to replace mode.
        # Until we have k candidates we never evict -- we simply collect.
        if len(heap) < k:
            # Why `-dist`? Negate distance: Python's min-heap becomes a max-heap.
            # The FARTHEST of the k closest sits at the root as gatekeeper.
            heapq.heappush(heap, (-dist, [x, y]))
        # Why `dist < -heap[0][0]`?
        # -heap[0][0] is the farthest distance in our k-closest set.
        # If this point is closer, it replaces the farthest. If it's farther,
        # it can never be in the k-closest, so we skip it.
        elif dist < -heap[0][0]:
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
    # Why `while heap`?
    # We process every unique character exactly once; when heap is empty
    # all characters have been appended in frequency-descending order.
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

    # Initialize with heads.
    # Why `if node`?
    # A None head means an empty linked list. Pushing None into the heap
    # would crash on comparison. Skipping it is safe -- empty lists contribute nothing.
    # We include index `i` as a tiebreaker: when two nodes have the
    # same val, Python compares the next tuple element. Without `i`,
    # it would try to compare ListNode objects, which crashes.
    for i, node in enumerate(lists):
        if node:
            heapq.heappush(heap, (node.val, i, node))

    dummy = ListNode(0)
    current = dummy

    # Why `while heap`?
    # The heap holds exactly one pointer per non-exhausted list. Once every
    # list is fully consumed, the heap empties and the merge is complete.
    while heap:
        val, idx, node = heapq.heappop(heap)
        current.next = node
        current = current.next

        # Why `if node.next`?
        # Only push the next node if this list isn't exhausted.
        # Pushing None would corrupt the heap with a non-comparable entry.
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
        # Why `col + 1 < n`?
        # Bounds check -- each row has n columns. Without this guard,
        # col+1 could index out of range when we reach the last column.
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

    # Why `while heap and len(result) < k`?
    # Two exit conditions: we've collected k pairs (done), or we've exhausted
    # all candidates (heap empty). Checking both prevents over-collection and
    # avoids popping from an empty heap.
    while heap and len(result) < k:
        sum_val, i, j = heapq.heappop(heap)
        result.append([nums1[i], nums2[j]])

        # Why `j + 1 < len(nums2)`?
        # Each pop for (i, j) spawns its successor (i, j+1) only when j+1
        # is within bounds. This incrementally explores the sorted space
        # without materializing all O(m*n) pairs upfront.
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

    # Why `while len(heap) == len(nums)`?
    # The heap holds exactly one element per list. Once any list is exhausted,
    # its representative is gone and the heap shrinks below len(nums).
    # At that point no range can cover all lists, so we stop immediately.
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
        # Why push to small first, then immediately move its max to large?
        # This two-step guarantees that every element in small is truly <=
        # every element in large, even if the new number is very large.
        heapq.heappush(self.small, -num)

        # Balance: move max of small to large
        heapq.heappush(self.large, -heapq.heappop(self.small))

        # Why `len(self.large) > len(self.small)` (not `>=`)?
        # Our invariant allows small to be 1 larger than large (holds median
        # for odd count). large must NEVER exceed small in size. If it does,
        # the median calculation breaks (we'd average the wrong elements).
        if len(self.large) > len(self.small):
            heapq.heappush(self.small, -heapq.heappop(self.large))

    def findMedian(self) -> float:
        # Why `len(self.small) > len(self.large)`?
        # When total count is odd, small holds one extra element -- the true median.
        # When even, both halves are equal and we average their tops.
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
        # Why `if not small or num <= -small[0]`?
        # -small[0] is the current maximum of the lower half.
        # Numbers at or below that max belong in small (lower half).
        # Numbers above it belong in large (upper half).
        # The `not small` guard handles the empty initial state.
        if not small or num <= -small[0]:
            heapq.heappush(small, -num)
        else:
            heapq.heappush(large, num)

    def remove(num):
        removed[num] += 1

    def balance():
        # Why `while len(small) > len(large) + 1`?
        # We allow small to hold at most one extra element (for odd windows).
        # If it has two or more extra, the median would be inside small but
        # we'd compute it incorrectly. Move extras to large.
        while len(small) > len(large) + 1:
            heapq.heappush(large, -heapq.heappop(small))
            prune(small, -1)
        # Why `while len(large) > len(small)`?
        # large must never exceed small. If it does, the lower half is
        # under-populated and small[0] no longer represents the median.
        while len(large) > len(small):
            heapq.heappush(small, -heapq.heappop(large))
            prune(large, 1)

    def prune(heap, sign):
        # Why `while heap` and not `if heap`?
        # Multiple logically-deleted elements may sit consecutively at the top.
        # We must drain all of them before the real minimum/maximum is exposed.
        while heap:
            val = -heap[0] if sign == -1 else heap[0]
            # Why `removed[val] > 0`?
            # removed tracks how many pending deletions exist for each value.
            # Decrement and discard until we hit a live element (count == 0).
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
        # Why `while available and available[0][0] <= w`?
        # available is a min-heap by capital required. Popping while the
        # cheapest project is within our budget moves ALL newly unlocked
        # projects into the affordable max-heap. Stopping early would miss
        # projects that become reachable only after the previous iteration.
        while available and available[0][0] <= w:
            cap, prof = heapq.heappop(available)
            heapq.heappush(affordable, -prof)

        # Why `if not affordable: break`?
        # If no project is within budget after collecting all affordable ones,
        # no future iteration can help either (capital only grows). Early exit
        # avoids k redundant loops that would pop from an empty heap.
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
    # Why `while heap`?
    # heap shrinks as tasks complete. When it's empty all tasks are scheduled.
    while heap:
        cycle = []
        for _ in range(n + 1):  # Each cycle is n+1 slots
            # Why `if heap`?
            # The cooldown window is always n+1 slots wide, but we may run
            # out of unique tasks before filling the window. Empty slots are
            # idle time; we still increment `time` to account for the gap.
            if heap:
                freq = heapq.heappop(heap)
                # Why `freq < -1`?
                # freq is stored negated. -1 means 1 remaining occurrence.
                # After using it, the count drops to 0 -- nothing to re-queue.
                # Only re-queue tasks that still have occurrences left (freq < -1).
                if freq < -1:  # More occurrences remaining
                    cycle.append(freq + 1)
            time += 1
            # Why `if not heap and not cycle: break`?
            # If no tasks remain in the heap AND no tasks are cooling down,
            # we've finished everything. Don't pad with unnecessary idle slots.
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
        # Why `if start >= rooms[0]`?
        # rooms[0] is the earliest finishing meeting. If the new meeting
        # starts at or after that finish time, the room is free to reuse.
        # We pop the old end time and push the new one (same room, new booking).
        # Without this check, every meeting would get its own room -- always wrong.
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
    # Why `max_freq > (len(s) + 1) // 2`?
    # If one character appears more than ceil(len/2) times it's impossible
    # to place it without two adjacent copies. Example: "aaab" -- 'a' appears
    # 3 times in a 4-char string; (4+1)//2 = 2, and 3 > 2, so return "".
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

        # Why `if prev_freq < 0`?
        # prev_freq is stored negated. 0 means the previous character's count
        # has reached 0 -- nothing left to re-enqueue. A value < 0 means it
        # still has occurrences and must be made available again after a one-
        # slot cooldown (which is satisfied because we just placed a different char).
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
            # Why `if new_ugly not in seen`?
            # The same ugly number can be generated multiple ways: e.g., 6 = 2*3 = 3*2.
            # Without the seen-set check, 6 would be pushed twice, causing duplicates
            # and returning a wrong nth ugly number.
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
            # Why `not visited[ni][nj]`?
            # A cell can be reached from multiple boundary directions. Without the
            # visited check, we'd process it multiple times, counting its water
            # contribution repeatedly and inflating the result.
            if 0 <= ni < m and 0 <= nj < n and not visited[ni][nj]:
                visited[ni][nj] = True
                # Why `max(0, height - heightMap[ni][nj])`?
                # `height` is the water level dictated by the surrounding boundary.
                # If the neighbor is taller, no water can pool there (max clamps to 0).
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

    # Why `while len(sticks) > 1`?
    # We need at least two sticks to perform a merge. When only one remains,
    # all merges are done. Using `> 1` (not `> 0`) prevents popping from a
    # single-element heap which would leave `second` undefined.
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

        # Why `if len(heap) > k`?
        # We can pick at most k engineers. When the heap exceeds k, evict
        # the slowest (min-heap root) to keep only the k fastest included so far.
        # This maximises speed_sum for any given efficiency lower bound.
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

---

## Appendix: Conditional Quick Reference

This table lists every key condition used in this handbook, its plain-English meaning, and the intuition behind it.

### A. Heap Size & Maintenance Conditions

| Condition | Plain English | Why it works |
|-----------|---------------|--------------|
| `if len(heap) < k` | Heap is not yet full | Fill to exactly k before entering replace mode; avoids evicting valid candidates too early |
| `if len(heap) > k` | Heap has one too many elements | After pushing, trim the excess immediately — keeps exactly k elements at all times (off-by-one guard) |
| `while len(sticks) > 1` | At least two sticks remain to merge | Need two operands for every combine step; `> 1` prevents an undefined second pop on the last element |
| `while heap and len(result) < k` | Heap non-empty AND haven't collected k results yet | Two exit conditions: collected enough (stop early) or exhausted all candidates (stop safely) |
| `if len(heap) > k` (maxPerformance) | Team size exceeds k | Evict the slowest engineer to keep the k-fastest for the current efficiency lower bound |
| `if elem_idx + 1 < len(lists[list_idx])` | Next element exists in this list | Bounds guard — only extend the heap pointer when the current list has more elements |
| `if col + 1 < n` | Next column exists in this matrix row | Bounds guard — prevents out-of-range access when a row is fully consumed |

### B. Min-Heap / Max-Heap Conversion Conditions

| Condition | Plain English | Why it works |
|-----------|---------------|--------------|
| `heapq.heappush(heap, -val)` | Store the negated value | Python only provides min-heap; negating flips the ordering so the largest value becomes the smallest stored key |
| `max_val = -heapq.heappop(heap)` | Negate the popped value back | Recovers the true positive magnitude after the heap has stored it negated |
| `elif num > heap[0]` | New number beats the current gatekeeper | heap[0] is the smallest of k largest; only larger numbers deserve entry — smaller ones are irrelevant |
| `elif num < -heap[0]` (k-smallest) | New number beats the k-smallest gatekeeper | -heap[0] is the largest of k smallest; only smaller numbers deserve entry into the k-smallest set |
| `(-freq, char)` tuple ordering | Frequency is compared first | Python sorts tuples left-to-right; putting freq (negated) first makes the heap ordered by frequency, not alphabetically |

### C. Balance & Median Conditions

| Condition | Plain English | Why it works |
|-----------|---------------|--------------|
| `if len(self.large) > len(self.small)` | Upper half is bigger than lower half | Invariant: small can be at most 1 larger, never smaller. Violation means the median pointer is on the wrong side |
| `if len(self.small) > len(self.large)` | Lower half holds the extra element | Odd total count means the middle element lives in small; return -small[0] directly instead of averaging |
| `while len(small) > len(large) + 1` | small has more than one extra element | The +1 tolerance allows for odd-count windows; more than +1 means small is over-filled and must donate to large |
| `while len(large) > len(small)` | Upper half exceeds lower half | large must never be bigger; if it is, the median would be miscalculated (averaging wrong tops) |
| `if not small or num <= -small[0]` | Number belongs in the lower half | Route incoming numbers: values at or below the current lower-half max stay in small; larger ones go to large |
| `while len(heap) == len(nums)` | All lists still have a representative | The moment any list is exhausted the covering range condition is broken; stop as soon as heap shrinks |

### D. Lazy Deletion & Staleness Conditions

| Condition | Plain English | Why it works |
|-----------|---------------|--------------|
| `while self.heap and self.deleted[self.heap[0]] > 0` | Top element is marked for deletion | Multiple consecutive deleted elements may stack up; loop (not if) drains ALL phantom tops before exposing the live minimum |
| `if removed[val] > 0` (sliding window prune) | This value has pending removals | Deletion counter tracks how many phantom copies exist; decrement and discard until a live element is found |
| `if new_ugly not in seen` | This ugly number hasn't been generated yet | The same value can be reached via multiple factor paths (e.g. 6 = 2×3 = 3×2); the seen-set prevents duplicate heap entries |
| `if node in visited: continue` (Dijkstra pattern) | A shorter path to this node was already processed | Lazy deletion in Dijkstra: stale entries with outdated distances remain in the heap; skip them to avoid reprocessing |
