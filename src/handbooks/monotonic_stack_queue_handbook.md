# The Complete Monotonic Stack & Queue Handbook
> A template-based approach for mastering monotonic stacks and queues in coding interviews

**Philosophy:** Monotonic stacks/queues are not about maintaining sorted order for fun. They're about **efficiently answering "what's the next/previous greater/smaller element?"** by maintaining a stack where elements are always in increasing or decreasing order.

---

## Table of Contents
1. [Understanding the Core Philosophy](#core-philosophy)
2. [The 3 Master Templates](#master-templates)
3. [Pattern Classification Guide](#pattern-guide)
4. [Complete Pattern Library](#patterns)
5. [Post-Processing Reference](#post-processing)
6. [Common Pitfalls & Solutions](#pitfalls)
7. [Problem Recognition Framework](#recognition)
8. [Interview Preparation Checklist](#checklist)
9. [Quick Reference Cards](#quick-ref)
10. [Complexity Reference](#complexity-ref)

---

<a name="core-philosophy"></a>
## 1. Understanding the Core Philosophy

### First Principles

- **The Tall Person in a Crowd**: You're in a crowd looking right. The next person taller than you blocks your view of everyone behind them. A monotonic stack finds these "view blockers." Shorter people standing between two tall people are irrelevant -- they'll never block anyone's view. The stack discards them immediately.
- **The Temperature Drop**: Each day you ask "when will it next be warmer?" You don't compare against every future day -- you keep a stack of unresolved days, and each new warm day resolves all cooler days on the stack. Each day enters the stack once and leaves once, giving O(n) total.

### No-Jargon Translation

- **Monotonic stack**: a stack where elements are always sorted (all increasing or all decreasing from bottom to top)
- **Next greater element**: the first value to the right that's bigger than the current one
- **Previous smaller element**: the first value to the left that's smaller than the current one
- **Monotonic deque**: a double-ended queue maintaining sorted order, useful for sliding window min/max
- **Span**: how far back you can go from an element before finding one that's greater (or smaller)

### Mental Model

> "A monotonic stack is a bouncer at a club door -- when someone taller shows up, everyone shorter in front gets kicked out because they'll never be relevant again."

---

### Why Monotonic Stack?

The naive approach to "next greater element" problems checks every pair:
```python
# O(n^2) - Check every pair
for i in range(n):
    for j in range(i + 1, n):
        if arr[j] > arr[i]:
            result[i] = arr[j]
            break
```

A monotonic stack reduces this to O(n) by **eliminating irrelevant comparisons**:

> "Instead of looking forward from each element, I process elements and resolve backward -- each new element answers the question for all smaller elements waiting on the stack."

### The Key Insight: Each Element is Processed Twice

Every element enters the stack exactly once and leaves the stack exactly once. That means the total number of push + pop operations across the entire array is at most 2n, which is O(n).

```
arr = [2, 1, 5, 6, 2, 3]

Step-by-step (finding next greater element):

Process 2: stack = [2]
Process 1: stack = [2, 1]          (1 < 2, just push)
Process 5: pop 1 → NGE[1] = 5     (5 > 1, resolve!)
           pop 2 → NGE[2] = 5     (5 > 2, resolve!)
           stack = [5]
Process 6: pop 5 → NGE[5] = 6
           stack = [6]
Process 2: stack = [6, 2]          (2 < 6, just push)
Process 3: pop 2 → NGE[2] = 3
           stack = [6, 3]

End: stack = [6, 3] → NGE = -1 for both (nothing greater to right)
```

### Visual: Monotonic Stack Invariant

```
Monotonic DECREASING stack (bottom to top):    Monotonic INCREASING stack (bottom to top):
Used for: Next Greater Element                 Used for: Next Smaller Element

Stack state:  bottom → top                     Stack state:  bottom → top
              [9, 7, 5, 3]                                   [1, 3, 5, 8]
                   ↓                                              ↓
        Each element < the one below            Each element > the one below

When new element 6 arrives:                    When new element 4 arrives:
Pop 3 (3 < 6) → 6 is NGE for 3               Pop 8 (8 > 4) → 4 is NSE for 8
Pop 5 (5 < 6) → 6 is NGE for 5               Pop 5 (5 > 4) → 4 is NSE for 5
Stop at 7 (7 > 6)                             Stop at 3 (3 < 4)
Push 6: [9, 7, 6]                             Push 4: [1, 3, 4]
```

### Three Flavors of Monotonic Structures

| Structure | Invariant | Used For |
|-----------|-----------|----------|
| Decreasing stack | bottom > top | Next/Previous Greater Element |
| Increasing stack | bottom < top | Next/Previous Smaller Element |
| Monotonic deque | sorted front to back | Sliding window min/max |

### The Duality: Next vs. Previous

A subtle but critical insight:

- **Next greater/smaller**: when element `x` is **popped** by element `y`, then `y` is the **next** greater/smaller of `x`
- **Previous greater/smaller**: when element `x` is **pushed**, whatever is currently on top of the stack is the **previous** greater/smaller of `x`

This means a single pass through the array with a monotonic stack can compute **both** the next and previous relationships simultaneously!

---

<a name="master-templates"></a>
## 2. The 3 Master Templates

### Template 1: Monotonic Decreasing Stack (Next Greater Element)

The stack holds elements in decreasing order from bottom to top. When a new element is larger than the top, it "resolves" the top element -- the new element is the next greater element for everything it pops.

```python
def next_greater_element(nums):
    """
    For each element, find the next element to the right that is greater.
    Returns an array where result[i] = next greater element for nums[i], or -1.

    Stack invariant: decreasing from bottom to top.
    We store indices so we can write into the result array.
    """
    n = len(nums)
    result = [-1] * n
    stack = []  # stores indices; nums[stack[-1]] is decreasing from bottom to top

    for i in range(n):
        # Pop all elements smaller than current -- current is their NGE.
        # Why `>` and not `>=`? We want STRICTLY greater. If we used `>=`,
        # an equal element would wrongly count as "greater."
        # Why `stack` check first? Short-circuit: if stack is empty, nothing to pop.
        # Can this loop run forever? No -- each element is pushed once and popped
        # at most once across the entire outer loop, so total pops <= n.
        while stack and nums[i] > nums[stack[-1]]:
            idx = stack.pop()
            result[idx] = nums[i]
        stack.append(i)

    # Elements remaining in stack have no next greater element (result stays -1).
    # Why? They were never popped, meaning no future element was larger.
    return result
```

**When to use:** "Next greater element," "days until warmer temperature," any forward-looking comparison.

---

### Template 2: Monotonic Increasing Stack (Next Smaller Element)

The stack holds elements in increasing order from bottom to top. When a new element is smaller than the top, it resolves the top element.

```python
def next_smaller_element(nums):
    """
    For each element, find the next element to the right that is smaller.
    Returns an array where result[i] = next smaller element for nums[i], or -1.

    Stack invariant: increasing from bottom to top.
    """
    n = len(nums)
    result = [-1] * n
    stack = []  # stores indices; nums[stack[-1]] is increasing from bottom to top

    for i in range(n):
        # Pop all elements larger than current -- current is their NSE.
        # Why `<` and not `<=`? We want STRICTLY smaller. Equal is not "smaller."
        # Mirror of Template 1: just flip the comparison direction.
        while stack and nums[i] < nums[stack[-1]]:
            idx = stack.pop()
            result[idx] = nums[i]
        stack.append(i)

    return result
```

**When to use:** "Largest rectangle in histogram" (finding boundaries where bars get shorter), "trapping rain water," any problem needing next smaller.

---

### Template 3: Monotonic Deque for Sliding Window Max/Min

A deque (double-ended queue) maintains the window's maximum (or minimum) candidates in decreasing (or increasing) order. The front of the deque is always the current window's max/min.

```python
from collections import deque

def sliding_window_maximum(nums, k):
    """
    Find the maximum in every contiguous window of size k.

    Deque invariant: decreasing from front to back.
    Front of deque is always the index of the current window's maximum.
    """
    result = []
    dq = deque()  # stores indices; nums[dq[0]] >= nums[dq[1]] >= ...

    for i in range(len(nums)):
        # Remove elements outside the window from the front.
        # Why `<= i - k` and not `< i - k`? The window [i-k+1, i] has leftmost
        # valid index i-k+1. So index i-k is OUT. dq[0] <= i-k means dq[0] < i-k+1,
        # i.e., it's outside the window. Using `<` would keep one stale element.
        while dq and dq[0] <= i - k:
            dq.popleft()

        # Remove elements smaller than current from the back.
        # Why `>=` and not just `>`? If nums[dq[-1]] == nums[i], the older equal
        # element will leave the window sooner, so we prefer keeping the newer one.
        # This is safe: we never lose a potential max because the new element is
        # equally large AND will stay in the window longer.
        while dq and nums[i] >= nums[dq[-1]]:
            dq.pop()

        dq.append(i)

        # Why `i >= k - 1`? The first full window spans indices [0, k-1].
        # Before that, we haven't seen k elements yet, so no result to record.
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result
```

**When to use:** "Sliding window maximum/minimum," "shortest subarray with sum at least K," any problem needing efficient min/max over a sliding range.

---

### Decision Matrix: Which Template?

| Problem Asks For | Template | Stack Order | Pop Condition |
|-----------------|----------|-------------|---------------|
| Next Greater Element | Template 1 | Decreasing | `nums[i] > stack top` |
| Next Smaller Element | Template 2 | Increasing | `nums[i] < stack top` |
| Previous Greater Element | Template 1 (check top before push) | Decreasing | Same |
| Previous Smaller Element | Template 2 (check top before push) | Increasing | Same |
| Sliding Window Max | Template 3 (decreasing deque) | Decreasing | Back < current |
| Sliding Window Min | Template 3 (increasing deque) | Increasing | Back > current |

**Quick Rule of Thumb:**
- Want **greater**? Use a **decreasing** stack (so larger elements cause pops).
- Want **smaller**? Use an **increasing** stack (so smaller elements cause pops).
- Want **window min/max**? Use a **monotonic deque**.

---

<a name="pattern-guide"></a>
## 3. Pattern Classification Guide

| Category | Core Idea | Representative Problems |
|----------|-----------|------------------------|
| **Next Greater Element** | For each element, find the first larger one to its right | LC 496, 503, 739 |
| **Previous Greater/Smaller** | For each element, find the first larger/smaller to its left | LC 901, 1063 |
| **Largest Rectangle** | Use stack to find left/right boundaries where height drops | LC 84, 85 |
| **Trapping Rain Water** | Stack tracks valleys; each pop computes trapped water | LC 42 |
| **Sliding Window Max/Min** | Deque maintains max/min candidates in a sliding window | LC 239, 862 |
| **Remove/Build Optimal Sequence** | Greedily remove elements to build lexicographically smallest/largest result | LC 402, 316, 321 |

### When NOT to Use Monotonic Stack

- **Sorted data needed globally**: Use actual sorting.
- **Kth largest/smallest**: Use a heap.
- **Range queries with updates**: Use segment trees.
- **No clear "next/previous greater/smaller" relationship**: Reconsider the approach.

---

<a name="patterns"></a>
## 4. Complete Pattern Library

---

### Pattern 1: Next Greater Element

The foundational monotonic stack pattern. For each element in the array, find the first element to its right that is strictly greater.

---

#### Problem 1.1: Next Greater Element I (LC 496)

**Problem Statement:** You have two arrays `nums1` and `nums2` where `nums1` is a subset of `nums2`. For each element in `nums1`, find the next greater element in `nums2`. If no such element exists, output -1.

**Example:**
```
Input: nums1 = [4, 1, 2], nums2 = [1, 3, 4, 2]
Output: [-1, 3, -1]
Explanation:
  4 → no greater element to its right in nums2 → -1
  1 → next greater in nums2 is 3 → 3
  2 → no greater element to its right in nums2 → -1
```

**Key Insight:** Compute next greater element for ALL elements in `nums2` using a monotonic stack, store in a hash map, then look up values for `nums1`.

**Solution:**
```python
def nextGreaterElement(nums1, nums2):
    """
    Time: O(n + m) where n = len(nums1), m = len(nums2)
    Space: O(m) for the hash map and stack
    """
    nge_map = {}  # element -> its next greater element in nums2
    stack = []    # monotonic decreasing stack (stores values, not indices)

    for num in nums2:
        # Pop all elements smaller than current; current is their NGE.
        # Why store values instead of indices here? Because nums2 has all distinct
        # elements and we only need a value-to-value mapping (no result array).
        while stack and num > stack[-1]:
            smaller = stack.pop()
            nge_map[smaller] = num
        stack.append(num)

    # Elements left in stack have no NGE
    # (they default to -1 via dict.get)

    return [nge_map.get(num, -1) for num in nums1]
```

**Visual Trace:**
```
nums2 = [1, 3, 4, 2]

Step 1: num = 1
  stack = []  →  push 1  →  stack = [1]

Step 2: num = 3
  3 > stack[-1]=1  →  pop 1, nge_map[1] = 3
  stack = []  →  push 3  →  stack = [3]

Step 3: num = 4
  4 > stack[-1]=3  →  pop 3, nge_map[3] = 4
  stack = []  →  push 4  →  stack = [4]

Step 4: num = 2
  2 < stack[-1]=4  →  push 2  →  stack = [4, 2]

End: stack = [4, 2]  →  no NGE for 4 and 2

nge_map = {1: 3, 3: 4}

Lookup nums1 = [4, 1, 2]:
  4 → nge_map.get(4, -1) = -1
  1 → nge_map.get(1, -1) = 3
  2 → nge_map.get(2, -1) = -1

Result: [-1, 3, -1]
```

**Complexity:** Time O(n + m), Space O(m)

---

#### Problem 1.2: Next Greater Element II (LC 503)

**Problem Statement:** Given a circular integer array `nums`, find the next greater element for every element. The search wraps around the array.

**Example:**
```
Input: nums = [1, 2, 1]
Output: [2, -1, 2]
Explanation:
  1 → next greater is 2
  2 → after wrapping around: 1, 2, 1 → no element > 2 → -1
  1 → wraps around, finds 2 → 2
```

**Key Insight:** Simulate the circular array by iterating through the array **twice** (indices 0 to 2n-1). Use `i % n` to wrap indices. Only write results for the first pass.

**Solution:**
```python
def nextGreaterElements(nums):
    """
    Time: O(n) -- each element pushed/popped at most twice
    Space: O(n) for the result and stack
    """
    n = len(nums)
    result = [-1] * n
    stack = []  # stores indices

    # Why iterate 2*n times? In a circular array, element at index n-1 might
    # have its NGE at index 0. Two passes simulate the wrap-around.
    for i in range(2 * n):
        # Why `i % n`? Maps indices n..2n-1 back to 0..n-1, simulating circularity.
        while stack and nums[i % n] > nums[stack[-1]]:
            idx = stack.pop()
            result[idx] = nums[i % n]
        # Why only push during the first pass (i < n)?
        # The second pass exists only to RESOLVE remaining elements on the stack.
        # Pushing again would create duplicate indices and waste work.
        if i < n:
            stack.append(i)

    return result
```

**Visual Trace:**
```
nums = [1, 2, 1], n = 3

First pass (i = 0..2):
  i=0: num=1, stack=[] → push 0 → stack=[0]
  i=1: num=2 > nums[0]=1 → pop 0, result[0]=2
       stack=[] → push 1 → stack=[1]
  i=2: num=1 < nums[1]=2 → push 2 → stack=[1, 2]

Second pass (i = 3..5):
  i=3: i%3=0, num=1 < nums[2]=1 → no pop (not strictly greater)
  i=4: i%3=1, num=2 > nums[2]=1 → pop 2, result[2]=2
       num=2 = nums[1]=2 → no pop (not strictly greater)
  i=5: i%3=2, num=1 < nums[1]=2 → no pop

result = [2, -1, 2]
```

**Complexity:** Time O(n), Space O(n)

---

#### Problem 1.3: Daily Temperatures (LC 739)

**Problem Statement:** Given an array `temperatures`, return an array `answer` where `answer[i]` is the number of days you have to wait after day `i` to get a warmer temperature. If no future day is warmer, put 0.

**Example:**
```
Input: temperatures = [73, 74, 75, 71, 69, 72, 76, 73]
Output: [1, 1, 4, 2, 1, 1, 0, 0]
```

**Key Insight:** This is exactly the "next greater element" pattern, but instead of returning the value, return the **distance** (index difference).

**Solution:**
```python
def dailyTemperatures(temperatures):
    """
    Time: O(n)
    Space: O(n)
    """
    n = len(temperatures)
    answer = [0] * n
    stack = []  # stores indices of days waiting for a warmer day

    for i in range(n):
        # Why `>` and not `>=`? "Warmer" means strictly higher temperature.
        # A day with the same temperature does NOT resolve the wait.
        while stack and temperatures[i] > temperatures[stack[-1]]:
            prev_day = stack.pop()
            # Why `i - prev_day` is always positive: i > prev_day is guaranteed
            # because we only push earlier indices onto the stack.
            answer[prev_day] = i - prev_day  # number of days waited
        stack.append(i)

    return answer
```

**Visual Trace:**
```
temps = [73, 74, 75, 71, 69, 72, 76, 73]

i=0 (73): stack=[] → push 0 → stack=[0]
i=1 (74): 74>73 → pop 0, answer[0]=1-0=1
           stack=[] → push 1 → stack=[1]
i=2 (75): 75>74 → pop 1, answer[1]=2-1=1
           stack=[] → push 2 → stack=[2]
i=3 (71): 71<75 → push 3 → stack=[2,3]
i=4 (69): 69<71 → push 4 → stack=[2,3,4]
i=5 (72): 72>69 → pop 4, answer[4]=5-4=1
           72>71 → pop 3, answer[3]=5-3=2
           72<75 → push 5 → stack=[2,5]
i=6 (76): 76>72 → pop 5, answer[5]=6-5=1
           76>75 → pop 2, answer[2]=6-2=4
           stack=[] → push 6 → stack=[6]
i=7 (73): 73<76 → push 7 → stack=[6,7]

End: stack=[6,7] → answer stays 0 for indices 6 and 7

answer = [1, 1, 4, 2, 1, 1, 0, 0]
```

**Complexity:** Time O(n), Space O(n)

---

### Pattern 2: Previous Greater/Smaller Element

Instead of looking right (next), we look left (previous). The trick: when you push onto the stack, whatever is currently on top is your previous greater/smaller.

---

#### Problem 2.1: Online Stock Span (LC 901)

**Problem Statement:** Design a class `StockSpanner` that collects daily stock prices and returns the "span" of the current day's price. The span is the maximum number of consecutive days (ending with today) where the price was less than or equal to today's price.

**Example:**
```
Input:  prices called in sequence: [100, 80, 60, 70, 60, 75, 85]
Output: spans:                      [1,   1,  1,  2,  1,  4,  6]

For price=75: looking back → 60, 70, 60 are all ≤ 75, so span = 4 (75 itself + 3 previous)
For price=85: looking back → 75, 60, 70, 60, 80 are all ≤ 85, so span = 6
```

**Key Insight:** This is a "previous greater element" problem in disguise. The span of today's price is `today's index - index of the previous greater price`. Use a monotonic decreasing stack. When a new price comes in, pop all smaller-or-equal prices.

**Solution:**
```python
class StockSpanner:
    """
    Time: O(1) amortized per call (each price pushed/popped at most once)
    Space: O(n) total for n calls
    """
    def __init__(self):
        self.stack = []  # (price, index) -- monotonic decreasing
        self.idx = 0

    def next(self, price):
        # Why `<=` and not `<`? The span includes days with EQUAL price.
        # A day with the same price doesn't "block" the span -- only a
        # strictly greater price does. So we pop equal prices too.
        while self.stack and self.stack[-1][0] <= price:
            self.stack.pop()

        # Span = distance from current index to the previous greater element.
        if self.stack:
            # Why `self.idx - self.stack[-1][1]`? The stack top is the most
            # recent day with a STRICTLY greater price. Everything between
            # that day and today has price <= today's price.
            span = self.idx - self.stack[-1][1]
        else:
            # Why `self.idx + 1`? No previous greater price exists, so the
            # span covers every day from day 0 to today (inclusive).
            span = self.idx + 1

        self.stack.append((price, self.idx))
        self.idx += 1
        return span
```

**Visual Trace:**
```
Call 1: price=100, idx=0
  stack=[] → span = 0+1 = 1
  push (100,0) → stack=[(100,0)]
  Return 1

Call 2: price=80, idx=1
  80 ≤ 100? No → don't pop
  stack top index = 0, span = 1-0 = 1
  push (80,1) → stack=[(100,0),(80,1)]
  Return 1

Call 3: price=60, idx=2
  60 ≤ 80? No → don't pop
  span = 2-1 = 1
  push (60,2) → stack=[(100,0),(80,1),(60,2)]
  Return 1

Call 4: price=70, idx=3
  70 > 60? Yes → pop (60,2)
  70 ≤ 80? No → stop
  span = 3-1 = 2
  push (70,3) → stack=[(100,0),(80,1),(70,3)]
  Return 2

Call 5: price=60, idx=4
  60 ≤ 70? No → don't pop
  span = 4-3 = 1
  push (60,4) → stack=[(100,0),(80,1),(70,3),(60,4)]
  Return 1

Call 6: price=75, idx=5
  75 > 60? pop (60,4)
  75 > 70? pop (70,3)
  75 ≤ 80? No → stop
  span = 5-1 = 4
  push (75,5) → stack=[(100,0),(80,1),(75,5)]
  Return 4

Call 7: price=85, idx=6
  85 > 75? pop (75,5)
  85 > 80? pop (80,1)
  85 ≤ 100? No → stop
  span = 6-0 = 6
  push (85,6) → stack=[(100,0),(85,6)]
  Return 6
```

**Complexity:** Time O(1) amortized per call, Space O(n)

---

#### Problem 2.2: Number of Valid Subarrays (LC 1063)

**Problem Statement:** Given an integer array `nums`, return the number of non-empty subarrays where the leftmost element is not larger than any other element in the subarray. In other words, for a subarray starting at index `i`, all elements in the subarray must be >= `nums[i]`.

**Example:**
```
Input: nums = [1, 4, 2, 5, 3]
Output: 11
Explanation: Valid subarrays (leftmost ≤ all others):
  [1], [1,4], [1,4,2], [1,4,2,5], [1,4,2,5,3]  → 5 starting at index 0
  [4], [4,2] is invalid (2<4), so only [4] → wait, let me recalculate
  Actually: [1]→1, [4]→1, [2]→1, [5]→1, [3]→1  (single elements = 5)
  [1,4]→valid, [1,4,2]→valid, [1,4,2,5]→valid, [1,4,2,5,3]→valid → 4
  [4,2]→invalid. [2,5]→valid, [2,5,3]→valid → 2
  [5,3]→invalid. Total = 5 + 4 + 2 = 11
```

**Key Insight:** For each element at index `i`, find the index of the **next smaller element** `j`. Then `i` can start `j - i` valid subarrays (subarrays ending at i, i+1, ..., j-1 are all valid since no element before j is smaller than nums[i]).

**Solution:**
```python
def validSubarrays(nums):
    """
    Time: O(n)
    Space: O(n)
    """
    n = len(nums)
    # For each i, find the index of the next smaller element
    # If no smaller element, use n (meaning all subarrays from i to end are valid)
    nse = [n] * n
    stack = []  # monotonic increasing stack (stores indices)

    for i in range(n):
        # Why `<` (strictly less)? A subarray starting at idx is invalid once we
        # hit an element SMALLER than nums[idx]. Equal elements are fine (>= nums[idx]).
        while stack and nums[i] < nums[stack[-1]]:
            idx = stack.pop()
            nse[idx] = i
        stack.append(i)

    # For element at index i, number of valid subarrays starting at i = nse[i] - i.
    # Why? Subarrays [i..i], [i..i+1], ..., [i..nse[i]-1] are all valid.
    # That's nse[i] - i subarrays. If nse[i] == n, all subarrays to the end are valid.
    result = 0
    for i in range(n):
        result += nse[i] - i

    return result
```

**Complexity:** Time O(n), Space O(n)

---

### Pattern 3: Largest Rectangle in Histogram

This is the crown jewel of monotonic stack problems. The key insight: for each bar, the largest rectangle using that bar as the shortest bar extends left and right until a shorter bar is found.

---

#### Problem 3.1: Largest Rectangle in Histogram (LC 84)

**Problem Statement:** Given an array `heights` representing the heights of bars in a histogram (each bar has width 1), find the area of the largest rectangle that can be formed within the histogram.

**Example:**
```
Input: heights = [2, 1, 5, 6, 2, 3]
Output: 10

Visualization:
        ___
       | 6 |
   ___ |   |
  | 5 ||   |         ___
  |   ||   |___     | 3 |
__|   ||   | 2 |    |   |
|2|   ||   |   |    |   |
|_|___|____|___|____|___|
  2  1   5   6   2    3

The largest rectangle (area=10) uses heights[2..3] = [5,6] with min height 5, width 2:
Area = 5 * 2 = 10
```

**Key Insight:** For each bar `i`, find:
- `left[i]`: index of the previous bar shorter than `heights[i]`
- `right[i]`: index of the next bar shorter than `heights[i]`
- Area using bar `i` as the shortest = `heights[i] * (right[i] - left[i] - 1)`

A monotonic increasing stack handles both boundaries in one pass. When a bar is popped, the current index is its right boundary, and the new stack top is its left boundary.

**Solution:**
```python
def largestRectangleArea(heights):
    """
    Time: O(n)
    Space: O(n)
    """
    stack = []  # monotonic increasing stack (stores indices)
    max_area = 0

    # Why append a sentinel of 0? Without it, bars that never meet a shorter bar
    # to their right stay in the stack forever and are never processed.
    # A height of 0 is shorter than everything, so it forces all remaining bars out.
    heights_extended = heights + [0]

    for i in range(len(heights_extended)):
        # Why `<`? We pop when the current bar is SHORTER than the stack top.
        # This means the popped bar's rectangle can't extend further right.
        # Why not `<=`? Using `<` keeps equal-height bars in the stack.
        # The rightmost equal bar will eventually compute the correct full width.
        while stack and heights_extended[i] < heights_extended[stack[-1]]:
            h = heights_extended[stack.pop()]
            # Why `i if not stack`? If the stack is empty after popping, this bar
            # was the shortest seen so far -- its rectangle extends from index 0
            # all the way to i-1, giving width = i.
            # Why `i - stack[-1] - 1`? The rectangle spans from (stack[-1] + 1)
            # to (i - 1). We subtract 1 to exclude both boundary bars.
            w = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, h * w)
        stack.append(i)

    return max_area
```

**Visual Trace:**
```
heights = [2, 1, 5, 6, 2, 3] + [0]  (sentinel appended)

i=0 (h=2): stack=[] → push 0 → stack=[0]
i=1 (h=1): 1 < 2 → pop 0
            h=2, w=1 (stack empty, w=i=1), area=2*1=2, max_area=2
            push 1 → stack=[1]
i=2 (h=5): 5 > 1 → push 2 → stack=[1,2]
i=3 (h=6): 6 > 5 → push 3 → stack=[1,2,3]
i=4 (h=2): 2 < 6 → pop 3
            h=6, w=4-2-1=1, area=6*1=6, max_area=6
            2 < 5 → pop 2
            h=5, w=4-1-1=2, area=5*2=10, max_area=10
            2 > 1 → push 4 → stack=[1,4]
i=5 (h=3): 3 > 2 → push 5 → stack=[1,4,5]
i=6 (h=0, sentinel): 0 < 3 → pop 5
            h=3, w=6-4-1=1, area=3*1=3, max_area=10
            0 < 2 → pop 4
            h=2, w=6-1-1=4, area=2*4=8, max_area=10
            0 < 1 → pop 1
            h=1, w=6 (stack empty), area=1*6=6, max_area=10

Result: 10
```

**Complexity:** Time O(n), Space O(n)

---

#### Problem 3.2: Maximal Rectangle (LC 85)

**Problem Statement:** Given a 2D binary matrix filled with '0's and '1's, find the largest rectangle containing only '1's and return its area.

**Example:**
```
Input: matrix = [
  ["1","0","1","0","0"],
  ["1","0","1","1","1"],
  ["1","1","1","1","1"],
  ["1","0","0","1","0"]
]
Output: 6

Explanation: The maximal rectangle is in rows 1-2, columns 2-4:
  1 1 1
  1 1 1
```

**Key Insight:** Build a histogram for each row (treating consecutive 1's above as bar heights), then apply LC 84 (Largest Rectangle in Histogram) for each row.

**Solution:**
```python
def maximalRectangle(matrix):
    """
    Time: O(rows * cols)
    Space: O(cols)
    """
    # Why check both `not matrix` and `not matrix[0]`?
    # `not matrix` catches empty input []. `not matrix[0]` catches [[]]
    # (a matrix with one empty row). Either way, no area is possible.
    if not matrix or not matrix[0]:
        return 0

    cols = len(matrix[0])
    heights = [0] * cols
    max_area = 0

    for row in matrix:
        # Build histogram heights
        for j in range(cols):
            # Why `+= 1` for '1' but `= 0` for '0'?
            # A '1' extends the bar from the row above (consecutive 1s stack up).
            # A '0' breaks the streak -- the bar resets to height 0, not -1,
            # because you can't build a rectangle through a gap.
            if row[j] == '1':
                heights[j] += 1
            else:
                heights[j] = 0

        # Apply largest rectangle in histogram
        max_area = max(max_area, _largest_rectangle(heights))

    return max_area

def _largest_rectangle(heights):
    """Same logic as LC 84 -- see detailed comments in Problem 3.1 above."""
    stack = []
    max_area = 0
    extended = heights + [0]  # Sentinel to flush remaining bars

    for i in range(len(extended)):
        while stack and extended[i] < extended[stack[-1]]:
            h = extended[stack.pop()]
            w = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, h * w)
        stack.append(i)

    return max_area
```

**Visual Trace:**
```
matrix:
  1 0 1 0 0
  1 0 1 1 1
  1 1 1 1 1
  1 0 0 1 0

Row-by-row histogram heights:
  Row 0: [1, 0, 1, 0, 0] → largest rect = 1
  Row 1: [2, 0, 2, 1, 1] → largest rect = 3 (height 1, width 3 at cols 2-4)
  Row 2: [3, 1, 3, 2, 2] → largest rect = 6 (height 2, width 3 at cols 2-4)
  Row 3: [4, 0, 0, 3, 0] → largest rect = 4 (height 4, width 1 at col 0)

Max across all rows: 6
```

**Complexity:** Time O(rows * cols), Space O(cols)

---

### Pattern 4: Trapping Rain Water

---

#### Problem 4.1: Trapping Rain Water (LC 42)

**Problem Statement:** Given `n` non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.

**Example:**
```
Input: height = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
Output: 6

Visualization:
                     ___
             ___    | 3 |     ___
    ___     | 2 |   |   |___| 2 |
   | 1 | ~ |   | ~ | ~ | 1 ||   |___
___|   |___|   |_1_|_0_|   ||   | 1 |
 0   1   0   2   1   0   1   3   2   1   2   1

Water trapped (marked with ~): 6 units
```

**Key Insight (stack approach):** Use a monotonic stack to find "valleys." When we encounter a bar taller than the stack's top, we've found the right wall of a valley. The popped element is the valley floor, and the new stack top is the left wall. Water trapped = `min(left_wall, right_wall) - valley_floor) * width`.

**Solution:**
```python
def trap(height):
    """
    Stack-based approach.
    Time: O(n)
    Space: O(n)
    """
    stack = []  # monotonic stack storing indices
    water = 0

    for i in range(len(height)):
        # Why `>`? We need a bar TALLER than the valley floor to form a right wall.
        # Equal height doesn't create a valley to trap water in.
        while stack and height[i] > height[stack[-1]]:
            valley = stack.pop()

            # Why check `if not stack`? After popping the valley, if the stack is
            # empty, there's no left wall. Water needs BOTH a left and right wall
            # to be trapped. Without a left wall, water flows off the left side.
            if not stack:
                break

            left_wall = stack[-1]
            # Why `min(left_wall_height, right_wall_height)`? Water level is
            # limited by the SHORTER wall (water would overflow the shorter one).
            # Why subtract `height[valley]`? The valley floor displaces water.
            # Can bounded_height be negative? No -- the valley was popped because
            # height[i] > height[valley], and left_wall >= valley (stack invariant),
            # so min(left, right) >= height[valley].
            bounded_height = min(height[left_wall], height[i]) - height[valley]
            # Why `i - left_wall - 1`? The water spans between the two walls,
            # not including the walls themselves.
            width = i - left_wall - 1
            water += bounded_height * width

        stack.append(i)

    return water
```

**Visual Trace:**
```
height = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]

i=0 (h=0): stack=[] → push 0 → stack=[0]
i=1 (h=1): 1>0 → pop 0 (valley=0), stack empty → break
            push 1 → stack=[1]
i=2 (h=0): 0<1 → push 2 → stack=[1,2]
i=3 (h=2): 2>0 → pop 2 (valley h=0)
            left=1 (h=1), right=3 (h=2)
            bounded_height = min(1,2)-0 = 1, width = 3-1-1 = 1
            water += 1*1 = 1. Total water = 1
            2>1 → pop 1 (valley h=1)
            stack empty → break
            push 3 → stack=[3]
i=4 (h=1): 1<2 → push 4 → stack=[3,4]
i=5 (h=0): 0<1 → push 5 → stack=[3,4,5]
i=6 (h=1): 1>0 → pop 5 (valley h=0)
            left=4 (h=1), right=6 (h=1)
            bounded_height = min(1,1)-0 = 1, width = 6-4-1 = 1
            water += 1. Total = 2
            1=1 → not strictly greater → stop
            push 6 → stack=[3,4,6]
i=7 (h=3): 3>1 → pop 6 (valley h=1)
            left=4 (h=1), bounded_height=min(1,3)-1=0, water+=0
            3>1 → pop 4 (valley h=1)
            left=3 (h=2), bounded_height=min(2,3)-1=1, width=7-3-1=3
            water += 3. Total = 5
            3>2 → pop 3 (valley h=2)
            stack empty → break
            push 7 → stack=[7]
i=8 (h=2): 2<3 → push 8 → stack=[7,8]
i=9 (h=1): 1<2 → push 9 → stack=[7,8,9]
i=10 (h=2): 2>1 → pop 9 (valley h=1)
             left=8 (h=2), bounded_height=min(2,2)-1=1, width=10-8-1=1
             water += 1. Total = 6
             2=2 → stop
             push 10 → stack=[7,8,10]
i=11 (h=1): 1<2 → push 11 → stack=[7,8,10,11]

Total water = 6
```

**Complexity:** Time O(n), Space O(n)

---

### Pattern 5: Sliding Window Maximum (Monotonic Deque)

When you need the maximum (or minimum) of every contiguous window of size k, a monotonic deque gives you O(n) total -- O(1) amortized per window.

---

#### Problem 5.1: Sliding Window Maximum (LC 239)

**Problem Statement:** Given an array `nums` and a window size `k`, return an array of the maximum value in each sliding window of size `k`.

**Example:**
```
Input: nums = [1, 3, -1, -3, 5, 3, 6, 7], k = 3
Output: [3, 3, 5, 5, 6, 7]

Window positions:
[1  3  -1] -3  5  3  6  7   → max = 3
 1 [3  -1  -3] 5  3  6  7   → max = 3
 1  3 [-1  -3  5] 3  6  7   → max = 5
 1  3  -1 [-3  5  3] 6  7   → max = 5
 1  3  -1  -3 [5  3  6] 7   → max = 6
 1  3  -1  -3  5 [3  6  7]  → max = 7
```

**Key Insight:** Maintain a deque of indices where values are in decreasing order. The front is always the max of the current window. Remove from the front when it falls outside the window, and remove from the back when the new element is larger (since those elements can never be the max while the new element is in the window).

**Solution:**
```python
from collections import deque

def maxSlidingWindow(nums, k):
    """
    Time: O(n) -- each element enqueued and dequeued at most once
    Space: O(k) for the deque
    """
    dq = deque()  # indices, with nums[dq[0]] >= nums[dq[1]] >= ...
    result = []

    for i in range(len(nums)):
        # Why `<= i - k`? Window covers [i-k+1, i]. Index i-k is one step
        # outside. So any index <= i-k is stale and must be removed.
        while dq and dq[0] <= i - k:
            dq.popleft()

        # Why `>=` (not `>`)? If the back element equals nums[i], the back
        # element is older and will expire from the window sooner. The new
        # element is equally large but fresher, so it's strictly better to keep.
        while dq and nums[i] >= nums[dq[-1]]:
            dq.pop()

        dq.append(i)

        # Why `i >= k - 1`? The first valid window is [0..k-1], which completes
        # when i reaches k-1. Before that, we don't have k elements yet.
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result
```

**Visual Trace:**
```
nums = [1, 3, -1, -3, 5, 3, 6, 7], k = 3

i=0 (val=1):
  dq=[] → append 0 → dq=[0]
  i < k-1 → no output

i=1 (val=3):
  dq[0]=0, not outside window
  nums[1]=3 >= nums[0]=1 → pop 0 → dq=[]
  append 1 → dq=[1]
  i < k-1 → no output

i=2 (val=-1):
  dq[0]=1, not outside window
  nums[2]=-1 < nums[1]=3 → don't pop
  append 2 → dq=[1, 2]
  i >= k-1 → result.append(nums[1]=3) → result=[3]

i=3 (val=-3):
  dq[0]=1, 1 <= 3-3=0? No
  nums[3]=-3 < nums[2]=-1 → don't pop
  append 3 → dq=[1, 2, 3]
  result.append(nums[1]=3) → result=[3, 3]

i=4 (val=5):
  dq[0]=1, 1 <= 4-3=1? Yes → popleft → dq=[2, 3]
  nums[4]=5 >= nums[3]=-3 → pop 3 → dq=[2]
  nums[4]=5 >= nums[2]=-1 → pop 2 → dq=[]
  append 4 → dq=[4]
  result.append(nums[4]=5) → result=[3, 3, 5]

i=5 (val=3):
  dq[0]=4, not outside
  nums[5]=3 < nums[4]=5 → don't pop
  append 5 → dq=[4, 5]
  result.append(nums[4]=5) → result=[3, 3, 5, 5]

i=6 (val=6):
  dq[0]=4, 4 <= 6-3=3? No
  nums[6]=6 >= nums[5]=3 → pop 5
  nums[6]=6 >= nums[4]=5 → pop 4 → dq=[]
  append 6 → dq=[6]
  result.append(nums[6]=6) → result=[3, 3, 5, 5, 6]

i=7 (val=7):
  dq[0]=6, not outside
  nums[7]=7 >= nums[6]=6 → pop 6 → dq=[]
  append 7 → dq=[7]
  result.append(nums[7]=7) → result=[3, 3, 5, 5, 6, 7]

Result: [3, 3, 5, 5, 6, 7]
```

**Complexity:** Time O(n), Space O(k)

---

#### Problem 5.2: Shortest Subarray with Sum at Least K (LC 862)

**Problem Statement:** Given an integer array `nums` and an integer `k`, return the length of the shortest non-empty subarray with a sum of at least `k`. If no such subarray exists, return -1. (Note: `nums` may contain negative numbers.)

**Example:**
```
Input: nums = [2, -1, 2], k = 3
Output: 3
Explanation: The only subarray with sum >= 3 is [2, -1, 2] with sum 3.
```

**Key Insight:** Use prefix sums. We want the shortest `j - i` such that `prefix[j] - prefix[i] >= k`. For each `j`, we want the largest `i < j` where `prefix[i] <= prefix[j] - k`. A monotonic deque on prefix sums (increasing order) lets us efficiently find this.

- **Front removal:** If `prefix[j] - prefix[dq[0]] >= k`, then `dq[0]` is a valid start. Record the length, then remove `dq[0]` (no future `j'` > `j` can give a shorter subarray with this start).
- **Back removal:** If `prefix[j] <= prefix[dq[-1]]`, then `dq[-1]` is useless as a start (any future index would prefer `j` over `dq[-1]` since `j` is farther right AND has a smaller-or-equal prefix sum).

**Solution:**
```python
from collections import deque

def shortestSubarray(nums, k):
    """
    Time: O(n)
    Space: O(n)
    """
    n = len(nums)
    # Compute prefix sums: prefix[j] = sum(nums[0..j-1])
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + nums[i]

    dq = deque()  # indices into prefix array, prefix values are increasing
    result = float('inf')

    for j in range(n + 1):
        # Check if current prefix minus front of deque >= k
        while dq and prefix[j] - prefix[dq[0]] >= k:
            result = min(result, j - dq.popleft())

        # Maintain increasing order: remove from back if current prefix is ≤ back
        while dq and prefix[j] <= prefix[dq[-1]]:
            dq.pop()

        dq.append(j)

    return result if result != float('inf') else -1
```

**Visual Trace:**
```
nums = [2, -1, 2], k = 3
prefix = [0, 2, 1, 3]

j=0: prefix[0]=0
  dq=[] → append 0 → dq=[0]

j=1: prefix[1]=2
  prefix[1]-prefix[0] = 2-0 = 2 < 3 → no front removal
  prefix[1]=2 > prefix[0]=0 → no back removal
  append 1 → dq=[0, 1]

j=2: prefix[2]=1
  prefix[2]-prefix[0] = 1-0 = 1 < 3 → no front removal
  prefix[2]=1 <= prefix[1]=2 → pop 1 → dq=[0]
  prefix[2]=1 > prefix[0]=0 → no more back removal
  append 2 → dq=[0, 2]

j=3: prefix[3]=3
  prefix[3]-prefix[0] = 3-0 = 3 >= 3 → result=min(inf, 3-0)=3, popleft 0
  prefix[3]-prefix[2] = 3-1 = 2 < 3 → stop front
  prefix[3]=3 > prefix[2]=1 → no back removal
  append 3 → dq=[2, 3]

Result: 3
```

**Complexity:** Time O(n), Space O(n)

---

### Pattern 6: Remove/Build Optimal Sequence

Use a monotonic stack to greedily build the smallest or largest possible subsequence by removing elements that violate the desired order, subject to constraints.

---

#### Problem 6.1: Remove K Digits (LC 402)

**Problem Statement:** Given a non-negative integer `num` represented as a string and an integer `k`, remove `k` digits from the number so that the resulting number is as small as possible.

**Example:**
```
Input: num = "1432219", k = 3
Output: "1219"
Explanation: Remove 4, 3, 2 → "1219"
```

**Key Insight:** Use a monotonic increasing stack. Iterate through digits: if the current digit is smaller than the stack's top, pop the top (that's a removal). This greedily removes the leftmost "peak" digits first, which is optimal for minimizing the result.

**Solution:**
```python
def removeKdigits(num, k):
    """
    Time: O(n)
    Space: O(n)
    """
    stack = []

    for digit in num:
        # While we can still remove and the top of stack is larger
        while k > 0 and stack and stack[-1] > digit:
            stack.pop()
            k -= 1
        stack.append(digit)

    # If we still have removals left, remove from the end
    # (the stack is now non-decreasing, so the largest are at the end)
    if k > 0:
        stack = stack[:-k]

    # Remove leading zeros and handle empty result
    result = ''.join(stack).lstrip('0')
    return result if result else '0'
```

**Visual Trace:**
```
num = "1432219", k = 3

digit='1': stack=[] → push → stack=['1']
digit='4': '4' > '1' → push → stack=['1','4']
digit='3': '3' < '4' → pop '4', k=2 → stack=['1']
           '3' > '1' → push → stack=['1','3']
digit='2': '2' < '3' → pop '3', k=1 → stack=['1']
           '2' > '1' → push → stack=['1','2']
digit='2': '2' = '2' → push → stack=['1','2','2']
digit='1': '1' < '2' → pop '2', k=0 → stack=['1','2']
           k=0, stop removing → push → stack=['1','2','1']
digit='9': push → stack=['1','2','1','9']

k=0, no more removals needed.
Result: ''.join(['1','2','1','9']) = "1219"
```

**Complexity:** Time O(n), Space O(n)

---

#### Problem 6.2: Remove Duplicate Letters (LC 316)

**Problem Statement:** Given a string `s`, remove duplicate letters so that every letter appears once and only once. The result must be the smallest in lexicographical order among all possible results.

**Example:**
```
Input: s = "bcabc"
Output: "abc"

Input: s = "cbacdcbc"
Output: "acdb"
```

**Key Insight:** Use a monotonic increasing stack with two extra constraints:
1. Track which characters are already in the stack (don't add duplicates).
2. Only pop a character if it appears later in the string (so we won't lose it forever).

**Solution:**
```python
def removeDuplicateLetters(s):
    """
    Time: O(n)
    Space: O(1) -- at most 26 characters in stack
    """
    # Count remaining occurrences of each character
    remaining = {}
    for ch in s:
        remaining[ch] = remaining.get(ch, 0) + 1

    stack = []
    in_stack = set()

    for ch in s:
        remaining[ch] -= 1

        # Skip if already in the result
        if ch in in_stack:
            continue

        # Pop characters that are:
        # 1) Greater than current (to get lexicographic order)
        # 2) Will appear again later (safe to remove)
        while stack and ch < stack[-1] and remaining[stack[-1]] > 0:
            removed = stack.pop()
            in_stack.remove(removed)

        stack.append(ch)
        in_stack.add(ch)

    return ''.join(stack)
```

**Visual Trace:**
```
s = "cbacdcbc"
Initial remaining: {c:4, b:2, a:1, d:1}

ch='c': remaining[c]=3, stack=[], push → stack=['c'], in_stack={c}
ch='b': remaining[b]=1, 'b'<'c' and remaining[c]=3>0 → pop 'c' → stack=[]
        push 'b' → stack=['b'], in_stack={b}
ch='a': remaining[a]=0, 'a'<'b' and remaining[b]=1>0 → pop 'b' → stack=[]
        push 'a' → stack=['a'], in_stack={a}
ch='c': remaining[c]=2, 'c'>'a' → push → stack=['a','c'], in_stack={a,c}
ch='d': remaining[d]=0, 'd'>'c' → push → stack=['a','c','d'], in_stack={a,c,d}
ch='c': remaining[c]=1, 'c' in in_stack → skip
ch='b': remaining[b]=0, 'b'<'d' but remaining[d]=0 → can't pop d
        push 'b' → stack=['a','c','d','b'], in_stack={a,c,d,b}
ch='c': remaining[c]=0, 'c' in in_stack → skip

Result: "acdb"
```

**Complexity:** Time O(n), Space O(1) (alphabet-bounded)

---

#### Problem 6.3: Create Maximum Number (LC 321)

**Problem Statement:** Given two integer arrays `nums1` and `nums2` of lengths `m` and `n` respectively, create the maximum number of length `k` by selecting digits from both arrays while preserving relative order within each array.

**Example:**
```
Input: nums1 = [3, 4, 6, 5], nums2 = [9, 1, 2, 5, 8, 3], k = 5
Output: [9, 8, 6, 5, 3]
```

**Key Insight:** Three sub-problems combined:
1. **Pick best subsequence of length `t` from a single array**: Use a monotonic decreasing stack with a budget of removals.
2. **Merge two subsequences into the largest possible**: Greedily pick the larger leading element (with tie-breaking by comparing the rest).
3. **Try all splits**: For `k` total digits, try taking `i` from `nums1` and `k-i` from `nums2` for all valid `i`.

**Solution:**
```python
def maxNumber(nums1, nums2, k):
    """
    Time: O(k * (m + n + k))
    Space: O(k)
    """
    def max_subsequence(nums, length):
        """Pick the largest subsequence of given length using monotonic stack."""
        drop = len(nums) - length  # number of elements we can drop
        stack = []
        for num in nums:
            while drop > 0 and stack and stack[-1] < num:
                stack.pop()
                drop -= 1
            stack.append(num)
        return stack[:length]

    def merge(sub1, sub2):
        """Merge two subsequences into the largest possible sequence."""
        result = []
        i, j = 0, 0
        while i < len(sub1) and j < len(sub2):
            # Compare remaining subsequences lexicographically
            if sub1[i:] >= sub2[j:]:
                result.append(sub1[i])
                i += 1
            else:
                result.append(sub2[j])
                j += 1
        result.extend(sub1[i:])
        result.extend(sub2[j:])
        return result

    best = []
    m, n = len(nums1), len(nums2)

    for i in range(k + 1):
        j = k - i
        if i > m or j > n:
            continue
        sub1 = max_subsequence(nums1, i)
        sub2 = max_subsequence(nums2, j)
        merged = merge(sub1, sub2)
        if merged > best:
            best = merged

    return best
```

**Visual Trace:**
```
nums1 = [3, 4, 6, 5], nums2 = [9, 1, 2, 5, 8, 3], k = 5

Try i=0 from nums1, j=5 from nums2:
  sub2 = max_subsequence([9,1,2,5,8,3], 5) = [9,2,5,8,3]
  merged = [9,2,5,8,3]

Try i=1, j=4:
  sub1 = max_subsequence([3,4,6,5], 1) = [6]
  sub2 = max_subsequence([9,1,2,5,8,3], 4) = [9,5,8,3]
  merged = [9,6,5,8,3]

Try i=2, j=3:
  sub1 = max_subsequence([3,4,6,5], 2) = [6,5]
  sub2 = max_subsequence([9,1,2,5,8,3], 3) = [9,8,3]
  merged = [9,8,6,5,3]

Try i=3, j=2:
  sub1 = max_subsequence([3,4,6,5], 3) = [4,6,5]
  sub2 = max_subsequence([9,1,2,5,8,3], 2) = [9,8]
  merged = [9,8,4,6,5]

Try i=4, j=1:
  sub1 = max_subsequence([3,4,6,5], 4) = [3,4,6,5]
  sub2 = max_subsequence([9,1,2,5,8,3], 1) = [9]
  merged = [9,3,4,6,5]

Best: [9,8,6,5,3] (from i=2, j=3)
```

**Complexity:** Time O(k * (m + n + k)), Space O(m + n + k)

---

<a name="post-processing"></a>
## 5. Post-Processing Reference

After building your monotonic stack/deque result, you often need to post-process. Here is a reference table:

| Raw Result | Post-Processing | Example Problem |
|------------|----------------|-----------------|
| Index array of next greater | Convert `result[i] = j` to `result[i] = j - i` (distance) | LC 739 Daily Temperatures |
| Index array of next greater | Look up `nums[result[i]]` to get actual values | LC 496 Next Greater Element I |
| Stack still has elements | Set their result to `-1` (no next greater/smaller found) | All NGE/NSE problems |
| Left and right boundaries | Compute area as `height * (right - left - 1)` | LC 84 Histogram |
| Deque front per window | Append `nums[dq[0]]` to result array | LC 239 Sliding Window Max |
| Stack of characters/digits | Join into string, strip leading zeros | LC 402 Remove K Digits |
| Prefix sum deque | Track `j - popleft()` for shortest subarray | LC 862 Shortest Subarray |
| Multiple subsequences | Merge step needed to combine results | LC 321 Create Maximum Number |
| Remaining removals after loop | Remove from end of stack: `stack[:-k]` | LC 402 Remove K Digits |
| Circular array | Iterate `2n` times, use `i % n` for index | LC 503 Next Greater II |

### Common Post-Processing Patterns

**Pattern A: Map result from index to value**
```python
# After computing NGE indices, convert to values
for i in range(n):
    if result[i] != -1:
        result[i] = nums[result[i]]
```

**Pattern B: Compute distance from index**
```python
# After computing NGE indices, convert to distances
for i in range(n):
    if result[i] != -1:
        result[i] = result[i] - i
    else:
        result[i] = 0  # or whatever the default is
```

**Pattern C: Handle remaining stack elements**
```python
# Elements left in stack never found a match
while stack:
    idx = stack.pop()
    result[idx] = -1  # or 0, or n, depending on problem
```

---

<a name="pitfalls"></a>
## 6. Common Pitfalls & Solutions

### Pitfall 1: Storing Values Instead of Indices

**Wrong:**
```python
stack = []
for i in range(n):
    while stack and nums[i] > stack[-1]:
        val = stack.pop()
        # How do we know WHERE val was? We can't write to result!
    stack.append(nums[i])
```

**Right:**
```python
stack = []
for i in range(n):
    while stack and nums[i] > nums[stack[-1]]:
        idx = stack.pop()
        result[idx] = nums[i]  # Now we know which position to update
    stack.append(i)
```

**Why:** Most problems need you to write back to a result array by index. Always store indices unless the problem only needs a hash map (like LC 496).

---

### Pitfall 2: Strict vs. Non-Strict Comparison

**Wrong (when you need strict greater):**
```python
while stack and nums[i] >= nums[stack[-1]]:  # >= includes equal
    idx = stack.pop()
    result[idx] = nums[i]
```

**Right:**
```python
while stack and nums[i] > nums[stack[-1]]:  # > for strictly greater
    idx = stack.pop()
    result[idx] = nums[i]
```

**Why:** Using `>=` instead of `>` (or vice versa) can cause incorrect results. For "next greater," use `>`. For stock span (where equal prices count as part of the span), use `<=` in the pop condition. Always match the comparison to the problem's definition.

---

### Pitfall 3: Forgetting the Sentinel in Histogram

**Wrong:**
```python
def largestRectangleArea(heights):
    stack = []
    max_area = 0
    for i in range(len(heights)):
        while stack and heights[i] < heights[stack[-1]]:
            h = heights[stack.pop()]
            w = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, h * w)
        stack.append(i)
    return max_area  # BUG: elements remaining in stack are never processed!
```

**Right:**
```python
def largestRectangleArea(heights):
    stack = []
    max_area = 0
    heights = heights + [0]  # Sentinel forces all bars to be processed
    for i in range(len(heights)):
        while stack and heights[i] < heights[stack[-1]]:
            h = heights[stack.pop()]
            w = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, h * w)
        stack.append(i)
    return max_area
```

**Why:** Without the sentinel value of 0 at the end, bars remaining in the stack (those that never encountered a shorter bar to their right) are never processed. The sentinel is shorter than everything, so it forces all remaining bars to pop.

---

### Pitfall 4: Off-by-One in Width Calculation

**Wrong:**
```python
w = i - stack[-1]  # Off by one
```

**Right:**
```python
w = i - stack[-1] - 1  # Exclude both boundaries
```

**Why:** The rectangle using the popped bar doesn't include the left boundary (`stack[-1]`) or the right boundary (`i`). It spans from `stack[-1] + 1` to `i - 1`, which is a width of `i - stack[-1] - 1`. When the stack is empty, the bar extends all the way to the left edge, so `w = i`.

---

### Pitfall 5: Deque Window Boundary Check

**Wrong:**
```python
while dq and dq[0] < i - k:  # Off by one: should be <=
    dq.popleft()
```

**Right:**
```python
while dq and dq[0] <= i - k:  # Remove elements outside window [i-k+1, i]
    dq.popleft()
```

**Why:** For a window of size `k` ending at index `i`, the leftmost valid index is `i - k + 1`. So any index `<= i - k` is outside the window.

---

### Pitfall 6: Not Handling Leading Zeros in Remove K Digits

**Wrong:**
```python
return ''.join(stack)
```

**Right:**
```python
result = ''.join(stack).lstrip('0')
return result if result else '0'
```

**Why:** After removing digits, the result might have leading zeros (e.g., "10200" with k=1 becomes "0200"). Also, the result might be empty (e.g., "10" with k=2), which should return "0" not "".

---

### Pitfall 7: Circular Array Without Double Pass

**Wrong:**
```python
for i in range(n):  # Only one pass -- misses wrap-around cases
    while stack and nums[i] > nums[stack[-1]]:
        ...
    stack.append(i)
```

**Right:**
```python
for i in range(2 * n):  # Two passes to handle circularity
    while stack and nums[i % n] > nums[stack[-1]]:
        idx = stack.pop()
        result[idx] = nums[i % n]
    if i < n:
        stack.append(i)  # Only push in the first pass
```

**Why:** In a circular array, an element near the end might have its next greater element near the beginning. The second pass allows wrapping around, and we only push indices in the first pass to avoid processing duplicates.

---

<a name="recognition"></a>
## 7. Problem Recognition Framework

### Decision Tree

```
                    Is the problem about finding relationships
                    between elements in an array/sequence?
                            |
                    ________|________
                   |                 |
                  YES                NO → Not a monotonic stack problem
                   |
        Does it involve "next" or "previous"
        greater/smaller element relationships?
                   |
            _______|_______
           |               |
          YES              NO
           |               |
    Use MONOTONIC      Is it about a sliding window
    STACK              min/max over a range?
           |               |
    Is it "greater"?    ___|___
     |         |       |       |
    YES       NO      YES      NO
     |         |       |       |
    Use       Use    Use      Does it involve building
  DECREASING INCREASING  MONOTONIC  the optimal subsequence?
    stack     stack    DEQUE        |
                                ___|___
                               |       |
                              YES      NO
                               |       |
                         Use GREEDY   Consider other
                         MONOTONIC    approaches
                         STACK        (DP, two-pointer, etc.)
```

### Quick Recognition Signals

| Signal in Problem Statement | Likely Pattern |
|----------------------------|----------------|
| "next greater/warmer/taller" | Pattern 1: Next Greater Element |
| "span" or "how far back" | Pattern 2: Previous Greater/Smaller |
| "largest rectangle" or "maximal area" | Pattern 3: Histogram |
| "trap water/rain" | Pattern 4: Trapping Rain Water |
| "sliding window" + "maximum/minimum" | Pattern 5: Monotonic Deque |
| "remove digits/letters" + "smallest/largest result" | Pattern 6: Greedy Build |
| "consecutive days where..." | Pattern 2: Stock Span variant |
| "subarray with sum at least K" + "shortest" | Pattern 5: Prefix Sum + Deque |
| Contains "circular array" | Add double-pass to any pattern |

---

<a name="checklist"></a>
## 8. Interview Preparation Checklist

### Before the Interview

- [ ] Can you implement the 3 master templates from memory?
- [ ] Can you explain why a monotonic stack is O(n) despite having a while loop inside a for loop?
- [ ] Can you draw the stack state for a 5-element example of next greater element?
- [ ] Do you know when to use decreasing vs. increasing stack?
- [ ] Can you solve Largest Rectangle in Histogram without looking at notes?
- [ ] Can you explain the difference between "next" and "previous" in stack terms?
- [ ] Can you implement sliding window maximum with a deque?

### During the Interview

1. **Recognize the pattern** (1-2 minutes):
   - Does the problem ask about next/previous greater/smaller?
   - Is there a sliding window min/max?
   - Is it about building an optimal subsequence?

2. **State the approach** (1-2 minutes):
   - "I'll use a monotonic [decreasing/increasing] stack because..."
   - "The key insight is that each element enters and leaves the stack exactly once, giving O(n)."

3. **Implement with template** (5-8 minutes):
   - Start with the appropriate template
   - Adapt the comparison operator and result recording
   - Handle edge cases (empty array, single element, all same values)

4. **Trace through an example** (2-3 minutes):
   - Walk through at least 3-4 iterations showing stack state
   - Verify the result matches expected output

5. **State complexity** (30 seconds):
   - Time: O(n) -- each element pushed and popped at most once
   - Space: O(n) -- for the stack and result array

### Common Follow-Up Questions

| Question | Answer |
|----------|--------|
| "Why not just sort?" | Sorting loses positional information (next/previous relationships) |
| "Can you do it with two pointers?" | Trapping Rain Water yes, but most NGE problems need a stack |
| "What if the array is circular?" | Double the iteration range, use `i % n` |
| "Can you do it in-place?" | Daily Temperatures can use backward iteration, but stack is cleaner |
| "What about duplicates?" | Depends on problem -- adjust `>` vs `>=` carefully |

---

<a name="quick-ref"></a>
## 9. Quick Reference Cards

### Card 1: Next Greater Element (Template)
```
┌─────────────────────────────────────────────────┐
│  NEXT GREATER ELEMENT                           │
│                                                 │
│  Stack type: DECREASING (bottom > top)          │
│  Pop when:   nums[i] > nums[stack[-1]]          │
│  Result:     result[popped_idx] = nums[i]       │
│  Default:    -1 (never popped = no NGE)         │
│                                                 │
│  for i in range(n):                             │
│      while stack and nums[i] > nums[stack[-1]]: │
│          result[stack.pop()] = nums[i]          │
│      stack.append(i)                            │
└─────────────────────────────────────────────────┘
```

### Card 2: Next Smaller Element (Template)
```
┌─────────────────────────────────────────────────┐
│  NEXT SMALLER ELEMENT                           │
│                                                 │
│  Stack type: INCREASING (bottom < top)          │
│  Pop when:   nums[i] < nums[stack[-1]]          │
│  Result:     result[popped_idx] = nums[i]       │
│  Default:    -1 (never popped = no NSE)         │
│                                                 │
│  for i in range(n):                             │
│      while stack and nums[i] < nums[stack[-1]]: │
│          result[stack.pop()] = nums[i]          │
│      stack.append(i)                            │
└─────────────────────────────────────────────────┘
```

### Card 3: Sliding Window Maximum (Template)
```
┌─────────────────────────────────────────────────────┐
│  SLIDING WINDOW MAXIMUM (Deque)                     │
│                                                     │
│  Deque order: DECREASING (front > back)             │
│  Front removal:  dq[0] <= i - k (outside window)    │
│  Back removal:   nums[i] >= nums[dq[-1]]            │
│  Window max:     nums[dq[0]]                        │
│                                                     │
│  for i in range(n):                                 │
│      while dq and dq[0] <= i - k: dq.popleft()     │
│      while dq and nums[i] >= nums[dq[-1]]: dq.pop()│
│      dq.append(i)                                   │
│      if i >= k-1: result.append(nums[dq[0]])        │
└─────────────────────────────────────────────────────┘
```

### Card 4: Largest Rectangle in Histogram
```
┌──────────────────────────────────────────────────────┐
│  LARGEST RECTANGLE IN HISTOGRAM                      │
│                                                      │
│  Stack type: INCREASING (bottom < top)               │
│  Pop when:   heights[i] < heights[stack[-1]]         │
│  On pop:     h = heights[popped]                     │
│              w = i if stack empty else i-stack[-1]-1  │
│              area = h * w                            │
│  Sentinel:   Append 0 to heights to flush stack      │
│                                                      │
│  heights += [0]                                      │
│  for i in range(len(heights)):                       │
│      while stack and heights[i] < heights[stack[-1]]:│
│          h = heights[stack.pop()]                    │
│          w = i if not stack else i - stack[-1] - 1   │
│          max_area = max(max_area, h * w)             │
│      stack.append(i)                                 │
└──────────────────────────────────────────────────────┘
```

### Card 5: Greedy Removal (Remove K Digits)
```
┌─────────────────────────────────────────────────┐
│  GREEDY REMOVAL (Monotonic Increasing Stack)    │
│                                                 │
│  Goal:    Build smallest number by removing k   │
│  Pop:     stack[-1] > current AND k > 0         │
│  Finish:  If k > 0, trim from end               │
│  Cleanup: lstrip('0'), handle empty → '0'       │
│                                                 │
│  for digit in num:                              │
│      while k and stack and stack[-1] > digit:   │
│          stack.pop(); k -= 1                    │
│      stack.append(digit)                        │
│  if k: stack = stack[:-k]                       │
│  return ''.join(stack).lstrip('0') or '0'       │
└─────────────────────────────────────────────────┘
```

---

<a name="complexity-ref"></a>
## 10. Complexity Reference

| Problem | Time | Space | Key Insight |
|---------|------|-------|-------------|
| Next Greater Element I (LC 496) | O(n + m) | O(m) | Hash map for lookups |
| Next Greater Element II (LC 503) | O(n) | O(n) | Double pass for circular |
| Daily Temperatures (LC 739) | O(n) | O(n) | Index difference = days waited |
| Online Stock Span (LC 901) | O(1) amortized | O(n) | Previous greater element |
| Number of Valid Subarrays (LC 1063) | O(n) | O(n) | Next smaller element bounds count |
| Largest Rectangle (LC 84) | O(n) | O(n) | Sentinel + width calculation |
| Maximal Rectangle (LC 85) | O(rows * cols) | O(cols) | Row-by-row histogram |
| Trapping Rain Water (LC 42) | O(n) | O(n) | Valley between walls |
| Sliding Window Maximum (LC 239) | O(n) | O(k) | Deque front = window max |
| Shortest Subarray Sum >= K (LC 862) | O(n) | O(n) | Prefix sum + monotonic deque |
| Remove K Digits (LC 402) | O(n) | O(n) | Greedy increasing stack |
| Remove Duplicate Letters (LC 316) | O(n) | O(1) | Remaining count check |
| Create Maximum Number (LC 321) | O(k*(m+n+k)) | O(m+n+k) | Split + merge strategy |

### Amortized O(n) Explanation

The key question interviewers ask: "There's a while loop inside a for loop -- isn't this O(n^2)?"

**No.** Here's the proof:
- Each element is pushed onto the stack **at most once** (during its iteration in the for loop).
- Each element is popped from the stack **at most once** (during some future iteration's while loop).
- Total pushes across entire loop: n
- Total pops across entire loop: at most n
- Total operations: push n + pop n = 2n = O(n)

Think of it this way: the while loop doesn't do n work each time. It does variable work, but the **total work across all iterations** is bounded by n pops.

---

## Final Thoughts

### The Three Things to Remember

1. **Monotonic stack = efficient next/previous greater/smaller finder.** Every time you see a problem asking "for each element, find the nearest element satisfying some comparison," think monotonic stack.

2. **Each element enters and leaves the stack exactly once.** This is why the time complexity is O(n) despite the nested loop. In an interview, state this clearly to show you understand the amortized analysis.

3. **The comparison operator determines the stack type.** Want to find "greater"? Use a decreasing stack (so larger elements trigger pops). Want to find "smaller"? Use an increasing stack. Getting the direction right is the most common mistake.

### When to Move Beyond Monotonic Stack

Monotonic stack handles one-dimensional relationships beautifully. But for:
- **2D problems**: Often reduce to 1D (like Maximal Rectangle reduces rows to histograms)
- **Multiple queries on ranges**: Consider segment trees or sparse tables
- **Dynamic data with insertions/deletions**: Consider balanced BSTs
- **K-th largest/smallest**: Use heaps instead

### The Monotonic Stack Family Tree

```
Monotonic Stack Family
├── Monotonic Decreasing Stack (pop when current > top)
│   ├── Next Greater Element (pop resolves NGE)
│   ├── Previous Greater Element (top before push is PGE)
│   ├── Stock Span (previous greater variant)
│   └── Trapping Rain Water (pop valleys when taller wall found)
├── Monotonic Increasing Stack (pop when current < top)
│   ├── Next Smaller Element (pop resolves NSE)
│   ├── Previous Smaller Element (top before push is PSE)
│   └── Largest Rectangle (both NSE boundaries)
├── Monotonic Deque
│   ├── Sliding Window Maximum (decreasing deque)
│   ├── Sliding Window Minimum (increasing deque)
│   └── Shortest Subarray Sum >= K (prefix sum + deque)
└── Greedy Monotonic Stack
    ├── Remove K Digits (build smallest)
    ├── Remove Duplicate Letters (smallest with constraints)
    └── Create Maximum Number (build largest + merge)
```

### Final Advice

In interviews, monotonic stack problems can feel intimidating because the logic is subtle. The best strategy:

1. **Identify the pattern** using the recognition framework above.
2. **Write the template** from memory.
3. **Adapt the comparison** and result-recording logic.
4. **Trace through a small example** to verify.
5. **State the O(n) complexity** with the push/pop amortization argument.

If you can do these five steps smoothly, you'll handle any monotonic stack problem thrown at you.

---

## Appendix: Practice Problem Set

### Tier 1: Foundation (Start Here)

| # | Problem | Difficulty | Pattern | Key Concept |
|---|---------|------------|---------|-------------|
| 1 | LC 496 - Next Greater Element I | Easy | Next Greater | Hash map + basic stack |
| 2 | LC 739 - Daily Temperatures | Medium | Next Greater | Index distance |
| 3 | LC 901 - Online Stock Span | Medium | Previous Greater | Span calculation |
| 4 | LC 402 - Remove K Digits | Medium | Greedy Build | Monotonic increasing |

### Tier 2: Core Competency

| # | Problem | Difficulty | Pattern | Key Concept |
|---|---------|------------|---------|-------------|
| 5 | LC 503 - Next Greater Element II | Medium | Next Greater | Circular array |
| 6 | LC 84 - Largest Rectangle in Histogram | Hard | Histogram | Sentinel + width calc |
| 7 | LC 42 - Trapping Rain Water | Hard | Water Trapping | Valley computation |
| 8 | LC 239 - Sliding Window Maximum | Hard | Monotonic Deque | Window maintenance |
| 9 | LC 316 - Remove Duplicate Letters | Medium | Greedy Build | Constraint management |

### Tier 3: Advanced Mastery

| # | Problem | Difficulty | Pattern | Key Concept |
|---|---------|------------|---------|-------------|
| 10 | LC 85 - Maximal Rectangle | Hard | Histogram | 2D to 1D reduction |
| 11 | LC 862 - Shortest Subarray with Sum >= K | Hard | Monotonic Deque | Prefix sum + deque |
| 12 | LC 321 - Create Maximum Number | Hard | Greedy Build | Split + merge |
| 13 | LC 1063 - Number of Valid Subarrays | Hard | Previous Smaller | Count from boundaries |

### Tier 4: Extended Practice

| # | Problem | Difficulty | Pattern | Key Concept |
|---|---------|------------|---------|-------------|
| 14 | LC 456 - 132 Pattern | Medium | Stack Variant | Track max popped |
| 15 | LC 907 - Sum of Subarray Minimums | Medium | Prev/Next Smaller | Contribution technique |
| 16 | LC 1475 - Final Prices With Discount | Easy | Next Smaller | Direct application |
| 17 | LC 768 - Max Chunks To Make Sorted II | Hard | Stack Variant | Merge on pop |
| 18 | LC 1856 - Maximum Subarray Min-Product | Medium | Histogram Variant | Min * sum of subarray |

### Recommended Study Order

**Week 1:** Problems 1-4 (Foundation)
- Focus on understanding the template and tracing through examples manually.

**Week 2:** Problems 5-9 (Core)
- Focus on recognizing patterns and adapting templates.

**Week 3:** Problems 10-13 (Advanced)
- Focus on combining monotonic stack with other techniques.

**Week 4:** Problems 14-18 (Extended)
- Focus on speed and handling novel variations.

**Daily Practice Routine:**
1. Pick one problem from your current tier.
2. Spend 5 minutes identifying the pattern before coding.
3. Implement using the template, then trace through an example.
4. If stuck for more than 15 minutes, review the pattern section above.
5. After solving, write down the key insight in one sentence.
