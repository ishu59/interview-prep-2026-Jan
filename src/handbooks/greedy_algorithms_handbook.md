# The Complete Greedy Algorithms Handbook
> A template-based approach for mastering greedy algorithms in coding interviews

**Philosophy:** Greedy algorithms are not about being short-sighted. They're about proving that the locally optimal choice at each step leads to the globally optimal solution — and knowing when that proof breaks down.

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
9. [Quick Reference Cards](#quick-reference)
10. [Complexity Reference](#complexity-reference)

---

<a name="core-philosophy"></a>
## 1. Understanding the Core Philosophy

### First Principles

- **The Coin Cashier**: When making change, you always pick the largest coin that fits. It works for US coins because of their denominations — but it fails for arbitrary coin sets. Greedy works only when the "biggest first" strategy is provably optimal.
- **The Buffet Strategy**: At an all-you-can-eat buffet with a small plate, you pick the highest-value items first. Each choice is independent and locally optimal. This is the fractional knapsack — and it works because you can take fractions.

### No-Jargon Translation

- **Greedy choice**: picking the best-looking option right now without thinking about the future
- **Greedy choice property**: proof that always picking locally best leads to globally best
- **Exchange argument**: proof technique — if you swap any choice for the greedy choice, the answer doesn't get worse
- **Optimal substructure**: after making one greedy choice, the remaining problem has the same structure

### Mental Model

> "A greedy algorithm is like a hiker who always takes the steepest path upward — it reaches the summit only when there's a single peak, but gets stuck on a local hill when there are multiple peaks."

---

### When Does Greedy Work?

A greedy algorithm requires two properties:

**1. Greedy Choice Property:** The locally optimal choice leads to a globally optimal solution.

**2. Optimal Substructure:** An optimal solution to the problem contains optimal solutions to its subproblems.

### Key Insight: Greedy vs DP

```
Greedy:  Make a choice → Solve the remaining subproblem
DP:      Solve all subproblems → Combine to make optimal choice

Greedy looks FORWARD (commit and never look back)
DP looks BACKWARD (consider all possibilities)
```

### When Greedy Fails

```
Coin Change: coins = [1, 3, 4], amount = 6
  Greedy: 4 + 1 + 1 = 3 coins       (WRONG)
  Optimal: 3 + 3   = 2 coins         (DP needed)

0/1 Knapsack: items = [(60, 10), (100, 20), (120, 30)], capacity = 50
  Greedy by value/weight: item1 + item2 = 160  (WRONG)
  Optimal: item2 + item3 = 220                 (DP needed)
```

### The Exchange Argument (Proof Technique)

The most powerful way to prove greedy correctness:

```
1. Assume an optimal solution O that differs from greedy solution G
2. Find the first place where O and G differ
3. Show you can "exchange" O's choice for G's choice
   without making the solution worse
4. Repeat until O = G, proving G is also optimal
```

### Visual Understanding

```
Greedy Works (single peak):       Greedy Fails (multiple peaks):

     *                                 *           *
    * *                               * *         * *
   *   *                             *   *       *   *
  *     *                           *     *     *     *
 *       *                         *       *   *       *
*         *                       *         * *         *
──────────────                   ─────────────────────────
  ^ always go up                  ^ gets stuck on local peak
    reaches top!                    misses global peak!
```

---

<a name="master-templates"></a>
## 2. The 4 Master Templates

### Template 1: Sort-Then-Iterate

**Use for:** Interval scheduling, activity selection, most greedy problems that need ordering.

```python
def sort_then_iterate(items):
    """
    Sort by the greedy criterion, then iterate making local choices.
    Works for: intervals, scheduling, assignment problems.
    """
    # Step 1: Sort by the right criterion
    # - By end time for "max non-overlapping" problems
    # - By start time for merging problems
    # - By some ratio or value for optimization problems
    items.sort(key=lambda x: greedy_criterion(x))

    result = 0  # or [] or other accumulator
    current_state = initial_state  # e.g., last_end = -inf

    # Step 2: Iterate and greedily decide
    for item in items:
        if can_include(item, current_state):
            # Include this item
            result = update_result(result, item)
            current_state = update_state(current_state, item)

    return result
```

### Template 2: Two-Pass Greedy (Left-Right Sweep)

**Use for:** Problems where constraints come from both directions (Candy, Trapping Rain Water).

```python
def two_pass_greedy(nums):
    """
    First pass left-to-right enforces one constraint.
    Second pass right-to-left enforces the other.
    Final answer combines both passes.
    """
    n = len(nums)

    # Pass 1: Left to right
    left = [1] * n  # or appropriate initial values
    for i in range(1, n):
        if left_condition(nums, i):
            left[i] = left[i - 1] + 1  # or some update

    # Pass 2: Right to left
    right = [1] * n
    for i in range(n - 2, -1, -1):
        if right_condition(nums, i):
            right[i] = right[i + 1] + 1  # or some update

    # Combine: take the stricter constraint at each position
    return combine(left, right)  # e.g., sum(max(l, r) for l, r in zip(left, right))
```

### Template 3: Priority-Queue Greedy

**Use for:** Problems where you always need the min/max available element (Huffman, scheduling with deadlines).

```python
import heapq

def priority_queue_greedy(items):
    """
    Use a heap to always pick the optimal next choice.
    Works for: Huffman coding, optimal merging, scheduling.
    """
    # Step 1: Initialize heap
    heap = []
    for item in items:
        heapq.heappush(heap, priority_key(item))

    result = 0  # or other accumulator

    # Step 2: Repeatedly pick the best available
    while len(heap) > 1:  # or while heap, depending on problem
        best = heapq.heappop(heap)

        # Process best choice
        result = update_result(result, best)

        # Possibly push modified item back
        new_item = transform(best)
        if should_reinsert(new_item):
            heapq.heappush(heap, new_item)

    return result
```

### Template 4: Exchange Argument Proof Template

**Use for:** Proving your greedy approach is correct during an interview.

```python
"""
Exchange Argument Proof Template (verbal, not code):

1. DEFINE the greedy strategy:
   "At each step, we choose [specific criterion]."

2. ASSUME a better solution exists:
   "Suppose optimal solution O differs from greedy solution G."

3. FIND the first difference:
   "Let position i be the first place O and G differ."

4. EXCHANGE O's choice for G's choice:
   "If we swap O[i] with G[i], the solution doesn't get worse because..."

5. CONCLUDE by induction:
   "Repeating this exchange transforms O into G without
    worsening the result, so G is also optimal."

Example for Activity Selection:
1. Greedy: always pick the activity that ends earliest.
2. Suppose optimal O picks activity a where G picks activity b (b ends earlier).
3. Since b ends no later than a, replacing a with b in O
   still leaves room for all subsequent activities in O.
4. So the modified O has the same number of activities.
5. Therefore greedy is optimal.
"""
```

### Decision Matrix: Which Template?

| Problem Characteristics | Template | Example |
|------------------------|----------|---------|
| Need to process items in sorted order | Sort-Then-Iterate | Interval scheduling, activity selection |
| Constraints from both left and right | Two-Pass Greedy | Candy, queue reconstruction |
| Need to repeatedly pick min/max | Priority-Queue | Connect sticks, Huffman coding |
| Need to prove correctness verbally | Exchange Argument | Any greedy interview question |
| Jump/reach from current position | Sort-Then-Iterate (variant) | Jump Game, Video Stitching |
| Remove/select digits or characters | Stack-based Greedy | Remove K Digits, Remove Duplicate Letters |

---

<a name="pattern-guide"></a>
## 3. Pattern Classification Guide

### Category 1: Interval Scheduling / Activity Selection
Sort by end time, greedily pick non-overlapping intervals.
- LC 435: Non-overlapping Intervals
- LC 452: Minimum Number of Arrows to Burst Balloons
- LC 56: Merge Intervals

### Category 2: Jump / Reachability
Track the farthest reachable position, decide greedily.
- LC 55: Jump Game
- LC 45: Jump Game II
- LC 1024: Video Stitching

### Category 3: Task Scheduling
Use frequency counting and greedy placement or heap-based selection.
- LC 621: Task Scheduler
- LC 767: Reorganize String

### Category 4: Huffman-style / Optimal Merge
Always merge the two smallest; use a min-heap.
- LC 1167: Minimum Cost to Connect Sticks

### Category 5: Circular / Gas Station
Track running surplus; reset start when deficit occurs.
- LC 134: Gas Station

### Category 6: Two-Pass Greedy
Sweep left-to-right then right-to-left, combine results.
- LC 135: Candy
- LC 406: Queue Reconstruction by Height

### Category 7: Digit / Character Manipulation (Stack-based Greedy)
Use a monotonic stack to greedily build the optimal sequence.
- LC 402: Remove K Digits
- LC 316: Remove Duplicate Letters
- LC 321: Create Maximum Number

---

<a name="patterns"></a>
## 4. Complete Pattern Library

---

### PATTERN 1: Interval Scheduling / Activity Selection

---

#### Pattern 1A: Non-overlapping Intervals

**Problem:** LeetCode 435 - Given an array of intervals, return the minimum number of intervals you need to remove to make the rest non-overlapping.

**Example:**
```
Input:  intervals = [[1,2],[2,3],[3,4],[1,3]]
Output: 1
Explanation: Remove [1,3] and the rest are non-overlapping.
```

**Key Insight:** This is the classic activity selection problem in disguise. Sort by end time. Greedily keep intervals that don't overlap with the last kept interval. The number to remove = total - number kept. Why greedy works: picking the interval that ends earliest leaves the most room for future intervals (exchange argument).

**Visual Trace:**
```
intervals = [[1,2],[2,3],[3,4],[1,3]]

After sorting by end time: [[1,2],[2,3],[1,3],[3,4]]

Step 1: Pick [1,2], last_end = 2, kept = 1
        |--|
        1  2  3  4

Step 2: [2,3] starts at 2 >= last_end 2 -> Pick it, last_end = 3, kept = 2
        |--|
           |--|
        1  2  3  4

Step 3: [1,3] starts at 1 < last_end 3 -> Skip (overlaps!)
        |--|
           |--|
        |-----|    <- skip this
        1  2  3  4

Step 4: [3,4] starts at 3 >= last_end 3 -> Pick it, last_end = 4, kept = 3
        |--|
           |--|
              |--|
        1  2  3  4

Answer: 4 total - 3 kept = 1 removal
```

**Solution:**
```python
def eraseOverlapIntervals(intervals: list[list[int]]) -> int:
    if not intervals:
        return 0

    # Why sort by END time (not start time)?
    # Picking the interval that ENDS earliest leaves the most room
    # for future intervals — greedy choice that's provably optimal.
    intervals.sort(key=lambda x: x[1])

    kept = 1
    last_end = intervals[0][1]

    for i in range(1, len(intervals)):
        # Why >= (not just >)? Two intervals like [1,2] and [2,3] share
        # endpoint 2, but they do NOT overlap — one ends as the other starts.
        # Using > would incorrectly skip valid non-overlapping pairs.
        if intervals[i][0] >= last_end:
            # No overlap, keep this interval
            kept += 1
            last_end = intervals[i][1]
        # else: overlaps, skip (remove) this interval

    return len(intervals) - kept
```

**Complexity:** Time O(n log n) for sorting, Space O(1) extra (O(n) for sort).

---

#### Pattern 1B: Minimum Number of Arrows to Burst Balloons

**Problem:** LeetCode 452 - Balloons are represented as intervals on the x-axis. An arrow shot at x bursts all balloons where x_start <= x <= x_end. Find the minimum number of arrows to burst all balloons.

**Example:**
```
Input:  points = [[10,16],[2,8],[1,6],[7,12]]
Output: 2
Explanation: Shoot at x=6 (bursts [2,8] and [1,6]) and x=11 (bursts [10,16] and [7,12]).
```

**Key Insight:** This is equivalent to finding the maximum number of overlapping groups. Sort by end time. An arrow placed at the end of the earliest-ending balloon bursts all balloons that overlap with it. Start a new arrow only when a balloon starts after the current arrow's position.

**Visual Trace:**
```
points = [[10,16],[2,8],[1,6],[7,12]]

After sorting by end: [[1,6],[2,8],[7,12],[10,16]]

Arrow 1 at x=6:
  [1,--------6]       <- burst (contains 6)
     [2,--------8]    <- burst (contains 6)
  arrows = 1, arrow_pos = 6

  [7,------12]        <- 7 > 6, need new arrow

Arrow 2 at x=12:
  [7,------12]        <- burst (contains 12)
      [10,----16]     <- burst (contains 12)
  arrows = 2, arrow_pos = 12

Answer: 2 arrows
```

**Solution:**
```python
def findMinArrowShots(points: list[list[int]]) -> int:
    if not points:
        return 0

    # Why sort by END position? Same reasoning as activity selection:
    # placing an arrow at the earliest-ending balloon's end ensures
    # it bursts as many overlapping balloons as possible.
    points.sort(key=lambda x: x[1])

    arrows = 1
    # Place the first arrow at the end of the first balloon —
    # this is the rightmost point that still bursts it.
    arrow_pos = points[0][1]

    for i in range(1, len(points)):
        # Why > (strict) instead of >=?
        # Unlike non-overlapping intervals, here touching counts as overlap:
        # an arrow at x bursts balloon [x_start, x_end] if x_start <= x <= x_end.
        # So if points[i][0] == arrow_pos, the current arrow still bursts it.
        # We only need a new arrow when the balloon starts AFTER the arrow position.
        if points[i][0] > arrow_pos:
            # This balloon is not burst by current arrow
            arrows += 1
            arrow_pos = points[i][1]

    return arrows
```

**Complexity:** Time O(n log n), Space O(1) extra.

---

#### Pattern 1C: Merge Intervals

**Problem:** LeetCode 56 - Given an array of intervals, merge all overlapping intervals.

**Example:**
```
Input:  intervals = [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
```

**Key Insight:** Sort by start time. Then greedily extend the current interval: if the next interval overlaps (its start <= current end), merge by extending the end. Otherwise, start a new merged interval. This is greedy because we commit to merging as aggressively as possible.

**Visual Trace:**
```
intervals = [[1,3],[2,6],[8,10],[15,18]]

After sorting by start: [[1,3],[2,6],[8,10],[15,18]]

Step 1: Start with [1,3]
        merged = [[1,3]]

Step 2: [2,6] -> 2 <= 3 (overlaps!) -> extend to [1,6]
        merged = [[1,6]]

Step 3: [8,10] -> 8 > 6 (no overlap) -> new interval
        merged = [[1,6],[8,10]]

Step 4: [15,18] -> 15 > 10 (no overlap) -> new interval
        merged = [[1,6],[8,10],[15,18]]
```

**Solution:**
```python
def merge(intervals: list[list[int]]) -> list[list[int]]:
    if not intervals:
        return []

    # Why sort by START time (not end time)?
    # For merging, we need to process intervals in the order they begin
    # so we can detect and extend overlaps as a contiguous sweep.
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]

    for i in range(1, len(intervals)):
        # Why <= (not <)? Intervals [1,3] and [3,5] touch at point 3.
        # They should merge into [1,5] because they share boundary 3.
        # Using < would leave them separate, which is wrong for most
        # "merge intervals" problem definitions.
        if intervals[i][0] <= merged[-1][1]:
            # Overlaps: extend the end.
            # Why max() instead of just assigning intervals[i][1]?
            # The new interval might be entirely contained within the last one.
            # e.g., merged[-1]=[1,10], intervals[i]=[2,5] -> end stays 10.
            merged[-1][1] = max(merged[-1][1], intervals[i][1])
        else:
            # No overlap: start new interval
            merged.append(intervals[i])

    return merged
```

**Complexity:** Time O(n log n), Space O(n) for output.

---

### PATTERN 2: Jump Game Variants

---

#### Pattern 2A: Jump Game

**Problem:** LeetCode 55 - Given an array where each element represents the max jump length from that position, determine if you can reach the last index.

**Example:**
```
Input:  nums = [2,3,1,1,4]
Output: True
Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.

Input:  nums = [3,2,1,0,4]
Output: False
Explanation: You always arrive at index 3, where the value is 0.
```

**Key Insight:** Track the farthest index reachable so far. At each position, update the farthest reach. If at any point the current index exceeds the farthest reach, you are stuck. Greedy works because if you can reach position i, you can also reach all positions before i.

**Visual Trace:**
```
nums = [2, 3, 1, 1, 4]
index:  0  1  2  3  4

i=0: farthest = max(0, 0+2) = 2
     Can reach indices 0,1,2

i=1: 1 <= 2 (reachable), farthest = max(2, 1+3) = 4
     Can reach indices 0,1,2,3,4
     4 >= 4 (last index) -> return True!

nums = [3, 2, 1, 0, 4]
index:  0  1  2  3  4

i=0: farthest = max(0, 0+3) = 3
i=1: farthest = max(3, 1+2) = 3
i=2: farthest = max(3, 2+1) = 3
i=3: farthest = max(3, 3+0) = 3
     3 < 4 (can't reach last index) -> return False
```

**Solution:**
```python
def canJump(nums: list[int]) -> bool:
    farthest = 0

    for i in range(len(nums)):
        # Why "i > farthest" means unreachable:
        # farthest tracks the maximum index reachable from any index [0..i-1].
        # If the current index i exceeds that, there is a gap — no earlier
        # position could jump far enough to reach i, so we are stuck.
        if i > farthest:
            return False
        # Why max() instead of just assigning i + nums[i]?
        # A previous position might already reach farther than i + nums[i].
        # We must keep the best reach seen so far, not just the current one.
        farthest = max(farthest, i + nums[i])
        # Why >= (not >)? The last index is len(nums)-1. If farthest
        # equals it exactly, we can land on it — reaching it is enough.
        if farthest >= len(nums) - 1:
            return True

    return True
```

**Complexity:** Time O(n), Space O(1).

---

#### Pattern 2B: Jump Game II

**Problem:** LeetCode 45 - Given an array where each element is the max jump length, return the minimum number of jumps to reach the last index. You can always reach the last index.

**Example:**
```
Input:  nums = [2,3,1,1,4]
Output: 2
Explanation: Jump 1 to index 1, then 3 to index 4.
```

**Key Insight:** Think of it as BFS in levels. Each "level" is the set of positions reachable with k jumps. Use `current_end` to track the boundary of the current level and `farthest` to track how far the next level reaches. When you hit `current_end`, you must take another jump.

**Visual Trace:**
```
nums = [2, 3, 1, 1, 4]
index:  0  1  2  3  4

jumps=0, current_end=0, farthest=0

i=0: farthest = max(0, 0+2) = 2
     i == current_end (0 == 0):
       jumps = 1, current_end = 2
     Level 1 can reach indices {1, 2}

i=1: farthest = max(2, 1+3) = 4
     4 >= 4 -> reached end!
     i != current_end, continue

i=2: farthest = max(4, 2+1) = 4
     i == current_end (2 == 2):
       jumps = 2, current_end = 4
       4 >= 4 -> break

Answer: 2 jumps
```

**Solution:**
```python
def jump(nums: list[int]) -> int:
    # Why <= 1? A single-element array means we are already at the last
    # index — no jumps needed. Also handles empty arrays safely.
    if len(nums) <= 1:
        return 0

    jumps = 0
    current_end = 0   # rightmost index reachable with 'jumps' jumps
    farthest = 0      # rightmost index reachable with 'jumps + 1' jumps

    # Why len(nums) - 1 (not len(nums))?
    # We never need to "jump from" the last index. If we reach it, we are done.
    # Including the last index would count an extra unnecessary jump.
    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])

        # Why "i == current_end" triggers a jump:
        # current_end is the boundary of the current BFS "level."
        # When i reaches this boundary, we have explored every position
        # reachable with the current number of jumps, so we MUST take
        # another jump to go farther. farthest becomes the new boundary.
        if i == current_end:
            jumps += 1
            current_end = farthest

            if current_end >= len(nums) - 1:
                break

    return jumps
```

**Complexity:** Time O(n), Space O(1).

---

#### Pattern 2C: Video Stitching

**Problem:** LeetCode 1024 - Given a collection of video clips `[start, end]`, return the minimum number of clips needed to cover the interval `[0, time]`, or -1 if impossible.

**Example:**
```
Input:  clips = [[0,2],[4,6],[8,10],[1,9],[1,5],[5,8]], time = 10
Output: 3
Explanation: [0,2] + [1,9] + [8,10] covers [0,10].
```

**Key Insight:** Sort clips by start time (break ties by longer end first). Greedily pick the clip that starts at or before the current coverage end and extends coverage the farthest. This is the interval covering problem — at each step, among all clips that overlap with your current coverage, pick the one that reaches farthest.

**Visual Trace:**
```
clips = [[0,2],[4,6],[8,10],[1,9],[1,5],[5,8]], time = 10

Sorted: [[0,2],[1,5],[1,9],[4,6],[5,8],[8,10]]

coverage = 0, clips_used = 0

Round 1: current coverage ends at 0
  Candidates (start <= 0): [0,2]
  Best extends to: 2
  coverage = 2, clips_used = 1

Round 2: current coverage ends at 2
  Candidates (start <= 2): [1,5], [1,9]
  Best extends to: 9
  coverage = 9, clips_used = 2

Round 3: current coverage ends at 9
  Candidates (start <= 9): [4,6], [5,8], [8,10]
  Best extends to: 10
  coverage = 10, clips_used = 3

10 >= time -> return 3
```

**Solution:**
```python
def videoStitching(clips: list[list[int]], time: int) -> int:
    # Why sort by start time, then by end time descending?
    # Start time first ensures we consider clips in order of where they begin.
    # Descending end time is a minor optimization: among clips with the same
    # start, the longest one comes first and will be picked by the max() below.
    clips.sort(key=lambda x: (x[0], -x[1]))

    count = 0
    coverage = 0
    i = 0

    # Why "coverage < time"? We need to cover [0, time]. Once coverage
    # reaches or exceeds 'time', the entire interval is covered.
    while coverage < time:
        farthest = coverage

        # Why "clips[i][0] <= coverage"?
        # A clip can extend our coverage only if it starts at or before
        # our current coverage endpoint — otherwise there is a gap.
        # Among all such overlapping clips, we greedily pick the one
        # that reaches farthest (maximizes coverage extension).
        while i < len(clips) and clips[i][0] <= coverage:
            farthest = max(farthest, clips[i][1])
            i += 1

        # Why "farthest == coverage" means impossible:
        # If no clip could extend coverage beyond its current value,
        # there is an uncoverable gap — no future clip can help either,
        # since they all start after coverage.
        if farthest == coverage:
            return -1

        coverage = farthest
        count += 1

    return count
```

**Complexity:** Time O(n log n), Space O(1) extra.

---

### PATTERN 3: Task Scheduling

---

#### Pattern 3A: Task Scheduler

**Problem:** LeetCode 621 - Given tasks with a cooldown period n between same tasks, return the minimum time to finish all tasks.

**Example:**
```
Input:  tasks = ["A","A","A","B","B","B"], n = 2
Output: 8
Explanation: A -> B -> idle -> A -> B -> idle -> A -> B
```

**Key Insight:** The most frequent task dictates the schedule. Arrange the most frequent task first with gaps of size n between them. Other tasks fill the gaps. The formula is: `(max_freq - 1) * (n + 1) + count_of_max_freq_tasks`. But if the total number of tasks exceeds this (gaps are all filled and more), the answer is simply `len(tasks)`.

**Visual Trace:**
```
tasks = ["A","A","A","B","B","B"], n = 2
Frequencies: A=3, B=3

max_freq = 3, tasks with max_freq = 2 (A and B)

Build the frame with most frequent task A:
  A _ _ | A _ _ | A
  (max_freq-1) groups of size (n+1), plus final group

Fill in B:
  A B _ | A B _ | A B

Formula: (3-1) * (2+1) + 2 = 2*3 + 2 = 8

Check: max(8, len(tasks)) = max(8, 6) = 8

Schedule: A B idle A B idle A B = 8 time units
```

**Solution:**
```python
from collections import Counter

def leastInterval(tasks: list[str], n: int) -> int:
    count = Counter(tasks)
    max_freq = max(count.values())
    # How many tasks have the maximum frequency?
    # We need this because all max-frequency tasks occupy the final "partial" cycle.
    # Why "if freq == max_freq"? Only tasks matching the peak frequency
    # appear in every cycle including the last one — others fit in the gaps.
    max_count = sum(1 for freq in count.values() if freq == max_freq)

    # Why (max_freq - 1) * (n + 1) + max_count?
    # Visualize: the most frequent task creates (max_freq - 1) full cycles,
    # each of length (n + 1) to respect cooldown. E.g., A _ _ | A _ _ | A
    # has (3-1)=2 full cycles of size (2+1)=3, plus 1 final A.
    # max_count accounts for all tasks tied at max frequency in that last slot.
    result = (max_freq - 1) * (n + 1) + max_count

    # Why max(result, len(tasks))?
    # When there are many distinct tasks, all idle slots get filled and we
    # may even need more time than the formula predicts. In that case, the
    # answer is simply the total number of tasks (no idle time at all).
    return max(result, len(tasks))
```

**Complexity:** Time O(n), Space O(1) (at most 26 task types).

---

#### Pattern 3B: Reorganize String

**Problem:** LeetCode 767 - Given a string s, rearrange so no two adjacent characters are the same. Return "" if impossible.

**Example:**
```
Input:  s = "aab"
Output: "aba"

Input:  s = "aaab"
Output: ""
```

**Key Insight:** Impossible if any character has frequency > (len(s) + 1) / 2. Otherwise, greedily place the most frequent character, alternating with the second most frequent. Use a max-heap to always pick the most available character that differs from the previously placed one.

**Visual Trace:**
```
s = "aab"
Frequencies: a=2, b=1

Check: max_freq=2 <= (3+1)/2=2 -> possible

Heap: [(-2,'a'), (-1,'b')]

Step 1: Pop (-2,'a'), place 'a', prev = (-1,'a')
        result = "a", push nothing yet (prev_freq was 0)

Step 2: Pop (-1,'b'), place 'b', push prev (-1,'a') back
        result = "ab", prev = (0,'b')

Step 3: Pop (-1,'a'), place 'a', push prev (0,'b') -> no, freq=0
        result = "aba"

Heap empty -> done. Answer: "aba"
```

**Solution:**
```python
import heapq
from collections import Counter

def reorganizeString(s: str) -> str:
    count = Counter(s)

    # Check if possible
    max_freq = max(count.values())
    # Why (len(s) + 1) // 2?
    # In a string of length n, you can place at most ceil(n/2) copies of one
    # character in non-adjacent positions (every other slot: positions 0,2,4,...).
    # If any character exceeds this, no valid arrangement exists.
    if max_freq > (len(s) + 1) // 2:
        return ""

    # Max-heap of (-frequency, character).
    # Negative because Python's heapq is a min-heap; negating makes it a max-heap.
    heap = [(-freq, char) for char, freq in count.items()]
    heapq.heapify(heap)

    result = []
    prev_freq, prev_char = 0, ''

    while heap:
        # Always pop the most frequent available character.
        # This is greedy: placing the most frequent character first
        # reduces the risk of being stuck with adjacent duplicates later.
        freq, char = heapq.heappop(heap)
        result.append(char)

        # Why push the PREVIOUS character back now (not the current one)?
        # We delay re-inserting a character by one step to guarantee it is
        # not placed in two consecutive positions. The character we just
        # placed becomes "prev" and waits one round before being eligible again.
        if prev_freq < 0:
            heapq.heappush(heap, (prev_freq, prev_char))

        # Update previous (used one occurrence, so freq moves toward 0).
        # freq is negative, so +1 means "one fewer remaining."
        prev_freq = freq + 1
        prev_char = char

    return ''.join(result)
```

**Complexity:** Time O(n log k) where k is the number of distinct characters (at most 26, so effectively O(n)), Space O(k).

---

### PATTERN 4: Huffman-style / Optimal Merge

---

#### Pattern 4A: Minimum Cost to Connect Sticks

**Problem:** LeetCode 1167 - You have sticks of various lengths. Each time you connect two sticks, the cost is the sum of their lengths. Return the minimum total cost to connect all sticks into one.

**Example:**
```
Input:  sticks = [2, 4, 3]
Output: 14
Explanation:
  Connect 2+3=5, cost=5, sticks=[5,4]
  Connect 4+5=9, cost=9, sticks=[9]
  Total = 5+9 = 14
```

**Key Insight:** This is exactly the Huffman coding problem. Always merge the two smallest sticks first. Why? Sticks merged earlier contribute to more future costs (they get re-summed every time). So you want the smallest sticks to be merged first, accumulating cost less. This is provable via the exchange argument.

**Visual Trace:**
```
sticks = [2, 4, 3]
Heap: [2, 3, 4]

Step 1: Pop 2 and 3, merge -> 5, cost = 5
        Heap: [4, 5]

Step 2: Pop 4 and 5, merge -> 9, cost = 9
        Heap: [9]

Total cost = 5 + 9 = 14

Why not merge 4+3=7 first?
  Step 1: 3+4=7, cost=7, sticks=[2,7]
  Step 2: 2+7=9, cost=9, sticks=[9]
  Total = 7+9 = 16 > 14  (worse!)
```

**Solution:**
```python
import heapq

def connectSticks(sticks: list[int]) -> int:
    # Why <= 1? Zero or one stick means no connections needed.
    if len(sticks) <= 1:
        return 0

    heapq.heapify(sticks)
    total_cost = 0

    # Why "len(sticks) > 1"? We need at least two sticks to merge.
    # When only one stick remains, all sticks have been connected.
    while len(sticks) > 1:
        # Why always pop the two SMALLEST?
        # Each merged stick gets re-added and will contribute to future merge costs.
        # Merging the smallest first minimizes how many times large values
        # get re-summed — this is the Huffman coding insight.
        first = heapq.heappop(sticks)
        second = heapq.heappop(sticks)
        combined = first + second
        total_cost += combined
        heapq.heappush(sticks, combined)

    return total_cost
```

**Complexity:** Time O(n log n), Space O(n).

---

### PATTERN 5: Gas Station / Circular

---

#### Pattern 5A: Gas Station

**Problem:** LeetCode 134 - There are n gas stations in a circle. You start with an empty tank. `gas[i]` is fuel available at station i, `cost[i]` is fuel needed to go from station i to i+1. Return the starting station index if you can complete the circuit, or -1.

**Example:**
```
Input:  gas  = [1,2,3,4,5]
        cost = [3,4,5,1,2]
Output: 3
Explanation: Start at station 3, tank = 0+4-1=3, then 3+5-2=6, then 6+1-3=4,
             then 4+2-4=2, then 2+3-5=0. Complete!
```

**Key Insight:** Two key observations: (1) If total gas >= total cost, a solution must exist. (2) If starting from station `start`, you run out of gas at station `j`, then no station between `start` and `j` can be a valid starting point either (because you arrived at each of those with a non-negative tank, and still couldn't make it past `j`). So you reset and try starting from `j+1`.

**Visual Trace:**
```
gas  = [1, 2, 3, 4, 5]
cost = [3, 4, 5, 1, 2]
net  = [-2,-2,-2, 3, 3]  (gas[i] - cost[i])

total = -2-2-2+3+3 = 0 >= 0, so solution exists

Try start = 0:
  tank = 0 + (-2) = -2 < 0 -> fail, move start to 1

Try start = 1:
  tank = 0 + (-2) = -2 < 0 -> fail, move start to 2

Try start = 2:
  tank = 0 + (-2) = -2 < 0 -> fail, move start to 3

Try start = 3:
  tank = 0 + 3 = 3 >= 0 -> continue
  tank = 3 + 3 = 6 >= 0 -> continue (wraps around)
  Reached end of array with tank >= 0

Answer: 3
```

**Solution:**
```python
def canCompleteCircuit(gas: list[int], cost: list[int]) -> int:
    # Why check total gas < total cost?
    # This is the global feasibility test. If the total fuel available across
    # ALL stations is less than the total fuel needed, no starting point can
    # work — you will always run out somewhere on the circuit.
    if sum(gas) < sum(cost):
        return -1

    tank = 0
    start = 0

    for i in range(len(gas)):
        tank += gas[i] - cost[i]

        # Why "tank < 0" triggers a reset:
        # If the running tank goes negative, we cannot reach station i+1
        # from the current start. Moreover, no station between start and i
        # can work either — because we arrived at each of those with a
        # non-negative tank and still ended up negative at i.
        # Why reset tank to 0 (not carry the deficit)?
        # We are "restarting" from station i+1 with a fresh empty tank.
        # The global check above guarantees that the deficit we abandon
        # here will be compensated by surplus elsewhere on the circuit.
        if tank < 0:
            start = i + 1
            tank = 0

    return start
```

**Complexity:** Time O(n), Space O(1).

---

### PATTERN 6: Two-Pass Greedy

---

#### Pattern 6A: Candy

**Problem:** LeetCode 135 - Children stand in a line with ratings. Each child gets at least 1 candy. Children with a higher rating than a neighbor must get more candies than that neighbor. Return the minimum total candies.

**Example:**
```
Input:  ratings = [1, 0, 2]
Output: 5
Explanation: Candies = [2, 1, 2]. Child 0 (rating 1) > child 1 (rating 0),
             so child 0 gets more. Child 2 (rating 2) > child 1 (rating 0),
             so child 2 gets more.
```

**Key Insight:** One pass can't handle both left and right neighbors simultaneously. Two passes solve this: (1) Left-to-right: if ratings[i] > ratings[i-1], give more than left neighbor. (2) Right-to-left: if ratings[i] > ratings[i+1], give more than right neighbor. At each position, take the maximum of both constraints.

**Visual Trace:**
```
ratings = [1, 2, 87, 87, 87, 2, 1]

Pass 1 (left to right, enforce left neighbor rule):
  candy = [1, 1, 1, 1, 1, 1, 1]  (start with 1 each)
  i=1: 2 > 1 -> candy[1] = candy[0]+1 = 2
  i=2: 87 > 2 -> candy[2] = candy[1]+1 = 3
  i=3: 87 == 87 -> no change
  i=4: 87 == 87 -> no change
  i=5: 2 < 87 -> no change
  i=6: 1 < 2 -> no change
  candy = [1, 2, 3, 1, 1, 1, 1]

Pass 2 (right to left, enforce right neighbor rule):
  i=5: 2 > 1 -> candy[5] = max(candy[5], candy[6]+1) = max(1, 2) = 2
  i=4: 87 > 2 -> candy[4] = max(1, candy[5]+1) = max(1, 3) = 3
  i=3: 87 == 87 -> no change
  i=2: 87 == 87 -> no change
  i=1: 2 < 87 -> no change
  i=0: 1 < 2 -> no change
  candy = [1, 2, 3, 1, 3, 2, 1]

Total = 1+2+3+1+3+2+1 = 13
```

**Solution:**
```python
def candy(ratings: list[int]) -> int:
    n = len(ratings)
    candies = [1] * n

    # Left to right: enforce left neighbor constraint
    for i in range(1, n):
        if ratings[i] > ratings[i - 1]:
            candies[i] = candies[i - 1] + 1

    # Right to left: enforce right neighbor constraint
    for i in range(n - 2, -1, -1):
        if ratings[i] > ratings[i + 1]:
            candies[i] = max(candies[i], candies[i + 1] + 1)

    return sum(candies)
```

**Complexity:** Time O(n), Space O(n).

---

#### Pattern 6B: Queue Reconstruction by Height

**Problem:** LeetCode 406 - People described by [height, k] where k is the number of people in front with height >= this person's height. Reconstruct the queue.

**Example:**
```
Input:  people = [[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]]
Output: [[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]]
```

**Key Insight:** Sort by height descending, then by k ascending. Insert people one by one at index k. Taller people are placed first, so when placing a shorter person at index k, exactly k taller people are already in front. This works because shorter people don't affect the k-count of taller people already placed.

**Visual Trace:**
```
people = [[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]]

After sorting by (-height, k):
  [[7,0],[7,1],[6,1],[5,0],[5,2],[4,4]]

Insert step by step:
  [7,0] at index 0: [[7,0]]
  [7,1] at index 1: [[7,0],[7,1]]
  [6,1] at index 1: [[7,0],[6,1],[7,1]]
  [5,0] at index 0: [[5,0],[7,0],[6,1],[7,1]]
  [5,2] at index 2: [[5,0],[7,0],[5,2],[6,1],[7,1]]
  [4,4] at index 4: [[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]]

Verify [4,4]: people in front with h>=4: [5,0],[7,0],[5,2],[6,1] -> 4 people. Correct!
```

**Solution:**
```python
def reconstructQueue(people: list[list[int]]) -> list[list[int]]:
    # Sort: tallest first, then by k ascending
    people.sort(key=lambda x: (-x[0], x[1]))

    queue = []
    for person in people:
        # Insert at position k
        queue.insert(person[1], person)

    return queue
```

**Complexity:** Time O(n^2) due to list insertions, Space O(n).

---

### PATTERN 7: Digit Manipulation Greedy

---

#### Pattern 7A: Remove K Digits

**Problem:** LeetCode 402 - Given a non-negative integer represented as a string, remove k digits to make the number as small as possible. Return as string.

**Example:**
```
Input:  num = "1432219", k = 3
Output: "1219"
Explanation: Remove 4, 3, 2 to get 1219.
```

**Key Insight:** Use a monotonic increasing stack. Scan left to right. If the current digit is smaller than the stack top, pop the stack (this "removes" a larger digit from a more significant position). This is greedy: removing a larger digit from a higher position always produces a smaller number than removing a smaller digit from a lower position.

**Visual Trace:**
```
num = "1432219", k = 3

Process digit by digit:
  '1': stack = [1]
  '4': 4 > 1, push. stack = [1, 4]
  '3': 3 < 4, pop 4 (k=2). stack = [1, 3]
  '2': 2 < 3, pop 3 (k=1). stack = [1, 2]
  '2': 2 == 2, push. stack = [1, 2, 2]
  '1': 1 < 2, pop 2 (k=0). stack = [1, 2, 1]
  '9': k=0, just push. stack = [1, 2, 1, 9]

Result: "1219"

Why does this work?
  "1432219" -> remove '4' -> "132219"
  "132219"  -> remove '3' -> "12219"
  "12219"   -> remove second '2' -> "1219"
  Each removal picks the leftmost digit that is larger than its right neighbor.
```

**Solution:**
```python
def removeKdigits(num: str, k: int) -> str:
    stack = []

    for digit in num:
        # While stack top is larger and we still have removals
        while k > 0 and stack and stack[-1] > digit:
            stack.pop()
            k -= 1
        stack.append(digit)

    # If k > 0, remove from the end (digits are in non-decreasing order)
    if k > 0:
        stack = stack[:-k]

    # Remove leading zeros and handle empty result
    result = ''.join(stack).lstrip('0')
    return result if result else '0'
```

**Complexity:** Time O(n), Space O(n).

---

#### Pattern 7B: Remove Duplicate Letters

**Problem:** LeetCode 316 - Given a string, remove duplicate letters so every letter appears once. Return the smallest lexicographic result among all possible answers.

**Example:**
```
Input:  s = "bcabc"
Output: "abc"

Input:  s = "cbacdcbc"
Output: "acdb"
```

**Key Insight:** Use a monotonic stack with two extra constraints: (1) don't add a character already in the stack, (2) only pop the stack top if that character appears again later. This greedy approach builds the lexicographically smallest string by keeping smaller characters as early as possible, but only removing larger characters that still have future occurrences.

**Visual Trace:**
```
s = "cbacdcbc"

Count remaining: c=4, b=2, a=1, d=1
in_stack = set()

'c': stack=[], push 'c'. stack=['c'], in_stack={c}
     remaining: c=3, b=2, a=1, d=1

'b': 'b' < 'c' and c appears later (count=3), pop 'c'.
     stack=[], in_stack={}
     push 'b'. stack=['b'], in_stack={b}
     remaining: c=3, b=1, a=1, d=1

'a': 'a' < 'b' and b appears later (count=1), pop 'b'.
     stack=[], in_stack={}
     push 'a'. stack=['a'], in_stack={a}
     remaining: c=3, b=1, a=0, d=1

'c': 'c' > 'a', push. stack=['a','c'], in_stack={a,c}
     remaining: c=2, b=1, a=0, d=1

'd': 'd' > 'c', push. stack=['a','c','d'], in_stack={a,c,d}
     remaining: c=2, b=1, a=0, d=0

'c': 'c' already in stack, skip.
     remaining: c=1, b=1, a=0, d=0

'b': 'b' < 'd' but d has count=0 (no future occurrence), can't pop.
     push 'b'. stack=['a','c','d','b'], in_stack={a,c,d,b}
     remaining: c=1, b=0, a=0, d=0

'c': 'c' already in stack, skip.
     remaining: c=0, b=0, a=0, d=0

Result: "acdb"
```

**Solution:**
```python
from collections import Counter

def removeDuplicateLetters(s: str) -> str:
    remaining = Counter(s)
    in_stack = set()
    stack = []

    for char in s:
        remaining[char] -= 1

        if char in in_stack:
            continue

        # Pop larger characters that appear later
        while stack and char < stack[-1] and remaining[stack[-1]] > 0:
            in_stack.remove(stack[-1])
            stack.pop()

        stack.append(char)
        in_stack.add(char)

    return ''.join(stack)
```

**Complexity:** Time O(n), Space O(1) (at most 26 characters in stack).

---

#### Pattern 7C: Create Maximum Number

**Problem:** LeetCode 321 - Given two integer arrays of length m and n, create the maximum number of length k using digits from both arrays while preserving the relative order within each array.

**Example:**
```
Input:  nums1 = [3,4,6,5], nums2 = [9,1,2,5,8,3], k = 5
Output: [9,8,6,5,3]
```

**Key Insight:** Break into three subproblems: (1) Choose i digits from nums1 and k-i digits from nums2, for all valid i. (2) For each array, find the maximum subsequence of given length (use the "remove k digits" stack technique, but maximize instead). (3) Merge two subsequences to form the largest number. Try all splits and return the best.

**Visual Trace:**
```
nums1 = [3,4,6,5], nums2 = [9,1,2,5,8,3], k = 5

Try i=0: 0 from nums1, 5 from nums2
  max_subseq(nums2, 5) = [9,2,5,8,3]  -> merge -> [9,2,5,8,3]

Try i=1: 1 from nums1, 4 from nums2
  max_subseq(nums1, 1) = [6]
  max_subseq(nums2, 4) = [9,5,8,3]
  merge([6],[9,5,8,3]) = [9,6,5,8,3]

Try i=2: 2 from nums1, 3 from nums2
  max_subseq(nums1, 2) = [6,5]
  max_subseq(nums2, 3) = [9,8,3]
  merge([6,5],[9,8,3]) = [9,8,6,5,3]

Try i=3: 3 from nums1, 2 from nums2
  max_subseq(nums1, 3) = [4,6,5]
  max_subseq(nums2, 2) = [9,8]
  merge([4,6,5],[9,8]) = [9,8,4,6,5]

Try i=4: 4 from nums1, 1 from nums2
  max_subseq(nums1, 4) = [3,4,6,5]
  max_subseq(nums2, 1) = [9]
  merge([3,4,6,5],[9]) = [9,3,4,6,5]

Best: [9,8,6,5,3]
```

**Solution:**
```python
def maxNumber(nums1: list[int], nums2: list[int], k: int) -> list[int]:

    def max_subsequence(nums, length):
        """Find the maximum subsequence of given length (stack-based greedy)."""
        drop = len(nums) - length
        stack = []
        for num in nums:
            while drop > 0 and stack and stack[-1] < num:
                stack.pop()
                drop -= 1
            stack.append(num)
        return stack[:length]

    def merge(subseq1, subseq2):
        """Merge two subsequences to form the largest number."""
        result = []
        i, j = 0, 0
        while i < len(subseq1) or j < len(subseq2):
            # Compare remaining subsequences lexicographically
            if subseq1[i:] >= subseq2[j:]:
                result.append(subseq1[i])
                i += 1
            else:
                result.append(subseq2[j])
                j += 1
        return result

    best = []
    for i in range(k + 1):
        j = k - i
        if i > len(nums1) or j > len(nums2):
            continue
        sub1 = max_subsequence(nums1, i)
        sub2 = max_subsequence(nums2, j)
        candidate = merge(sub1, sub2)
        if candidate > best:
            best = candidate

    return best
```

**Complexity:** Time O(k * (m + n + k)), Space O(m + n + k). The outer loop runs O(k) times, and each iteration does O(m + n) for subsequences and O(k) for merge.

---

<a name="post-processing"></a>
## 5. Post-Processing Reference

| Problem Type | Return Value | Post-Processing Notes |
|--------------|-------------|----------------------|
| **Interval scheduling** | Count (kept or removed) | `removed = total - kept` |
| **Merge intervals** | List of intervals | Build result list during scan |
| **Jump reachability** | Boolean or count | Track farthest reach |
| **Task scheduling** | Minimum time | `max(formula_result, len(tasks))` |
| **Reorganize string** | String or "" | Check feasibility first, then build |
| **Optimal merge** | Minimum cost | Accumulate cost at each merge |
| **Gas station** | Starting index or -1 | Check `sum(gas) >= sum(cost)` first |
| **Candy / two-pass** | Sum or list | `max(left_pass, right_pass)` at each index |
| **Digit removal** | String | Strip leading zeros, handle empty -> "0" |
| **Create max number** | List of digits | Try all splits, return lexicographic max |

---

<a name="pitfalls"></a>
## 6. Common Pitfalls & Solutions

### Pitfall 1: Sorting by the Wrong Criterion

```python
# WRONG: Sorting intervals by start for "max non-overlapping"
intervals.sort(key=lambda x: x[0])

# This fails because a long interval starting early blocks many short ones
# Example: [[1,10],[2,3],[4,5],[6,7]]
# Sorting by start picks [1,10] first, can only fit 1 interval
```

**Solution:** Sort by end time for selection problems, start time for merging problems.
```python
# CORRECT: Sort by end time for activity selection
intervals.sort(key=lambda x: x[1])
# Now picks [2,3], [4,5], [6,7] = 3 intervals
```

---

### Pitfall 2: Not Handling Edge Cases in Remove K Digits

```python
# WRONG: Missing edge cases
def removeKdigits(num, k):
    stack = []
    for d in num:
        while k and stack and stack[-1] > d:
            stack.pop()
            k -= 1
        stack.append(d)
    return ''.join(stack)

# Fails on: num="10", k=2 -> should return "0"
# Fails on: num="1111", k=2 -> stack never pops, need to trim end
# Fails on: num="10200", k=1 -> leading zeros not removed
```

**Solution:** Handle remaining k, leading zeros, and empty result.
```python
# CORRECT
def removeKdigits(num, k):
    stack = []
    for d in num:
        while k and stack and stack[-1] > d:
            stack.pop()
            k -= 1
        stack.append(d)
    # Trim remaining k from end
    if k:
        stack = stack[:-k]
    # Remove leading zeros
    result = ''.join(stack).lstrip('0')
    return result if result else '0'
```

---

### Pitfall 3: Greedy When DP is Required

```python
# WRONG: Greedy for 0/1 knapsack
def knapsack_greedy(values, weights, capacity):
    items = sorted(zip(values, weights), key=lambda x: x[0]/x[1], reverse=True)
    total = 0
    for v, w in items:
        if w <= capacity:
            total += v
            capacity -= w
    return total

# Fails: values=[60,100,120], weights=[10,20,30], capacity=50
# Greedy: 60+100=160 (picks by ratio: 6,5,4 -> items 0,1)
# Optimal: 100+120=220 (items 1,2)
```

**Solution:** Recognize that 0/1 knapsack needs DP, not greedy. Greedy only works for fractional knapsack.
```python
# CORRECT: Use DP for 0/1 knapsack
def knapsack_dp(values, weights, capacity):
    n = len(values)
    dp = [0] * (capacity + 1)
    for i in range(n):
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    return dp[capacity]
```

---

### Pitfall 4: Off-by-One in Jump Game II

```python
# WRONG: Iterating through ALL indices including the last
def jump(nums):
    jumps = 0
    current_end = 0
    farthest = 0
    for i in range(len(nums)):  # Bug: includes last index
        farthest = max(farthest, i + nums[i])
        if i == current_end:
            jumps += 1
            current_end = farthest
    return jumps

# For nums=[2,3,1,1,4], this counts an extra jump at the end
```

**Solution:** Iterate only up to `len(nums) - 1` (exclusive of last index).
```python
# CORRECT
def jump(nums):
    jumps = 0
    current_end = 0
    farthest = 0
    for i in range(len(nums) - 1):  # Stop before last index
        farthest = max(farthest, i + nums[i])
        if i == current_end:
            jumps += 1
            current_end = farthest
            if current_end >= len(nums) - 1:
                break
    return jumps
```

---

### Pitfall 5: Forgetting Feasibility Check Before Greedy

```python
# WRONG: Not checking if reorganization is possible
def reorganizeString(s):
    count = Counter(s)
    heap = [(-f, c) for c, f in count.items()]
    heapq.heapify(heap)
    result = []
    prev = (0, '')
    while heap:
        freq, char = heapq.heappop(heap)
        result.append(char)
        if prev[0] < 0:
            heapq.heappush(heap, prev)
        prev = (freq + 1, char)
    return ''.join(result)  # Returns wrong answer for "aaab"!
```

**Solution:** Always check feasibility first.
```python
# CORRECT
def reorganizeString(s):
    count = Counter(s)
    max_freq = max(count.values())
    if max_freq > (len(s) + 1) // 2:
        return ""  # Impossible!
    # ... rest of the solution
```

---

### Pitfall 6: Gas Station - Not Verifying Total Feasibility

```python
# WRONG: Only checking local feasibility
def canCompleteCircuit(gas, cost):
    tank = 0
    start = 0
    for i in range(len(gas)):
        tank += gas[i] - cost[i]
        if tank < 0:
            start = i + 1
            tank = 0
    return start  # May return invalid start if total gas < total cost!
```

**Solution:** Check that total gas >= total cost first.
```python
# CORRECT
def canCompleteCircuit(gas, cost):
    if sum(gas) < sum(cost):
        return -1  # Impossible regardless of starting point
    tank = 0
    start = 0
    for i in range(len(gas)):
        tank += gas[i] - cost[i]
        if tank < 0:
            start = i + 1
            tank = 0
    return start
```

---

### Pitfall 7: Queue Reconstruction - Wrong Sort Order

```python
# WRONG: Sorting by height ascending
people.sort(key=lambda x: (x[0], x[1]))
# Inserting shorter people first means taller people shift them,
# breaking the k-count invariant.
```

**Solution:** Sort by height descending, k ascending. Insert tallest first so shorter insertions don't disrupt them.
```python
# CORRECT
people.sort(key=lambda x: (-x[0], x[1]))
queue = []
for person in people:
    queue.insert(person[1], person)
```

---

<a name="recognition"></a>
## 7. Problem Recognition Framework

### Step 1: Is Greedy Appropriate?

**Strong greedy signals:**
1. Problem asks for minimum/maximum of something
2. You can sort and make a single pass
3. Choices are independent or have greedy choice property
4. Exchange argument is easy to construct
5. Problem involves intervals, scheduling, or sequencing

**Greedy is likely wrong when:**
1. Choices have complex dependencies (use DP)
2. You need to consider all subsets (use backtracking)
3. Coin change with arbitrary denominations (use DP)
4. 0/1 selection problems where items can't be split (often DP)

### Step 2: Identify the Greedy Criterion

| Problem Type | Sort/Select By |
|-------------|----------------|
| Max non-overlapping intervals | Earliest end time |
| Merge intervals | Start time |
| Task scheduling | Highest frequency first |
| Minimum cost merge | Smallest elements first |
| Digit removal for min number | Remove larger digits from left |
| Digit selection for max number | Keep larger digits toward left |
| Candy distribution | Two-pass (left then right) |
| Gas station | Running surplus, reset on deficit |

### Step 3: Decision Tree

```
                     Optimization problem?
                            |
                     +------+------+
                     |             |
                    Yes           No -> Not greedy
                     |
              Can you sort and
              make one pass?
                     |
              +------+------+
              |             |
             Yes           No
              |             |
        Greedy choice    Need all
        property holds?  subproblems?
              |             |
        +-----+-----+    +--+--+
        |           |    |     |
       Yes         No   DP   Backtrack
        |           |
     GREEDY     Try DP
        |
   +----+----+----+----+
   |    |    |    |    |
Intervals? Jump? Digits? Scheduling? Circular?
   |    |    |    |         |
Sort by  Track  Stack  Frequency   Running
end/start farthest based  count    surplus
```

### Quick Pattern Matching

```
"non-overlapping" / "minimum removal"  ->  Sort by end, activity selection
"merge intervals"                      ->  Sort by start, extend
"jump" / "reach" / "can you get to"    ->  Track farthest reachable
"minimum jumps"                        ->  BFS-style greedy levels
"task scheduler" / "cooldown"          ->  Frequency formula
"reorganize" / "no adjacent same"      ->  Max-heap greedy
"connect" / "merge cost"              ->  Min-heap Huffman-style
"gas station" / "circular route"       ->  Running surplus + reset
"candy" / "both neighbors"            ->  Two-pass left-right
"remove digits" / "smallest number"    ->  Monotonic stack
"create maximum"                       ->  Stack subsequence + merge
```

---

<a name="checklist"></a>
## 8. Interview Preparation Checklist

### Before the Interview

**Master the fundamentals:**
- [ ] Can explain greedy choice property in plain English
- [ ] Can construct an exchange argument proof
- [ ] Know when greedy fails (0/1 knapsack, arbitrary coin change)
- [ ] Can identify which sorting criterion to use for interval problems

**Know the templates:**
- [ ] Sort-then-iterate (interval scheduling)
- [ ] Two-pass greedy (Candy)
- [ ] Priority-queue greedy (connect sticks, reorganize string)
- [ ] Stack-based digit manipulation (remove K digits)

**Practice pattern recognition:**
- [ ] Can distinguish greedy from DP in under 30 seconds
- [ ] Know that "minimum removal for non-overlapping" = activity selection
- [ ] Recognize jump game variants immediately
- [ ] Identify two-pass greedy when constraints come from both sides

**Common problems solved:**
- [ ] LC 55: Jump Game
- [ ] LC 45: Jump Game II
- [ ] LC 56: Merge Intervals
- [ ] LC 435: Non-overlapping Intervals
- [ ] LC 134: Gas Station
- [ ] LC 135: Candy
- [ ] LC 402: Remove K Digits
- [ ] LC 621: Task Scheduler
- [ ] LC 316: Remove Duplicate Letters

### During the Interview

**1. Clarify (30 seconds)**
- What is being optimized? (min cost, max count, etc.)
- Are items sortable? By what criterion?
- Can items be split/fractioned or are they 0/1?

**2. Identify pattern (30 seconds)**
- Intervals? -> Sort by end or start
- Reachability? -> Track farthest
- Digit manipulation? -> Monotonic stack
- Constraints from both sides? -> Two-pass

**3. Prove greedy works (1 minute)**
- State the greedy choice
- Quick exchange argument: "If we swapped this choice for any other, the result wouldn't improve because..."
- Mention optimal substructure if relevant

**4. Code (3-4 minutes)**
- Sort if needed
- Single pass (or two passes)
- Track state variables
- Return result

**5. Test (1-2 minutes)**
- Empty input
- Single element
- Already optimal (no work needed)
- All elements identical
- Worst case / boundary conditions

**6. Analyze (30 seconds)**
- Time: usually O(n log n) for sort-based, O(n) for scan-based
- Space: usually O(1) extra for scan, O(n) for stack-based

---

<a name="quick-reference"></a>
## 9. Quick Reference Cards

### Activity Selection (Max Non-overlapping)
```python
intervals.sort(key=lambda x: x[1])  # Sort by END
kept, last_end = 0, float('-inf')
for start, end in intervals:
    if start >= last_end:
        kept += 1
        last_end = end
# removed = len(intervals) - kept
```

### Jump Game (Reachability)
```python
farthest = 0
for i in range(len(nums)):
    if i > farthest:
        return False
    farthest = max(farthest, i + nums[i])
return True
```

### Jump Game II (Min Jumps)
```python
jumps = current_end = farthest = 0
for i in range(len(nums) - 1):
    farthest = max(farthest, i + nums[i])
    if i == current_end:
        jumps += 1
        current_end = farthest
return jumps
```

### Gas Station
```python
if sum(gas) < sum(cost): return -1
tank = start = 0
for i in range(len(gas)):
    tank += gas[i] - cost[i]
    if tank < 0:
        start = i + 1
        tank = 0
return start
```

### Two-Pass (Candy)
```python
candy = [1] * n
for i in range(1, n):           # Left to right
    if ratings[i] > ratings[i-1]:
        candy[i] = candy[i-1] + 1
for i in range(n-2, -1, -1):    # Right to left
    if ratings[i] > ratings[i+1]:
        candy[i] = max(candy[i], candy[i+1] + 1)
return sum(candy)
```

### Remove K Digits (Monotonic Stack)
```python
stack = []
for d in num:
    while k and stack and stack[-1] > d:
        stack.pop(); k -= 1
    stack.append(d)
if k: stack = stack[:-k]
return ''.join(stack).lstrip('0') or '0'
```

### Task Scheduler Formula
```python
max_freq = max(Counter(tasks).values())
max_count = sum(1 for f in Counter(tasks).values() if f == max_freq)
return max((max_freq - 1) * (n + 1) + max_count, len(tasks))
```

### Huffman / Connect Sticks
```python
heapq.heapify(sticks)
cost = 0
while len(sticks) > 1:
    a, b = heapq.heappop(sticks), heapq.heappop(sticks)
    cost += a + b
    heapq.heappush(sticks, a + b)
return cost
```

---

<a name="complexity-reference"></a>
## 10. Complexity Reference

| Problem | Time | Space | Bottleneck |
|---------|------|-------|------------|
| Non-overlapping Intervals (LC 435) | O(n log n) | O(1) | Sorting |
| Min Arrows (LC 452) | O(n log n) | O(1) | Sorting |
| Merge Intervals (LC 56) | O(n log n) | O(n) | Sorting + output |
| Jump Game (LC 55) | O(n) | O(1) | Single scan |
| Jump Game II (LC 45) | O(n) | O(1) | Single scan |
| Video Stitching (LC 1024) | O(n log n) | O(1) | Sorting |
| Task Scheduler (LC 621) | O(n) | O(1) | Counting (26 chars) |
| Reorganize String (LC 767) | O(n) | O(1) | Heap with 26 entries |
| Connect Sticks (LC 1167) | O(n log n) | O(n) | Heap operations |
| Gas Station (LC 134) | O(n) | O(1) | Single scan |
| Candy (LC 135) | O(n) | O(n) | Two linear scans |
| Queue Reconstruction (LC 406) | O(n^2) | O(n) | List insertions |
| Remove K Digits (LC 402) | O(n) | O(n) | Stack operations |
| Remove Duplicate Letters (LC 316) | O(n) | O(1) | Stack (26 chars max) |
| Create Max Number (LC 321) | O(k(m+n+k)) | O(m+n+k) | All splits + merge |

### Greedy vs DP Complexity Comparison

| Problem | Greedy | DP | Which to Use |
|---------|--------|-----|-------------|
| Activity Selection | O(n log n) | O(n log n) | Greedy (simpler) |
| Fractional Knapsack | O(n log n) | N/A | Greedy |
| 0/1 Knapsack | N/A (wrong) | O(nW) | DP |
| Coin Change (US) | O(n) | O(nW) | Greedy |
| Coin Change (arbitrary) | N/A (wrong) | O(nW) | DP |
| Jump Game | O(n) | O(n) | Greedy (simpler) |
| Huffman Coding | O(n log n) | N/A | Greedy |

---

## Final Thoughts

**Remember:**
1. Greedy is not just "pick the biggest/smallest." It requires proof that the local choice leads to the global optimum.
2. The exchange argument is your best friend for proving correctness. Practice stating it in one sentence.
3. If greedy seems too easy, double-check with a counterexample. If you find one, switch to DP.
4. Most greedy problems involve sorting first. Ask yourself: "What should I sort by?"
5. Stack-based greedy (monotonic stack) is a powerful subpattern for digit/character manipulation.
6. Two-pass greedy handles problems where constraints come from both directions.
7. When in doubt between greedy and DP, try greedy first (it is simpler), then verify with small examples.

**When stuck:**
1. Ask: "Can I sort and make a single pass?"
2. Ask: "Does the locally optimal choice ever lead to a globally suboptimal result?" (Find a counterexample)
3. Try the exchange argument: "If I swap my greedy choice for any other choice, does the answer get worse?"
4. If greedy fails, don't force it -- switch to DP or backtracking.
5. Draw the problem: number lines for intervals, decision trees for scheduling, stacks for digit problems.

**The greedy mindset:**
- Greedy = commit and never look back
- DP = consider all options before committing
- If you can prove "never looking back" works, greedy gives you the simplest, fastest solution.

---

## Appendix: Practice Problem Set

### Easy
- 455. Assign Cookies
- 860. Lemonade Change
- 1005. Maximize Sum of Array After K Negations

### Medium
- 45. Jump Game II
- 55. Jump Game
- 56. Merge Intervals
- 134. Gas Station
- 135. Candy
- 316. Remove Duplicate Letters
- 402. Remove K Digits
- 406. Queue Reconstruction by Height
- 435. Non-overlapping Intervals
- 452. Minimum Number of Arrows to Burst Balloons
- 621. Task Scheduler
- 767. Reorganize String
- 1024. Video Stitching

### Hard
- 321. Create Maximum Number
- 330. Patching Array
- 1167. Minimum Cost to Connect Sticks
- 765. Couples Holding Hands
- 757. Set Intersection Size At Least Two

**Recommended Practice Order:**
1. Start with reachability: 55 -> 45 (jump game basics)
2. Interval selection: 435 -> 452 -> 56 (sort-based greedy)
3. Gas station and circular: 134 (single-scan greedy)
4. Two-pass: 135 -> 406 (bidirectional constraints)
5. Digit manipulation: 402 -> 316 -> 321 (stack-based greedy)
6. Scheduling: 621 -> 767 (frequency-based greedy)
7. Optimal merge: 1167 (heap-based greedy)
8. Wrap up with easy problems for speed: 455, 860, 1005

Good luck with your interview preparation!
