# The Complete Intervals Handbook
> A template-based approach for mastering interval problems in coding interviews

**Philosophy:** Interval problems are not about complex algorithms. They're about **sorting by the right endpoint** and then making greedy or linear scan decisions.

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

- **The Calendar**: Every interval is a meeting on a calendar -- it has a start time and an end time. Two meetings overlap if one starts before the other ends. Interval problems are really scheduling problems: "Do any meetings conflict?", "What's the minimum number of rooms?", "Can I fit a new meeting in?"
- **The Sort-Then-Scan**: The universal trick is: sort all meetings by start time, then walk through them left to right. Once they're sorted, you only ever need to compare each meeting with the one right before it (or the current "merged" result). This turns chaos into order.

### No-Jargon Translation

- **Interval**: a pair [start, end] representing a range on a number line -- like a meeting from 2pm to 4pm
- **Overlap**: two intervals share some common range -- two meetings conflict
- **Merge**: combining overlapping intervals into one larger interval -- "2-4pm and 3-5pm" becomes "2-5pm"
- **Sweep line**: an imaginary vertical line that sweeps left to right across the number line, processing events as it hits them
- **Greedy**: making the locally optimal choice at each step and trusting it leads to the global best

### Mental Model

> "Interval problems are a messy pile of calendar appointments: sort them by start time, then scan left to right, deciding at each step whether the next appointment overlaps, fits in a gap, or needs a new room."

---

### What is an Interval?

An interval represents a continuous range: `[start, end]`

**Types:**
- **Closed:** `[1, 5]` includes both endpoints
- **Open:** `(1, 5)` excludes both endpoints
- **Half-open:** `[1, 5)` includes start, excludes end

### Key Concepts

**Overlap:** Two intervals overlap if they share at least one point
```
[1, 5] and [3, 7] overlap (share [3, 5])
[1, 5] and [6, 10] don't overlap
[1, 5] and [5, 10] → depends on problem (touching vs overlapping)
```

**Merge:** Combine overlapping intervals into one
```
[1, 5] + [3, 7] → [1, 7]
```

### The Sorting Strategy

Almost every interval problem requires sorting. The question is: **sort by what?**

| Sort By | When to Use |
|---------|-------------|
| **Start time** | Merging intervals, finding overlaps |
| **End time** | Maximum non-overlapping intervals |
| **Both (custom)** | Complex scenarios |

### Visual Understanding

```
Intervals: [1,3], [2,6], [8,10], [15,18]

Number line:
1---3
  2------6
          8--10
                    15--18

After merge: [1,6], [8,10], [15,18]
```

---

<a name="master-templates"></a>
## 2. The Master Templates

### Template A: Merge Overlapping Intervals

```python
def merge(intervals: list[list[int]]) -> list[list[int]]:
    """
    Merge all overlapping intervals.
    Sort by start, then greedily extend.
    """
    # Edge case: nothing to merge
    if not intervals:
        return []

    # Sort by start time so we only need to compare neighbors
    intervals.sort(key=lambda x: x[0])

    merged = [intervals[0]]

    for current in intervals[1:]:
        last = merged[-1]

        # Why `<=` and not `<`?
        # Touching intervals like [1,3] and [3,5] should merge into [1,5].
        # With `<`, we would miss the case where current starts exactly
        # where last ends, and they'd stay separate incorrectly.
        if current[0] <= last[1]:  # Overlapping or touching
            # Why max and not just replace?
            # The previous interval might already extend beyond the current
            # one. E.g., last=[1,10], current=[2,5] -- replacing would shrink
            # the merged interval. max keeps the farthest-reaching end.
            last[1] = max(last[1], current[1])
        else:  # Gap between intervals, no overlap
            merged.append(current)

    return merged
```

---

### Template B: Insert Interval

```python
def insert(intervals: list[list[int]], newInterval: list[int]) -> list[list[int]]:
    """
    Insert and merge a new interval into sorted intervals.
    """
    result = []
    i = 0
    n = len(intervals)

    # PHASE 1: Add all intervals that end before the new one starts.
    # Why `intervals[i][1] < newInterval[0]`?
    # We compare the existing interval's END with the new interval's START.
    # If the existing one ends before the new one even begins, there is
    # zero overlap -- it belongs entirely to the "before" region.
    # Why `<` and not `<=`? If they touch (end == start), that is still
    # considered overlapping and should be merged in the next phase.
    while i < n and intervals[i][1] < newInterval[0]:
        result.append(intervals[i])
        i += 1

    # PHASE 2: Merge all intervals that overlap with newInterval.
    # Why `intervals[i][0] <= newInterval[1]`?
    # An interval overlaps with newInterval if it starts before (or at)
    # where newInterval ends. Combined with Phase 1 already skipping
    # intervals that end before newInterval starts, this captures
    # exactly the overlapping set.
    while i < n and intervals[i][0] <= newInterval[1]:
        # Why min/max instead of just taking the current interval's values?
        # We are building one big merged interval that spans all overlaps.
        # min picks the earliest start, max picks the latest end.
        newInterval[0] = min(newInterval[0], intervals[i][0])
        newInterval[1] = max(newInterval[1], intervals[i][1])
        i += 1
    result.append(newInterval)

    # PHASE 3: Add remaining intervals (all start after newInterval ends).
    # No condition needed beyond bounds -- everything left is after.
    while i < n:
        result.append(intervals[i])
        i += 1

    return result
```

---

### Template C: Non-Overlapping Intervals (Greedy)

```python
def eraseOverlapIntervals(intervals: list[list[int]]) -> int:
    """
    Minimum intervals to remove for non-overlapping.
    Sort by END time and greedily keep earliest ending.
    """
    if not intervals:
        return 0

    # Why sort by END time, not start time?
    # We want to keep as many intervals as possible. An interval that
    # ends earlier leaves more room for future intervals. Sorting by
    # end time lets the greedy choice (keep earliest-ending) be optimal.
    intervals.sort(key=lambda x: x[1])

    count = 0
    prev_end = float('-inf')

    for start, end in intervals:
        # Why `>=` and not `>`?
        # If the current interval starts exactly where the previous one
        # ended (e.g., [1,3] and [3,5]), they do NOT overlap -- one ends
        # as the other begins. So `>=` correctly treats touching as
        # non-overlapping.
        if start >= prev_end:  # No overlap -- keep this interval
            prev_end = end
        else:  # Overlap -- remove current (keep previous, it ends earlier)
            count += 1

    return count
```

---

### Template D: Meeting Rooms (Overlap Count)

```python
def minMeetingRooms(intervals: list[list[int]]) -> int:
    """
    Count maximum simultaneous overlapping intervals.
    Use sweep line: +1 at start, -1 at end.
    """
    events = []
    for start, end in intervals:
        events.append((start, 1))   # Meeting begins: need +1 room
        events.append((end, -1))    # Meeting ends: free -1 room

    # Why sort by (time, delta) where delta is +1 or -1?
    # At the same timestamp, we process end events (-1) before start
    # events (+1). If a meeting ends at 10 and another starts at 10,
    # the room is freed first, then reused -- so we don't over-count.
    events.sort(key=lambda x: (x[0], x[1]))

    max_rooms = current_rooms = 0

    for time, delta in events:
        current_rooms += delta
        max_rooms = max(max_rooms, current_rooms)

    return max_rooms
```

**Alternative using heap:**
```python
import heapq

def minMeetingRooms_heap(intervals: list[list[int]]) -> int:
    if not intervals:
        return 0

    intervals.sort(key=lambda x: x[0])

    # Heap tracks end times of meetings currently in progress.
    # The smallest end time is always at heap[0] (min-heap property).
    heap = []

    for start, end in intervals:
        # Why `heap[0] <= start` and not `< start`?
        # If the earliest-ending meeting ends at exactly the same time
        # this one starts, the room is free -- that meeting is over.
        # `<=` correctly allows reuse in this case. With `<`, we would
        # allocate an unnecessary extra room for back-to-back meetings.
        if heap and heap[0] <= start:
            heapq.heappop(heap)
        heapq.heappush(heap, end)

    # Each entry in the heap is a room still in use.
    # The heap size = number of rooms needed at peak.
    return len(heap)
```

---

### Template E: Interval Intersection

```python
def intervalIntersection(
    firstList: list[list[int]], secondList: list[list[int]]
) -> list[list[int]]:
    """
    Find intersection of two sorted interval lists.
    Two-pointer approach.
    """
    result = []
    i, j = 0, 0

    # Why `and` not `or`? Once either list is exhausted, there can be
    # no more intersections -- an intersection requires one interval
    # from each list. So we stop as soon as one list runs out.
    while i < len(firstList) and j < len(secondList):
        a_start, a_end = firstList[i]
        b_start, b_end = secondList[j]

        # The intersection of two intervals (if it exists) starts at the
        # later of the two starts and ends at the earlier of the two ends.
        start = max(a_start, b_start)
        end = min(a_end, b_end)

        # Why `start <= end`?
        # If the computed start is after the computed end, the two
        # intervals do not actually overlap -- there is a gap between
        # them. Only when start <= end is there a valid (possibly
        # single-point) intersection to record.
        if start <= end:
            result.append([start, end])

        # Why advance the pointer with the smaller end?
        # The interval that ends first cannot overlap with anything
        # further in the other list (which only has later intervals).
        # The interval that ends later might still overlap with the
        # next interval in the other list, so we keep it.
        if a_end < b_end:
            i += 1
        else:
            j += 1

    return result
```

---

### Template F: Interval Coverage

```python
def minIntervalsToCover(intervals: list[list[int]], target: list[int]) -> int:
    """
    Minimum intervals to cover target range.
    Sort by start, greedily pick farthest reaching.
    """
    intervals.sort()
    target_start, target_end = target

    count = 0
    i = 0
    current_end = target_start

    # Why `<` and not `<=`?
    # We need to cover up to target_end. Once current_end reaches
    # target_end, the range is fully covered -- no more work needed.
    while current_end < target_end:
        max_end = current_end
        # Why `intervals[i][0] <= current_end`?
        # An interval can only extend our coverage if it starts at or
        # before where we currently are. If it starts after current_end,
        # there would be a gap in coverage. Among all intervals that
        # start in time, we greedily pick the one reaching farthest.
        while i < len(intervals) and intervals[i][0] <= current_end:
            max_end = max(max_end, intervals[i][1])
            i += 1

        # If no interval could extend our reach, there is an
        # uncoverable gap -- the target range cannot be fully covered.
        if max_end == current_end:  # Can't extend
            return -1

        current_end = max_end
        count += 1

    return count
```

---

### Quick Decision Matrix

| Problem Type | Sort By | Template |
|--------------|---------|----------|
| Merge overlapping | Start | A |
| Insert interval | (already sorted) | B |
| Min to remove | End | C |
| Max non-overlapping | End | C (count kept) |
| Concurrent count | Sweep line | D |
| Intersection | (two pointers) | E |
| Coverage | Start | F |

---

<a name="pattern-guide"></a>
## 3. Pattern Classification Guide

### Category 1: Merge Intervals
- Combine overlapping intervals
- **Template A**
- Sort by start, merge if overlap

### Category 2: Insert Interval
- Add new interval to sorted list
- **Template B**
- Three phases: before, overlap, after

### Category 3: Non-Overlapping Selection
- Max intervals that don't overlap
- **Template C**
- Sort by END, greedy selection

### Category 4: Overlap Counting
- Max simultaneous overlaps
- **Template D**
- Sweep line or heap

### Category 5: Intersection
- Common parts of interval lists
- **Template E**
- Two pointers

### Category 6: Coverage
- Cover target range
- **Template F**
- Greedy farthest reach

---

<a name="patterns"></a>
## 4. Complete Pattern Library

### PATTERN 1: Merge Intervals

---

#### Pattern 1A: Merge Intervals

**Problem:** LeetCode 56 - Merge all overlapping intervals

```python
def merge(intervals: list[list[int]]) -> list[list[int]]:
    intervals.sort(key=lambda x: x[0])
    merged = []

    for interval in intervals:
        # Why `merged[-1][1] < interval[0]` (strict `<`)?
        # This means: "last merged interval ends BEFORE current starts."
        # If they touch (e.g., [1,3] and [3,6]), `<` is false so we
        # fall into the else branch and merge them. This treats touching
        # intervals as overlapping, which is the standard for LC 56.
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            # Why max? The current interval might end before the last
            # merged one (it's fully contained). max ensures we keep
            # the farthest-reaching end.
            merged[-1][1] = max(merged[-1][1], interval[1])

    return merged
```

**Visual:**
```
Input: [[1,3],[2,6],[8,10],[15,18]]

Sorted: [[1,3],[2,6],[8,10],[15,18]]

Process:
- [1,3]: merged = [[1,3]]
- [2,6]: overlaps (2 <= 3), merged = [[1,6]]
- [8,10]: no overlap (8 > 6), merged = [[1,6],[8,10]]
- [15,18]: no overlap, merged = [[1,6],[8,10],[15,18]]
```

---

#### Pattern 1B: Insert Interval

**Problem:** LeetCode 57 - Insert new interval into sorted list

```python
def insert(intervals: list[list[int]], newInterval: list[int]) -> list[list[int]]:
    result = []
    i = 0
    n = len(intervals)

    # Phase 1: Collect intervals entirely before newInterval.
    # Why `intervals[i][1] < newInterval[0]`?
    # The existing interval's end is before the new interval's start,
    # meaning no overlap. Strict `<` because if they touch, we merge.
    while i < n and intervals[i][1] < newInterval[0]:
        result.append(intervals[i])
        i += 1

    # Phase 2: Merge all intervals overlapping with newInterval.
    # Why `intervals[i][0] <= newInterval[1]`?
    # The existing interval starts before (or exactly where) the new
    # interval ends -- so they overlap or touch. We absorb it.
    while i < n and intervals[i][0] <= newInterval[1]:
        newInterval = [
            min(newInterval[0], intervals[i][0]),
            max(newInterval[1], intervals[i][1])
        ]
        i += 1
    result.append(newInterval)

    # Phase 3: Everything remaining starts after newInterval -- no overlap.
    result.extend(intervals[i:])

    return result
```

---

### PATTERN 2: Non-Overlapping Selection

---

#### Pattern 2A: Non-overlapping Intervals

**Problem:** LeetCode 435 - Minimum intervals to remove

```python
def eraseOverlapIntervals(intervals: list[list[int]]) -> int:
    if not intervals:
        return 0

    # Why sort by end time? Intervals ending earlier leave more space
    # for future intervals. This greedy choice minimizes removals.
    intervals.sort(key=lambda x: x[1])

    removals = 0
    prev_end = intervals[0][1]

    for i in range(1, len(intervals)):
        # Why `<` and not `<=`?
        # If current starts exactly where previous ends (e.g., [1,3]
        # and [3,5]), they do NOT overlap -- they are back-to-back.
        # Strict `<` means: "starts before previous ends" = true overlap.
        if intervals[i][0] < prev_end:  # Overlap -- remove current
            removals += 1
        else:
            # No overlap -- keep this interval, update prev_end
            prev_end = intervals[i][1]

    return removals
```

**Why sort by end?**
- Intervals ending earlier leave more room for others
- Greedy: always keep the interval that ends first

---

#### Pattern 2B: Maximum Number of Events

**Problem:** LeetCode 1353 - Max events that can be attended

```python
import heapq

def maxEvents(events: list[list[int]]) -> int:
    events.sort()
    heap = []  # End times of available events
    count = 0
    day = 0
    i = 0
    n = len(events)

    # Why `i < n or heap`?
    # We continue as long as there are unprocessed events (i < n) OR
    # events already in the heap waiting to be attended.
    while i < n or heap:
        # If the heap is empty, skip forward to the next event's start
        # day. This avoids iterating through empty days one by one.
        if not heap:
            day = events[i][0]

        # Add all events that have started by today.
        # Why `events[i][0] <= day`? An event is available to attend
        # on any day from its start through its end. We push all
        # events whose start day has arrived.
        while i < n and events[i][0] <= day:
            heapq.heappush(heap, events[i][1])
            i += 1

        # Why `heap[0] < day` (strict `<`)?
        # An event with end time equal to today can still be attended
        # today. Only events that ended strictly before today are
        # expired and should be discarded.
        while heap and heap[0] < day:
            heapq.heappop(heap)

        # Greedily attend the event ending soonest (min-heap top).
        # This maximizes the chance of attending future events.
        if heap:
            heapq.heappop(heap)
            count += 1

        day += 1

    return count
```

---

### PATTERN 3: Meeting Rooms

---

#### Pattern 3A: Can Attend All Meetings

**Problem:** LeetCode 252 - Check if person can attend all meetings

```python
def canAttendMeetings(intervals: list[list[int]]) -> bool:
    intervals.sort(key=lambda x: x[0])

    for i in range(1, len(intervals)):
        # Why `<` and not `<=`?
        # If meeting i starts exactly when meeting i-1 ends (e.g.,
        # [1,3] and [3,5]), the person finishes one and walks into
        # the next -- no conflict. Strict `<` catches only true
        # overlaps where a meeting starts before the previous ends.
        if intervals[i][0] < intervals[i-1][1]:
            return False

    return True
```

---

#### Pattern 3B: Meeting Rooms II

**Problem:** LeetCode 253 - Minimum meeting rooms required

**Sweep Line:**
```python
def minMeetingRooms(intervals: list[list[int]]) -> int:
    events = []
    for start, end in intervals:
        events.append((start, 1))
        events.append((end, -1))

    # Why just events.sort() (sorting by tuple default)?
    # Default tuple sort: first by time, then by delta.
    # Since -1 < 1, at the same timestamp, end events (-1) are
    # processed before start events (+1). This means if one meeting
    # ends at time T and another starts at T, the room is freed first
    # -- so we don't over-count concurrent rooms.
    events.sort()

    max_rooms = rooms = 0
    for _, delta in events:
        rooms += delta
        max_rooms = max(max_rooms, rooms)

    return max_rooms
```

**Heap:**
```python
import heapq

def minMeetingRooms_heap(intervals: list[list[int]]) -> int:
    if not intervals:
        return 0

    intervals.sort(key=lambda x: x[0])
    heap = []  # End times of meetings currently using rooms

    for start, end in intervals:
        # Why `heap[0] <= start`?
        # heap[0] is the earliest end time among all ongoing meetings.
        # If that meeting ends at or before the current meeting starts,
        # the room is free to reuse. We pop it (freeing the room) and
        # then push the new meeting's end time.
        # `<=` (not `<`) because a meeting ending at time 5 and one
        # starting at time 5 do not conflict -- the room is available.
        if heap and heap[0] <= start:
            heapq.heappop(heap)
        heapq.heappush(heap, end)

    # The heap size equals the number of rooms in use at peak overlap.
    return len(heap)
```

---

#### Pattern 3C: Employee Free Time

**Problem:** LeetCode 759 - Find common free time intervals

```python
def employeeFreeTime(schedule: list[list[list[int]]]) -> list[list[int]]:
    # Flatten and sort all intervals
    all_intervals = []
    for employee in schedule:
        all_intervals.extend(employee)

    all_intervals.sort(key=lambda x: x[0])

    # Merge to find consolidated busy times
    merged = [all_intervals[0]]
    for interval in all_intervals[1:]:
        # Why `<=`? If one employee's shift ends at 5 and another's
        # starts at 5, there is no free time gap between them --
        # the company is continuously busy. So touching = merge.
        if interval[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], interval[1])
        else:
            merged.append(interval)

    # Gaps between merged busy intervals = everyone's free time.
    # Each gap [merged[i-1] end, merged[i] start] is a window where
    # no employee is working.
    result = []
    for i in range(1, len(merged)):
        result.append([merged[i-1][1], merged[i][0]])

    return result
```

---

### PATTERN 4: Interval Intersection

---

#### Pattern 4A: Interval List Intersections

**Problem:** LeetCode 986 - Intersection of two interval lists

```python
def intervalIntersection(
    firstList: list[list[int]], secondList: list[list[int]]
) -> list[list[int]]:
    result = []
    i, j = 0, 0

    # Both pointers must be in bounds. Once one list is exhausted,
    # no more intersections are possible.
    while i < len(firstList) and j < len(secondList):
        # The intersection starts at the later start and ends at the
        # earlier end. This is the region both intervals share.
        start = max(firstList[i][0], secondList[j][0])
        end = min(firstList[i][1], secondList[j][1])

        # Why `start <= end`?
        # If start > end, the intervals don't actually overlap (there
        # is a gap). Only record an intersection when start <= end.
        # Note: `<=` (not `<`) to include single-point intersections
        # like [3,3] where two intervals just touch.
        if start <= end:
            result.append([start, end])

        # Advance the pointer whose interval ends first. That interval
        # cannot intersect with anything further in the other list.
        if firstList[i][1] < secondList[j][1]:
            i += 1
        else:
            j += 1

    return result
```

---

### PATTERN 5: Coverage Problems

---

#### Pattern 5A: Minimum Number of Arrows

**Problem:** LeetCode 452 - Min arrows to burst all balloons

```python
def findMinArrowPoints(points: list[list[int]]) -> int:
    if not points:
        return 0

    # Why sort by end? We want to shoot an arrow as late as possible
    # within the first balloon's range, to also hit as many subsequent
    # overlapping balloons as possible. Sorting by end and placing the
    # arrow at current_end is optimal for this greedy strategy.
    points.sort(key=lambda x: x[1])

    arrows = 1
    current_end = points[0][1]

    for start, end in points[1:]:
        # Why `>` and not `>=`?
        # Balloons are inclusive ranges. If a balloon starts exactly
        # at current_end (e.g., arrow at 3, balloon [3,5]), the arrow
        # at position 3 still pops it. Only if it starts strictly
        # after does it need a new arrow.
        if start > current_end:  # Need new arrow
            arrows += 1
            current_end = end
        # else: current arrow at current_end also pops this balloon

    return arrows
```

---

#### Pattern 5B: Video Stitching

**Problem:** LeetCode 1024 - Min clips to cover [0, time]

```python
def videoStitching(clips: list[list[int]], time: int) -> int:
    clips.sort()

    count = 0
    current_end = 0
    farthest = 0
    i = 0

    # Why `current_end < time`?
    # We need to cover [0, time]. Once current_end reaches time,
    # the full range is covered.
    while current_end < time:
        # Why `clips[i][0] <= current_end`?
        # A clip can only extend our coverage if it starts at or
        # before our current coverage boundary. If it starts after,
        # there would be a gap. Among all valid clips, we greedily
        # pick the one whose end reaches farthest.
        while i < len(clips) and clips[i][0] <= current_end:
            farthest = max(farthest, clips[i][1])
            i += 1

        # If no clip could push farthest beyond current_end,
        # there is an uncoverable gap in the timeline.
        if farthest == current_end:  # Can't extend
            return -1

        count += 1
        current_end = farthest

    return count
```

---

#### Pattern 5C: Jump Game II (as intervals)

**Problem:** LeetCode 45 - Minimum jumps to reach end

```python
def jump(nums: list[int]) -> int:
    # Already at or past the end -- no jumps needed.
    if len(nums) <= 1:
        return 0

    jumps = 0
    current_end = 0    # Farthest index reachable with `jumps` jumps
    farthest = 0       # Farthest index reachable with `jumps + 1` jumps

    # Why `len(nums) - 1` and not `len(nums)`?
    # We never need to "jump from" the last index. If we reach it,
    # we are done. Iterating to len(nums) could cause an unnecessary
    # extra jump count.
    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])

        # Why `i == current_end`?
        # current_end is the boundary of what the current jump covers.
        # When i reaches this boundary, we have explored all positions
        # reachable by the current number of jumps. We must now "use"
        # the next jump, extending our reach to farthest.
        if i == current_end:
            jumps += 1
            current_end = farthest

            # Why `>=` and not `==`?
            # farthest might overshoot the last index. We do not need
            # to land exactly on it -- reaching or passing it is enough.
            if current_end >= len(nums) - 1:
                break

    return jumps
```

---

### PATTERN 6: Calendar Problems

---

#### Pattern 6A: My Calendar I

**Problem:** LeetCode 729 - Book if no double booking

```python
class MyCalendar:
    def __init__(self):
        self.bookings = []

    def book(self, start: int, end: int) -> bool:
        for s, e in self.bookings:
            if start < e and end > s:  # Overlap
                return False

        self.bookings.append((start, end))
        return True
```

**Using binary search (sorted):**
```python
import bisect

class MyCalendar:
    def __init__(self):
        self.starts = []
        self.ends = []

    def book(self, start: int, end: int) -> bool:
        # Find where to insert
        i = bisect.bisect_right(self.starts, start)

        # Check overlap with previous booking (if exists)
        if i > 0 and self.ends[i-1] > start:
            return False

        # Check overlap with next booking (if exists)
        if i < len(self.starts) and end > self.starts[i]:
            return False

        self.starts.insert(i, start)
        self.ends.insert(i, end)
        return True
```

---

#### Pattern 6B: My Calendar II

**Problem:** LeetCode 731 - Allow double booking, not triple

```python
class MyCalendarTwo:
    def __init__(self):
        self.bookings = []
        self.overlaps = []

    def book(self, start: int, end: int) -> bool:
        # Check for triple booking
        for s, e in self.overlaps:
            if start < e and end > s:
                return False

        # Add overlaps with existing bookings
        for s, e in self.bookings:
            if start < e and end > s:
                self.overlaps.append((max(start, s), min(end, e)))

        self.bookings.append((start, end))
        return True
```

---

#### Pattern 6C: My Calendar III

**Problem:** LeetCode 732 - Return max concurrent bookings

```python
from collections import defaultdict

class MyCalendarThree:
    def __init__(self):
        self.timeline = defaultdict(int)

    def book(self, start: int, end: int) -> int:
        self.timeline[start] += 1
        self.timeline[end] -= 1

        max_booking = current = 0
        for time in sorted(self.timeline.keys()):
            current += self.timeline[time]
            max_booking = max(max_booking, current)

        return max_booking
```

---

### PATTERN 7: Advanced Interval Problems

---

#### Pattern 7A: Remove Covered Intervals

**Problem:** LeetCode 1288 - Remove intervals covered by another

```python
def removeCoveredIntervals(intervals: list[list[int]]) -> int:
    # Sort by start ascending, then by end descending
    # This ensures if starts are equal, longer interval comes first
    intervals.sort(key=lambda x: (x[0], -x[1]))

    count = 0
    prev_end = 0

    for start, end in intervals:
        if end > prev_end:
            count += 1
            prev_end = end
        # else: current is covered by previous

    return count
```

---

#### Pattern 7B: Data Stream Disjoint Intervals

**Problem:** LeetCode 352 - Merge numbers into intervals as they arrive

```python
import bisect

class SummaryRanges:
    def __init__(self):
        self.intervals = []

    def addNum(self, val: int) -> None:
        new_interval = [val, val]
        result = []
        inserted = False

        for interval in self.intervals:
            if interval[1] < new_interval[0] - 1:
                result.append(interval)
            elif new_interval[1] < interval[0] - 1:
                if not inserted:
                    result.append(new_interval)
                    inserted = True
                result.append(interval)
            else:
                new_interval = [
                    min(new_interval[0], interval[0]),
                    max(new_interval[1], interval[1])
                ]

        if not inserted:
            result.append(new_interval)

        self.intervals = result

    def getIntervals(self) -> list[list[int]]:
        return self.intervals
```

---

<a name="post-processing"></a>
## 5. Post-Processing Reference

| Problem Type | Result | Notes |
|--------------|--------|-------|
| **Merge** | List of intervals | Non-overlapping |
| **Insert** | List of intervals | Sorted, merged |
| **Max non-overlap** | Count | Sort by end |
| **Max concurrent** | Integer | Sweep line |
| **Min to remove** | Count | Total - max non-overlap |

---

<a name="pitfalls"></a>
## 6. Common Pitfalls & Solutions

### Pitfall 1: Wrong Sorting Criteria

```python
# For merge: sort by START
intervals.sort(key=lambda x: x[0])

# For max non-overlapping: sort by END
intervals.sort(key=lambda x: x[1])
```

### Pitfall 2: Overlap Condition

```python
# Overlap: a_start < b_end AND b_start < a_end
# Equivalent: max(starts) < min(ends)

# Non-overlap: a_end <= b_start OR b_end <= a_start
```

### Pitfall 3: Inclusive vs Exclusive Boundaries

```python
# If [1,5] and [5,10] should NOT overlap:
if intervals[i][0] < intervals[i-1][1]:  # Overlap

# If [1,5] and [5,10] SHOULD overlap:
if intervals[i][0] <= intervals[i-1][1]:  # Overlap
```

### Pitfall 4: Not Modifying In Place

```python
# WRONG: Creates shallow copy issues
merged[-1] = [merged[-1][0], max(merged[-1][1], interval[1])]

# CORRECT: Modify in place
merged[-1][1] = max(merged[-1][1], interval[1])
```

---

<a name="recognition"></a>
## 7. Problem Recognition Framework

### Step 1: Identify as Interval Problem

**Indicators:**
- Time ranges, schedules, meetings
- Start/end pairs
- Overlapping, merging, covering
- "Maximum number that can..."

### Step 2: Choose Strategy

| Goal | Strategy | Sort By |
|------|----------|---------|
| Merge overlapping | Linear merge | Start |
| Max non-overlapping | Greedy | End |
| Count concurrent | Sweep line | Events |
| Cover range | Greedy extend | Start |
| Find intersection | Two pointers | - |

### Decision Tree

```
         Interval Problem
               ↓
    ┌──────────┼──────────┐
    │          │          │
  Merge    Selection   Counting
    ↓          ↓          ↓
Sort by    Sort by     Sweep
 Start      End        Line
```

---

<a name="checklist"></a>
## 8. Interview Preparation Checklist

### Before the Interview

**Master the fundamentals:**
- [ ] Can merge intervals from memory
- [ ] Know when to sort by start vs end
- [ ] Understand sweep line technique
- [ ] Can detect overlap in O(1)

**Know the patterns:**
- [ ] Merge intervals
- [ ] Insert interval
- [ ] Non-overlapping (sort by end)
- [ ] Meeting rooms (sweep/heap)
- [ ] Interval intersection

**Common problems solved:**
- [ ] LC 56: Merge Intervals
- [ ] LC 57: Insert Interval
- [ ] LC 435: Non-overlapping Intervals
- [ ] LC 253: Meeting Rooms II
- [ ] LC 986: Interval Intersection
- [ ] LC 452: Minimum Arrows

### During the Interview

**1. Clarify (30 seconds)**
- Inclusive or exclusive boundaries?
- Already sorted?
- Touching = overlapping?

**2. Choose approach (30 seconds)**
- Merge → sort by start
- Selection → sort by end
- Concurrent count → sweep line

**3. Code (3-4 minutes)**
- Sort if needed
- Process intervals
- Return result

**4. Test (1-2 minutes)**
- Empty input
- Single interval
- All overlapping
- None overlapping

---

## 9. Quick Reference Cards

### Merge Intervals
```python
intervals.sort(key=lambda x: x[0])
merged = [intervals[0]]
for curr in intervals[1:]:
    if curr[0] <= merged[-1][1]:
        merged[-1][1] = max(merged[-1][1], curr[1])
    else:
        merged.append(curr)
```

### Non-overlapping (Max Keep)
```python
intervals.sort(key=lambda x: x[1])
count, end = 0, float('-inf')
for s, e in intervals:
    if s >= end:
        count += 1
        end = e
```

### Sweep Line
```python
events = []
for s, e in intervals:
    events.append((s, 1))
    events.append((e, -1))
events.sort()
```

---

## 10. Complexity Reference

| Pattern | Time | Space |
|---------|------|-------|
| Merge | O(n log n) | O(n) |
| Insert | O(n) | O(n) |
| Non-overlapping | O(n log n) | O(1) |
| Meeting Rooms | O(n log n) | O(n) |
| Intersection | O(m + n) | O(min) |

---

## Final Thoughts

**Remember:**
1. Most interval problems need sorting first
2. Sort by start for merging, by end for selection
3. Sweep line for counting concurrent
4. Two pointers for intersection of sorted lists
5. Greedy works for most interval problems

**When stuck:**
1. Draw intervals on number line
2. Ask: "What should I sort by?"
3. Try greedy approach
4. Consider sweep line for counting

---

## Appendix: Practice Problem Set

### Easy
- 252. Meeting Rooms

### Medium
- 56. Merge Intervals
- 57. Insert Interval
- 253. Meeting Rooms II
- 435. Non-overlapping Intervals
- 452. Minimum Arrows to Burst Balloons
- 729. My Calendar I
- 731. My Calendar II
- 986. Interval List Intersections
- 1024. Video Stitching
- 1288. Remove Covered Intervals

### Hard
- 352. Data Stream as Disjoint Intervals
- 732. My Calendar III
- 759. Employee Free Time

**Recommended Practice Order:**
1. Basic: 56 → 57 → 252
2. Selection: 435 → 452
3. Counting: 253 → 732
4. Advanced: 986 → 1024 → 759

Good luck with your interview preparation!
