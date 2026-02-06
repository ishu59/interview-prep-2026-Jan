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
    if not intervals:
        return []

    # Sort by start time
    intervals.sort(key=lambda x: x[0])

    merged = [intervals[0]]

    for current in intervals[1:]:
        last = merged[-1]

        if current[0] <= last[1]:  # Overlapping
            last[1] = max(last[1], current[1])
        else:  # Not overlapping
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

    # Add all intervals that end before new one starts
    while i < n and intervals[i][1] < newInterval[0]:
        result.append(intervals[i])
        i += 1

    # Merge overlapping intervals
    while i < n and intervals[i][0] <= newInterval[1]:
        newInterval[0] = min(newInterval[0], intervals[i][0])
        newInterval[1] = max(newInterval[1], intervals[i][1])
        i += 1
    result.append(newInterval)

    # Add remaining intervals
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

    # Sort by end time
    intervals.sort(key=lambda x: x[1])

    count = 0
    prev_end = float('-inf')

    for start, end in intervals:
        if start >= prev_end:  # No overlap
            prev_end = end
        else:  # Overlap, remove current (keep previous)
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
        events.append((start, 1))   # Start event
        events.append((end, -1))    # End event

    # Sort by time, end events before start events at same time
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

    # Heap tracks end times of meetings in progress
    heap = []

    for start, end in intervals:
        # If earliest ending meeting is done, reuse that room
        if heap and heap[0] <= start:
            heapq.heappop(heap)
        heapq.heappush(heap, end)

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

    while i < len(firstList) and j < len(secondList):
        a_start, a_end = firstList[i]
        b_start, b_end = secondList[j]

        # Find overlap
        start = max(a_start, b_start)
        end = min(a_end, b_end)

        if start <= end:
            result.append([start, end])

        # Move pointer of interval that ends first
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

    while current_end < target_end:
        # Find interval that starts <= current_end and extends farthest
        max_end = current_end
        while i < len(intervals) and intervals[i][0] <= current_end:
            max_end = max(max_end, intervals[i][1])
            i += 1

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
        # If empty or no overlap with last
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            # Merge with last
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

    # Add intervals completely before newInterval
    while i < n and intervals[i][1] < newInterval[0]:
        result.append(intervals[i])
        i += 1

    # Merge overlapping intervals
    while i < n and intervals[i][0] <= newInterval[1]:
        newInterval = [
            min(newInterval[0], intervals[i][0]),
            max(newInterval[1], intervals[i][1])
        ]
        i += 1
    result.append(newInterval)

    # Add remaining intervals
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

    # Sort by end time
    intervals.sort(key=lambda x: x[1])

    removals = 0
    prev_end = intervals[0][1]

    for i in range(1, len(intervals)):
        if intervals[i][0] < prev_end:  # Overlap
            removals += 1
        else:
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

    while i < n or heap:
        if not heap:
            day = events[i][0]

        # Add all events starting today
        while i < n and events[i][0] <= day:
            heapq.heappush(heap, events[i][1])
            i += 1

        # Remove events that have ended
        while heap and heap[0] < day:
            heapq.heappop(heap)

        # Attend event ending earliest
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
    heap = []  # End times

    for start, end in intervals:
        if heap and heap[0] <= start:
            heapq.heappop(heap)
        heapq.heappush(heap, end)

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

    # Merge to find busy times
    merged = [all_intervals[0]]
    for interval in all_intervals[1:]:
        if interval[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], interval[1])
        else:
            merged.append(interval)

    # Gaps between merged = free time
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

    while i < len(firstList) and j < len(secondList):
        # Find intersection
        start = max(firstList[i][0], secondList[j][0])
        end = min(firstList[i][1], secondList[j][1])

        if start <= end:
            result.append([start, end])

        # Move pointer with smaller end
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

    # Sort by end position
    points.sort(key=lambda x: x[1])

    arrows = 1
    current_end = points[0][1]

    for start, end in points[1:]:
        if start > current_end:  # Need new arrow
            arrows += 1
            current_end = end
        # else: current arrow covers this balloon

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

    while current_end < time:
        # Find clip that starts <= current_end and extends farthest
        while i < len(clips) and clips[i][0] <= current_end:
            farthest = max(farthest, clips[i][1])
            i += 1

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
    if len(nums) <= 1:
        return 0

    jumps = 0
    current_end = 0
    farthest = 0

    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])

        if i == current_end:
            jumps += 1
            current_end = farthest

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
