# The Complete Binary Search Handbook
> A template-based approach for mastering binary search in coding interviews

**Philosophy:** Binary search is not about finding a value. It's about **partitioning a search space** and eliminating half of it with each decision.

---

## Table of Contents
1. [Understanding the Core Philosophy](#core-philosophy)
2. [The Two Master Templates](#master-templates)
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

- **The Phone Book Rip**: When you look up "Smith" in a phone book, you don't start at page 1. You open roughly to the middle, see "M", and rip away the wrong half. Each rip halves the remaining pages.
- **The Yes/No Boundary**: Every binary search is really finding the boundary where the answer flips from "no" to "yes" (or vice versa). The sorted order guarantees this boundary exists in exactly one place.

### No-Jargon Translation

- **Search space**: the chunk of possibilities still in play
- **Monotonic condition**: a yes/no test that once it flips, stays flipped
- **lo/hi pointers**: bookmarks for the left and right edges of what's left
- **mid**: the page you open to next

### Mental Model

> "Binary search is a game of 20 Questions where every answer eliminates exactly half the remaining possibilities."

---

### Why Do We Struggle with Binary Search?

The #1 reason: **Off-by-one errors** and confusion about loop conditions.
- Should I use `lo <= hi` or `lo < hi`?
- Should I update with `hi = mid` or `hi = mid - 1`?
- When do I return `lo` vs `hi` vs `mid`?

### The Solution: Think in Terms of "Search Space"

Instead of thinking "find the target", think:
> **"I'm maintaining a window `[lo, hi]` that MIGHT contain my answer. Each iteration, I eliminate half the space that DEFINITELY doesn't contain the answer."**

This leads to our core principles:

#### Principle 1: Loop Until One Element Remains

```python
# Why `lo < hi` not `lo <= hi`?
# With `lo <= hi`, you must handle returns inside the loop — easy to mess up.
# With `lo < hi`, the loop ends when lo == hi: a single candidate to validate.
# Simpler post-processing, fewer off-by-one errors.
while lo < hi:
```

#### Principle 2: Never Lose a Valid Candidate

```python
if condition:
    hi = mid      # Why `hi = mid` not `hi = mid - 1`?
                  # mid MIGHT be the answer — never drop a valid candidate.
                  # Use `hi = mid - 1` ONLY when you are CERTAIN mid cannot win.
else:
    lo = mid + 1  # mid is DEFINITELY not the answer, so safely exclude it.
```

#### Principle 3: Avoid Integer Overflow

```python
# Why `lo + (hi - lo) // 2` not `(lo + hi) // 2`?
# Python integers don't overflow, but in Java/C++, (lo + hi) overflows when
# lo + hi > 2,147,483,647. This formula is equivalent but safe in all languages.
mid = lo + (hi - lo) // 2  # Safe
# NOT: (lo + hi) // 2     <- Overflows in statically-typed languages!
```

---

<a name="master-templates"></a>
## 2. The Two Master Templates

There are fundamentally **TWO** templates you need to master. The key difference is the **direction** you're searching.

### Template A: MINIMIZE (Find First / Lower Bound)

**Use when:** Finding the FIRST valid element, or MINIMIZING a parameter

```python
def binary_search_minimize(lo: int, hi: int) -> int:
    # Why `lo < hi`?
    # Loop terminates when lo == hi — one candidate remaining.
    # No ambiguity about what to return: always return lo (== hi).
    while lo < hi:
        # Why left-biased mid (no +1)?
        # When 2 elements remain (lo=2, hi=3): mid = 2 (LEFT element).
        # With `hi = mid`: hi becomes 2, window shrinks.
        # With `lo = mid + 1`: lo becomes 3, window shrinks.
        # No infinite loop possible — `hi = mid` ALWAYS shrinks the window.
        mid = lo + (hi - lo) // 2

        if is_valid(mid):
            # Why `hi = mid` not `hi = mid - 1`?
            # mid IS valid. There might be a SMALLER valid value to its left.
            # Keep mid in the search space — we don't know if it's the first.
            hi = mid
        else:
            # mid is NOT valid. Answer must be strictly LARGER than mid.
            # Safe to exclude mid entirely.
            lo = mid + 1

    # lo == hi: this is our single remaining candidate.
    return lo
```

#### Visual: How it works
```
[X X X V V V V V]  <- We want the FIRST V (valid)
          ^
       We keep shrinking toward the LEFT boundary
       hi = mid keeps valid candidates
```

---

### Template B: MAXIMIZE (Find Last / Upper Bound)

**Use when:** Finding the LAST valid element, or MAXIMIZING a parameter

```python
def binary_search_maximize(lo: int, hi: int) -> int:
    while lo < hi:
        # Why RIGHT-biased mid (+1)? CRITICAL -- avoids infinite loop!
        #
        # WITHOUT +1: lo=2, hi=3 -> mid = 2 + (3-2)//2 = 2 (LEFT)
        #   If valid: lo = mid = 2. Next iter: lo=2, hi=3 -> SAME STATE -> INFINITE LOOP
        #
        # WITH +1:    lo=2, hi=3 -> mid = 2 + (3-2+1)//2 = 3 (RIGHT)
        #   If valid: lo = mid = 3. Next iter: lo=3, hi=3 -> loop exits
        #
        # Rule: `lo = mid` REQUIRES right-biased mid. No exceptions!
        mid = lo + (hi - lo + 1) // 2

        if is_valid(mid):
            # mid IS valid. There might be a LARGER valid value to its right.
            # Keep mid in the search space — we don't know if it's the last.
            lo = mid
        else:
            # mid is NOT valid. Answer must be strictly SMALLER than mid.
            # Safe to exclude mid entirely.
            hi = mid - 1

    # lo == hi: this is our single remaining candidate.
    return lo
```

#### Visual: How it works
```
[V V V V V X X X]  <- We want the LAST V (valid)
              ^
       We keep shrinking toward the RIGHT boundary
       lo = mid keeps valid candidates
```

---

### Quick Decision Matrix

| What are you looking for? | Template | Mid Calculation | Update Logic |
|---------------------------|----------|-----------------|--------------|
| **First** occurrence | A (Minimize) | `lo + (hi-lo)//2` | `if valid: hi=mid else: lo=mid+1` |
| **Last** occurrence | B (Maximize) | `lo + (hi-lo+1)//2` | `if valid: lo=mid else: hi=mid-1` |
| **Minimize** parameter | A (Minimize) | `lo + (hi-lo)//2` | `if works: hi=mid else: lo=mid+1` |
| **Maximize** parameter | B (Maximize) | `lo + (hi-lo+1)//2` | `if works: lo=mid else: hi=mid-1` |

**Memory Aid:**
- Moving LEFT (`hi = mid`) -> Use LEFT-biased mid (no +1)
- Moving RIGHT (`lo = mid`) -> Use RIGHT-biased mid (+1)

---

<a name="pattern-guide"></a>
## 3. Pattern Classification Guide

Binary search problems fall into these categories:

### Category 1: Index Search (Searching in Arrays)
- You have an array (possibly modified)
- You're searching for an **index** or **value**
- Examples: Find first/last occurrence, rotated array, peak finding

### Category 2: Answer Search (Binary Search on Parameter)
- You're searching for a **parameter value** (capacity, speed, distance)
- No actual array of that parameter exists
- You have a **feasibility function**: "Can I achieve the goal with value X?"
- Examples: Minimum ship capacity, maximum distance, square root

### Category 3: Count-Based Search
- You have a structure where you can **count** elements
- You're finding the K-th element
- Examples: K-th smallest in matrix, K-th pair distance

### Category 4: 2D Space Search
- Searching in a 2D matrix or grid
- Can often be converted to 1D search
- Examples: Search 2D matrix, search in row-col sorted matrix

**Recognition Questions:**
1. **Is there a sorted array?** -> Likely Category 1
2. **Question asks "minimum X such that..." or "maximum X such that..."?** -> Likely Category 2
3. **Question asks for "K-th smallest/largest"?** -> Likely Category 3
4. **Involves a matrix?** -> Likely Category 4

---

<a name="patterns"></a>
## 4. Complete Pattern Library

### PATTERN 1: Finding Boundaries in Sorted Arrays

---

#### Pattern 1A: First Occurrence (Lower Bound)

**Problem:** Find the FIRST index where `nums[i] == target`

**Example:** `[1, 2, 2, 2, 3]`, target = `2` -> Answer: `1`

**Which Template?** Template A (Minimize) - finding FIRST

```python
def first_occurrence(nums: list[int], target: int) -> int:
    lo, hi = 0, len(nums) - 1

    while lo < hi:
        mid = lo + (hi - lo) // 2  # Left-biased

        # Why `nums[mid] >= target` not `nums[mid] == target`?
        # We want the FIRST position where value >= target.
        # Using `==` creates three cases (less, equal, greater) -- messy.
        # Using `>=` gives a clean two-way partition:
        #   "At or past the boundary?" -> hi = mid (search left)
        #   "Before the boundary?" -> lo = mid + 1 (search right)
        # The exact match is captured naturally when lo converges on it.
        if nums[mid] >= target:
            hi = mid      # mid might be first, or first is to the left
        else:
            lo = mid + 1  # mid is too small, go right

    # Why TWO checks in post-processing?
    # (1) `lo < len(nums)` guards against target being larger than all elements,
    #     where lo converges to last index but no actual match exists.
    # (2) `nums[lo] == target` confirms we found the target, not merely the
    #     first element >= target (which could be a different, larger value).
    if lo < len(nums) and nums[lo] == target:
        return lo
    return -1
```

**Why `nums[mid] >= target`?**

Because we're finding a **boundary**:
```
[1, 2, 2, 2, 3]  target = 2
 X  V  V  V  X
    ^
  First >= target is what we want!
```

If we use `nums[mid] == target`, we'd have three cases to handle. Using `>=` gives us a clean two-way split.

---

#### Pattern 1B: Last Occurrence (Upper Bound - 1)

**Problem:** Find the LAST index where `nums[i] == target`

**Example:** `[1, 2, 2, 2, 3]`, target = `2` -> Answer: `3`

**Which Template?** Template B (Maximize) - finding LAST

```python
def last_occurrence(nums: list[int], target: int) -> int:
    lo, hi = 0, len(nums) - 1

    while lo < hi:
        mid = lo + (hi - lo + 1) // 2  # Right-biased

        # Why `nums[mid] <= target` not `nums[mid] == target`?
        # We want the LAST position where value <= target.
        # `<=` gives a clean two-way partition:
        #   "At or before the boundary?" -> lo = mid (search right)
        #   "Past the boundary?" -> hi = mid - 1 (search left)
        # The exact match is captured naturally when lo converges on it.
        if nums[mid] <= target:
            lo = mid      # mid might be last, or last is to the right
        else:
            hi = mid - 1  # mid is too large, go left

    # Why `lo >= 0 and nums[lo] == target`?
    # `lo >= 0` is defensive -- lo is always valid in this setup.
    # `nums[lo] == target` confirms we found the target, not merely the
    # last element <= target (which could be a smaller, non-target value).
    if lo >= 0 and nums[lo] == target:
        return lo
    return -1
```

**Why `nums[mid] <= target`?**

We're finding the RIGHT boundary:
```
[1, 2, 2, 2, 3]  target = 2
 X  V  V  V  X
          ^
     Last <= target is what we want!
```

---

#### Pattern 1C: Insert Position

**Problem:** LeetCode 35 - Find index where target should be inserted

**Example:** `[1, 3, 5, 6]`, target = `2` -> Answer: `1`

**Which Template?** Template A (Minimize) - finding first position >= target

```python
def search_insert(nums: list[int], target: int) -> int:
    lo, hi = 0, len(nums) - 1

    while lo < hi:
        mid = lo + (hi - lo) // 2

        if nums[mid] >= target:
            hi = mid
        else:
            lo = mid + 1

    # Why post-processing is different from exact match?
    # Insert position ALWAYS has a valid answer (even beyond the end).
    # If target is larger than all elements, lo converges to the last index
    # but nums[lo] < target -- the correct insert position is one past the end.
    # Example: [1,3,5], target=6 -> lo=2, nums[2]=5 < 6 -> return 3.
    if lo < len(nums) and nums[lo] < target:
        return lo + 1
    return lo
```

---

#### Pattern 1D: Search Range (Both Bounds)

**Problem:** LeetCode 34 - Find both first and last occurrence

**Example:** `[5, 7, 7, 8, 8, 10]`, target = `8` -> Answer: `[3, 4]`

**Solution:** Combine Pattern 1A + 1B

```python
def search_range(nums: list[int], target: int) -> list[int]:
    # Why guard against empty array separately?
    # With len 0, lo=0, hi=-1: while loop never runs and lo=0 may look valid.
    # Fail fast to avoid a misleading candidate check.
    if not nums:
        return [-1, -1]

    # Find first occurrence (Template A)
    first = first_occurrence(nums, target)

    # Why check `first == -1` before searching for last?
    # If the target doesn't exist, running a second binary search is wasted work.
    # Early exit saves O(log n) and is a correctness guard.
    if first == -1:
        return [-1, -1]

    # Find last occurrence (Template B)
    last = last_occurrence(nums, target)

    return [first, last]
```

**Interview Tip:** This demonstrates that you understand BOTH templates!

---

### PATTERN 2: Rotated Sorted Array

---

#### Pattern 2A: Without Duplicates (Clean O(log n))

**Problem:** LeetCode 33 - Search in `[4, 5, 6, 7, 0, 1, 2]`

**Key Insight:** At any point, one half is ALWAYS sorted. Use that sorted half to make decisions.

```python
def search_rotated(nums: list[int], target: int) -> int:
    lo, hi = 0, len(nums) - 1

    while lo < hi:
        mid = lo + (hi - lo) // 2

        # Why `nums[lo] <= nums[mid]` with `<=` not `<`?
        # When only 2 elements remain, lo == mid. nums[lo] == nums[mid] trivially.
        # The left "half" is a single element -- trivially sorted.
        # Using `<` would wrongly classify this 1-element half as "unsorted".
        if nums[lo] <= nums[mid]:
            # LEFT half [lo...mid] is sorted.

            # Why double-bound check `nums[lo] <= target <= nums[mid]`?
            # The left half is sorted, so we can reliably range-check it.
            # If target falls within [nums[lo], nums[mid]], it's there.
            # If not, it MUST be in the other (possibly unsorted) half.
            if nums[lo] <= target <= nums[mid]:
                hi = mid        # Target in sorted left half
            else:
                lo = mid + 1    # Target in rotated right half
        else:
            # RIGHT half [mid+1...hi] is sorted.

            # Why `nums[mid] < target` (strict) but `target <= nums[hi]`?
            # Strict `<` at mid: if nums[mid] equaled target, the left-half
            # branch would have caught it -- we've moved past mid.
            # `<=` at hi: hi is inclusive in our search window.
            # Together: target lies in the open-closed range (nums[mid], nums[hi]].
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1    # Target in sorted right half
            else:
                hi = mid        # Target in rotated left half

    # Why validate after convergence?
    # Binary search narrows to a candidate -- it doesn't guarantee target exists.
    return lo if nums[lo] == target else -1
```

**Visual:**
```
[4, 5, 6, 7, 0, 1, 2]
 ^        ^        ^
lo      mid       hi

nums[lo]=4 <= nums[mid]=7 -> Left half is sorted!
If target is in [4,7], search left. Otherwise, search right.
```

---

#### Pattern 2B: With Duplicates (Worst Case O(n))

**Problem:** LeetCode 81 - Search in `[1, 0, 1, 1, 1]`

**Challenge:** When `nums[lo] == nums[mid] == nums[hi]`, we can't determine which half is sorted.

```python
def search_rotated_duplicates(nums: list[int], target: int) -> bool:
    lo, hi = 0, len(nums) - 1

    while lo < hi:
        mid = lo + (hi - lo) // 2

        # Why shrink BOTH ends when all three are equal?
        # If nums[lo] == nums[mid] == nums[hi], we cannot tell which half
        # contains the rotation point. Example: [1,1,1,0,1] -- rotation could
        # be on either side of mid. Shrinking only one end might skip the answer.
        # Trimming one element from each end safely reduces the window:
        # the answer can't be at lo or hi (they equal mid, re-checked next iter).
        if nums[lo] == nums[mid] == nums[hi]:
            lo += 1
            hi -= 1
            continue  # Re-evaluate with smaller window

        # Same logic as Pattern 2A from here
        if nums[lo] <= nums[mid]:
            if nums[lo] <= target <= nums[mid]:
                hi = mid
            else:
                lo = mid + 1
        else:
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1
            else:
                hi = mid

    return nums[lo] == target
```

**Why can't we do better than O(n)?**

Consider: `[1, 1, 1, 1, 0, 1, 1, 1]`, target = `0`
- Almost all elements are `1`
- We can't eliminate half the array without checking
- Fundamental limitation, not a bug in our algorithm!

**Interview Tip:** Mention this is a known limitation. Shows depth of understanding.

---

### PATTERN 3: Binary Search on Answer Space

**Core Concept:** We're not searching an array. We're searching a **range of possible answers**.

**Mental Model:**
```
Capacity:  [10, 11, 12, 13, 14, 15, 16, ...]
Can ship?  [ N   N   N   Y   Y   Y   Y  ...]
                      ^
               Find this boundary!
```

**Template Structure:**
1. Define `lo` = minimum possible answer
2. Define `hi` = maximum possible answer
3. Write `can_achieve(x)` feasibility function
4. Use Template A (minimize) or B (maximize)

---

#### Pattern 3A: Minimize Maximum

**Use when:** "Find the minimum X such that we can achieve the goal"

**Example:** LeetCode 410 - Split Array Largest Sum

**Problem:** Split `[7,2,5,10,8]` into `3` subarrays to minimize the maximum sum.

```python
def split_array(nums: list[int], m: int) -> int:
    # Why `lo = max(nums)`?
    # Any valid capacity must be >= the largest single element, otherwise
    # that element alone exceeds the limit and no valid split exists.
    lo = max(nums)
    # Why `hi = sum(nums)`?
    # Putting everything in one subarray is always feasible -- it's the
    # upper bound. We never need capacity larger than the total sum.
    hi = sum(nums)

    # Binary search on capacity (Template A -- minimize)
    while lo < hi:
        mid = lo + (hi - lo) // 2

        if can_split(nums, mid, m):
            hi = mid        # This capacity works, try smaller
        else:
            lo = mid + 1    # Too small, need larger capacity

    return lo


def can_split(nums: list[int], max_sum: int, m: int) -> bool:
    """Greedy check: can we split into <= m subarrays with max_sum limit?"""
    subarrays = 1
    current_sum = 0

    for num in nums:
        # Why `current_sum + num > max_sum` (strict `>`) not `>=`?
        # A subarray sum exactly equal to max_sum is VALID -- don't open a new one.
        # Only start a new subarray when adding `num` would EXCEED the limit.
        if current_sum + num > max_sum:
            subarrays += 1
            current_sum = num
            # Why early return when `subarrays > m`?
            # If we already need more than m subarrays with elements remaining,
            # there's no way to fit the rest -- fail fast, skip unnecessary work.
            if subarrays > m:
                return False
        else:
            current_sum += num

    return True
```

**Why this is Template A (Minimize):**
- We want the MINIMUM capacity that works
- `can_split(14)` works -> Try `can_split(13)`, `can_split(12)`...
- This is finding the FIRST valid value -> Template A

---

#### Pattern 3B: Maximize Minimum

**Use when:** "Find the maximum X such that we can achieve the goal"

**Example:** LeetCode 1552 - Magnetic Force Between Two Balls

**Problem:** Place `m` balls in `positions = [1,2,3,4,7]` to maximize minimum distance.

```python
def max_distance(position: list[int], m: int) -> int:
    position.sort()

    # lo = 1: minimum possible distance between any two positions
    # hi = spread: maximum possible distance across all positions
    lo = 1
    hi = position[-1] - position[0]

    # Binary search on distance (Template B -- maximize)
    while lo < hi:
        mid = lo + (hi - lo + 1) // 2  # Right-biased

        if can_place(position, mid, m):
            lo = mid        # This distance works, try larger
        else:
            hi = mid - 1    # Too large, need smaller distance

    return lo


def can_place(position: list[int], min_dist: int, m: int) -> bool:
    """Greedy check: can we place m balls with at least min_dist apart?"""
    count = 1           # Greedily place first ball at leftmost position
    last_pos = position[0]

    # Why start at index 1?
    # We already placed the first ball at position[0] (leftmost after sorting).
    # Check remaining positions starting from the second one.
    for i in range(1, len(position)):
        # Why `position[i] - last_pos >= min_dist` with `>=` not `>`?
        # Exactly min_dist apart IS valid -- that's the minimum distance constraint.
        # Using `>` would wrongly reject placements that exactly meet the requirement.
        if position[i] - last_pos >= min_dist:
            count += 1
            last_pos = position[i]
            # Why `count >= m` not `count == m`?
            # Once we've placed m balls, we're done -- no need to check further.
            if count >= m:
                return True

    return False
```

**Why this is Template B (Maximize):**
- We want the MAXIMUM distance that works
- `can_place(3)` works -> Try `can_place(4)`, `can_place(5)`...
- This is finding the LAST valid value -> Template B
- **Critical:** Must use RIGHT-biased mid!

---

#### Pattern 3C: Classic Examples

**Square Root (LC 69)**

Finding `sqrt(8)` = `2` (floor value)

```python
def my_sqrt(x: int) -> int:
    # Why handle x < 2 separately?
    # For x=0: sqrt=0. For x=1: sqrt=1. Both trivially equal x.
    # Also, hi = x // 2 would be 0 for x=1, making hi < lo -- loop never runs.
    if x < 2:
        return x

    lo = 1
    # Why `hi = x // 2`?
    # For any x >= 4, sqrt(x) <= x/2. Searching up to x itself is wasteful.
    # Proof: if k = x/2, then k^2 = x^2/4 >= x when x >= 4.
    hi = x // 2

    while lo < hi:
        mid = lo + (hi - lo + 1) // 2  # Template B (maximize)

        # Why `mid <= x // mid` instead of `mid * mid <= x`?
        # `mid * mid` can overflow 32-bit integers in other languages.
        # `mid <= x // mid` is algebraically equivalent (checks mid^2 <= x)
        # but uses division -- safe from overflow in any language.
        if mid <= x // mid:
            lo = mid        # mid^2 <= x, try larger
        else:
            hi = mid - 1    # mid^2 > x, too large

    return lo
```

**Why Template B?** We want the LARGEST number whose square <= x.

---

**Koko Eating Bananas (LC 875)**

```python
def min_eating_speed(piles: list[int], h: int) -> int:
    lo = 1
    hi = max(piles)  # Fastest useful speed: eat the largest pile in one hour

    while lo < hi:
        mid = lo + (hi - lo) // 2  # Template A (minimize)

        if can_finish(piles, mid, h):
            hi = mid        # This speed works, try slower
        else:
            lo = mid + 1    # Too slow, need faster

    return lo


def can_finish(piles: list[int], speed: int, h: int) -> bool:
    hours = 0
    for pile in piles:
        # Why ceiling division `(pile + speed - 1) // speed`?
        # Koko spends a WHOLE hour on each pile, even if she finishes early.
        # A pile of 7 at speed 3 takes ceil(7/3) = 3 hours, not 2.
        # Formula: ceil(a/b) = (a + b - 1) // b for positive integers.
        # Avoids floating point entirely -- pure integer arithmetic.
        hours += (pile + speed - 1) // speed

    # Why `hours <= h` with `<=` not `<`?
    # Koko CAN finish early. Fewer hours than h is perfectly acceptable.
    return hours <= h
```

**Why Template A?** We want the MINIMUM speed that works.

---

### PATTERN 4: K-th Smallest (Count-Based Search)

**Core Concept:** Instead of indexing, we pick a VALUE and ask: "How many elements are <= this value?"

**When to use:** Finding K-th smallest in a matrix, multiplication table, or pair distances.

**Example:** LeetCode 378 - K-th Smallest Element in Sorted Matrix

```python
def kth_smallest(matrix: list[list[int]], k: int) -> int:
    n = len(matrix)
    lo = matrix[0][0]       # Smallest possible value
    hi = matrix[n-1][n-1]   # Largest possible value

    while lo < hi:
        mid = lo + (hi - lo) // 2  # Template A

        count = count_less_equal(matrix, mid)

        # Why `count < k` (strict `<`) not `count <= k`?
        # If count == k, exactly k values are <= mid.
        # mid COULD be the k-th smallest, or a smaller value might be.
        # By using `hi = mid`, we search LEFT for the smallest value with count >= k.
        # Using `count <= k` would push lo past the true answer when count == k.
        if count < k:
            lo = mid + 1    # Not enough values <= mid, need larger
        else:
            hi = mid        # Have >= k values, try smaller

    return lo


def count_less_equal(matrix: list[list[int]], target: int) -> int:
    """Count elements <= target in a row-col sorted matrix."""
    n = len(matrix)
    count = 0

    # Why start at bottom-left corner (row=n-1, col=0)?
    # It's the unique position where moving RIGHT increases value
    # and moving UP decreases value. This gives binary-search-like
    # navigation of 2D space without backtracking.
    row, col = n - 1, 0

    # Why two loop conditions `row >= 0 and col < n`?
    # `row >= 0`: stops us from going above the top of the matrix.
    # `col < n`: stops us from going past the right edge.
    # Together they keep us in-bounds as we staircase through.
    while row >= 0 and col < n:
        # Why `<= target` and add `row + 1` elements at once?
        # If matrix[row][col] <= target, ALL elements above it in the same
        # column are also <= target (column sorted smallest at top).
        # We count the entire column slice in one O(1) step.
        if matrix[row][col] <= target:
            count += row + 1  # All elements from top of column down to row
            col += 1          # Move right: next column to check
        else:
            row -= 1          # Too large, move up to smaller values

    return count
```

**Key insight:** We're searching in the VALUE space, not index space!

---

### PATTERN 5: Peak Finding

**Core Concept:** In a mountain/bitonic array, follow the slope upward.

**Example:** LeetCode 162 - Find Peak Element

```python
def find_peak_element(nums: list[int]) -> int:
    lo, hi = 0, len(nums) - 1

    while lo < hi:
        mid = lo + (hi - lo) // 2

        # Why is accessing `nums[mid + 1]` safe here?
        # `lo < hi` guarantees mid < hi, so mid + 1 <= hi -- always valid.
        # No out-of-bounds risk as long as the loop condition holds.
        #
        # Why `nums[mid] < nums[mid + 1]`?
        # If the slope is ascending (mid+1 is higher), mid CANNOT be a peak.
        # A peak requires being higher than BOTH neighbors.
        # The peak must therefore exist somewhere to the RIGHT of mid.
        if nums[mid] < nums[mid + 1]:
            lo = mid + 1    # Ascending: peak is to the right
        else:
            # Why `hi = mid` not `hi = mid - 1`?
            # Descending slope means mid COULD be the peak (its left neighbor
            # is guaranteed smaller by the problem's "-infinity" boundary).
            # Keep mid in the search space -- don't drop a valid candidate.
            hi = mid        # Descending: peak is at mid or to the left

    return lo  # lo == hi is the peak index
```

**Why this works:**
```
      /\
     /  \
    /    \___

If mid is ascending (nums[mid] < nums[mid+1]):
    We're going up, peak must be to the right!
```

**Interview Tip:** This works even if there are multiple peaks. We're guaranteed to find ONE peak.

---

### PATTERN 6: 2D Matrix Binary Search

**Core Concept:** Treat a 2D matrix as a flattened 1D array.

**Example:** LeetCode 74 - Search 2D Matrix

**Matrix structure:**
```
[ 1,  3,  5,  7]
[10, 11, 16, 20]
[23, 30, 34, 60]
```
Treat as: `[1, 3, 5, 7, 10, 11, 16, 20, 23, 30, 34, 60]`

```python
def search_matrix(matrix: list[list[int]], target: int) -> bool:
    # Why guard against empty matrix or empty rows?
    # Accessing matrix[0] on an empty matrix raises IndexError.
    if not matrix or not matrix[0]:
        return False

    m, n = len(matrix), len(matrix[0])
    lo = 0
    # Why `m * n - 1`?
    # We treat the 2D matrix as a flat 1D array of length m*n.
    # The last valid 0-indexed position is m*n - 1.
    hi = m * n - 1

    while lo < hi:
        mid = lo + (hi - lo) // 2

        # Why `mid // n` for row and `mid % n` for column?
        # Think of the flat array "wrapped" every n elements.
        # Division: how many complete rows have we passed? -> row index.
        # Modulo: how far along the current row? -> column index.
        # Example: index 7 in a 3x4 matrix -> row=7//4=1, col=7%4=3.
        row = mid // n
        col = mid % n

        # Why `>= target` (Template A pattern)?
        # We're finding the FIRST position where value >= target.
        # If it equals target we found it. If greater, target might be left.
        if matrix[row][col] >= target:
            hi = mid
        else:
            lo = mid + 1

    # Check the single converged candidate
    row, col = lo // n, lo % n
    return matrix[row][col] == target
```

**Why the conversion works:**
```
Index:  0  1  2  3  4   5   6   7   8   9  10  11
Array: [1, 3, 5, 7, 10, 11, 16, 20, 23, 30, 34, 60]

Index 7 -> row=7//4=1, col=7%4=3 -> matrix[1][3]=20
```

---

<a name="post-processing"></a>
## 5. Post-Processing Reference

After the loop ends with `lo == hi`, you need to validate/return based on problem type:

| Problem Type | Post-Processing | Why |
|--------------|-----------------|-----|
| **Exact Match** | `return lo if nums[lo] == target else -1` | Answer might not exist |
| **First Occurrence** | `return lo if lo < n and nums[lo] == target else -1` | Need bounds check + match |
| **Last Occurrence** | `return lo if lo >= 0 and nums[lo] == target else -1` | Need bounds check + match |
| **Insert Position** | `return lo + 1 if lo < n and nums[lo] < target else lo` | Might insert at end |
| **BS on Answer** | `return lo` | Boundary always exists |
| **Peak Finding** | `return lo` | Peak always exists |
| **K-th Smallest** | `return lo` | We found the k-th value |
| **Rotated Array** | `return lo if nums[lo] == target else -1` | Answer might not exist |

**Key Principle:** Different problems have different guarantees about answer existence!

---

<a name="pitfalls"></a>
## 6. Common Pitfalls & Solutions

### Pitfall 1: Infinite Loop with Wrong Mid Rounding

**WRONG:**
```python
while lo < hi:
    mid = lo + (hi - lo) // 2  # Left-biased
    if condition:
        lo = mid  # DANGER!
    else:
        hi = mid - 1
# When lo=2, hi=3: mid=2, then lo=2 again -> INFINITE LOOP
```

**CORRECT:**
```python
while lo < hi:
    mid = lo + (hi - lo + 1) // 2  # Right-biased
    if condition:
        lo = mid  # Now safe -- mid picks the RIGHT element when 2 remain
    else:
        hi = mid - 1
```

**Rule:** `lo = mid` requires RIGHT-biased mid. No exceptions!

---

### Pitfall 2: Losing Valid Answer

**WRONG:**
```python
if nums[mid] >= target:
    hi = mid - 1  # Lost the answer if nums[mid] == target!
else:
    lo = mid + 1
```

**CORRECT:**
```python
if nums[mid] >= target:
    hi = mid  # Keep mid in search space -- it might be the answer
else:
    lo = mid + 1
```

**Rule:** Only exclude `mid` when you're CERTAIN it can't be the answer.

---

### Pitfall 3: Integer Overflow in Mid Calculation

**WRONG:**
```python
mid = (lo + hi) // 2  # Overflows in Java/C++ if lo + hi > INT_MAX
```

**CORRECT:**
```python
mid = lo + (hi - lo) // 2  # Safe from overflow in any language
```

---

### Pitfall 4: Integer Overflow in Answer Space

**WRONG (in Java/C++):**
```python
hi = sum(weights)  # Can overflow 32-bit int in Java/C++
```

**CORRECT (Python handles big ints natively; use `long` in Java):**
```python
hi = 0
for w in weights:
    hi += w  # Explicit accumulation; in Java, declare hi as long
```

---

### Pitfall 5: Not Checking Array Bounds

**WRONG:**
```python
return lo if nums[lo] == target else -1  # What if lo >= len(nums)?
```

**CORRECT:**
```python
if lo < len(nums) and nums[lo] == target:
    return lo
return -1
```

---

### Pitfall 6: Using Wrong Template for the Problem

**Symptom:** Getting first occurrence when you need last, or vice versa.

**Solution:** Ask yourself:
- Am I finding the FIRST valid? -> Template A (left-biased)
- Am I finding the LAST valid? -> Template B (right-biased)

---

<a name="recognition"></a>
## 7. Problem Recognition Framework

### Step 1: Identify if Binary Search is Applicable

Ask: **Is there a monotonic property?**

**The IFTTT Test (If-This-Then-That):**
> "If value X works/is valid, does X+1 also work?"
> OR
> "If value X fails/is invalid, does X-1 also fail?"

**Examples:**

- Ship Capacity: If capacity 10 works -> capacity 11 works -> **Monotonic**
- Maximum Distance: If distance 5 fails -> distance 6 fails -> **Monotonic**
- Sorted Array: If arr[5] > target -> arr[6] > target -> **Monotonic**
- Unsorted Array: No guarantees -> **Not monotonic**

---

### Step 2: Identify the Search Space

| Searching for | Pattern Type | Example |
|---------------|--------------|---------|
| **Array Index** | Array Search | First/Last occurrence, rotated array |
| **Parameter Value** | Answer Search | Capacity, speed, distance |
| **K-th Element** | Count-Based | K-th smallest in matrix |
| **Peak/Valley** | Bitonic Search | Mountain peak |
| **Matrix Position** | 2D Search | Search 2D matrix |

---

### Step 3: Determine Direction (Minimize vs Maximize)

| Keywords in Problem | Direction | Template |
|---------------------|-----------|----------|
| "Find the **first**..." | Minimize | Template A (left-biased) |
| "Find the **minimum**..." | Minimize | Template A (left-biased) |
| "Lower bound" | Minimize | Template A (left-biased) |
| "Find the **last**..." | Maximize | Template B (right-biased) |
| "Find the **maximum**..." | Maximize | Template B (right-biased) |
| "Upper bound" | Maximize | Template B (right-biased) |

---

### Step 4: Design the Feasibility Function

For BS-on-answer problems, ask: **How do I check if value X works?**

**Key:** The feasibility function should be:
1. **Deterministic:** Same input -> same output
2. **Fast:** Ideally O(n) or better
3. **Correct:** Accurately determines if X works

---

### Decision Tree

```
                    Can I use Binary Search?
                           |
                   Is there monotonicity?
                           |
                   +-------+-------+
                  Yes              No
                   |                |
            What am I           Try other
            searching?          approaches
                   |
      +------------+------------+
      |            |            |
   Index        Parameter    K-th/2D
      |            |            |
      v            v            v
  Template A/B  Template A/B  Special
  (First/Last)  (Min/Max)     Pattern
```

---

<a name="checklist"></a>
## 8. Interview Preparation Checklist

### Before the Interview

**Master the fundamentals:**
- [ ] Can write Template A (minimize) from memory in Python
- [ ] Can write Template B (maximize) from memory in Python
- [ ] Understand why right-biased needs `+1` in mid
- [ ] Know when to use `hi = mid` vs `lo = mid`
- [ ] Understand the loop invariant `lo < hi`

**Practice pattern recognition:**
- [ ] Can identify monotonic properties (IFTTT test)
- [ ] Can distinguish minimize vs maximize problems
- [ ] Can map problem keywords to templates
- [ ] Know which post-processing to use

**Know the patterns:**
- [ ] First/Last occurrence
- [ ] Rotated array (with and without duplicates)
- [ ] BS on answer (minimize max, maximize min)
- [ ] K-th smallest
- [ ] Peak finding
- [ ] 2D matrix search

**Common problems solved:**
- [ ] LC 33: Search in Rotated Sorted Array
- [ ] LC 34: Find First and Last Position
- [ ] LC 410: Split Array Largest Sum
- [ ] LC 1552: Magnetic Force Between Two Balls
- [ ] LC 378: K-th Smallest in Matrix
- [ ] LC 162: Find Peak Element
- [ ] LC 74: Search 2D Matrix

### During the Interview

**1. Clarify the problem (30 seconds)**
- Array sorted? Modified? Duplicates?
- What should I return if not found?
- Constraints? (Array size, value ranges)

**2. Identify the pattern (30 seconds)**
- Run through decision tree
- State which template you'll use
- Explain the monotonic property

**3. Code (3-4 minutes)**
- Write template from memory
- Customize the condition
- Add post-processing

**4. Test (1-2 minutes)**
- Test with simple case
- Test edge cases (empty, single element)
- Test boundary conditions

**5. Analyze (30 seconds)**
- State time complexity: O(log n) or O(n log k)
- State space complexity: O(1)

---

## 9. Quick Reference Cards

### Template A: MINIMIZE / FIRST
```python
lo, hi = start, end
while lo < hi:
    mid = lo + (hi - lo) // 2      # LEFT-biased
    if is_valid(mid):
        hi = mid                    # Keep mid, search left
    else:
        lo = mid + 1                # Exclude mid, search right
return lo  # or with validation
```

### Template B: MAXIMIZE / LAST
```python
lo, hi = start, end
while lo < hi:
    mid = lo + (hi - lo + 1) // 2  # RIGHT-biased
    if is_valid(mid):
        lo = mid                    # Keep mid, search right
    else:
        hi = mid - 1                # Exclude mid, search left
return lo  # or with validation
```

---

## 10. Complexity Reference

| Pattern | Time | Space | Notes |
|---------|------|-------|-------|
| Basic Array Search | O(log n) | O(1) | Standard binary search |
| Rotated (no dup) | O(log n) | O(1) | One half always sorted |
| Rotated (with dup) | O(n) worst | O(1) | Degrades when many duplicates |
| BS on Answer | O(n log k) | O(1) | k = search space size, n = validation cost |
| K-th Smallest | O(n log k) | O(1) | k = value range, n = count cost |
| Peak Finding | O(log n) | O(1) | Single pass |
| 2D Matrix | O(log(mn)) | O(1) | m x n matrix as 1D array |

**Note:** In "BS on Answer", if your feasibility function is O(n), overall is O(n log k).

---

## Final Thoughts

**Remember:**
1. Binary search is about **partitioning search space**, not just "finding a value"
2. There are TWO core templates: Minimize (left-biased) and Maximize (right-biased)
3. The `+1` in right-biased mid is CRITICAL to avoid infinite loops
4. Post-processing depends on the problem type
5. Recognition is as important as implementation

**When stuck in an interview:**
1. Draw the array and mark lo, mid, hi
2. Ask: "What property remains true in my search space?"
3. Ask: "Am I finding first or last?"
4. Choose the template and trust it!

---

## Appendix: Practice Problem Set

### Easy (Master these first)
- 704. Binary Search
- 35. Search Insert Position
- 69. Sqrt(x)
- 278. First Bad Version
- 852. Peak Index in Mountain Array

### Medium (Core interview level)
- 33. Search in Rotated Sorted Array
- 34. Find First and Last Position
- 74. Search a 2D Matrix
- 162. Find Peak Element
- 875. Koko Eating Bananas
- 1011. Capacity To Ship Packages
- 410. Split Array Largest Sum
- 378. K-th Smallest Element in Sorted Matrix

### Hard (Advanced patterns)
- 81. Search in Rotated Sorted Array II
- 4. Median of Two Sorted Arrays
- 719. Find K-th Smallest Pair Distance
- 1552. Magnetic Force Between Two Balls
- 1044. Longest Duplicate Substring

**Recommended Practice Order:**
1. Solve 2-3 problems from each difficulty tier
2. Revisit patterns you find challenging
3. Time yourself: aim for 15-20 minutes per medium problem
4. Focus on recognizing patterns quickly

Good luck!

---

## Appendix: Conditional Quick Reference

A consolidated table of every key conditional in binary search, what it means, and when to use it.

### Loop Conditions

| Conditional | Meaning | When to Use |
|---|---|---|
| `while lo < hi:` | Loop until one candidate remains (`lo == hi`). | The standard for both Template A and B. Use this in nearly all binary search problems. The loop guarantees convergence to a single candidate. |
| `while lo <= hi:` | Loop until the search space is empty (`lo > hi`). | Classic textbook binary search that returns inside the loop on match. More error-prone. Prefer `lo < hi` in interviews. |

### Mid Calculation

| Conditional | Meaning | When to Use |
|---|---|---|
| `mid = lo + (hi - lo) // 2` | Left-biased mid. When two elements remain, picks the LEFT one. | Template A (Minimize/Find First). Pair with `hi = mid` and `lo = mid + 1`. Prevents infinite loops because `hi = mid` always shrinks the window. |
| `mid = lo + (hi - lo + 1) // 2` | Right-biased mid. When two elements remain, picks the RIGHT one. | Template B (Maximize/Find Last). Pair with `lo = mid` and `hi = mid - 1`. The `+1` is CRITICAL: without it, `lo = mid` with left-biased mid causes an infinite loop when two elements remain. |

### Pointer Updates

| Conditional | Meaning | When to Use |
|---|---|---|
| `hi = mid` | Keep mid in the search space (mid might be the answer). | When condition is satisfied and you want to search LEFT for a smaller answer. Always pair with left-biased mid. |
| `hi = mid - 1` | Exclude mid from the search space (mid is definitely not the answer). | When you are CERTAIN mid cannot be the answer. Always pair with right-biased mid (Template B). |
| `lo = mid + 1` | Exclude mid, search right. | When mid fails the condition and the answer must be strictly larger. Safe with both mid formulas. |
| `lo = mid` | Keep mid in the search space, search right. | When condition is satisfied and you want to search RIGHT for a larger answer. MUST use right-biased mid to avoid infinite loops. |

### Boundary Initialization

| Conditional | Meaning | When to Use |
|---|---|---|
| `lo, hi = 0, len(arr) - 1` | Search space is the entire array (inclusive on both ends). | Standard for index-based searches (first/last occurrence, rotated array, peak finding). |
| `lo, hi = 0, len(arr)` | Search space includes one position past the array end. | Insert position problems where the answer could be at index `len` (appending). |
| `lo, hi = min_val, max_val` | Search space is a range of VALUES, not indices. | Binary search on answer space (ship capacity, eating speed, distance). |

### Common Condition Checks

| Conditional | Meaning | When to Use |
|---|---|---|
| `if nums[mid] >= target: hi = mid` | Find first position where value >= target (lower bound). | First occurrence, insert position. `>=` creates a clean two-way partition. |
| `if nums[mid] <= target: lo = mid` | Find last position where value <= target (upper bound). | Last occurrence. `<=` captures the exact target AND values before it. |
| `if nums[mid] < target: lo = mid + 1` | Mid is strictly too small, exclude it. | When you need to skip past values definitively less than your target. |
| `if nums[lo] <= nums[mid]:` | Left half `[lo..mid]` is sorted. | Rotated array. `<=` handles `lo == mid` (two elements) where left "half" is trivially sorted. |
| `if nums[lo] <= target <= nums[mid]:` | Target falls within the sorted left half's range. | Rotated array: reliable range check once you know left half is sorted. |
| `if nums[mid] < nums[mid + 1]:` | Ascending slope -- peak must exist to the right. | Peak finding. If next element is higher, mid cannot be a peak. |
| `if count < k: lo = mid + 1` | Fewer than k elements at or below mid. | K-th smallest. Strict `<` because `count == k` means mid could be the answer. |
| `if current_sum + num > max_sum:` | Adding element would exceed subarray limit. | Greedy feasibility (split array / ship packages). Strict `>` because equaling is valid. |
| `if position[i] - last_pos >= min_dist:` | Gap meets the minimum distance requirement. | Greedy feasibility for maximize-minimum-distance problems. `>=` because exact min is valid. |
| `if mid <= x // mid:` | Equivalent to `mid^2 <= x` without overflow. | Integer square root. Avoids `mid * mid` which overflows 32-bit integers. |
| `return hours <= h` | Koko finishes within the allowed time. | Eating speed feasibility. `<=` because finishing early is fine. |

### Return Values

| Conditional | Meaning | When to Use |
|---|---|---|
| `return lo` | After `while lo < hi:`, `lo == hi` is the converged answer. | BS on answer, peak finding, K-th smallest -- problems where answer is guaranteed. |
| `return lo if nums[lo] == target else -1` | Validate converged candidate actually matches. | Exact match (standard search, rotated array) where target might not exist. |
| `return lo if lo < n and nums[lo] == target else -1` | Bounds check + match validation. | First occurrence. `lo < n` guards against target being larger than all elements. |
| `return lo + 1` | Insert after the converged position. | Insert position when target is larger than all elements in the array. |
