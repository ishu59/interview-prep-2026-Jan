# The Complete Two Pointers Handbook
> A template-based approach for mastering two pointers in coding interviews

**Philosophy:** Two pointers is not about having two variables. It's about **reducing a nested O(n²) search to O(n) by exploiting structure** — typically sorted order or a constraint that lets both pointers move in one direction.

---

## Table of Contents
1. [Understanding the Core Philosophy](#core-philosophy)
2. [The Three Master Templates](#master-templates)
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

- **The Pinch**: Imagine squeezing a rubber band from both ends toward the center. Each squeeze eliminates one end because you can prove it can't be part of the answer. This is the "opposite ends" pattern.
- **The Tortoise and Hare**: Two runners on a track moving at different speeds. If there's a loop, the fast one laps the slow one. If there's no loop, the fast one hits the finish first. This is the "fast/slow" pattern.
- **The Merge Lane**: Two sorted conveyor belts feeding into one. You always pick the smaller item from whichever belt has it. This is the "two arrays" pattern.

### No-Jargon Translation

- **Monotonic elimination**: proving one end can't help, so you skip it
- **In-place**: rearranging without making a copy
- **Convergence**: the two pointers will always eventually meet
- **Partition**: splitting into groups by some rule

### Mental Model

> "Two pointers are two fingers walking along the data, and at every step you can prove at least one finger should move, so you never waste time."

---

### Why Two Pointers?

The naive approach to many problems involves checking all pairs:
```python
# O(n²) - Check every pair
for i in range(n):
    for j in range(i+1, n):
        if some_condition(arr[i], arr[j]):
            return result
```

Two pointers reduces this to O(n) by using **problem structure** to eliminate unnecessary comparisons.

### The Key Insight: Monotonic Elimination

Two pointers works when you can **eliminate possibilities** based on pointer positions:

> "If the current pair doesn't satisfy the condition, I know which direction to move — and I'll never need to revisit the eliminated pairs."

**Example:** Finding two numbers that sum to target in a sorted array.
```
arr = [1, 3, 5, 7, 9], target = 8

Brute force: Check (1,3), (1,5), (1,7), (1,9), (3,5), (3,7), (3,9), (5,7), (5,9), (7,9)
             = 10 pairs

Two pointers:
  left=0, right=4: 1+9=10 > 8 → right-- (eliminate all pairs with 9)
  left=0, right=3: 1+7=8 = target → Found!
             = 2 comparisons
```

**Why can we eliminate pairs with 9?**
Because the array is sorted! If `1 + 9 > 8`, then `3 + 9`, `5 + 9`, `7 + 9` are all > 8 too.

### The Three Mental Models

#### Model 1: Opposite Ends (Converging)
```
[  |  |  |  |  |  |  ]
   ↑                 ↑
  left             right

  ←←← Both move inward →→→
```
**Use for:** Sum problems, container problems, palindrome checking

#### Model 2: Same Direction (Fast/Slow)
```
[  |  |  |  |  |  |  ]
   ↑  ↑
 slow fast

  →→→ Both move same direction →→→
```
**Use for:** Removing duplicates, cycle detection, partitioning

#### Model 3: Two Arrays (Merge)
```
Array 1: [  |  |  |  ]
            ↑
            i

Array 2: [  |  |  |  |  ]
            ↑
            j
```
**Use for:** Merging sorted arrays, intersection, comparison

---

<a name="master-templates"></a>
## 2. The Three Master Templates

### Template A: Opposite Ends (Converging Pointers)

**Use when:** Array is sorted, looking for pairs with specific sum/property

```python
def opposite_ends(arr, condition):
    """
    Template for converging pointers from both ends.
    Works when: sorted array, need to find pair satisfying condition
    """
    left = 0
    right = len(arr) - 1

    while left < right:
        # Evaluate current pair
        current = evaluate(arr[left], arr[right])

        if current == target:
            return (left, right)  # Found answer
        elif current < target:
            left += 1   # Need larger sum → move left pointer right
        else:
            right -= 1  # Need smaller sum → move right pointer left

    return None  # No valid pair found
```

**Why this works:**
- Each iteration eliminates one pointer position permanently
- At most `n` iterations (each step moves at least one pointer)
- **Time: O(n), Space: O(1)**

**Key Decision:** Which pointer to move?
- If result is **too small** → move left pointer (to get larger values)
- If result is **too large** → move right pointer (to get smaller values)

---

### Template B: Same Direction (Fast/Slow)

**Use when:** Removing elements, finding duplicates, partitioning array

```python
def same_direction(arr, condition):
    """
    Template for fast/slow pointers moving same direction.
    slow = position to write / boundary of "good" elements
    fast = position to read / scanner
    """
    slow = 0  # Points to next position to write

    for fast in range(len(arr)):
        if condition(arr[fast]):
            arr[slow] = arr[fast]
            slow += 1

    return slow  # Length of processed array
```

**Why this works:**
- `slow` maintains invariant: `arr[0:slow]` contains valid elements
- `fast` scans through all elements once
- **Time: O(n), Space: O(1)**

**Key Insight:** Think of `slow` as the "writer" and `fast` as the "reader":
```
Original: [0, 1, 2, 2, 3, 0, 4, 2]  (Remove 2s)

Step by step:
fast=0: arr[0]=0 ≠ 2, write at slow=0, slow++
fast=1: arr[1]=1 ≠ 2, write at slow=1, slow++
fast=2: arr[2]=2 = 2, skip
fast=3: arr[3]=2 = 2, skip
fast=4: arr[4]=3 ≠ 2, write at slow=2, slow++
...

Result: [0, 1, 3, 0, 4, _, _, _], return slow=5
```

---

### Template C: Two Arrays (Merge Pattern)

**Use when:** Processing two sorted arrays simultaneously

```python
def two_arrays(arr1, arr2, combine):
    """
    Template for processing two sorted arrays.
    """
    i, j = 0, 0
    result = []

    while i < len(arr1) and j < len(arr2):
        if arr1[i] <= arr2[j]:
            result.append(arr1[i])
            i += 1
        else:
            result.append(arr2[j])
            j += 1

    # Handle remaining elements
    while i < len(arr1):
        result.append(arr1[i])
        i += 1
    while j < len(arr2):
        result.append(arr2[j])
        j += 1

    return result
```

**Why this works:**
- Both arrays are sorted, so we always pick the smaller element
- Each element is processed exactly once
- **Time: O(n + m), Space: O(n + m)** for result (O(1) if in-place possible)

---

### Quick Decision Matrix

| Problem Type | Template | Pointer Movement | Example |
|--------------|----------|------------------|---------|
| Two Sum (sorted) | A: Opposite | Converge based on sum | LC 167 |
| 3Sum | A: Opposite | Fix one, two-sum the rest | LC 15 |
| Container Water | A: Opposite | Move smaller height | LC 11 |
| Remove Duplicates | B: Fast/Slow | slow=write, fast=read | LC 26 |
| Move Zeros | B: Fast/Slow | slow=non-zero pos | LC 283 |
| Merge Arrays | C: Two Arrays | Compare and advance | LC 88 |
| Array Intersection | C: Two Arrays | Equal→both advance | LC 349 |

---

<a name="pattern-guide"></a>
## 3. Pattern Classification Guide

### Category 1: Sum Problems
- Array is sorted (or can be sorted)
- Finding pairs/triplets with specific sum
- **Template A** (Opposite ends)
- Examples: Two Sum II, 3Sum, 4Sum, 3Sum Closest

### Category 2: Container/Area Problems
- Maximize/minimize area between two boundaries
- Decision based on limiting factor (smaller height, shorter distance)
- **Template A** (Opposite ends)
- Examples: Container With Most Water, Trapping Rain Water

### Category 3: Array Modification
- Remove/modify elements in-place
- Maintain relative order
- **Template B** (Fast/Slow)
- Examples: Remove Duplicates, Remove Element, Move Zeros

### Category 4: Palindrome Problems
- Check symmetry from both ends
- **Template A** (Opposite ends)
- Examples: Valid Palindrome, Palindrome Linked List

### Category 5: Merge/Intersection
- Two sorted arrays
- Combine or find common elements
- **Template C** (Two Arrays)
- Examples: Merge Sorted Array, Intersection of Arrays

### Category 6: Partition Problems
- Rearrange array based on condition
- Dutch National Flag type
- **Template B** (Fast/Slow) or custom
- Examples: Sort Colors, Partition Array

---

<a name="patterns"></a>
## 4. Complete Pattern Library

### PATTERN 1: Two Sum (Sorted Array)

---

#### Pattern 1A: Basic Two Sum

**Problem:** LeetCode 167 - Two Sum II (Input array is sorted)

**Example:** `numbers = [2, 7, 11, 15]`, target = `9` → `[1, 2]` (1-indexed)

**Why Two Pointers?**
- Array is **sorted** — this is the key!
- We can eliminate half the remaining pairs with each comparison

```python
def twoSum(numbers: list[int], target: int) -> list[int]:
    left, right = 0, len(numbers) - 1

    while left < right:
        current_sum = numbers[left] + numbers[right]

        if current_sum == target:
            return [left + 1, right + 1]  # 1-indexed
        elif current_sum < target:
            # Sum too small → need larger numbers → move left pointer right
            left += 1
        else:
            # Sum too large → need smaller numbers → move right pointer left
            right -= 1

    return []  # No solution found
```

**Visual Trace:**
```
[2, 7, 11, 15], target = 9

Step 1: left=0, right=3
        2 + 15 = 17 > 9 → right--

Step 2: left=0, right=2
        2 + 11 = 13 > 9 → right--

Step 3: left=0, right=1
        2 + 7 = 9 = target → Found! Return [1, 2]
```

**Why move `left` when sum is too small?**
- `numbers[left]` is the smallest unused number
- If `numbers[left] + numbers[right] < target`, then `numbers[left]` with ANY smaller right value will also be < target
- So we can safely discard `numbers[left]` and try a larger number

**Complexity:** Time O(n), Space O(1)

---

#### Pattern 1B: 3Sum

**Problem:** LeetCode 15 - Find all unique triplets that sum to zero

**Example:** `nums = [-1, 0, 1, 2, -1, -4]` → `[[-1, -1, 2], [-1, 0, 1]]`

**Key Insight:** Fix one element, then use Two Sum on the rest!

```python
def threeSum(nums: list[int]) -> list[list[int]]:
    nums.sort()  # Essential for two-pointer approach
    result = []
    n = len(nums)

    for i in range(n - 2):
        # Skip duplicates for the first element
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        # Early termination: if smallest triplet > 0, no solution
        if nums[i] > 0:
            break

        # Two-pointer for remaining two elements
        left, right = i + 1, n - 1
        target = -nums[i]  # We need nums[left] + nums[right] = -nums[i]

        while left < right:
            current_sum = nums[left] + nums[right]

            if current_sum == target:
                result.append([nums[i], nums[left], nums[right]])

                # Skip duplicates for second and third elements
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1

                left += 1
                right -= 1
            elif current_sum < target:
                left += 1
            else:
                right -= 1

    return result
```

**Why sort first?**
1. Enables two-pointer technique
2. Makes duplicate skipping easy (duplicates are adjacent)
3. Allows early termination when `nums[i] > 0`

**Why skip duplicates?**
```
[-1, -1, 0, 1, 2]
      ↑
If we don't skip, we'd find [-1, 0, 1] twice!
```

**Complexity:** Time O(n²), Space O(1) excluding output

---

#### Pattern 1C: 3Sum Closest

**Problem:** LeetCode 16 - Find triplet with sum closest to target

**Key Difference:** Don't need exact match, track closest distance

```python
def threeSumClosest(nums: list[int], target: int) -> int:
    nums.sort()
    n = len(nums)
    closest_sum = float('inf')

    for i in range(n - 2):
        left, right = i + 1, n - 1

        while left < right:
            current_sum = nums[i] + nums[left] + nums[right]

            # Update closest if this sum is better
            if abs(current_sum - target) < abs(closest_sum - target):
                closest_sum = current_sum

            if current_sum == target:
                return target  # Can't get closer than exact match!
            elif current_sum < target:
                left += 1
            else:
                right -= 1

    return closest_sum
```

**Complexity:** Time O(n²), Space O(1)

---

#### Pattern 1D: 4Sum

**Problem:** LeetCode 18 - Find all unique quadruplets that sum to target

**Key Insight:** Fix two elements, then use Two Sum

```python
def fourSum(nums: list[int], target: int) -> list[list[int]]:
    nums.sort()
    n = len(nums)
    result = []

    for i in range(n - 3):
        # Skip duplicates
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        for j in range(i + 1, n - 2):
            # Skip duplicates
            if j > i + 1 and nums[j] == nums[j - 1]:
                continue

            left, right = j + 1, n - 1
            remaining = target - nums[i] - nums[j]

            while left < right:
                current_sum = nums[left] + nums[right]

                if current_sum == remaining:
                    result.append([nums[i], nums[j], nums[left], nums[right]])

                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1

                    left += 1
                    right -= 1
                elif current_sum < remaining:
                    left += 1
                else:
                    right -= 1

    return result
```

**Generalization to K-Sum:**
- Fix k-2 elements with nested loops
- Use two-pointer for last two elements
- Time: O(n^(k-1))

**Complexity:** Time O(n³), Space O(1) excluding output

---

### PATTERN 2: Container and Area Problems

---

#### Pattern 2A: Container With Most Water

**Problem:** LeetCode 11 - Find two lines that form container with most water

**Example:** `height = [1, 8, 6, 2, 5, 4, 8, 3, 7]` → `49` (between indices 1 and 8)

**Key Insight:** Area = width × min(height[left], height[right])

Why move the shorter line? Because:
- Width will decrease regardless of which we move
- Moving the taller line can only decrease or maintain the height (limited by shorter)
- Moving the shorter line gives us a chance to find a taller line

```python
def maxArea(height: list[int]) -> int:
    left, right = 0, len(height) - 1
    max_area = 0

    while left < right:
        # Calculate current area
        width = right - left
        h = min(height[left], height[right])
        current_area = width * h
        max_area = max(max_area, current_area)

        # Move the shorter line
        # Why? Moving the taller line can never increase area
        # (it's limited by the shorter line anyway)
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1

    return max_area
```

**Visual Explanation:**
```
height = [1, 8, 6, 2, 5, 4, 8, 3, 7]
          ↑                       ↑
        left=0               right=8

Area = 8 × min(1, 7) = 8 × 1 = 8

height[left]=1 < height[right]=7
Move left → gives chance to find taller left wall
Move right → area can only decrease (limited by 1)
```

**Proof of Correctness:**
When we move a pointer, we're saying "I don't need to consider any pair involving this pointer position with any position on the other side."

- If `height[left] < height[right]`:
  - Any pair `(left, j)` where `j < right` has:
    - Smaller width (j < right)
    - Same or smaller height (limited by height[left])
  - So all pairs `(left, j)` are strictly worse → safe to discard

**Complexity:** Time O(n), Space O(1)

---

#### Pattern 2B: Trapping Rain Water

**Problem:** LeetCode 42 - Calculate trapped rain water

**Example:** `height = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]` → `6`

**Key Insight:** Water at position i = min(max_left, max_right) - height[i]

**Two-Pointer Approach:** Track max from left and right, process the side with smaller max.

```python
def trap(height: list[int]) -> int:
    if not height:
        return 0

    left, right = 0, len(height) - 1
    left_max, right_max = height[left], height[right]
    water = 0

    while left < right:
        if height[left] < height[right]:
            # Water level at left is determined by left_max
            # (because we know there's something taller on the right)
            left += 1
            left_max = max(left_max, height[left])
            water += left_max - height[left]
        else:
            # Water level at right is determined by right_max
            right -= 1
            right_max = max(right_max, height[right])
            water += right_max - height[right]

    return water
```

**Why this works:**

The key insight is: **water level at any position is limited by the SMALLER of the two maxes.**

- If `height[left] < height[right]`:
  - We know `right_max >= height[right] > height[left]`
  - So water at `left` is determined by `left_max` (the limiting factor)
  - We can safely calculate water at `left` and move on

```
[0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
    ↑                          ↑
   left                      right
left_max=1                  right_max=1

height[left]=1 >= height[right]=1
Process right side:
  right-- → right=10, height=2
  right_max = max(1, 2) = 2
  water += 2 - 2 = 0
```

**Alternative: Stack-based approach** (for reference)
```python
def trap_stack(height: list[int]) -> int:
    stack = []  # Store indices
    water = 0

    for i, h in enumerate(height):
        while stack and height[stack[-1]] < h:
            bottom = stack.pop()
            if not stack:
                break
            width = i - stack[-1] - 1
            bounded_height = min(h, height[stack[-1]]) - height[bottom]
            water += width * bounded_height
        stack.append(i)

    return water
```

**Complexity:** Time O(n), Space O(1) for two-pointer

---

### PATTERN 3: Array Modification (In-Place)

---

#### Pattern 3A: Remove Duplicates from Sorted Array

**Problem:** LeetCode 26 - Remove duplicates in-place, return new length

**Example:** `nums = [1, 1, 2]` → `2`, nums becomes `[1, 2, _]`

**Key Insight:** Use slow pointer as "write position", fast pointer scans

```python
def removeDuplicates(nums: list[int]) -> int:
    if not nums:
        return 0

    # slow = position to write next unique element
    # nums[0] is always unique, so start slow at 1
    slow = 1

    for fast in range(1, len(nums)):
        # If current element differs from previous, it's unique
        if nums[fast] != nums[fast - 1]:
            nums[slow] = nums[fast]
            slow += 1

    return slow
```

**Visual Trace:**
```
nums = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
        ↑  ↑
      slow fast

fast=1: nums[1]=0 == nums[0]=0, skip
fast=2: nums[2]=1 != nums[1]=0, write at slow=1, slow++
        [0, 1, 1, 1, 1, 2, 2, 3, 3, 4]
            ↑     ↑
          slow  fast

fast=3: nums[3]=1 == nums[2]=1, skip
fast=4: nums[4]=1 == nums[3]=1, skip
fast=5: nums[5]=2 != nums[4]=1, write at slow=2, slow++
        [0, 1, 2, 1, 1, 2, 2, 3, 3, 4]
               ↑           ↑
             slow        fast
...

Result: [0, 1, 2, 3, 4, _, _, _, _, _], return 5
```

**Invariant:** `nums[0:slow]` contains all unique elements seen so far

**Complexity:** Time O(n), Space O(1)

---

#### Pattern 3B: Remove Element

**Problem:** LeetCode 27 - Remove all instances of value in-place

**Example:** `nums = [3, 2, 2, 3]`, val = `3` → `2`, nums becomes `[2, 2, _, _]`

```python
def removeElement(nums: list[int], val: int) -> int:
    slow = 0

    for fast in range(len(nums)):
        if nums[fast] != val:
            nums[slow] = nums[fast]
            slow += 1

    return slow
```

**Alternative: Two-pointer from both ends** (when element order doesn't matter)
```python
def removeElement_swap(nums: list[int], val: int) -> int:
    left, right = 0, len(nums) - 1

    while left <= right:
        if nums[left] == val:
            nums[left] = nums[right]
            right -= 1
        else:
            left += 1

    return left
```

This version does fewer writes when `val` is rare.

**Complexity:** Time O(n), Space O(1)

---

#### Pattern 3C: Move Zeroes

**Problem:** LeetCode 283 - Move all zeroes to end, maintaining order

**Example:** `nums = [0, 1, 0, 3, 12]` → `[1, 3, 12, 0, 0]`

```python
def moveZeroes(nums: list[int]) -> None:
    # slow = position to write next non-zero
    slow = 0

    # Move all non-zeros to front
    for fast in range(len(nums)):
        if nums[fast] != 0:
            nums[slow] = nums[fast]
            slow += 1

    # Fill rest with zeros
    for i in range(slow, len(nums)):
        nums[i] = 0
```

**Optimized: Swap instead of overwrite** (single pass)
```python
def moveZeroes_swap(nums: list[int]) -> None:
    slow = 0

    for fast in range(len(nums)):
        if nums[fast] != 0:
            nums[slow], nums[fast] = nums[fast], nums[slow]
            slow += 1
```

**Why swap works:**
- When `slow == fast`, we swap element with itself (no-op)
- When `slow < fast`, `nums[slow]` is always 0 (by invariant)
- So we're always swapping non-zero with zero

**Complexity:** Time O(n), Space O(1)

---

#### Pattern 3D: Remove Duplicates II (Allow K duplicates)

**Problem:** LeetCode 80 - Allow at most 2 duplicates

**Example:** `nums = [1, 1, 1, 2, 2, 3]` → `5`, nums becomes `[1, 1, 2, 2, 3, _]`

**Key Insight:** Compare with element k positions back

```python
def removeDuplicates(nums: list[int], k: int = 2) -> int:
    if len(nums) <= k:
        return len(nums)

    slow = k  # First k elements always kept

    for fast in range(k, len(nums)):
        # Compare with element k positions back in result
        if nums[fast] != nums[slow - k]:
            nums[slow] = nums[fast]
            slow += 1

    return slow
```

**Why compare with `nums[slow - k]`?**
- `nums[0:slow]` is our result array
- If `nums[fast] != nums[slow - k]`, then adding `nums[fast]` won't create > k duplicates
- Because there are exactly k-1 positions between `slow-k` and `slow-1`

**Visual for k=2:**
```
nums = [1, 1, 1, 2, 2, 3]
              ↑  ↑
           slow fast

fast=2: nums[2]=1 == nums[slow-2]=nums[0]=1, skip (would be 3rd '1')
fast=3: nums[3]=2 != nums[slow-2]=nums[0]=1, write at slow=2, slow++
        [1, 1, 2, 2, 2, 3]
                 ↑     ↑
               slow  fast
```

**Complexity:** Time O(n), Space O(1)

---

### PATTERN 4: Palindrome Problems

---

#### Pattern 4A: Valid Palindrome

**Problem:** LeetCode 125 - Check if string is palindrome (alphanumeric only)

```python
def isPalindrome(s: str) -> bool:
    left, right = 0, len(s) - 1

    while left < right:
        # Skip non-alphanumeric from left
        while left < right and not s[left].isalnum():
            left += 1

        # Skip non-alphanumeric from right
        while left < right and not s[right].isalnum():
            right -= 1

        # Compare characters (case-insensitive)
        if s[left].lower() != s[right].lower():
            return False

        left += 1
        right -= 1

    return True
```

**Why two pointers for palindrome?**
- Palindrome = same forwards and backwards
- Compare character at position i with character at position n-1-i
- Two pointers from ends naturally do this

**Complexity:** Time O(n), Space O(1)

---

#### Pattern 4B: Valid Palindrome II (Delete at most one)

**Problem:** LeetCode 680 - Can become palindrome by removing at most one character

```python
def validPalindrome(s: str) -> bool:
    def is_palindrome(left: int, right: int) -> bool:
        while left < right:
            if s[left] != s[right]:
                return False
            left += 1
            right -= 1
        return True

    left, right = 0, len(s) - 1

    while left < right:
        if s[left] != s[right]:
            # Try skipping either left or right character
            return is_palindrome(left + 1, right) or is_palindrome(left, right - 1)
        left += 1
        right -= 1

    return True
```

**Key Insight:** When mismatch found, we have two choices:
1. Skip left character and check if rest is palindrome
2. Skip right character and check if rest is palindrome

Either one working means we can form palindrome with one deletion.

**Complexity:** Time O(n), Space O(1)

---

### PATTERN 5: Sorted Array Merge/Intersection

---

#### Pattern 5A: Merge Sorted Array

**Problem:** LeetCode 88 - Merge nums2 into nums1 in-place

**Example:** `nums1 = [1, 2, 3, 0, 0, 0]`, `nums2 = [2, 5, 6]` → `[1, 2, 2, 3, 5, 6]`

**Key Insight:** Merge from the END to avoid overwriting

```python
def merge(nums1: list[int], m: int, nums2: list[int], n: int) -> None:
    # Start from the end of both arrays
    p1 = m - 1      # Pointer for nums1's actual elements
    p2 = n - 1      # Pointer for nums2
    p = m + n - 1   # Pointer for write position

    # Merge backwards
    while p1 >= 0 and p2 >= 0:
        if nums1[p1] > nums2[p2]:
            nums1[p] = nums1[p1]
            p1 -= 1
        else:
            nums1[p] = nums2[p2]
            p2 -= 1
        p -= 1

    # If nums2 has remaining elements, copy them
    # (If nums1 has remaining, they're already in place)
    while p2 >= 0:
        nums1[p] = nums2[p2]
        p2 -= 1
        p -= 1
```

**Why merge backwards?**
- Merging forwards would overwrite nums1's elements before we use them
- Merging backwards fills the empty space first
- nums1's actual elements are always to the left of write position

**Visual:**
```
nums1 = [1, 2, 3, 0, 0, 0]
               ↑        ↑
              p1        p
nums2 = [2, 5, 6]
               ↑
              p2

nums1[p1]=3 < nums2[p2]=6 → write 6
nums1 = [1, 2, 3, 0, 0, 6]
               ↑     ↑
              p1     p
nums2[p2]=5 → write 5
...
```

**Complexity:** Time O(m + n), Space O(1)

---

#### Pattern 5B: Intersection of Two Arrays

**Problem:** LeetCode 349 - Find intersection (unique values)

```python
def intersection(nums1: list[int], nums2: list[int]) -> list[int]:
    # Sort both arrays
    nums1.sort()
    nums2.sort()

    result = []
    i, j = 0, 0

    while i < len(nums1) and j < len(nums2):
        if nums1[i] == nums2[j]:
            # Avoid duplicates in result
            if not result or result[-1] != nums1[i]:
                result.append(nums1[i])
            i += 1
            j += 1
        elif nums1[i] < nums2[j]:
            i += 1
        else:
            j += 1

    return result
```

**Why this works:**
- Both arrays sorted → smallest unprocessed elements are at pointers
- If equal → found intersection, advance both
- If not equal → advance the smaller one (can't find match for it ahead)

**Complexity:** Time O(n log n + m log m), Space O(1) excluding output

**Alternative using set:** Time O(n + m) with O(n) space

---

#### Pattern 5C: Intersection of Two Arrays II (With duplicates)

**Problem:** LeetCode 350 - Include duplicates in intersection

```python
def intersect(nums1: list[int], nums2: list[int]) -> list[int]:
    nums1.sort()
    nums2.sort()

    result = []
    i, j = 0, 0

    while i < len(nums1) and j < len(nums2):
        if nums1[i] == nums2[j]:
            result.append(nums1[i])  # Include every match
            i += 1
            j += 1
        elif nums1[i] < nums2[j]:
            i += 1
        else:
            j += 1

    return result
```

**Difference from Pattern 5B:** Don't skip duplicates, include every matching pair.

**Follow-up: What if nums2 is on disk?**
- Sort nums1 in memory
- Stream nums2 from disk, use binary search for each element
- Or: External merge sort both, then stream intersection

**Complexity:** Time O(n log n + m log m), Space O(1) excluding output

---

### PATTERN 6: Partition and Sort

---

#### Pattern 6A: Sort Colors (Dutch National Flag)

**Problem:** LeetCode 75 - Sort array with only 0, 1, 2

**Example:** `nums = [2, 0, 2, 1, 1, 0]` → `[0, 0, 1, 1, 2, 2]`

**Three Pointers:** low (0s boundary), mid (scanner), high (2s boundary)

```python
def sortColors(nums: list[int]) -> None:
    low = 0           # Boundary for 0s (next position to place 0)
    mid = 0           # Current element being processed
    high = len(nums) - 1  # Boundary for 2s (next position to place 2)

    while mid <= high:
        if nums[mid] == 0:
            # Swap with low boundary, advance both
            nums[low], nums[mid] = nums[mid], nums[low]
            low += 1
            mid += 1
        elif nums[mid] == 1:
            # 1 is in correct region, just advance
            mid += 1
        else:  # nums[mid] == 2
            # Swap with high boundary, only decrease high
            # Don't advance mid - need to process swapped element
            nums[mid], nums[high] = nums[high], nums[mid]
            high -= 1
```

**Why not advance mid when swapping with high?**
- When we swap `nums[mid]` with `nums[high]`, we don't know what value came from `high`
- It could be 0, 1, or 2 — we need to process it
- When swapping with `low`, we know `nums[low]` was either 1 or we haven't passed it yet

**Invariant:**
- `nums[0:low]` contains all 0s
- `nums[low:mid]` contains all 1s
- `nums[mid:high+1]` is unprocessed
- `nums[high+1:]` contains all 2s

**Visual:**
```
[2, 0, 2, 1, 1, 0]
 ↑              ↑
low            high
mid

nums[mid]=2 → swap with high
[0, 0, 2, 1, 1, 2]
 ↑           ↑
low         high
mid

nums[mid]=0 → swap with low, advance both
[0, 0, 2, 1, 1, 2]
    ↑        ↑
  low       high
   mid
...
```

**Complexity:** Time O(n), Space O(1)

---

#### Pattern 6B: Partition Array

**Problem:** Move all elements < pivot before elements >= pivot

```python
def partition(nums: list[int], pivot: int) -> int:
    """
    Rearrange so that nums[:result] < pivot and nums[result:] >= pivot
    Returns the partition index
    """
    slow = 0

    for fast in range(len(nums)):
        if nums[fast] < pivot:
            nums[slow], nums[fast] = nums[fast], nums[slow]
            slow += 1

    return slow
```

**This is the core of QuickSort's partition step!**

**Complexity:** Time O(n), Space O(1)

---

### PATTERN 7: Linked List Two Pointers

---

#### Pattern 7A: Detect Cycle (Floyd's Algorithm)

**Problem:** LeetCode 141 - Determine if linked list has cycle

**Key Insight:** Fast pointer moves 2 steps, slow moves 1 step. If cycle exists, they meet.

```python
def hasCycle(head: ListNode) -> bool:
    if not head or not head.next:
        return False

    slow = head
    fast = head

    while fast and fast.next:
        slow = slow.next        # Move 1 step
        fast = fast.next.next   # Move 2 steps

        if slow == fast:
            return True

    return False
```

**Why they meet if cycle exists:**
- Think of it as a race on a circular track
- Fast gains 1 position per iteration
- Eventually fast "laps" slow

**Mathematical proof:**
- Let cycle length = C
- When slow enters cycle, fast is some distance d ahead
- Distance between them decreases by 1 each iteration
- They meet after C - d iterations

**Complexity:** Time O(n), Space O(1)

---

#### Pattern 7B: Find Cycle Start

**Problem:** LeetCode 142 - Find the node where cycle begins

```python
def detectCycle(head: ListNode) -> ListNode:
    if not head or not head.next:
        return None

    # Phase 1: Detect cycle
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break
    else:
        return None  # No cycle

    # Phase 2: Find cycle start
    # Reset one pointer to head, move both at same speed
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next

    return slow
```

**Why does Phase 2 work?**

Let:
- `a` = distance from head to cycle start
- `b` = distance from cycle start to meeting point
- `c` = distance from meeting point back to cycle start

When they meet:
- Slow traveled: `a + b`
- Fast traveled: `a + b + (b + c)` = `a + 2b + c`
- Fast traveled 2× slow: `a + 2b + c = 2(a + b)`
- Solving: `c = a`

So if we start one pointer at head and one at meeting point, both moving at speed 1, they'll meet at cycle start!

**Complexity:** Time O(n), Space O(1)

---

#### Pattern 7C: Find Middle of Linked List

**Problem:** LeetCode 876 - Return middle node

```python
def middleNode(head: ListNode) -> ListNode:
    slow = fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    return slow
```

**Why this works:**
- When fast reaches end (2n or 2n+1 steps), slow is at n (middle)
- For even length: returns second middle node
- For odd length: returns exact middle

**Variant for first middle in even-length list:**
```python
def middleNode_first(head: ListNode) -> ListNode:
    slow = fast = head
    while fast.next and fast.next.next:  # Different condition
        slow = slow.next
        fast = fast.next.next
    return slow
```

**Complexity:** Time O(n), Space O(1)

---

#### Pattern 7D: Remove Nth Node From End

**Problem:** LeetCode 19 - Remove nth node from end

```python
def removeNthFromEnd(head: ListNode, n: int) -> ListNode:
    # Dummy node handles edge case of removing head
    dummy = ListNode(0)
    dummy.next = head

    slow = fast = dummy

    # Advance fast by n+1 steps
    # This creates gap of n between slow and fast
    for _ in range(n + 1):
        fast = fast.next

    # Move both until fast reaches end
    while fast:
        slow = slow.next
        fast = fast.next

    # slow is now pointing to node BEFORE the one to delete
    slow.next = slow.next.next

    return dummy.next
```

**Why n+1 steps?**
- We want slow to point to the node BEFORE the one to delete
- If gap is n, when fast is at end (null), slow is at nth from end
- We need gap of n+1 so slow is one node earlier

**Complexity:** Time O(n), Space O(1)

---

<a name="post-processing"></a>
## 5. Post-Processing Reference

| Problem Type | Return Value | Validation Needed |
|--------------|--------------|-------------------|
| **Find pair with sum** | `[left, right]` or `None` | Check if found |
| **Find all pairs/triplets** | Accumulated `result` list | Duplicate handling |
| **Max/min area** | Accumulated `max_val` | None |
| **In-place modification** | `slow` (new length) | None |
| **Palindrome check** | `True/False` | None |
| **Merge arrays** | Modified array | None |
| **Cycle detection** | `True/False` or node | None |

---

<a name="pitfalls"></a>
## 6. Common Pitfalls & Solutions

### Pitfall 1: Forgetting to Sort

**Problem:** Using opposite-ends two pointers on unsorted array

```python
# WRONG: Array not sorted
nums = [3, 1, 2, 5, 4]
left, right = 0, 4
# Moving based on sum comparison doesn't work without sorted order!
```

**Solution:** Always sort first (if problem allows), or use hash map instead.

---

### Pitfall 2: Infinite Loop in Same-Direction

**Problem:** Fast pointer doesn't advance properly

```python
# WRONG: Missing increment
slow = 0
for fast in range(len(nums)):
    if condition:
        nums[slow] = nums[fast]
        # Forgot: slow += 1
```

**Solution:** Always increment slow when writing.

---

### Pitfall 3: Duplicate Handling in N-Sum

**Problem:** Not skipping duplicates leads to duplicate results

```python
# WRONG: Missing duplicate skip
for i in range(n):
    # Should check: if i > 0 and nums[i] == nums[i-1]: continue
```

**Solution:** After sorting, skip elements equal to previous.

---

### Pitfall 4: Off-by-One in Merge

**Problem:** Wrong handling of remaining elements

```python
# WRONG: Only copying nums2 remainder
while p2 >= 0:
    nums1[p] = nums2[p2]
    p2 -= 1
    # Forgot: p -= 1
```

**Solution:** Always decrement write pointer.

---

### Pitfall 5: Wrong Loop Condition for Linked List

**Problem:** Null pointer exception with fast pointer

```python
# WRONG: Can cause null pointer
while fast.next:
    fast = fast.next.next  # fast.next could be null!
```

**Solution:** Check both `fast` and `fast.next`:
```python
while fast and fast.next:
    fast = fast.next.next
```

---

### Pitfall 6: Not Using Dummy Node for Linked List

**Problem:** Special case when removing head node

```python
# WRONG: Fails when removing head
slow = fast = head
# ... remove slow.next
return head  # What if head was removed?
```

**Solution:** Use dummy node pointing to head:
```python
dummy = ListNode(0)
dummy.next = head
# ... operations
return dummy.next
```

---

<a name="recognition"></a>
## 7. Problem Recognition Framework

### Step 1: Is Two Pointers Applicable?

Ask these questions:
1. **Is the array sorted or can be sorted?** → Likely two pointers
2. **Looking for pairs with specific sum/property?** → Opposite ends
3. **Need to modify array in-place?** → Fast/slow
4. **Working with two sorted arrays?** → Two arrays pattern
5. **Linked list cycle or middle?** → Fast/slow

### Step 2: Identify the Template

| Clue in Problem | Template |
|-----------------|----------|
| "sorted array" + "find pair" | Opposite ends |
| "sum equals target" | Opposite ends |
| "container", "area", "water" | Opposite ends |
| "remove in-place" | Fast/slow |
| "move elements" | Fast/slow |
| "merge sorted" | Two arrays |
| "intersection" | Two arrays |
| "linked list cycle" | Fast/slow |
| "middle of linked list" | Fast/slow |
| "palindrome" | Opposite ends |

### Step 3: Determine Pointer Movement

For opposite ends:
- Too small → move left pointer right
- Too large → move right pointer left

For fast/slow:
- Condition true → write and advance slow
- Always advance fast

For two arrays:
- Pick smaller element, advance that pointer

### Decision Tree

```
                    Two Pointers Problem?
                           ↓
           ┌───────────────┼───────────────┐
           │               │               │
    Sorted Array    In-place Modify   Two Arrays
           │               │               │
           ↓               ↓               ↓
    ┌──────┴──────┐   Fast/Slow      Merge Pattern
    │             │
Sum/Pair     Area/Container
    │             │
    ↓             ↓
Opposite      Opposite
  Ends         Ends
(sum-based)  (min-based)
```

---

<a name="checklist"></a>
## 8. Interview Preparation Checklist

### Before the Interview

**Master the fundamentals:**
- [ ] Can write all three templates from memory
- [ ] Understand when to use each template
- [ ] Know duplicate handling in N-Sum problems
- [ ] Understand why merge works backwards

**Practice pattern recognition:**
- [ ] Can identify two-pointer problems quickly
- [ ] Know which template fits which problem type
- [ ] Understand the monotonic property requirement

**Know the patterns:**
- [ ] Two Sum / 3Sum / 4Sum
- [ ] Container With Most Water
- [ ] Trapping Rain Water
- [ ] Remove Duplicates (I and II)
- [ ] Move Zeros
- [ ] Merge Sorted Array
- [ ] Sort Colors
- [ ] Linked list cycle detection

**Common problems solved:**
- [ ] LC 167: Two Sum II
- [ ] LC 15: 3Sum
- [ ] LC 11: Container With Most Water
- [ ] LC 42: Trapping Rain Water
- [ ] LC 26: Remove Duplicates
- [ ] LC 283: Move Zeroes
- [ ] LC 88: Merge Sorted Array
- [ ] LC 75: Sort Colors
- [ ] LC 141/142: Linked List Cycle

### During the Interview

**1. Clarify (30 seconds)**
- Is array sorted?
- Can I sort it?
- Modify in-place or return new array?
- Handle duplicates how?

**2. Identify pattern (30 seconds)**
- State which template
- Explain why it works

**3. Code (3-4 minutes)**
- Write template
- Customize condition
- Handle edge cases

**4. Test (1-2 minutes)**
- Empty array
- Single element
- All same elements
- Already sorted (for problems requiring sort)

**5. Analyze (30 seconds)**
- Time: Usually O(n) or O(n²) for N-Sum
- Space: Usually O(1)

---

## 9. Quick Reference Cards

### Template A: Opposite Ends
```python
left, right = 0, len(arr) - 1
while left < right:
    current = f(arr[left], arr[right])
    if current == target:
        return result
    elif current < target:
        left += 1
    else:
        right -= 1
```

### Template B: Fast/Slow
```python
slow = 0
for fast in range(len(arr)):
    if condition(arr[fast]):
        arr[slow] = arr[fast]
        slow += 1
return slow
```

### Template C: Two Arrays
```python
i, j = 0, 0
while i < len(arr1) and j < len(arr2):
    if arr1[i] <= arr2[j]:
        process(arr1[i])
        i += 1
    else:
        process(arr2[j])
        j += 1
# Handle remaining
```

---

## 10. Complexity Reference

| Pattern | Time | Space | Notes |
|---------|------|-------|-------|
| Two Sum (sorted) | O(n) | O(1) | Single pass |
| 3Sum | O(n²) | O(1) | Sort + nested loop |
| 4Sum | O(n³) | O(1) | Sort + 2 nested loops |
| Container Water | O(n) | O(1) | Single pass |
| Trapping Rain | O(n) | O(1) | Single pass, two pointers |
| Remove Duplicates | O(n) | O(1) | Single pass |
| Merge Arrays | O(n + m) | O(1) | In-place |
| Sort Colors | O(n) | O(1) | Single pass |
| Cycle Detection | O(n) | O(1) | Fast/slow |

---

## Final Thoughts

**Remember:**
1. Two pointers works because of **structure** (sorted order, constraints)
2. Three templates cover 95% of problems: opposite ends, fast/slow, two arrays
3. For sum problems on sorted arrays → opposite ends
4. For in-place modification → fast/slow
5. Always think about which pointer to move and why

**When stuck:**
1. Draw the array and mark pointer positions
2. Ask: "What property lets me eliminate possibilities?"
3. Try both templates mentally and see which fits
4. For N-Sum, reduce to 2Sum as base case

---

## Appendix: Practice Problem Set

### Easy (Master these first)
- 167. Two Sum II - Input Array Is Sorted
- 26. Remove Duplicates from Sorted Array
- 27. Remove Element
- 283. Move Zeroes
- 125. Valid Palindrome
- 88. Merge Sorted Array
- 141. Linked List Cycle
- 876. Middle of the Linked List

### Medium (Core interview level)
- 15. 3Sum
- 16. 3Sum Closest
- 18. 4Sum
- 11. Container With Most Water
- 75. Sort Colors
- 80. Remove Duplicates II
- 142. Linked List Cycle II
- 19. Remove Nth Node From End
- 680. Valid Palindrome II
- 349. Intersection of Two Arrays
- 350. Intersection of Two Arrays II

### Hard
- 42. Trapping Rain Water
- 287. Find the Duplicate Number
- 838. Push Dominoes

**Recommended Practice Order:**
1. Start with easy problems to build intuition
2. Master 3Sum (most common interview question)
3. Practice Container With Most Water for area problems
4. Do linked list problems for fast/slow mastery
5. Attempt Trapping Rain Water as advanced challenge

Good luck with your interview preparation!
