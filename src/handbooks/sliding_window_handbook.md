# The Complete Sliding Window Handbook
> A template-based approach for mastering sliding window in coding interviews

**Philosophy:** Sliding window is not about moving a box across an array. It's about **maintaining a valid window state efficiently** — adding elements on one side and removing from the other, avoiding redundant computation.

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

- **The Train Window**: You're on a train looking through a fixed-width window. The scenery changes one frame at a time -- one new thing enters the right side, one old thing exits the left. You don't re-examine the entire landscape each time.
- **The Rubber Band Window**: Now imagine the window can stretch or shrink. You stretch it (add from the right) when you want more, and shrink it (remove from the left) when you have too much. You're hunting for the sweet spot.

### No-Jargon Translation

- **Window**: a contiguous chunk of the array you're currently examining
- **Window state**: a running summary -- like a total or a frequency count -- of what's inside the window right now
- **Shrink condition**: the rule that tells you your window has grown too large or has violated a constraint
- **Subarray**: a contiguous slice of the original array -- no skipping elements

### Mental Model

> "A sliding window is a picture frame that you drag across a shelf of books, and instead of re-counting every book in the frame each time you slide, you just subtract the book that left and add the book that entered."

---

### Why Sliding Window?

The naive approach to subarray/substring problems checks all possible windows:
```python
# O(n²) or O(n³) - Check every subarray
for start in range(n):
    for end in range(start, n):
        if is_valid(arr[start:end+1]):
            update_result()
```

Sliding window reduces this to O(n) by **reusing computation** from the previous window:

> "Instead of recalculating everything for each window, I only account for the element entering and the element leaving."

### The Key Insight: Incremental Updates

**Example:** Finding maximum sum of subarray of size k.

```
arr = [1, 3, 2, 6, 4, 2], k = 3

Naive:
  sum([1,3,2]) = 6
  sum([3,2,6]) = 11  ← Recalculate from scratch
  sum([2,6,4]) = 12  ← Recalculate from scratch
  ...

Sliding Window:
  sum([1,3,2]) = 6
  sum - arr[0] + arr[3] = 6 - 1 + 6 = 11  ← Only 2 operations!
  sum - arr[1] + arr[4] = 11 - 3 + 4 = 12
  ...
```

**Why is this O(n)?**
- Each element enters the window exactly once
- Each element leaves the window exactly once
- Total operations: 2n = O(n)

### Two Types of Windows

#### Fixed-Size Window
```
[  |  |  |  ]        Window size = k (constant)
   └──────┘
     size k

Slide one position:
   [  |  |  |  ]
      └──────┘
        size k
```
**Use for:** Maximum/minimum/average of fixed-size subarray

#### Variable-Size Window
```
[  |  |  |  |  |  |  ]
   └────────────┘
   Window expands/contracts based on condition
```
**Use for:** Longest/shortest subarray satisfying a condition

### The Window State

The power of sliding window comes from maintaining **state** about the current window:

| State Type | Data Structure | Example Problem |
|------------|---------------|-----------------|
| Sum | Single variable | Max sum of size k |
| Count | Integer/HashMap | At most k distinct |
| Frequency | HashMap/Array | Anagram matching |
| Min/Max | Deque/Heap | Sliding window maximum |

**Key Principle:** Choose state that allows O(1) updates when adding/removing elements.

---

<a name="master-templates"></a>
## 2. The Two Master Templates

### Template A: Fixed-Size Window

**Use when:** Window size k is given

```python
def fixed_window(arr: list, k: int):
    """
    Template for fixed-size sliding window.
    Window always has exactly k elements.
    """
    n = len(arr)
    # Why `n < k`?
    # We cannot form even one window of size k if the array is shorter than k.
    # Without this guard, arr[:k] would silently succeed but the loop would
    # never execute, and we'd return an uninitialized result.
    if n < k:
        return None  # Not enough elements

    # Initialize: Build first window
    window_state = initialize_state(arr[:k])
    result = evaluate(window_state)

    # Slide: Remove left element, add right element
    # Why `range(k, n)` and not `range(k, n+1)`?
    # The last valid index is n-1. When right = n-1, the window is
    # [n-1-k+1, n-1], which is the last possible window. right = n would
    # be out of bounds.
    for right in range(k, n):
        left = right - k  # Element leaving the window

        # Remove arr[left] from state
        remove_from_state(window_state, arr[left])

        # Add arr[right] to state
        add_to_state(window_state, arr[right])

        # Update result
        result = update_result(result, window_state)

    return result
```

**Concrete Example: Maximum Sum of Size K**

```python
def max_sum_subarray(arr: list[int], k: int) -> int:
    if len(arr) < k:
        return 0

    # Initialize first window
    window_sum = sum(arr[:k])
    max_sum = window_sum

    # Slide window
    for right in range(k, len(arr)):
        left = right - k
        window_sum = window_sum - arr[left] + arr[right]
        max_sum = max(max_sum, window_sum)

    return max_sum
```

**Time Complexity:** O(n) — each element processed twice (enter and leave)

**Space Complexity:** O(1) for sum, O(k) for more complex state

---

### Template B: Variable-Size Window (Shrinkable)

**Use when:** Finding longest/shortest subarray satisfying a condition

```python
def variable_window(arr: list, condition):
    """
    Template for variable-size sliding window.
    Expand right to add elements, shrink left to maintain validity.
    """
    left = 0
    window_state = initial_state()
    result = initial_result()

    # Why `range(len(arr))` (i.e., right < len(arr)) and not `right <= len(arr)`?
    # `right` is an index into the array. The last valid index is len(arr)-1.
    # Using `<=` would make right = len(arr), which is out of bounds.
    for right in range(len(arr)):
        # EXPAND: Add arr[right] to window
        add_to_state(window_state, arr[right])

        # SHRINK: While window is invalid, remove from left
        # Why `while` and not `if`? One removal may not be enough to restore
        # validity. Example: window has 5 distinct chars and limit is 2 --
        # you may need to remove several elements from the left before the
        # window becomes valid again.
        # Why can't this loop run forever? Because left can never pass right.
        # Each iteration removes one element and increments left. Once
        # left > right the window is empty, which is always valid.
        while not is_valid(window_state):
            remove_from_state(window_state, arr[left])
            left += 1

        # UPDATE: Window [left, right] is now valid
        # Why is the window guaranteed valid here? The while-loop above
        # keeps shrinking until validity is restored (or the window is empty).
        result = update_result(result, left, right)

    return result
```

**Why this works:**

1. **Right pointer** always moves forward (expansion)
2. **Left pointer** only moves forward (contraction)
3. Neither pointer ever moves backward
4. Total movements: at most 2n = O(n)

**The Shrink Condition:**

The key insight is knowing **when** to shrink:

| Problem Type | Shrink When |
|--------------|-------------|
| At most K distinct | distinct_count > k |
| Sum ≤ target | sum > target |
| At most K zeros | zeros > k |
| Valid anagram | frequency mismatch |

---

### Template C: Variable Window with Two Passes (Exactly K)

**Use when:** Finding subarrays with **exactly** K of something

**Key Insight:** `exactly(K) = atMost(K) - atMost(K-1)`

```python
def exactly_k(arr: list, k: int) -> int:
    """
    Count subarrays with exactly k of something.
    Uses the identity: exactly(k) = atMost(k) - atMost(k-1)
    """
    return at_most(arr, k) - at_most(arr, k - 1)

def at_most(arr: list, k: int) -> int:
    """Count subarrays with at most k of something."""
    # Why check k < 0? When computing exactly(k) = atMost(k) - atMost(k-1),
    # if k=0 we call atMost(-1). Without this guard, the while-loop below
    # would shrink forever (window_state > -1 is almost always true),
    # making left shoot past right and producing garbage counts.
    if k < 0:
        return 0

    left = 0
    count = 0
    window_state = 0  # Track relevant property

    for right in range(len(arr)):
        # Add arr[right]
        window_state = update_add(window_state, arr[right])

        # Shrink if needed
        # Why `>` and not `>=`? We want "at most k", meaning k itself is
        # still valid. We only shrink when we EXCEED k.
        while window_state > k:
            window_state = update_remove(window_state, arr[left])
            left += 1

        # Count all valid subarrays ending at right.
        # Why `right - left + 1`?
        # Every subarray [i, right] where left <= i <= right is valid
        # (it's a subset of the valid window [left, right]).
        # The number of such starting positions is right - left + 1.
        # Why +1? Because both endpoints are inclusive: the count of
        # integers from left to right inclusive is right - left + 1.
        # Think of it as: [3,3] has 1 element (3-3+1=1), not 0.
        count += right - left + 1

    return count
```

**Why `right - left + 1`?**

For each valid window `[left, right]`, we count all subarrays ending at `right`:
- `[left, right]`
- `[left+1, right]`
- `[left+2, right]`
- ...
- `[right, right]`

That's `right - left + 1` subarrays, all valid (since they're subsets of a valid window).

---

### Quick Decision Matrix

| Problem Type | Template | Window Size | Key Operation |
|--------------|----------|-------------|---------------|
| Max/min of size k | A: Fixed | k (given) | Slide and compare |
| Longest valid | B: Variable | Expands | Shrink when invalid |
| Shortest valid | B: Variable | Contracts | Track min length |
| Count with exactly k | C: Two-pass | Varies | atMost(k) - atMost(k-1) |
| Count with at most k | B: Variable | Varies | Count subarrays |

---

<a name="pattern-guide"></a>
## 3. Pattern Classification Guide

### Category 1: Fixed-Size Window
- Window size k is explicitly given
- Find max/min/average of size-k subarray
- **Template A**
- Examples: Max sum size k, sliding window average

### Category 2: Longest Subarray/Substring
- Find longest window satisfying condition
- Window expands to try longer, shrinks when invalid
- **Template B** with max tracking
- Examples: Longest without repeating, longest with at most k distinct

### Category 3: Shortest Subarray/Substring
- Find shortest window satisfying condition
- Window shrinks aggressively once valid
- **Template B** with min tracking (different shrink logic)
- Examples: Minimum window substring, shortest subarray with sum

### Category 4: Count Subarrays
- Count number of valid subarrays
- Either "at most k" or "exactly k"
- **Template B or C**
- Examples: Subarrays with k distinct, subarrays with bounded max

### Category 5: String Pattern Matching
- Find anagrams, permutations in string
- Use frequency map as window state
- **Template A** (fixed size = pattern length)
- Examples: Find all anagrams, permutation in string

---

<a name="patterns"></a>
## 4. Complete Pattern Library

### PATTERN 1: Fixed-Size Window

---

#### Pattern 1A: Maximum Sum Subarray of Size K

**Problem:** Find maximum sum of any contiguous subarray of size k

**Example:** `arr = [2, 1, 5, 1, 3, 2]`, k = `3` → `9` (subarray [5, 1, 3])

```python
def max_sum_subarray_k(arr: list[int], k: int) -> int:
    # Why `len(arr) < k`?
    # If the array has fewer elements than k, no window of size k exists.
    # Returning 0 early avoids an empty sum and an idle loop.
    if len(arr) < k:
        return 0

    # Build first window
    window_sum = sum(arr[:k])
    max_sum = window_sum

    # Why `range(k, len(arr))`?
    # The first window [0, k-1] is already initialized above.
    # We start sliding from index k, where arr[i-k] is the element leaving
    # and arr[i] is the element entering, keeping the window at exactly size k.
    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i - k]  # Add new, remove old
        max_sum = max(max_sum, window_sum)

    return max_sum
```

**Visual:**
```
[2, 1, 5, 1, 3, 2], k=3

Window 1: [2, 1, 5] = 8
Window 2: [1, 5, 1] = 7  (8 - 2 + 1)
Window 3: [5, 1, 3] = 9  (7 - 1 + 3) ← Maximum
Window 4: [1, 3, 2] = 6  (9 - 5 + 2)
```

**Complexity:** Time O(n), Space O(1)

---

#### Pattern 1B: Sliding Window Maximum (Monotonic Deque)

**Problem:** LeetCode 239 - Maximum in each window of size k

**Example:** `nums = [1, 3, -1, -3, 5, 3, 6, 7]`, k = `3` → `[3, 3, 5, 5, 6, 7]`

**Key Insight:** Maintain a monotonic decreasing deque of indices

```python
from collections import deque

def maxSlidingWindow(nums: list[int], k: int) -> list[int]:
    result = []
    dq = deque()  # Store indices, values are monotonically decreasing

    for i in range(len(nums)):
        # Remove elements outside current window.
        # Why `dq[0] <= i - k`? The window covers indices [i-k+1, i].
        # Any index <= i-k is to the LEFT of the window, so it's stale.
        # Example: k=3, i=5 -> window is [3,4,5], so index 2 (<=5-3) is out.
        while dq and dq[0] <= i - k:
            dq.popleft()

        # Remove smaller elements (they can never be max).
        # Why `<` and not `<=`? Equal elements are kept because they might
        # outlast the current front of the deque. Keeping duplicates is safe
        # and avoids losing a valid maximum.
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()

        dq.append(i)

        # Why `i >= k - 1` and not `i >= k`?
        # The first complete window of size k ends at index k-1 (indices
        # 0 through k-1). So we start recording results from i = k-1.
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result
```

**Why monotonic decreasing?**
- Front of deque is always the maximum
- When new element arrives, smaller elements can never be max (new element is larger AND stays longer)
- So we remove all smaller elements from back

**Visual:**
```
nums = [1, 3, -1, -3, 5, 3, 6, 7], k=3

i=0: dq=[0], val=[1]
i=1: 3>1, pop 0, dq=[1], val=[3]
i=2: -1<3, dq=[1,2], val=[3,-1], output 3
i=3: -3<-1, dq=[1,2,3], val=[3,-1,-3], but 1 is out of window!
     dq=[2,3], output 3 (which is nums[2]=-1? No wait...)
```

Let me trace more carefully:
```
i=0: dq=[0]
i=1: nums[0]=1 < nums[1]=3, pop. dq=[1]
i=2: nums[1]=3 > nums[2]=-1, keep. dq=[1,2]. Window complete: max=nums[1]=3
i=3: Index 1 still in window [1,3]. nums[2]=-1 > nums[3]=-3, keep. dq=[1,2,3]. max=3
i=4: Index 1 out of window [2,4], remove. dq=[2,3]. nums[3]=-3 < 5, pop. nums[2]=-1 < 5, pop. dq=[4]. max=5
i=5: nums[4]=5 > nums[5]=3, keep. dq=[4,5]. max=5
i=6: nums[5]=3 < 6, pop. nums[4]=5 < 6, pop. dq=[6]. max=6
i=7: nums[6]=6 < 7, pop. dq=[7]. max=7

Output: [3, 3, 5, 5, 6, 7]
```

**Complexity:** Time O(n), Space O(k)

---

#### Pattern 1C: Find All Anagrams

**Problem:** LeetCode 438 - Find all start indices of anagrams of p in s

**Example:** `s = "cbaebabacd"`, `p = "abc"` → `[0, 6]`

**Key Insight:** Fixed window of size len(p), track character frequencies

```python
from collections import Counter

def findAnagrams(s: str, p: str) -> list[int]:
    # Why `len(s) < len(p)`?
    # A window of size len(p) cannot even fit inside s, so no anagram
    # is possible. Returning early avoids Counter comparisons on empty data.
    if len(s) < len(p):
        return []

    p_count = Counter(p)
    window_count = Counter(s[:len(p)])
    result = []

    # Check first window
    if window_count == p_count:
        result.append(0)

    # Slide window
    # Why start at len(p)? Indices 0..len(p)-1 are the first window,
    # already built above. Now we slide one character at a time.
    for i in range(len(p), len(s)):
        # Add new character
        window_count[s[i]] += 1

        # Remove old character
        old_char = s[i - len(p)]
        window_count[old_char] -= 1
        # Why `del` when count reaches 0? Python's Counter equality check
        # considers {a:1, b:0} != {a:1}. If we leave zero-count keys,
        # the `==` comparison with p_count would fail even when the
        # frequencies actually match. Deleting zeros keeps both dicts
        # in the same "shape" for a clean equality check.
        if window_count[old_char] == 0:
            del window_count[old_char]

        # Check if anagram.
        # Why `i - len(p) + 1`? The window currently spans
        # [i - len(p) + 1, i], so the start index is i - len(p) + 1.
        if window_count == p_count:
            result.append(i - len(p) + 1)

    return result
```

**Optimized version with match counting:**

```python
def findAnagrams_optimized(s: str, p: str) -> list[int]:
    if len(s) < len(p):
        return []

    p_count = Counter(p)
    result = []
    matches = 0  # Characters with correct frequency
    needed = len(p_count)  # Unique characters in p

    for i in range(len(s)):
        # Add s[i] to window
        char = s[i]
        # Why `if char in p_count`?
        # Characters not in p cannot contribute to or break an anagram match.
        # We only track "debt" for characters that p actually requires.
        if char in p_count:
            p_count[char] -= 1
            # Why check `== 0` specifically? p_count[char] tracks the
            # "debt" for this character. When it hits 0, the window has
            # exactly the right count for this char -- one more match.
            if p_count[char] == 0:
                matches += 1

        # Remove s[i - len(p)] from window.
        # Why `i >= len(p)` and not `i >= len(p) - 1`?
        # The first full window occupies indices [0, len(p)-1]. We only
        # start removing the leftmost element when adding index len(p),
        # because that is the first time the window exceeds size len(p).
        if i >= len(p):
            old_char = s[i - len(p)]
            if old_char in p_count:
                # Why check `== 0` BEFORE incrementing? If the count is
                # currently 0, this char was perfectly matched. Adding 1
                # will break that match, so we decrement matches first.
                if p_count[old_char] == 0:
                    matches -= 1
                p_count[old_char] += 1

        # Why `matches == needed`?
        # `needed` is the number of UNIQUE characters in p. `matches` counts
        # how many of those have exactly the right frequency in the window.
        # When they are equal, every unique character in p is satisfied --
        # the window is an anagram. Comparing counts (not the full dict)
        # is O(1) instead of O(alphabet).
        if matches == needed:
            result.append(i - len(p) + 1)

    return result
```

**Complexity:** Time O(n), Space O(1) — at most 26 characters

---

### PATTERN 2: Longest Subarray/Substring

---

#### Pattern 2A: Longest Substring Without Repeating Characters

**Problem:** LeetCode 3 - Find length of longest substring without repeating characters

**Example:** `s = "abcabcbb"` → `3` ("abc")

```python
def lengthOfLongestSubstring(s: str) -> int:
    char_index = {}  # Last index of each character
    left = 0
    max_length = 0

    for right in range(len(s)):
        char = s[right]

        # If char was seen and is in current window, shrink.
        # Why TWO conditions joined by `and`?
        # 1. `char in char_index` -- the character was seen before.
        # 2. `char_index[char] >= left` -- that previous occurrence is
        #    INSIDE our current window [left, right]. If the old index
        #    is to the LEFT of `left`, it was already evicted and is
        #    irrelevant -- we don't want to accidentally move left backward.
        # Why `char_index[char] + 1`? We jump left past the duplicate,
        # landing on the character right after it, so the window is clean.
        if char in char_index and char_index[char] >= left:
            left = char_index[char] + 1

        # Why always update `char_index[char] = right` (even after a jump)?
        # We need to remember the LATEST position of every character so that
        # future duplicates can jump `left` correctly. If we skip this update
        # after a jump, a repeated character later would reference a stale
        # (too-early) position, causing `left` to move backward -- wrong!
        char_index[char] = right
        max_length = max(max_length, right - left + 1)

    return max_length
```

**Alternative using set:**

```python
def lengthOfLongestSubstring_set(s: str) -> int:
    char_set = set()
    left = 0
    max_length = 0

    for right in range(len(s)):
        # Shrink until no duplicate.
        # Why `while` and not `if`? The duplicate of s[right] might not
        # be at position `left`. We must keep removing from the left until
        # we've removed the conflicting character. Example: window "abcd"
        # and s[right]='b' -- we remove 'a' first, then 'b', then stop.
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1

        char_set.add(s[right])
        max_length = max(max_length, right - left + 1)

    return max_length
```

**Visual:**
```
s = "abcabcbb"

right=0: 'a', window="a", max=1
right=1: 'b', window="ab", max=2
right=2: 'c', window="abc", max=3
right=3: 'a' duplicate! left jumps to 1, window="bca", max=3
right=4: 'b' duplicate! left jumps to 2, window="cab", max=3
right=5: 'c' duplicate! left jumps to 3, window="abc", max=3
right=6: 'b' duplicate! left jumps to 5, window="cb", max=3
right=7: 'b' duplicate! left jumps to 7, window="b", max=3

Answer: 3
```

**Complexity:** Time O(n), Space O(min(n, alphabet size))

---

#### Pattern 2B: Longest Substring with At Most K Distinct Characters

**Problem:** LeetCode 340 - Longest substring with at most k distinct characters

**Example:** `s = "eceba"`, k = `2` → `3` ("ece")

```python
from collections import defaultdict

def lengthOfLongestSubstringKDistinct(s: str, k: int) -> int:
    # Why `k == 0`?
    # With zero distinct characters allowed, no non-empty substring is valid.
    # Without this guard, the while-loop would shrink forever on any non-empty
    # window, making `left` overshoot `right` and producing garbage results.
    if k == 0:
        return 0

    char_count = defaultdict(int)
    left = 0
    max_length = 0
    distinct = 0

    for right in range(len(s)):
        # Add s[right].
        # Why check `== 0` before incrementing? If this character's count
        # is currently 0, it's not in the window yet -- adding it
        # introduces a NEW distinct character, so we increment `distinct`.
        if char_count[s[right]] == 0:
            distinct += 1
        char_count[s[right]] += 1

        # Shrink while too many distinct.
        # Why `>` and not `>=`? We want "at most k" distinct, so having
        # exactly k is still valid. We only shrink when we EXCEED k.
        while distinct > k:
            char_count[s[left]] -= 1
            # Why check `== 0` after decrementing? If the count just
            # dropped to 0, this character has been fully removed from the
            # window -- one fewer distinct character.
            if char_count[s[left]] == 0:
                distinct -= 1
            left += 1

        max_length = max(max_length, right - left + 1)

    return max_length
```

**Visual:**
```
s = "eceba", k=2

right=0: 'e', distinct=1, window="e", max=1
right=1: 'c', distinct=2, window="ec", max=2
right=2: 'e', distinct=2, window="ece", max=3
right=3: 'b', distinct=3 > k!
         shrink: remove 'e', distinct still 3 (have another 'e')
         shrink: remove 'c', distinct=2, window="eb", max=3
right=4: 'a', distinct=3 > k!
         shrink: remove 'e', distinct=2, window="ba", max=3

Answer: 3
```

**Complexity:** Time O(n), Space O(k)

---

#### Pattern 2C: Longest Repeating Character Replacement

**Problem:** LeetCode 424 - Longest substring with same letter after at most k changes

**Example:** `s = "AABABBA"`, k = `1` → `4` ("AABA" → "AAAA")

**Key Insight:** Window is valid if `window_length - max_char_count <= k`

```python
from collections import defaultdict

def characterReplacement(s: str, k: int) -> int:
    char_count = defaultdict(int)
    left = 0
    max_count = 0  # Count of most frequent char in window
    max_length = 0

    for right in range(len(s)):
        char_count[s[right]] += 1
        # Why update `max_count` immediately after adding?
        # `max_count` tracks the highest frequency of any single character in
        # the current window. We use it to compute how many replacements are
        # needed. If we skip this update, `max_count` lags behind and we may
        # fail to recognize a newly valid (longer) window.
        max_count = max(max_count, char_count[s[right]])

        # Window invalid if we need more than k replacements.
        # Why `window_size - max_count`? To make all characters in the
        # window the same, we keep the most frequent character and replace
        # the rest. The "rest" is window_size - max_count characters.
        # Why `if` and not `while`? This is a deliberate optimization.
        # We only shrink by 1 because we only grew by 1 (adding s[right]).
        # The window can be at most 1 character too large, so a single
        # shrink restores validity. This also means the window size never
        # decreases -- it either stays the same or grows, acting as a
        # "high water mark" that records our best length so far.
        window_size = right - left + 1
        if window_size - max_count > k:
            # Shrink by 1
            char_count[s[left]] -= 1
            left += 1

        max_length = max(max_length, right - left + 1)

    return max_length
```

**Why don't we decrease max_count when shrinking?**

This is a subtle optimization:
- We only care about finding a LONGER valid window
- A window can only be longer if it has a higher max_count
- So keeping max_count at its historical maximum is fine
- If we find a window where current max equals historical max, it might be longer

**Complexity:** Time O(n), Space O(26) = O(1)

---

#### Pattern 2D: Max Consecutive Ones III

**Problem:** LeetCode 1004 - Longest subarray of 1s after flipping at most k 0s

**Example:** `nums = [1,1,1,0,0,0,1,1,1,1,0]`, k = `2` → `6`

```python
def longestOnes(nums: list[int], k: int) -> int:
    left = 0
    zeros = 0
    max_length = 0

    for right in range(len(nums)):
        # Add nums[right]
        # Why `nums[right] == 0`?
        # We only count zeros because they represent flips. A 1 costs nothing
        # and can always be included; only 0s consume our flip budget.
        if nums[right] == 0:
            zeros += 1

        # Shrink while too many zeros.
        # Why `zeros > k` and not `zeros >= k`? We're allowed to flip
        # AT MOST k zeros. Having exactly k flipped zeros is valid.
        # We only shrink when we've used MORE flips than allowed.
        while zeros > k:
            # Why `if nums[left] == 0: zeros -= 1` before `left += 1`?
            # Removing a 1 from the left doesn't free up a flip slot.
            # Only when a 0 leaves the window does our flip count decrease.
            if nums[left] == 0:
                zeros -= 1
            left += 1

        max_length = max(max_length, right - left + 1)

    return max_length
```

**Visual:**
```
[1,1,1,0,0,0,1,1,1,1,0], k=2

Window expands: [1,1,1,0,0] zeros=2, length=5
Add 0: [1,1,1,0,0,0] zeros=3 > k=2
Shrink: [1,1,0,0,0] zeros=3
Shrink: [1,0,0,0] zeros=3
Shrink: [0,0,0] zeros=3
Shrink: [0,0] zeros=2, valid!
Continue: [0,0,1,1,1,1] zeros=2, length=6 ← max
```

**Complexity:** Time O(n), Space O(1)

---

### PATTERN 3: Shortest Subarray/Substring

---

#### Pattern 3A: Minimum Size Subarray Sum

**Problem:** LeetCode 209 - Shortest subarray with sum >= target

**Example:** `nums = [2,3,1,2,4,3]`, target = `7` → `2` (subarray [4,3])

```python
def minSubArrayLen(target: int, nums: list[int]) -> int:
    left = 0
    current_sum = 0
    min_length = float('inf')

    for right in range(len(nums)):
        current_sum += nums[right]

        # Shrink while valid (sum >= target).
        # Why shrink when VALID (not invalid)?
        # For "shortest" problems, a valid window means we HAVE a candidate.
        # We record its length, then try to make it shorter by removing
        # from the left. We keep shrinking as long as the window stays valid
        # because an even shorter valid window would be a better answer.
        # This is the opposite of "longest" problems where we shrink when
        # INVALID to restore validity.
        while current_sum >= target:
            # Why record `min_length` BEFORE shrinking?
            # At this moment the window [left, right] is valid. If we shrink
            # first and then record, we'd miss this valid window entirely.
            # The update must precede the removal of arr[left].
            min_length = min(min_length, right - left + 1)
            current_sum -= nums[left]
            left += 1

    # Why this ternary? If min_length was never updated from infinity,
    # no valid subarray exists, so we return 0 per the problem spec.
    return min_length if min_length != float('inf') else 0
```

**Key Difference from Longest:**
- For **longest**: shrink when **invalid**, track length **after** shrinking
- For **shortest**: shrink when **valid**, track length **before** shrinking

**Visual:**
```
[2,3,1,2,4,3], target=7

Expand to [2,3,1,2] = 8 >= 7, min=4
Shrink: [3,1,2] = 6 < 7, stop shrinking
Expand: [3,1,2,4] = 10 >= 7, min=4
Shrink: [1,2,4] = 7 >= 7, min=3
Shrink: [2,4] = 6 < 7, stop
Expand: [2,4,3] = 9 >= 7, min=3
Shrink: [4,3] = 7 >= 7, min=2 ← answer
Shrink: [3] = 3 < 7, stop

Answer: 2
```

**Complexity:** Time O(n), Space O(1)

---

#### Pattern 3B: Minimum Window Substring

**Problem:** LeetCode 76 - Shortest substring of s containing all characters of t

**Example:** `s = "ADOBECODEBANC"`, `t = "ABC"` → `"BANC"`

```python
from collections import Counter

def minWindow(s: str, t: str) -> str:
    # Why `not t or not s`?
    # An empty target t means every window matches (ambiguous), and an empty
    # s means no window exists. Both are handled as "no valid window" to
    # avoid dividing by zero or iterating over empty strings.
    if not t or not s:
        return ""

    t_count = Counter(t)
    # Why `len(t_count)` and not `len(t)`?
    # We care about unique characters, not total characters. If t = "AAB",
    # we need A (count 2) and B (count 1) -- that's 2 unique chars, not 3.
    # `formed` tracks how many unique chars have met their target frequency.
    required = len(t_count)  # Unique characters needed

    left = 0
    formed = 0  # Unique characters with correct frequency
    window_count = {}

    min_len = float('inf')
    min_left = 0

    for right in range(len(s)):
        # Add s[right]
        char = s[right]
        window_count[char] = window_count.get(char, 0) + 1

        # Why `== t_count[char]` specifically (not `>=`)?
        # We only increment `formed` at the exact moment this character's
        # count REACHES the required frequency. If we already had enough
        # (count was already >= required), formed was already incremented
        # for this character on a previous step. Using `>=` would
        # double-count.
        if char in t_count and window_count[char] == t_count[char]:
            formed += 1

        # Shrink while valid (same "shortest" pattern as Pattern 3A).
        # Why `formed == required`? This means every unique character in t
        # has met its required frequency in our window -- the window
        # contains all characters of t. Time to record and try shorter.
        while formed == required:
            # Update minimum
            if right - left + 1 < min_len:
                min_len = right - left + 1
                min_left = left

            # Remove s[left]
            left_char = s[left]
            window_count[left_char] -= 1
            # Why `<` and not `<=` or `==`?
            # After decrementing, if the count drops BELOW what t needs,
            # we just lost a required character. The `<` catches the exact
            # moment we go from "enough" to "not enough" for this character.
            if left_char in t_count and window_count[left_char] < t_count[left_char]:
                formed -= 1
            left += 1

    return "" if min_len == float('inf') else s[min_left:min_left + min_len]
```

**Key Insight:** Track two things:
1. `required`: how many unique characters we need
2. `formed`: how many we currently have with sufficient frequency

**Visual:**
```
s = "ADOBECODEBANC", t = "ABC"
t_count = {A:1, B:1, C:1}, required = 3

Expand until formed=3:
"ADOBEC" has A:1, B:1, C:1, formed=3
→ min="ADOBEC" (6)

Shrink:
Remove A: "DOBEC", formed=2, stop

Expand:
"DOBECODEBA" has A:1, B:2, C:1, formed=3
→ min="ADOBEC" (6)

Shrink:
"OBECODEBA" formed=2 (lost one A wait no, we got another A)
Actually let me retrace...

This gets complex - the key is tracking when formed == required.
Final answer: "BANC" (4)
```

**Complexity:** Time O(|s| + |t|), Space O(|s| + |t|)

---

### PATTERN 4: Counting Subarrays

---

#### Pattern 4A: Subarrays with K Different Integers

**Problem:** LeetCode 992 - Count subarrays with exactly k distinct integers

**Example:** `nums = [1,2,1,2,3]`, k = `2` → `7`

**Key Formula:** `exactly(k) = atMost(k) - atMost(k-1)`

```python
from collections import defaultdict

def subarraysWithKDistinct(nums: list[int], k: int) -> int:
    def at_most_k(k: int) -> int:
        if k < 0:
            return 0

        count = defaultdict(int)
        left = 0
        result = 0
        distinct = 0

        for right in range(len(nums)):
            # Why check `count[nums[right]] == 0` BEFORE incrementing?
            # If the count is 0, this element isn't in the window yet.
            # Adding it will introduce a new distinct value, so we increment
            # `distinct` first. After the increment, the count becomes 1.
            if count[nums[right]] == 0:
                distinct += 1
            count[nums[right]] += 1

            # Why `distinct > k` and not `distinct >= k`?
            # "At most k" means k distinct values are still valid. We only
            # shrink when we have strictly MORE than k distinct values.
            while distinct > k:
                count[nums[left]] -= 1
                # Why check `count[nums[left]] == 0` AFTER decrementing?
                # Only when the count drops to exactly 0 has this element
                # been fully removed from the window. Until then, duplicates
                # remain and the distinct count must not change.
                if count[nums[left]] == 0:
                    distinct -= 1
                left += 1

            # Why `right - left + 1`?
            # Every subarray ending at `right` with start index in
            # [left, right] is a sub-window of our valid window, so it
            # also has at most k distinct values. There are right-left+1
            # such start positions (both endpoints inclusive).
            result += right - left + 1

        return result

    return at_most_k(k) - at_most_k(k - 1)
```

**Why `right - left + 1`?**

For window `[left, right]` with at most k distinct:
- All subarrays ending at `right` with start in `[left, right]` are valid
- That's `right - left + 1` subarrays

**Example breakdown:**
```
nums = [1,2,1,2,3], k=2

atMost(2):
right=0: [1], +1, total=1
right=1: [1,2], [2], +2, total=3
right=2: [1,2,1], [2,1], [1], +3, total=6
right=3: [1,2,1,2], [2,1,2], [1,2], [2], +4, total=10
right=4: [3] too many distinct, shrink to [2,3], [3], +2, total=12

atMost(1):
right=0: [1], +1, total=1
right=1: shrink to [2], +1, total=2
right=2: [2,1] has 2 distinct, shrink to [1], +1, total=3
right=3: shrink to [2], +1, total=4
right=4: shrink to [3], +1, total=5

exactly(2) = 12 - 5 = 7
```

**Complexity:** Time O(n), Space O(n)

---

#### Pattern 4B: Count Number of Nice Subarrays

**Problem:** LeetCode 1248 - Count subarrays with exactly k odd numbers

**Example:** `nums = [1,1,2,1,1]`, k = `3` → `2`

```python
def numberOfSubarrays(nums: list[int], k: int) -> int:
    def at_most(k: int) -> int:
        if k < 0:
            return 0

        left = 0
        odds = 0
        result = 0

        for right in range(len(nums)):
            # Why `nums[right] % 2 == 1`?
            # Odd numbers are what we're counting. Even numbers don't affect
            # our budget, so we skip them. Only odds consume a slot in k.
            if nums[right] % 2 == 1:
                odds += 1

            # Why `odds > k` and not `odds >= k`?
            # At most k odds are allowed. Having exactly k is valid.
            # We shrink only when we've exceeded the budget.
            while odds > k:
                # Why `if nums[left] % 2 == 1: odds -= 1` before `left += 1`?
                # Only an odd number leaving the window reduces the count.
                # An even number leaving costs nothing from our budget.
                if nums[left] % 2 == 1:
                    odds -= 1
                left += 1

            result += right - left + 1

        return result

    return at_most(k) - at_most(k - 1)
```

**Complexity:** Time O(n), Space O(1)

---

#### Pattern 4C: Binary Subarrays With Sum

**Problem:** LeetCode 930 - Count subarrays with sum equal to goal

**Example:** `nums = [1,0,1,0,1]`, goal = `2` → `4`

```python
def numSubarraysWithSum(nums: list[int], goal: int) -> int:
    def at_most(goal: int) -> int:
        if goal < 0:
            return 0

        left = 0
        current_sum = 0
        result = 0

        for right in range(len(nums)):
            current_sum += nums[right]

            # Why `current_sum > goal` and not `>= goal`?
            # "At most goal" means a sum equal to `goal` is valid.
            # We only shrink when the sum EXCEEDS the allowed maximum.
            # (This works here because nums is binary -- all 0s and 1s --
            # so the sum changes monotonically as the window grows.)
            while current_sum > goal:
                current_sum -= nums[left]
                left += 1

            result += right - left + 1

        return result

    return at_most(goal) - at_most(goal - 1)
```

**Alternative: Prefix Sum + HashMap** (for non-sliding-window approach)

```python
from collections import defaultdict

def numSubarraysWithSum_prefix(nums: list[int], goal: int) -> int:
    prefix_count = defaultdict(int)
    prefix_count[0] = 1
    current_sum = 0
    result = 0

    for num in nums:
        current_sum += num
        result += prefix_count[current_sum - goal]
        prefix_count[current_sum] += 1

    return result
```

**Complexity:** Time O(n), Space O(1) for sliding window

---

### PATTERN 5: String Matching with Constraints

---

#### Pattern 5A: Permutation in String

**Problem:** LeetCode 567 - Check if s2 contains permutation of s1

**Example:** `s1 = "ab"`, `s2 = "eidbaooo"` → `True` ("ba" is permutation)

```python
from collections import Counter

def checkInclusion(s1: str, s2: str) -> bool:
    # Why `len(s1) > len(s2)`?
    # A permutation of s1 cannot fit inside s2 if s1 is longer.
    # Without this guard we'd iterate over an empty range and always return
    # False anyway, but the early exit makes the intent explicit.
    if len(s1) > len(s2):
        return False

    s1_count = Counter(s1)
    window_count = Counter()
    required = len(s1_count)
    formed = 0

    for i in range(len(s2)):
        # Add s2[i]
        char = s2[i]
        window_count[char] += 1

        # Why `window_count[char] == s1_count[char]`?
        # At the exact moment the window count REACHES the required count,
        # this character is satisfied. Any further additions would over-supply
        # it (handled by the elif below), so we count the match exactly once.
        if char in s1_count and window_count[char] == s1_count[char]:
            formed += 1
        # Why `== s1_count[char] + 1`?
        # The window now has ONE MORE than required for this character.
        # It just crossed from "satisfied" to "over-supplied", so we
        # decrement `formed` to reflect the lost match.
        elif char in s1_count and window_count[char] == s1_count[char] + 1:
            formed -= 1  # We now have too many

        # Remove s2[i - len(s1)] if window is too large
        # Why `i >= len(s1)` and not `i > len(s1)`?
        # When i == len(s1), the window spans [0, len(s1)], which is one
        # element too large (size len(s1)+1). We must remove index 0 now.
        if i >= len(s1):
            left_char = s2[i - len(s1)]
            # Why check `== s1_count[left_char]` BEFORE decrementing?
            # The count is currently exactly right (matched). Removing one
            # will break the match, so we decrement `formed` first.
            if left_char in s1_count and window_count[left_char] == s1_count[left_char]:
                formed -= 1
            # Why `== s1_count[left_char] + 1` here?
            # The window had one too many of this char (over-supplied).
            # Removing one brings it back to exactly the required count --
            # a match is restored, so we increment `formed`.
            elif left_char in s1_count and window_count[left_char] == s1_count[left_char] + 1:
                formed += 1  # We had too many, now correct
            window_count[left_char] -= 1

        # Why `formed == required`?
        # Every unique character in s1 has exactly the right frequency in
        # the current window. This is precisely the definition of a permutation.
        if formed == required:
            return True

    return False
```

**Simpler version comparing counters:**

```python
def checkInclusion_simple(s1: str, s2: str) -> bool:
    # Why `len(s1) > len(s2)` again here?
    # Same guard as in the main version: permutation cannot fit, fail fast.
    if len(s1) > len(s2):
        return False

    s1_count = Counter(s1)
    window_count = Counter(s2[:len(s1)])

    # Why check the first window before the loop?
    # The loop starts at index len(s1) and slides the window forward.
    # The initial window [0, len(s1)-1] is never checked inside the loop,
    # so we must check it here or we'd miss it entirely.
    if window_count == s1_count:
        return True

    for i in range(len(s1), len(s2)):
        # Add new char
        window_count[s2[i]] += 1
        # Remove old char
        old_char = s2[i - len(s1)]
        window_count[old_char] -= 1
        # Why `del window_count[old_char]` when count reaches 0?
        # Counter equality checks consider {a:1, b:0} != {a:1}.
        # Leaving zero-count keys causes false negatives when comparing
        # with s1_count, so we remove them to keep the dict in canonical form.
        if window_count[old_char] == 0:
            del window_count[old_char]

        if window_count == s1_count:
            return True

    return False
```

**Complexity:** Time O(n), Space O(1) — at most 26 characters

---

### PATTERN 6: Advanced Window State

---

#### Pattern 6A: Sliding Window Median

**Problem:** LeetCode 480 - Median of each window of size k

This requires maintaining sorted order in the window.

```python
import bisect

def medianSlidingWindow(nums: list[int], k: int) -> list[float]:
    window = sorted(nums[:k])
    result = []

    def get_median():
        if k % 2 == 1:
            return float(window[k // 2])
        return (window[k // 2 - 1] + window[k // 2]) / 2

    result.append(get_median())

    for i in range(k, len(nums)):
        # Remove outgoing element
        outgoing = nums[i - k]
        idx = bisect.bisect_left(window, outgoing)
        window.pop(idx)

        # Add incoming element
        incoming = nums[i]
        bisect.insort(window, incoming)

        result.append(get_median())

    return result
```

**Better approach: Two heaps** (max-heap for lower half, min-heap for upper half)

```python
import heapq
from collections import defaultdict

def medianSlidingWindow_heaps(nums: list[int], k: int) -> list[float]:
    # small: max-heap (negate values), large: min-heap
    small = []  # Lower half
    large = []  # Upper half
    removed = defaultdict(int)  # Lazy removal

    def add(num):
        if not small or num <= -small[0]:
            heapq.heappush(small, -num)
        else:
            heapq.heappush(large, num)

    def remove(num):
        removed[num] += 1
        # Lazy removal: don't actually remove, just mark

    def balance():
        # Balance sizes: small can have at most 1 more than large
        while len(small) > len(large) + 1:
            heapq.heappush(large, -heapq.heappop(small))
        while len(large) > len(small):
            heapq.heappush(small, -heapq.heappop(large))

    def prune(heap, sign):
        # Remove marked elements from top
        while heap and removed[-sign * heap[0] if sign == -1 else heap[0]] > 0:
            removed[-sign * heap[0] if sign == -1 else heap[0]] -= 1
            heapq.heappop(heap)

    def get_median():
        if k % 2 == 1:
            return float(-small[0])
        return (-small[0] + large[0]) / 2

    # Initialize
    for i in range(k):
        add(nums[i])
        balance()

    result = [get_median()]

    for i in range(k, len(nums)):
        add(nums[i])
        remove(nums[i - k])

        # Rebalance after lazy removal
        balance()
        prune(small, -1)
        prune(large, 1)

        result.append(get_median())

    return result
```

**Complexity:** Time O(n log k), Space O(k)

---

<a name="post-processing"></a>
## 5. Post-Processing Reference

| Problem Type | Return Value | Edge Cases |
|--------------|--------------|------------|
| **Max in window** | Single value or array | Empty array, k > n |
| **Longest valid** | Length (integer) | No valid window → 0 |
| **Shortest valid** | Length or 0 | No valid window → 0 |
| **Count subarrays** | Count (integer) | k = 0, empty array |
| **Find substring** | String or "" | Not found → "" |
| **Check existence** | Boolean | False if not found |

---

<a name="pitfalls"></a>
## 6. Common Pitfalls & Solutions

### Pitfall 1: Off-by-One in Window Boundaries

**Problem:** Confusion about window indices

```python
# WRONG: Window size calculation
window_size = right - left  # Missing +1!
```

**Solution:** Window `[left, right]` has size `right - left + 1`

---

### Pitfall 2: Not Handling Empty Result

**Problem:** Returning infinity or garbage for "not found"

```python
# WRONG: Return infinity
min_len = float('inf')
return min_len  # Should return 0!
```

**Solution:**
```python
return min_len if min_len != float('inf') else 0
```

---

### Pitfall 3: Shrinking Logic for Longest vs Shortest

**Problem:** Using wrong shrink condition

```python
# For LONGEST: shrink when INVALID
while invalid_condition:
    shrink()
# Record length AFTER shrinking (window is valid)

# For SHORTEST: shrink when VALID
while valid_condition:
    record_length()  # BEFORE shrinking!
    shrink()
```

---

### Pitfall 4: Counter Goes Negative

**Problem:** Not handling zero counts properly

```python
# WRONG: Distinct count becomes wrong
if char_count[char] == 0:
    distinct -= 1
char_count[char] -= 1  # Now it's -1!
```

**Solution:** Check before decrementing or delete zero entries
```python
char_count[char] -= 1
if char_count[char] == 0:
    distinct -= 1
    del char_count[char]  # Optional: remove zero entries
```

---

### Pitfall 5: Fixed Window First Element

**Problem:** Forgetting to initialize first window

```python
# WRONG: Starts from index 0, misses initialization
for i in range(len(nums)):
    window_sum += nums[i] - nums[i - k]  # i-k is negative!
```

**Solution:** Initialize first k elements, then start sliding from k
```python
window_sum = sum(nums[:k])
for i in range(k, len(nums)):
    window_sum += nums[i] - nums[i - k]
```

---

### Pitfall 6: atMost with Negative K

**Problem:** atMost(-1) should return 0, not crash

```python
# WRONG: No check for negative k
def at_most(k):
    while current > k:  # Infinite loop if k < 0!
```

**Solution:**
```python
def at_most(k):
    if k < 0:
        return 0
    # ... rest of function
```

---

<a name="recognition"></a>
## 7. Problem Recognition Framework

### Step 1: Is Sliding Window Applicable?

**Key indicators:**
1. "Subarray" or "substring" mentioned
2. "Contiguous" elements
3. "Window of size k"
4. "Longest/shortest satisfying condition"
5. "Count subarrays with property"

**NOT sliding window if:**
- Need non-contiguous elements
- Array is not processed linearly
- Optimal substructure suggests DP
- **Negative numbers + sum target**: Sliding window assumes adding elements makes the sum larger (monotonicity). Negative numbers break this — use Prefix Sum + HashMap instead.

### Step 2: Fixed or Variable Window?

| Clue | Window Type |
|------|-------------|
| "Window of size k" | Fixed |
| "Exactly k elements" | Fixed or Variable (with trick) |
| "Longest..." | Variable |
| "Shortest..." | Variable |
| "At most k..." | Variable |
| "Count subarrays" | Variable |

### Step 3: What State to Maintain?

| Problem Type | State |
|--------------|-------|
| Sum-based | Running sum |
| Distinct count | HashMap + counter |
| Frequency match | Two HashMaps or one with delta |
| Max in window | Monotonic deque |
| Median | Two heaps |

### Decision Tree

```
              Is it a subarray/substring problem?
                           ↓
                    ┌──────┴──────┐
                   Yes            No
                    ↓              ↓
            Window size given?   Other approach
                    ↓
             ┌──────┴──────┐
            Yes            No
             ↓              ↓
      Template A      Longest or Shortest?
       (Fixed)              ↓
                     ┌──────┴──────┐
                 Longest        Shortest
                     ↓              ↓
              Shrink when     Shrink when
               INVALID          VALID
                     ↓              ↓
                Template B    Template B
                             (modified)
```

---

<a name="checklist"></a>
## 8. Interview Preparation Checklist

### Before the Interview

**Master the fundamentals:**
- [ ] Can write fixed-window template from memory
- [ ] Can write variable-window template from memory
- [ ] Understand shrink conditions for longest vs shortest
- [ ] Know the `exactly(k) = atMost(k) - atMost(k-1)` trick

**Practice pattern recognition:**
- [ ] Can identify sliding window problems quickly
- [ ] Know what state to maintain for each problem type
- [ ] Understand time complexity analysis

**Know the patterns:**
- [ ] Max sum of size k
- [ ] Sliding window maximum (monotonic deque)
- [ ] Longest without repeating
- [ ] Longest with at most k distinct
- [ ] Minimum window substring
- [ ] Count subarrays with exactly k

**Common problems solved:**
- [ ] LC 3: Longest Substring Without Repeating Characters
- [ ] LC 76: Minimum Window Substring
- [ ] LC 239: Sliding Window Maximum
- [ ] LC 438: Find All Anagrams
- [ ] LC 567: Permutation in String
- [ ] LC 209: Minimum Size Subarray Sum
- [ ] LC 992: Subarrays with K Different Integers

### During the Interview

**1. Clarify (30 seconds)**
- Contiguous required?
- What defines valid window?
- Return length or actual subarray?

**2. Identify pattern (30 seconds)**
- Fixed or variable window?
- Longest, shortest, or count?
- What state to maintain?

**3. Code (3-4 minutes)**
- Write template
- Define add/remove state operations
- Define valid/invalid condition
- Handle edge cases

**4. Test (1-2 minutes)**
- Empty input
- Window larger than array
- All elements same
- No valid window exists

**5. Analyze (30 seconds)**
- Time: Usually O(n)
- Space: O(1) or O(k) or O(alphabet)

---

## 9. Quick Reference Cards

### Template A: Fixed Window
```python
window = initialize(arr[:k])
result = evaluate(window)
for i in range(k, n):
    remove(window, arr[i-k])
    add(window, arr[i])
    result = update(result, window)
return result
```

### Template B: Variable Window (Longest)
```python
left = 0
for right in range(n):
    add(arr[right])
    while invalid():
        remove(arr[left])
        left += 1
    result = max(result, right - left + 1)
return result
```

### Template B: Variable Window (Shortest)
```python
left = 0
for right in range(n):
    add(arr[right])
    while valid():
        result = min(result, right - left + 1)
        remove(arr[left])
        left += 1
return result
```

### Template C: Exactly K
```python
def exactly_k(k):
    return at_most(k) - at_most(k-1)
```

---

## 10. Complexity Reference

| Pattern | Time | Space | Notes |
|---------|------|-------|-------|
| Fixed window sum | O(n) | O(1) | Simple accumulator |
| Fixed window max | O(n) | O(k) | Monotonic deque |
| Longest valid | O(n) | O(k) or O(1) | Each element enters/leaves once |
| Shortest valid | O(n) | O(k) or O(1) | Same as longest |
| Count subarrays | O(n) | O(k) | Two passes for exactly k |
| Anagram finding | O(n) | O(26) | Fixed alphabet size |
| Sliding median | O(n log k) | O(k) | Heap operations |

---

## Final Thoughts

**Remember:**
1. Sliding window works because **elements enter and leave exactly once**
2. Fixed window: size is constant, slide by adding one and removing one
3. Variable window: expand right, shrink left based on validity
4. For "exactly k": use `atMost(k) - atMost(k-1)`
5. Choose state that allows O(1) updates

**When stuck:**
1. Draw the array and mark window boundaries
2. Ask: "What makes a window valid/invalid?"
3. Ask: "What state do I need to check validity in O(1)?"
4. Trace through a small example step by step

---

## Appendix: Practice Problem Set

### Easy
- 643. Maximum Average Subarray I
- 219. Contains Duplicate II
- 1984. Minimum Difference Between Highest and Lowest of K Scores

### Medium (Core interview level)
- 3. Longest Substring Without Repeating Characters
- 159. Longest Substring with At Most Two Distinct Characters
- 340. Longest Substring with At Most K Distinct Characters
- 424. Longest Repeating Character Replacement
- 438. Find All Anagrams in a String
- 567. Permutation in String
- 209. Minimum Size Subarray Sum
- 713. Subarray Product Less Than K
- 904. Fruit Into Baskets
- 1004. Max Consecutive Ones III
- 1248. Count Number of Nice Subarrays

### Hard
- 76. Minimum Window Substring
- 239. Sliding Window Maximum
- 480. Sliding Window Median
- 992. Subarrays with K Different Integers
- 995. Minimum Number of K Consecutive Bit Flips

**Recommended Practice Order:**
1. Start with LC 643 (fixed window basics)
2. Master LC 3 (most common interview question)
3. Do LC 209 for shortest window pattern
4. Practice LC 438 and 567 for string matching
5. Attempt LC 76 as the classic hard problem
6. Try LC 992 for the "exactly k" trick

Good luck with your interview preparation!

---

## Appendix: Conditional Quick Reference

This table lists every key condition used in this handbook, its plain-English meaning, and the intuition behind it.

### A. Window Validity & Shrink Conditions

| Condition | Plain English | Why it works |
|-----------|---------------|--------------|
| `while not is_valid(window_state)` | Keep shrinking until the window obeys the constraint | One removal may not be enough; `while` ensures full restoration before recording the result |
| `while distinct > k` | The window has more unique characters than allowed | Strict `>` lets exactly k pass; shrinking removes elements from the left until we're back within budget |
| `while zeros > k` | More zeros have been flipped than the budget allows | Each `left` step potentially frees one flip slot; we loop until we're within k flips |
| `while current_sum >= target` (shortest) | The window sum is still valid — keep shrinking to find a shorter answer | For "shortest", validity triggers shrinking (opposite of "longest"); we record length before each shrink |
| `while current_sum > goal` (count) | Sum exceeds the "at most" ceiling | `>` not `>=` because equality is still a valid window for counting subarrays |
| `while odds > k` | More odd numbers in the window than the budget | Strict `>` keeps windows with exactly k odds valid; shrink only on over-budget |
| `window_size - max_count > k` (`if`, not `while`) | More replacements needed than k allows | Only one shrink needed because the window grew by exactly one; using `if` preserves the high-water mark |
| `while window_state > k` in `at_most` | Generic "at most k" shrink | Shrink until the tracked property is back within the allowed limit |

### B. Frequency / Count Tracking Conditions

| Condition | Plain English | Why it works |
|-----------|---------------|--------------|
| `if count[elem] == 0` before `count[elem] += 1` then `distinct += 1` | Element is new to the window | Checking before the increment catches the exact moment a new unique value enters; avoids double-counting |
| `if count[elem] == 0` after `count[elem] -= 1` then `distinct -= 1` | Element has been fully removed from the window | Checking after the decrement catches the exact moment the last copy leaves; distinct count drops by one |
| `if window_count[old_char] == 0: del window_count[old_char]` | Zero-count key must be purged before equality check | Python's Counter treats `{a:1, b:0}` as unequal to `{a:1}`; deleting zeros keeps the dict in canonical form |
| `if p_count[char] == 0: matches += 1` | Window has now exactly satisfied the required count for this character | Only fire at the exact crossing point (debt reaches 0); `>=` would double-count earlier satisfactions |
| `if p_count[old_char] == 0: matches -= 1` before restoring | Character was perfectly matched; removing one copy breaks the match | We must decrement before incrementing back, preserving the correct moment the match is lost |
| `if char in t_count and window_count[char] == t_count[char]: formed += 1` | The window has just met the required frequency for this character | `==` not `>=` ensures we count the transition exactly once, avoiding inflation of `formed` |
| `if left_char in t_count and window_count[left_char] < t_count[left_char]: formed -= 1` | A required character has dropped below its needed frequency | `<` detects the precise moment we lose a character's match when shrinking |

### C. Result Update Conditions

| Condition | Plain English | Why it works |
|-----------|---------------|--------------|
| `result = max(result, right - left + 1)` after shrink loop | Window is valid; record if it's the longest seen | The `while` loop above guarantees validity at this point, so this is always a safe update |
| `min_length = min(min_length, right - left + 1)` BEFORE shrinking | Window is currently valid; record it before making it smaller | For "shortest" problems we record then shrink — reversing the order would miss this candidate |
| `result += right - left + 1` (count subarrays) | Every subarray ending at `right` with start in `[left, right]` is valid | Sub-windows of a valid window are also valid; there are exactly `right - left + 1` such subarrays |
| `if i >= k - 1: result.append(nums[dq[0]])` | The first complete window of size k ends at index k-1 | Starting output at k-1 (not k) captures the result for the very first full window |
| `return min_length if min_length != float('inf') else 0` | If no valid window was ever found, return 0 | `float('inf')` is the sentinel for "never updated"; the ternary converts it to the required "not found" value |

### D. Edge Case & Termination Conditions

| Condition | Plain English | Why it works |
|-----------|---------------|--------------|
| `if n < k: return 0` / `return None` | Array too short to form even one window | Prevents `arr[:k]` from silently building a shorter-than-k window and producing a garbage result |
| `if k < 0: return 0` in `at_most` | Called with a negative limit (e.g., `atMost(-1)` from `exactly(0)`) | Without this guard the shrink loop runs forever because any non-negative state exceeds a negative target |
| `if k == 0: return 0` (distinct version) | Zero distinct characters means no non-empty window is valid | Prevents infinite shrinking and makes the intent explicit |
| `if not t or not s: return ""` | Empty target or empty source — no meaningful window to search | Avoids division-by-zero-style errors from `len(t_count) == 0` and keeps `required` well-defined |
| `if char in char_index and char_index[char] >= left` | Duplicate is inside the current window (not behind it) | The `>= left` guard prevents `left` from moving backward to a stale index that was already evicted |
