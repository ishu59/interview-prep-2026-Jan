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
```java
while (lo < hi) {  // Stop when lo == hi (one element)
```

**Why `lo < hi` instead of `lo <= hi`?**
- With `lo <= hi`, you need to figure out when to return (inside loop? after loop?)
- With `lo < hi`, the loop always ends with `lo == hi` = your candidate answer
- Simpler post-processing, fewer edge cases

#### Principle 2: Never Lose a Valid Candidate
```java
if (condition) {
    hi = mid;      // mid MIGHT be the answer, keep it in [lo, hi]
} else {
    lo = mid + 1;  // mid is DEFINITELY not the answer, exclude it
}
```

**Why this matters:**
- `hi = mid - 1` would lose `mid` even if it's the answer
- Only use `hi = mid - 1` when you're CERTAIN `mid` can't be the answer

#### Principle 3: Avoid Integer Overflow
```java
int mid = lo + (hi - lo) / 2;  // Safe
// NOT: (lo + hi) / 2  ← Can overflow!
```

**Why it can overflow:**
If `lo = 2,000,000,000` and `hi = 2,000,000,000`, then `lo + hi = 4,000,000,000` which exceeds `Integer.MAX_VALUE` (2,147,483,647).

---

<a name="master-templates"></a>
## 2. The Two Master Templates

There are fundamentally **TWO** templates you need to master. The key difference is the **direction** you're searching.

### 🔵 Template A: MINIMIZE (Find First / Lower Bound)

**Use when:** Finding the FIRST valid element, or MINIMIZING a parameter

```java
public int minimize(int[] nums, Condition condition) {
    int lo = 0;
    int hi = nums.length - 1;
    
    while (lo < hi) {
        // LEFT-biased: When 2 elements remain, pick the LEFT one
        int mid = lo + (hi - lo) / 2;
        
        if (condition.isValid(mid)) {
            // mid is valid, but there might be a SMALLER valid value
            hi = mid;
        } else {
            // mid is NOT valid, answer must be LARGER
            lo = mid + 1;
        }
    }
    
    // lo == hi, this is our candidate
    return lo;
}
```

#### Why LEFT-biased (no +1 in mid)?

**Example:** `lo = 2, hi = 3` (two elements remain)
- `mid = 2 + (3-2)/2 = 2` ← Picks LEFT element
- If we do `hi = mid`: `hi = 2`, progress made ✓
- If we do `lo = mid + 1`: `lo = 3`, progress made ✓
- **No infinite loop possible!**

#### Visual: How it works
```
[❌ ❌ ❌ ✅ ✅ ✅ ✅ ✅]  ← We want the FIRST ✅
          ↑
       We keep shrinking toward the LEFT boundary
       hi = mid keeps valid candidates
```

---

### 🔴 Template B: MAXIMIZE (Find Last / Upper Bound)

**Use when:** Finding the LAST valid element, or MAXIMIZING a parameter

```java
public int maximize(int[] nums, Condition condition) {
    int lo = 0;
    int hi = nums.length - 1;
    
    while (lo < hi) {
        // RIGHT-biased: When 2 elements remain, pick the RIGHT one
        // ⚠️ CRITICAL: The +1 is ESSENTIAL!
        int mid = lo + (hi - lo + 1) / 2;
        
        if (condition.isValid(mid)) {
            // mid is valid, but there might be a LARGER valid value
            lo = mid;
        } else {
            // mid is NOT valid, answer must be SMALLER
            hi = mid - 1;
        }
    }
    
    // lo == hi, this is our candidate
    return lo;
}
```

#### Why RIGHT-biased (+1 in mid)? ⚠️ CRITICAL CONCEPT

**Without +1 (WRONG):**
```
lo = 2, hi = 3 (two elements)
mid = 2 + (3-2)/2 = 2  ← Still picks LEFT
If condition is valid: lo = mid = 2
Next iteration: lo = 2, hi = 3  ← SAME STATE!
→ INFINITE LOOP! ❌
```

**With +1 (CORRECT):**
```
lo = 2, hi = 3 (two elements)
mid = 2 + (3-2+1)/2 = 2 + 1 = 3  ← Picks RIGHT
If condition is valid: lo = mid = 3
Next iteration: lo = 3, hi = 3  ← Loop exits! ✓
```

#### Visual: How it works
```
[✅ ✅ ✅ ✅ ✅ ❌ ❌ ❌]  ← We want the LAST ✅
              ↑
       We keep shrinking toward the RIGHT boundary
       lo = mid keeps valid candidates
```

---

### 📊 Quick Decision Matrix

| What are you looking for? | Template | Mid Calculation | Update Logic |
|---------------------------|----------|-----------------|--------------|
| **First** occurrence | A (Minimize) | `lo + (hi-lo)/2` | `if (valid) hi=mid else lo=mid+1` |
| **Last** occurrence | B (Maximize) | `lo + (hi-lo+1)/2` | `if (valid) lo=mid else hi=mid-1` |
| **Minimize** parameter | A (Minimize) | `lo + (hi-lo)/2` | `if (works) hi=mid else lo=mid+1` |
| **Maximize** parameter | B (Maximize) | `lo + (hi-lo+1)/2` | `if (works) lo=mid else hi=mid-1` |

**Memory Aid:**
- Moving LEFT (`hi = mid`) → Use LEFT-biased mid (no +1)
- Moving RIGHT (`lo = mid`) → Use RIGHT-biased mid (+1)

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
1. **Is there a sorted array?** → Likely Category 1
2. **Question asks "minimum X such that..." or "maximum X such that..."?** → Likely Category 2
3. **Question asks for "K-th smallest/largest"?** → Likely Category 3
4. **Involves a matrix?** → Likely Category 4

---

<a name="patterns"></a>
## 4. Complete Pattern Library

### PATTERN 1: Finding Boundaries in Sorted Arrays

---

#### Pattern 1A: First Occurrence (Lower Bound)

**Problem:** Find the FIRST index where `nums[i] == target`

**Example:** `[1, 2, 2, 2, 3]`, target = `2` → Answer: `1`

**Which Template?** Template A (Minimize) - finding FIRST

```java
public int firstOccurrence(int[] nums, int target) {
    int lo = 0;
    int hi = nums.length - 1;
    
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;  // LEFT-biased
        
        // Key insight: We want the FIRST >=
        // So keep ALL candidates >= target
        if (nums[mid] >= target) {
            hi = mid;  // mid might be first, or first is to the left
        } else {
            lo = mid + 1;  // mid is too small, go right
        }
    }

    // Validate: Did we actually find the target?
    // Why TWO checks? (1) lo < nums.length guards against searching
    // past the array end (e.g., target larger than all elements).
    // (2) nums[lo] == target confirms we found the target, not just
    // the first element >= target (which could be a larger value).
    if (lo < nums.length && nums[lo] == target) {
        return lo;
    }
    return -1;
}
```

**Why `nums[mid] >= target` and not `nums[mid] == target`?**

Because we're finding a **boundary**:
```
[1, 2, 2, 2, 3]  target = 2
 ❌ ✅ ✅ ✅ ❌
    ↑
  First >= target is what we want!
```

If we use `nums[mid] == target`, we'd have three cases to handle. Using `>=` gives us a clean two-way split.

---

#### Pattern 1B: Last Occurrence (Upper Bound - 1)

**Problem:** Find the LAST index where `nums[i] == target`

**Example:** `[1, 2, 2, 2, 3]`, target = `2` → Answer: `3`

**Which Template?** Template B (Maximize) - finding LAST

```java
public int lastOccurrence(int[] nums, int target) {
    int lo = 0;
    int hi = nums.length - 1;
    
    while (lo < hi) {
        int mid = lo + (hi - lo + 1) / 2;  // RIGHT-biased ⚠️
        
        // Key insight: We want the LAST <=
        // So keep ALL candidates <= target
        if (nums[mid] <= target) {
            lo = mid;  // mid might be last, or last is to the right
        } else {
            hi = mid - 1;  // mid is too large, go left
        }
    }

    // Validate: Did we actually find the target?
    // Why lo >= 0? If the array has no elements <= target, lo could
    // still be 0 (the initial value), but we need to confirm it
    // actually matches. The bounds check is defensive against edge cases.
    if (lo >= 0 && nums[lo] == target) {
        return lo;
    }
    return -1;
}
```

**Why `nums[mid] <= target`?**

We're finding the RIGHT boundary:
```
[1, 2, 2, 2, 3]  target = 2
 ❌ ✅ ✅ ✅ ❌
          ↑
     Last <= target is what we want!
```

---

#### Pattern 1C: Insert Position

**Problem:** LeetCode 35 - Find index where target should be inserted

**Example:** `[1, 3, 5, 6]`, target = `2` → Answer: `1`

**Which Template?** Template A (Minimize) - finding first position >= target

```java
public int searchInsert(int[] nums, int target) {
    int lo = 0;
    int hi = nums.length - 1;
    
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        
        if (nums[mid] >= target) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    
    // Special case: target larger than all elements.
    // Our loop converges to the last index, but if target exceeds
    // nums[lo], the correct insert position is one past the end.
    // Example: [1,3,5], target=6 -> lo=2, nums[2]=5 < 6, return 3.
    if (lo < nums.length && nums[lo] < target) {
        return lo + 1;
    }
    return lo;
}
```

**Why is post-processing different?**

Unlike exact match, insert position ALWAYS has an answer (even if at the end).

---

#### Pattern 1D: Search Range (Both Bounds)

**Problem:** LeetCode 34 - Find both first and last occurrence

**Example:** `[5, 7, 7, 8, 8, 10]`, target = `8` → Answer: `[3, 4]`

**Solution:** Combine Pattern 1A + 1B

```java
public int[] searchRange(int[] nums, int target) {
    // Empty array has no elements to search -- return early.
    if (nums.length == 0) {
        return new int[]{-1, -1};
    }

    // Find first occurrence (Template A)
    int first = firstOccurrence(nums, target);
    // Why check first before searching for last?
    // If the target doesn't exist at all, there's no point running
    // a second binary search. This is both an optimization and
    // a correctness guard.
    if (first == -1) {
        return new int[]{-1, -1};  // Target not found
    }
    
    // Find last occurrence (Template B)
    int last = lastOccurrence(nums, target);
    
    return new int[]{first, last};
}
```

**Interview Tip:** This demonstrates that you understand BOTH templates!

---

### PATTERN 2: Rotated Sorted Array

---

#### Pattern 2A: Without Duplicates (Clean O(log n))

**Problem:** LeetCode 33 - Search in `[4, 5, 6, 7, 0, 1, 2]`

**Key Insight:** At any point, one half is ALWAYS sorted. Use that sorted half to make decisions.

```java
public int searchRotated(int[] nums, int target) {
    int lo = 0;
    int hi = nums.length - 1;
    
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        
        // Step 1: Determine which half is sorted.
        // Why `<=` and not `<`?
        // When lo == mid (only 2 elements left), nums[lo] == nums[mid]
        // trivially. The left "half" is just one element, which IS
        // sorted. Using `<` would wrongly treat it as unsorted.
        if (nums[lo] <= nums[mid]) {
            // LEFT half [lo...mid] is sorted

            // Step 2: Is target in the sorted half?
            // Why double-bound check `nums[lo] <= target && target <= nums[mid]`?
            // Because the left half is sorted, we can reliably check
            // if target falls within its range. This is the one half
            // where a simple range test works. If target is outside
            // this range, it MUST be in the other (possibly unsorted) half.
            if (nums[lo] <= target && target <= nums[mid]) {
                hi = mid;  // Yes, search sorted left half
            } else {
                lo = mid + 1;  // No, search rotated right half
            }
        } else {
            // RIGHT half [mid+1...hi] is sorted

            // Step 2: Is target in the sorted half?
            // Why `nums[mid] < target` (strict) but `target <= nums[hi]`?
            // We already know nums[mid] != target would have been handled
            // if mid were the answer. Using `<` for mid excludes mid itself,
            // while `<=` for hi includes the right boundary. Together they
            // define the open-closed range (mid, hi] of the sorted right half.
            if (nums[mid] < target && target <= nums[hi]) {
                lo = mid + 1;  // Yes, search sorted right half
            } else {
                hi = mid;  // No, search rotated left half
            }
        }
    }
    
    // After convergence, lo == hi. Verify we actually found the target,
    // since binary search only narrows to a candidate -- it does not
    // guarantee the target exists in the array.
    return (nums[lo] == target) ? lo : -1;
}
```

**Why check the sorted half first?**

In a rotated array like `[4,5,6,7,0,1,2]`:
- We can't rely on `nums[mid]` comparison with `target` alone
- But we CAN identify which half is sorted: `nums[lo] <= nums[mid]` means left is sorted
- Once we know a half is sorted, we can use normal binary search logic on that half

**Visual:**
```
[4, 5, 6, 7, 0, 1, 2]
 ↑        ↑        ↑
lo      mid       hi

nums[lo]=4 <= nums[mid]=7 → Left half is sorted!
If target is in [4,7], search left. Otherwise, search right.
```

---

#### Pattern 2B: With Duplicates (Worst Case O(n))

**Problem:** LeetCode 81 - Search in `[1, 0, 1, 1, 1]`

**Challenge:** When `nums[lo] == nums[mid] == nums[hi]`, we can't determine which half is sorted.

```java
public boolean searchRotatedDuplicates(int[] nums, int target) {
    int lo = 0;
    int hi = nums.length - 1;
    
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        
        // Handle ambiguity: can't determine sorted half.
        // When all three values are equal (e.g., [1,1,1,0,1]),
        // there is no way to know which side the rotation point is on.
        // Why shrink BOTH ends? Shrinking only one side could skip
        // the rotation point on the other side. By trimming one
        // element from each end, we safely reduce the window while
        // guaranteeing we don't jump over the answer (it can't be
        // at lo or hi since they equal mid, which we'll re-check).
        if (nums[lo] == nums[mid] && nums[mid] == nums[hi]) {
            lo++;
            hi--;
            continue;  // Re-evaluate with smaller window
        }
        
        // Same logic as Pattern 2A
        if (nums[lo] <= nums[mid]) {
            if (nums[lo] <= target && target <= nums[mid]) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        } else {
            if (nums[mid] < target && target <= nums[hi]) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
    }
    
    return nums[lo] == target;
}
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
Can ship?  [❌  ❌  ❌  ✅  ✅  ✅  ✅  ...]
                      ↑
               Find this boundary!
```

**Template Structure:**
1. Define `lo` = minimum possible answer
2. Define `hi` = maximum possible answer  
3. Write `canAchieve(x)` feasibility function
4. Use Template A (minimize) or B (maximize)

---

#### Pattern 3A: Minimize Maximum

**Use when:** "Find the minimum X such that we can achieve the goal"

**Example:** LeetCode 410 - Split Array Largest Sum

**Problem:** Split `[7,2,5,10,8]` into `3` subarrays to minimize the maximum sum.

**Insight:** 
- Too small capacity → Can't split into 3 subarrays
- Large enough capacity → Can split into 3 subarrays
- Find the MINIMUM capacity that works

```java
public int splitArray(int[] nums, int m) {
    // Bounds:
    // lo = max element (can't split an element) -- any valid capacity
    //   must be at least as large as the biggest single element,
    //   otherwise that element alone would exceed the limit.
    // hi = sum of all (one subarray) -- putting everything in one
    //   subarray is always feasible, so this is the upper bound.
    int lo = 0;
    int hi = 0;
    for (int num : nums) {
        lo = Math.max(lo, num);
        hi += num;
    }
    
    // Binary search on capacity
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;  // Template A
        
        if (canSplit(nums, mid, m)) {
            hi = mid;  // This capacity works, try smaller
        } else {
            lo = mid + 1;  // Too small, need larger capacity
        }
    }
    
    return lo;
}

// Greedy check: Can we split into <= m subarrays with maxSum?
private boolean canSplit(int[] nums, int maxSum, int m) {
    int subarrays = 1;  // Start with first subarray
    int currentSum = 0;
    
    for (int num : nums) {
        // Why `>` and not `>=`? A subarray sum exactly equal to
        // maxSum is still valid. We only need a new subarray when
        // adding this element would EXCEED the allowed maximum.
        if (currentSum + num > maxSum) {
            // Need new subarray
            subarrays++;
            currentSum = num;
            // Why early return here? If we already need more than m
            // subarrays and haven't finished iterating, there's no
            // way to fit the remaining elements -- fail fast.
            if (subarrays > m) return false;  // Too many subarrays
        } else {
            currentSum += num;
        }
    }
    
    return true;  // We can split into <= m subarrays
}
```

**Why this is Template A (Minimize):**
- We want the MINIMUM capacity that works
- `canSplit(14)` works → Try `canSplit(13)`, `canSplit(12)`...
- This is finding the FIRST valid value → Template A

---

#### Pattern 3B: Maximize Minimum

**Use when:** "Find the maximum X such that we can achieve the goal"

**Example:** LeetCode 1552 - Magnetic Force Between Two Balls

**Problem:** Place `m` balls in `positions = [1,2,3,4,7]` to maximize minimum distance.

**Insight:**
- Small distance → Easy to place all balls
- Large distance → Hard to place all balls  
- Find the MAXIMUM distance that still works

```java
public int maxDistance(int[] position, int m) {
    Arrays.sort(position);
    
    // Bounds:
    // lo = 1 (minimum possible distance)
    // hi = max position - min position (all distance available)
    int lo = 1;
    int hi = position[position.length - 1] - position[0];
    
    // Binary search on distance
    while (lo < hi) {
        int mid = lo + (hi - lo + 1) / 2;  // Template B ⚠️
        
        if (canPlace(position, mid, m)) {
            lo = mid;  // This distance works, try larger
        } else {
            hi = mid - 1;  // Too large, need smaller distance
        }
    }
    
    return lo;
}

// Greedy check: Can we place m balls with minDist?
private boolean canPlace(int[] position, int minDist, int m) {
    int count = 1;  // Place first ball at position[0]
    int lastPos = position[0];
    
    // Why start at i = 1? We greedily placed the first ball at
    // position[0] (the leftmost position after sorting), so we
    // check remaining positions starting from the second one.
    for (int i = 1; i < position.length; i++) {
        // Why `>=` and not `>`? The minimum distance constraint
        // means balls must be AT LEAST minDist apart. Exactly
        // minDist apart is valid, so we use >= (not strict >).
        if (position[i] - lastPos >= minDist) {
            // Can place another ball here
            count++;
            lastPos = position[i];
            // Why `>=` here? We need exactly m balls. Once we've
            // placed m, we're done -- no need to check further.
            if (count >= m) return true;  // Placed all balls!
        }
    }
    
    return false;  // Couldn't place all balls
}
```

**Why this is Template B (Maximize):**
- We want the MAXIMUM distance that works
- `canPlace(3)` works → Try `canPlace(4)`, `canPlace(5)`...
- This is finding the LAST valid value → Template B
- **Critical:** Must use RIGHT-biased mid!

---

#### Pattern 3C: Classic Examples

**Square Root (LC 69)**

Finding `sqrt(8)` = `2` (floor value)

```java
public int mySqrt(int x) {
    // Why x < 2? For x = 0, sqrt = 0. For x = 1, sqrt = 1.
    // These are the only cases where sqrt(x) == x, and they'd
    // cause issues with our hi = x/2 bound (x/2 = 0 for x = 1).
    if (x < 2) return x;

    int lo = 1;
    // Why x/2? For any x >= 4, sqrt(x) <= x/2.
    // Proof: (x/2)^2 = x^2/4 >= x when x >= 4.
    // This tightens the search space vs using hi = x.
    int hi = x / 2;
    
    while (lo < hi) {
        int mid = lo + (hi - lo + 1) / 2;  // Template B
        
        // Avoid overflow: mid*mid can exceed Integer.MAX_VALUE,
        // but mid <= x/mid is algebraically equivalent and safe.
        // This checks: is mid^2 <= x? i.e., is mid a valid sqrt floor?
        if (mid <= x / mid) {
            lo = mid;  // mid^2 <= x, try larger
        } else {
            hi = mid - 1;  // mid^2 > x, too large
        }
    }
    
    return lo;
}
```

**Why Template B?** We want the LARGEST number whose square <= x.

---

**Koko Eating Bananas (LC 875)**

Can Koko eat all bananas in `h` hours with speed `k`?

```java
public int minEatingSpeed(int[] piles, int h) {
    int lo = 1;
    int hi = 0;
    for (int pile : piles) {
        hi = Math.max(hi, pile);
    }
    
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;  // Template A
        
        if (canFinish(piles, mid, h)) {
            hi = mid;  // This speed works, try slower
        } else {
            lo = mid + 1;  // Too slow, need faster
        }
    }
    
    return lo;
}

private boolean canFinish(int[] piles, int speed, int h) {
    long hours = 0;
    for (int pile : piles) {
        // Why ceiling division `(pile + speed - 1) / speed`?
        // Koko must spend a whole hour on each pile, even if she
        // finishes early. A pile of 7 bananas at speed 3 takes
        // ceil(7/3) = 3 hours, not 2. The formula avoids floating
        // point: ceil(a/b) == (a + b - 1) / b for positive integers.
        hours += (pile + speed - 1) / speed;
    }
    // Why `<=` and not `==`? Koko can finish EARLY. If she can
    // eat everything in fewer than h hours, that speed still works.
    return hours <= h;
}
```

**Why Template A?** We want the MINIMUM speed that works.

---

### PATTERN 4: K-th Smallest (Count-Based Search)

**Core Concept:** Instead of indexing, we pick a VALUE and ask: "How many elements are <= this value?"

**When to use:** Finding K-th smallest in a matrix, multiplication table, or pair distances.

**Example:** LeetCode 378 - K-th Smallest Element in Sorted Matrix

```java
public int kthSmallest(int[][] matrix, int k) {
    int n = matrix.length;
    int lo = matrix[0][0];           // Smallest value
    int hi = matrix[n-1][n-1];       // Largest value
    
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;  // Template A
        
        // Count how many numbers <= mid
        int count = countLessEqual(matrix, mid);

        // Why `count < k` (strict) and not `count <= k`?
        // If count == k, there are exactly k values <= mid, meaning
        // mid COULD be the k-th smallest (or mid might not even exist
        // in the matrix but a smaller value does). By going to hi = mid,
        // we keep searching left to find the smallest value with count >= k.
        if (count < k) {
            lo = mid + 1;  // Not enough numbers, need larger value
        } else {
            hi = mid;  // Have >= k numbers, try smaller value
        }
    }
    
    return lo;
}

// Count elements <= target in row-col sorted matrix
private int countLessEqual(int[][] matrix, int target) {
    int count = 0;
    int n = matrix.length;
    
    // Start from bottom-left corner.
    // Why bottom-left? It's the unique position where moving RIGHT
    // increases the value and moving UP decreases it. This gives
    // us a binary-search-like ability to navigate the 2D space.
    int row = n - 1;
    int col = 0;

    // Why two conditions? `row >= 0` stops us from going above the
    // matrix. `col < n` stops us from going past the right edge.
    // Together they keep us within bounds as we staircase through.
    while (row >= 0 && col < n) {
        // Why `<= target`? If the bottom-left element of this
        // sub-matrix is <= target, then ALL elements above it in
        // the same column are also <= target (column is sorted).
        // That's why we add (row + 1) elements at once.
        if (matrix[row][col] <= target) {
            count += row + 1;  // All elements in this column up to row
            col++;             // Move right to count the next column
        } else {
            row--;             // Too large, move up to smaller values
        }
    }
    
    return count;
}
```

**Why this works:**

If there are 5 numbers <= mid, and k=3:
- We need a SMALLER number
- Binary search brings us closer to the 3rd smallest

**Key insight:** We're searching in the VALUE space, not index space!

---

### PATTERN 5: Peak Finding

**Core Concept:** In a mountain/bitonic array, follow the slope upward.

**Example:** LeetCode 162 - Find Peak Element

```java
public int findPeakElement(int[] nums) {
    int lo = 0;
    int hi = nums.length - 1;
    
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        
        // Compare mid with its right neighbor.
        // Why is `mid + 1` safe (no out-of-bounds)?
        // Because `lo < hi` guarantees mid < hi, so mid + 1 <= hi,
        // which is always a valid index.
        if (nums[mid] < nums[mid + 1]) {
            // Ascending slope: peak is to the RIGHT.
            // mid is definitely NOT the peak (its neighbor is higher).
            lo = mid + 1;
        } else {
            // Descending slope → peak is LEFT or AT mid
            // mid COULD be the peak
            hi = mid;
        }
    }
    
    return lo;  // lo == hi is the peak index
}
```

**Why this works:**

```
      /\
     /  \
    /    \___
   
If mid is here and mid < mid+1:
       ↑
    We're ascending, peak must be to the right!
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

```java
public boolean searchMatrix(int[][] matrix, int target) {
    // Guard against empty matrix or empty rows -- no elements to search.
    if (matrix.length == 0 || matrix[0].length == 0) {
        return false;
    }

    int m = matrix.length;
    int n = matrix[0].length;
    int lo = 0;
    // Why m * n - 1? We're treating the 2D matrix as a flat 1D array
    // of length m*n. The last valid index is m*n - 1 (0-indexed).
    int hi = m * n - 1;
    
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        
        // Convert 1D index to 2D coordinates.
        // Why division and modulo? Think of the flat array being
        // "wrapped" every n elements. Division tells you which row
        // you've wrapped into; modulo tells you how far along that row.
        int row = mid / n;  // Which row?
        int col = mid % n;  // Which column?

        // Why `>=` (Template A pattern)? We want the FIRST position
        // where the value >= target. If it equals the target, we found it.
        // If it's greater, the target might be further left.
        if (matrix[row][col] >= target) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    
    // Check final position
    int row = lo / n;
    int col = lo % n;
    return matrix[row][col] == target;
}
```

**Why the conversion works:**

```
Index mapping:
  0  1  2  3  4  5  6  7  8  9  10 11
[ 1, 3, 5, 7, 10, 11, 16, 20, 23, 30, 34, 60]

Index 7 → row = 7/4 = 1, col = 7%4 = 3 → matrix[1][3] = 20 ✓
```

---

<a name="post-processing"></a>
## 5. Post-Processing Reference

After the loop ends with `lo == hi`, you need to validate/return based on problem type:

| Problem Type | Post-Processing | Why |
|--------------|-----------------|-----|
| **Exact Match** | `return (nums[lo] == target) ? lo : -1;` | Answer might not exist |
| **First Occurrence** | `return (lo < n && nums[lo] == target) ? lo : -1;` | Need bounds check + match |
| **Last Occurrence** | `return (lo >= 0 && nums[lo] == target) ? lo : -1;` | Need bounds check + match |
| **Insert Position** | `return (lo < n && nums[lo] < target) ? lo + 1 : lo;` | Might insert at end |
| **BS on Answer** | `return lo;` | Boundary always exists |
| **Peak Finding** | `return lo;` | Peak always exists |
| **K-th Smallest** | `return lo;` | We found the k-th value |
| **Rotated Array** | `return (nums[lo] == target) ? lo : -1;` | Answer might not exist |

**Key Principle:** Different problems have different guarantees about answer existence!

---

<a name="pitfalls"></a>
## 6. Common Pitfalls & Solutions

### Pitfall 1: Infinite Loop with Wrong Mid Rounding

**❌ WRONG:**
```java
while (lo < hi) {
    int mid = lo + (hi - lo) / 2;  // LEFT-biased
    if (condition) {
        lo = mid;  // ⚠️ DANGER!
    } else {
        hi = mid - 1;
    }
}
// When lo=2, hi=3: mid=2, then lo=2 again → INFINITE LOOP
```

**✅ CORRECT:**
```java
while (lo < hi) {
    int mid = lo + (hi - lo + 1) / 2;  // RIGHT-biased
    if (condition) {
        lo = mid;  // Now safe!
    } else {
        hi = mid - 1;
    }
}
```

**Rule:** `lo = mid` requires RIGHT-biased mid. No exceptions!

---

### Pitfall 2: Losing Valid Answer

**❌ WRONG:**
```java
if (nums[mid] >= target) {
    hi = mid - 1;  // ⚠️ Lost the answer if mid == target!
} else {
    lo = mid + 1;
}
```

**✅ CORRECT:**
```java
if (nums[mid] >= target) {
    hi = mid;  // Keep mid in search space
} else {
    lo = mid + 1;
}
```

**Rule:** Only exclude `mid` when you're CERTAIN it can't be the answer.

---

### Pitfall 3: Integer Overflow in Mid Calculation

**❌ WRONG:**
```java
int mid = (lo + hi) / 2;  // Overflows if lo + hi > INT_MAX
```

**✅ CORRECT:**
```java
int mid = lo + (hi - lo) / 2;  // Safe from overflow
```

---

### Pitfall 4: Integer Overflow in Answer Space

**❌ WRONG:**
```java
int hi = Arrays.stream(weights).sum();  // Can overflow!
```

**✅ CORRECT:**
```java
long hi = 0;
for (int weight : weights) {
    hi += weight;
}
```

**Or use long in the binary search:**
```java
long lo = 0, hi = sum;
while (lo < hi) {
    long mid = lo + (hi - lo) / 2;
    // ...
}
```

---

### Pitfall 5: Not Checking Array Bounds

**❌ WRONG:**
```java
return nums[lo] == target ? lo : -1;  // What if lo >= nums.length?
```

**✅ CORRECT:**
```java
if (lo < nums.length && nums[lo] == target) {
    return lo;
}
return -1;
```

---

### Pitfall 6: Using Wrong Template for the Problem

**Symptom:** Getting first occurrence when you need last, or vice versa.

**Solution:** Ask yourself:
- Am I finding the FIRST valid? → Template A (left-biased)
- Am I finding the LAST valid? → Template B (right-biased)

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

✅ Ship Capacity: If capacity 10 works → capacity 11 works → **Monotonic**
✅ Maximum Distance: If distance 5 fails → distance 6 fails → **Monotonic**
✅ Sorted Array: If arr[5] > target → arr[6] > target → **Monotonic**
❌ Unsorted Array: No guarantees → **Not monotonic**

---

### Step 2: Identify the Search Space

Ask: **What am I searching for?**

| Searching for | Pattern Type | Example |
|---------------|--------------|---------|
| **Array Index** | Array Search | First/Last occurrence, rotated array |
| **Parameter Value** | Answer Search | Capacity, speed, distance |
| **K-th Element** | Count-Based | K-th smallest in matrix |
| **Peak/Valley** | Bitonic Search | Mountain peak |
| **Matrix Position** | 2D Search | Search 2D matrix |

---

### Step 3: Determine Direction (Minimize vs Maximize)

Ask: **Am I finding first or last?**

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

**Example:** Ship capacity problem
- Feasibility: "Can I ship all packages in D days with capacity X?"
- Implementation: Greedy simulation

**Key:** The feasibility function should be:
1. **Deterministic:** Same input → same output
2. **Fast:** Ideally O(n) or better
3. **Correct:** Accurately determines if X works

---

### Decision Tree

```
                    Can I use Binary Search?
                           ↓
                   Is there monotonicity?
                           ↓
                   ┌───────┴───────┐
                  Yes              No
                   ↓                ↓
            What am I           Try other
            searching?          approaches
                   ↓
      ┌────────────┼────────────┐
      │            │            │
   Index        Parameter    K-th/2D
      │            │            │
      ↓            ↓            ↓
  Template A/B  Template A/B  Special
  (Based on    (Based on     Pattern
   First/Last)  Min/Max)
```

---

<a name="checklist"></a>
## 8. Interview Preparation Checklist

### Before the Interview

**Master the fundamentals:**
- [ ] Can write Template A (minimize) from memory
- [ ] Can write Template B (maximize) from memory
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
```java
int lo = start, hi = end;
while (lo < hi) {
    int mid = lo + (hi - lo) / 2;      // LEFT-biased
    if (isValid(mid)) {
        hi = mid;                       // Keep mid, search left
    } else {
        lo = mid + 1;                   // Exclude mid, search right
    }
}
return lo;  // or with validation
```

### Template B: MAXIMIZE / LAST
```java
int lo = start, hi = end;
while (lo < hi) {
    int mid = lo + (hi - lo + 1) / 2;  // RIGHT-biased ⚠️
    if (isValid(mid)) {
        lo = mid;                       // Keep mid, search right
    } else {
        hi = mid - 1;                   // Exclude mid, search left
    }
}
return lo;  // or with validation
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
| 2D Matrix | O(log(mn)) | O(1) | m×n matrix as 1D array |

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

**You're ready to ace binary search problems! 🚀**

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
| `while (lo < hi)` | Loop until one candidate remains (`lo == hi`). | The standard for both Template A and B. Use this in nearly all binary search problems. The loop guarantees convergence to a single candidate. |
| `while (lo <= hi)` | Loop until the search space is empty (`lo > hi`). | Classic textbook binary search that returns inside the loop on match. More error-prone because you must handle the return inside the loop AND after it. Prefer `lo < hi` in interviews. |

### Mid Calculation

| Conditional | Meaning | When to Use |
|---|---|---|
| `mid = lo + (hi - lo) / 2` | Left-biased mid. When two elements remain, picks the LEFT one. | Template A (Minimize/Find First). Pair with `hi = mid` and `lo = mid + 1`. Prevents infinite loops because `hi = mid` always shrinks the window. |
| `mid = lo + (hi - lo + 1) / 2` | Right-biased mid. When two elements remain, picks the RIGHT one. | Template B (Maximize/Find Last). Pair with `lo = mid` and `hi = mid - 1`. The `+1` is CRITICAL: without it, `lo = mid` with left-biased mid causes an infinite loop when two elements remain. |

### Pointer Updates

| Conditional | Meaning | When to Use |
|---|---|---|
| `hi = mid` | Keep mid in the search space (mid might be the answer). | When the condition at mid is satisfied and you want to search LEFT for a potentially better (smaller) answer. Always pair with left-biased mid. |
| `hi = mid - 1` | Exclude mid from the search space (mid is definitely not the answer). | When you are CERTAIN mid cannot be the answer. Always pair with right-biased mid (Template B). |
| `lo = mid + 1` | Exclude mid, search right. | When mid fails the condition and the answer must be strictly larger. Safe with both left-biased and right-biased mid. |
| `lo = mid` | Keep mid in the search space, search right. | When mid satisfies the condition and you want to search RIGHT for a potentially better (larger) answer. MUST use right-biased mid to avoid infinite loops. |

### Boundary Initialization

| Conditional | Meaning | When to Use |
|---|---|---|
| `lo = 0, hi = len - 1` | Search space is the entire array (inclusive on both ends). | Standard for index-based searches (first/last occurrence, rotated array, peak finding). |
| `lo = 0, hi = len` | Search space includes one position past the array end. | Insert position problems where the answer could be at index `len` (appending). |
| `lo = min_val, hi = max_val` | Search space is a range of VALUES, not indices. | Binary search on answer space (ship capacity, eating speed, distance). Bounds come from the problem constraints. |

### Common Condition Checks

| Conditional | Meaning | When to Use |
|---|---|---|
| `nums[mid] >= target` then `hi = mid` | Find first position where value >= target (lower bound). | First occurrence, insert position. Uses `>=` to capture the exact target AND values after it, creating a clean two-way partition. |
| `nums[mid] <= target` then `lo = mid` | Find last position where value <= target (upper bound). | Last occurrence. Uses `<=` to capture the exact target AND values before it. |
| `nums[mid] < target` then `lo = mid + 1` | Mid is strictly too small, exclude it. | When you need to skip past values that are definitively less than what you want. |
| `nums[lo] <= nums[mid]` | Left half `[lo..mid]` is sorted. | Rotated array search. The `<=` (not `<`) handles the case where `lo == mid` (two elements remaining), where the single-element left "half" is trivially sorted. |
| `nums[lo] <= target && target <= nums[mid]` | Target falls within the sorted left half's range. | Rotated array: once you know the left half is sorted, this range check reliably determines if the target is there. |
| `nums[mid] < nums[mid + 1]` | Ascending slope -- a peak must exist to the right. | Peak finding. If the next element is higher, mid cannot be a peak and a peak is guaranteed to the right (by the constraint that boundaries are negative infinity). |
| `count < k` then `lo = mid + 1` | Fewer than k elements exist at or below mid. | K-th smallest problems. Strict `<` because `count == k` means mid could be the answer (we keep it via `hi = mid` in the else branch). |
| `currentSum + num > maxSum` | Adding this element would exceed the subarray limit. | Greedy feasibility checks in "split array" / "ship packages" problems. Strict `>` because equaling the limit is still valid. |
| `position[i] - lastPos >= minDist` | Gap between current and last placed item meets the minimum. | Greedy feasibility checks for "maximize minimum distance" problems. Uses `>=` because exactly meeting the minimum distance is valid. |
| `mid <= x / mid` | Equivalent to `mid^2 <= x` without overflow risk. | Integer square root. Rearranging avoids `mid * mid` which can overflow 32-bit integers. |
| `hours <= h` | Koko finishes within the allowed time. | Feasibility check for eating speed. Uses `<=` because finishing early is acceptable. |

### Return Values

| Conditional | Meaning | When to Use |
|---|---|---|
| `return lo` (equivalently `return hi`) | After `while (lo < hi)`, `lo == hi` is the converged answer. | BS on answer, peak finding, K-th smallest -- problems where an answer is guaranteed to exist. |
| `return (nums[lo] == target) ? lo : -1` | Validate that the converged candidate actually matches. | Exact match problems (standard search, rotated array) where the target might not exist in the array. |
| `return (lo < n && nums[lo] == target) ? lo : -1` | Bounds check + match validation. | First occurrence. The bounds check `lo < n` guards against the case where the target is larger than all elements (lo could equal n). |
| `return lo + 1` | Insert after the converged position. | Insert position when the target is larger than all elements in the array. |
