
# The Universal Guide to Binary Search
> "Binary Search is not about finding a value. It is about narrowing the search space."

## 1. The Philosophy: "Minimize k" (The Universal Template)
Most people struggle with Binary Search because of **Off-By-One Errors** (e.g., should I use `lo <= hi`? `mid - 1`? `mid + 1`?).

We solve this by using the **"Left-Biased" / "Minimize k"** template.
* **No Early Return:** We do not return inside the loop. We shrink the window until only **one** element remains.
* **Universal Exit:** The loop always ends when `lo == hi`.
* **Overflow Safe:** Handles large integers correctly.

### The Master Template (Java)
```java
public int binarySearch(int[] nums, int target) {
    // 1. Search Space: [lo, hi]
    int lo = 0;
    int hi = nums.length - 1; 

    // 2. Loop Condition: lo < hi
    // We stop when lo == hi. This prevents infinite loops and ambiguity.
    while (lo < hi) {
        
        // 3. Mid Calculation: Safe from Overflow & Left-Biased
        // In Java/C++, (lo + hi) / 2 can overflow.
        // "Left-Biased" means if 2 elements remain, mid picks the LEFT one.
        int mid = lo + (hi - lo) / 2;

        // 4. The Decision (The Predicate)
        // This is the ONLY line that changes between problems.
        // Ask: "Is it possible the answer is at mid (or to the left)?"
        if (condition(mid)) {
            hi = mid;      // Valid candidate. Keep it. Try to find better to the left.
        } else {
            lo = mid + 1;  // Invalid. The answer MUST be to the right.
        }
    }

    // 5. Post-Processing
    // lo == hi. This is our single best candidate. Check if it's valid.
    if (nums[lo] == target) return lo;
    return -1;
}

```

---

## 2. Pattern I: Array Index Search

This is for searching in standard arrays, even if they are modified.

### A. First Occurrence (Lower Bound)

**Goal:** Find the *first* index where `nums[i] == target`.

* **Logic:** If `nums[mid] >= target`, then `mid` *could* be the first occurrence. We can't discard it, so `hi = mid`.
* **Snippet:**
```java
if (nums[mid] >= target) {
    hi = mid;
} else {
    lo = mid + 1;
}
// Check if nums[lo] == target at the end.

```



### B. Rotated Sorted Array (The "Hard" Case)

**Goal:** Search in `[4, 5, 6, 7, 0, 1, 2]`.
**The Challenge:** We have two sorted segments. We must decide which side is "clean" (sorted) to make a decision.

#### The Robust Logic (Works with Duplicates)

This handles the worst case: `[1, 0, 1, 1, 1]`. Here, `lo`, `mid`, and `hi` are all `1`. We cannot tell which side is sorted.
**Fix:** Linearly shrink the window (`lo++`, `hi--`) only when ambiguous.

```java
while (lo < hi) {
    int mid = lo + (hi - lo) / 2;
    if (nums[mid] == target) return true;

    // 1. Handle Duplicates (The "Tricky" Part)
    // If edges match mid, we can't determine the sorted half. Shrink inward.
    if (nums[lo] == nums[mid] && nums[mid] == nums[hi]) {
        lo++; hi--;
        continue;
    }

    // 2. Identify Sorted Half
    if (nums[lo] <= nums[mid]) { // Left side is sorted
        if (nums[lo] <= target && target <= nums[mid]) {
            hi = mid;      // Target is in the left half
        } else {
            lo = mid + 1;  // Target is in the right half
        }
    } else { // Right side is sorted
        if (nums[mid] < target && target <= nums[hi]) {
            lo = mid + 1;  // Target is in the right half
        } else {
            hi = mid;      // Target is in the left half
        }
    }
}

```

---

## 3. Pattern II: Search on Solution Space (Binary Search on Answer)

**Context:** You aren't searching for an index. You are searching for a **parameter** (Speed, Capacity, Time) that satisfies a condition.
**Identification:**

* "Find the **Minimum** capacity to..."
* "Find the **Maximum** number of..."
* The answer space is monotonic: `[Fail, Fail, Fail, Pass, Pass, Pass]`.

### The Logic

1. **Define Range:** `lo` = Minimum possible answer, `hi` = Maximum possible answer.
2. **Define Function:** `canDo(val)` returns `true` if the value works.
3. **Minimize/Maximize:** Use the standard template to find the boundary.

**Example: Capacity To Ship Packages (LC 1011)**

```java
public int shipWithinDays(int[] weights, int days) {
    // Range: Max single weight (min capacity) -> Sum of all weights (max capacity)
    int lo = Arrays.stream(weights).max().getAsInt();
    int hi = Arrays.stream(weights).sum();

    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        
        if (canShip(weights, mid, days)) {
            hi = mid;      // Capacity 'mid' works. Try smaller?
        } else {
            lo = mid + 1;  // Capacity 'mid' fails. Need more.
        }
    }
    return lo; // The smallest capacity that worked.
}

```

---

## 4. Pattern III: K-th Smallest (Count-Based Search)

**Context:** Finding the K-th smallest element in a Matrix, Multiplication Table, or Pair Distances.
**Why it's unique:** You don't have a sorted array to index. You have a **structure** where you can count.

**The Logic:**
Instead of looking for the index `k`, we pick a number `x` and ask: **"How many numbers in the matrix are <= x?"**

* If `count < k`: `x` is too small. We need a larger number.
* If `count >= k`: `x` might be the answer (or something smaller is).

**Template (LC 378 - Kth Smallest in Matrix):**

```java
while (lo < hi) {
    int mid = lo + (hi - lo) / 2;
    int count = countLessEqual(matrix, mid); // O(N) helper function
    
    if (count < k) {
        lo = mid + 1; // Need more numbers -> Increase value
    } else {
        hi = mid;     // Found enough numbers -> Try smaller value
    }
}

```

---

## 5. Pattern IV: Peak Finding (Bitonic Search)

**Context:** The data increases then decreases (like a mountain). You need the peak index.
**Logic:** Compare `mid` with its neighbor `mid + 1`.

```java
while (lo < hi) {
    int mid = lo + (hi - lo) / 2;
    
    if (nums[mid] < nums[mid + 1]) {
        // Rising slope. The peak is to the RIGHT.
        // mid is definitely NOT the peak.
        lo = mid + 1;
    } else {
        // Falling slope (or peak). The peak is LEFT or Current.
        // mid COULD be the peak.
        hi = mid;
    }
}
return lo;

```

---

## 6. Advanced Pattern: Search on Length

**Context:** "Find the longest repeating substring" or "Longest common subarray".
**Logic:**
Instead of searching for the substring itself, we Binary Search on the **Length** `L`.

* "Does a repeating substring of length 10 exist?" -> Yes.
* "Does a repeating substring of length 11 exist?" -> No.
* The answer is 10.

**Implementation Note:**
When searching for a **Maximum** (Upper Bound), we often tweak `mid` to round UP to avoid infinite loops when `lo` and `hi` are adjacent.

* `mid = lo + (hi - lo + 1) / 2`
* If `check(mid)` is true: `lo = mid` (Keep trying larger).
* Else: `hi = mid - 1` (Too big).

---

## Cheatsheet: Which Condition to Use?

| Problem Type | Template Logic | Helper Thought |
| --- | --- | --- |
| **Basic / First Index** | `if (arr[mid] >= target) hi = mid; else lo = mid + 1;` | Keep candidates that match or are larger (left-side). |
| **Search Answer (Min)** | `if (works(mid)) hi = mid; else lo = mid + 1;` | If it works, try to minimize it. |
| **Search Answer (Max)** | `if (works(mid)) lo = mid; else hi = mid - 1;` | If it works, try to maximize it. (Note `mid` rounding up). |
| **Peak Finding** | `if (arr[mid] < arr[mid+1]) lo = mid + 1; else hi = mid;` | Follow the slope upwards. |
| **Rotated Array** | Check `nums[lo] <= nums[mid]` first. | Determine sorted side, then check range. |

```

```