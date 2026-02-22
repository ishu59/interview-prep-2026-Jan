# The Complete Backtracking Handbook
> A template-based approach for mastering backtracking in coding interviews

**Philosophy:** Backtracking is not about trying everything blindly. It's about **systematically exploring the solution space** by building candidates incrementally and abandoning paths that cannot lead to valid solutions.

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

- **The Choose-Your-Own-Adventure Book**: At each page, you pick a choice. If the story leads to a bad ending, you flip back to the last choice and try a different option. You're systematically exploring every storyline.
- **The Pruning Shears**: You don't have to read every storyline. If you can tell a choice will definitely lead to a bad ending (e.g., you've already exceeded a budget), you skip that entire branch. This is pruning -- it turns an impossibly large book into a manageable one.

### No-Jargon Translation

- **Decision tree**: the map of all possible choices and where they lead
- **Candidate**: one option you could pick at this step
- **Constraint**: a rule that says "this choice is not allowed"
- **Pruning**: skipping a branch early because you already know it won't work -- like not exploring a road if the bridge is out
- **Backtrack**: undo your last choice and try the next option
- **State space**: the set of all possible combinations of choices

### Mental Model

> "Backtracking is reading a Choose-Your-Own-Adventure book cover-to-cover: at every fork, pick a path, read until you hit a dead end or success, then flip back and try the next path -- skipping any path you can already tell leads nowhere."

---

### What is Backtracking?

Backtracking is a systematic way to explore all possible configurations of a search space. It:
1. **Builds solutions incrementally** — one element at a time
2. **Abandons partial solutions** — when they can't lead to valid complete solutions
3. **Backtracks** — undoes the last choice and tries the next option

### The Decision Tree Mental Model

Every backtracking problem can be visualized as exploring a **decision tree**:

```
Subsets of [1, 2, 3]:

                    []
           /        |        \
         [1]       [2]       [3]
        /   \       |
     [1,2] [1,3]  [2,3]
       |
    [1,2,3]

Each path from root to any node is a valid subset!
```

### The Three Key Questions

For any backtracking problem, answer these:

1. **What are the choices at each step?**
   - Subsets: include or exclude each element
   - Permutations: which unused element to place next
   - Combinations: which element to add from remaining

2. **What are the constraints?**
   - Size limits (k elements)
   - Value constraints (sum = target)
   - Validity rules (N-Queens, Sudoku)

3. **When do we have a complete solution?**
   - Reached desired size
   - All elements placed
   - All constraints satisfied

### Backtracking vs DFS vs Recursion

| Concept | Definition | Example |
|---------|-----------|---------|
| **Recursion** | Function calls itself | Fibonacci |
| **DFS** | Traverse graph/tree depth-first | Tree traversal |
| **Backtracking** | DFS + undo choices when invalid | N-Queens |

**Key:** Backtracking = DFS with state modification + reversal

### The Pruning Advantage

```
Without pruning:          With pruning:
Explore all 2^n paths     Skip invalid branches early

    *                         *
   / \                       / \
  *   *                     *   X (invalid, skip entire subtree)
 / \ / \                   / \
*  * *  *                 *   *
```

**Pruning** dramatically reduces the search space by abandoning paths early.

---

<a name="master-templates"></a>
## 2. The Master Templates

### Template A: Basic Backtracking

```python
def backtrack(result, current, choices, start):
    """
    Basic backtracking template.

    Args:
        result: List to store all valid solutions
        current: Current partial solution being built
        choices: Available options (the input)
        start: Where to start considering choices (for combinations)
    """
    # Base case: check if current is a complete solution
    if is_valid_solution(current):
        result.append(current.copy())  # Save a copy!
        return  # or continue if multiple solutions possible at this state

    # Try each choice
    for i in range(start, len(choices)):
        # Make choice
        current.append(choices[i])

        # Recurse
        # Why `i + 1` vs `i`?
        # `i + 1` = "move forward, no reuse" (subsets, combos with unique elements).
        # `i` = "stay here, allow reuse" (Combination Sum where same element can repeat).
        # Think of it like a buffet line: i+1 means you pass a dish and can't go back;
        # i means you can scoop from the same dish again.
        backtrack(result, current, choices, i + 1)  # or i for reuse

        # Undo choice (backtrack)
        current.pop()
```

---

### Template B: Subsets (Include/Exclude Pattern)

```python
def subsets(nums):
    """
    Generate all subsets.
    At each element, choose to include or exclude.
    """
    result = []

    def backtrack(index, current):
        # Why `index == len(nums)` as the base case?
        # We make one binary decision (include/exclude) per element.
        # When index reaches the end, we've decided on every element --
        # whatever is in `current` is a complete subset.
        if index == len(nums):
            result.append(current.copy())
            return

        # Choice 1: Exclude nums[index]
        backtrack(index + 1, current)

        # Choice 2: Include nums[index]
        current.append(nums[index])
        backtrack(index + 1, current)
        current.pop()

    backtrack(0, [])
    return result
```

**Alternative (iteration-based):**
```python
def subsets_iterative(nums):
    result = []

    def backtrack(start, current):
        result.append(current.copy())

        for i in range(start, len(nums)):
            current.append(nums[i])
            backtrack(i + 1, current)
            current.pop()

    backtrack(0, [])
    return result
```

---

### Template C: Permutations (Use All Elements)

```python
def permutations(nums):
    """
    Generate all permutations.
    Track which elements are used.
    """
    result = []
    used = [False] * len(nums)

    def backtrack(current):
        # Why `len(current) == len(nums)`?
        # A permutation must use every element exactly once.
        # When current's length matches nums, all elements are placed.
        if len(current) == len(nums):
            result.append(current.copy())
            return

        for i in range(len(nums)):
            # Why `used[i]`?
            # Permutations need each element exactly once. The `used` array
            # tracks which elements are already in `current` so we don't
            # pick the same index twice. (Unlike subsets where `start`
            # prevents revisiting, permutations consider ALL indices each time.)
            if used[i]:
                continue

            # Choose
            used[i] = True
            current.append(nums[i])

            # Recurse
            backtrack(current)

            # Unchoose
            current.pop()
            used[i] = False

    backtrack([])
    return result
```

**Alternative (swap-based):**
```python
def permutations_swap(nums):
    result = []

    def backtrack(start):
        if start == len(nums):
            result.append(nums.copy())
            return

        for i in range(start, len(nums)):
            nums[start], nums[i] = nums[i], nums[start]
            backtrack(start + 1)
            nums[start], nums[i] = nums[i], nums[start]

    backtrack(0)
    return result
```

---

### Template D: Combinations (Choose K from N)

```python
def combinations(n, k):
    """
    Generate all combinations of k elements from 1 to n.
    """
    result = []

    def backtrack(start, current):
        # Complete when k elements chosen
        if len(current) == k:
            result.append(current.copy())
            return

        # Pruning: not enough elements left
        remaining = n - start + 1
        needed = k - len(current)
        # Why `remaining < needed`?
        # If there are fewer elements left than slots to fill, there's no way
        # to complete the combination. Example: n=5, k=3, current=[1], start=5.
        # remaining = 5-5+1 = 1, needed = 3-1 = 2. Only one number left (5),
        # but we need two more -- impossible, so prune the entire branch.
        if remaining < needed:
            return

        for i in range(start, n + 1):
            current.append(i)
            backtrack(i + 1, current)
            current.pop()

    backtrack(1, [])
    return result
```

---

### Template E: Constraint Satisfaction (N-Queens, Sudoku)

```python
def solve_constraint(board):
    """
    Template for constraint satisfaction problems.
    """
    result = []

    def is_valid(board, row, col):
        """Check if placing at (row, col) is valid."""
        # Check constraints specific to problem
        return True  # Implement specific validation

    def backtrack(row):
        # Base case: all rows filled
        if row == len(board):
            result.append(serialize(board))
            return

        for col in range(len(board)):
            if is_valid(board, row, col):
                # Make choice
                board[row][col] = 'Q'  # or appropriate value

                # Recurse
                backtrack(row + 1)

                # Undo choice
                board[row][col] = '.'

    backtrack(0)
    return result
```

---

### Template F: Partition/Split Pattern

```python
def partition(s):
    """
    Template for partitioning/splitting problems.
    """
    result = []

    def is_valid_part(substring):
        """Check if this part is valid."""
        return True  # Implement specific validation

    def backtrack(start, current):
        # Why `start == len(s)`?
        # `start` is our read cursor in the string. When it reaches the end,
        # we've partitioned the entire string -- every character is covered.
        if start == len(s):
            result.append(current.copy())
            return

        # Why `range(start + 1, len(s) + 1)`?
        # `start + 1` because the smallest slice has length 1 (s[start:start+1]).
        # `len(s) + 1` because Python slicing is exclusive -- s[start:len(s)]
        # grabs through the last character. This tries every possible "next chunk"
        # from length 1 up to the rest of the string.
        for end in range(start + 1, len(s) + 1):
            part = s[start:end]
            if is_valid_part(part):
                current.append(part)
                backtrack(end, current)
                current.pop()

    backtrack(0, [])
    return result
```

---

### Quick Decision Matrix

| Problem Type | Template | Key Pattern |
|--------------|----------|-------------|
| All subsets | B | Include/exclude each |
| All permutations | C | Use each once |
| Choose k from n | D | Start index advances |
| With duplicates | Modified | Skip duplicates |
| Constraint satisfaction | E | Validate before recurse |
| String partition | F | Try all split points |
| Sum to target | A + pruning | Track remaining sum |

---

<a name="pattern-guide"></a>
## 3. Pattern Classification Guide

### Category 1: Subsets
- Generate all subsets
- With or without duplicates
- **Template B**

### Category 2: Permutations
- Use all elements exactly once
- With or without duplicates
- **Template C**

### Category 3: Combinations
- Choose k elements from n
- Order doesn't matter
- **Template D**

### Category 4: Combination Sum
- Elements can be reused or not
- Must sum to target
- **Template A with sum tracking**

### Category 5: Constraint Satisfaction
- N-Queens, Sudoku
- Must satisfy multiple constraints
- **Template E**

### Category 6: String Partitioning
- Split string into valid parts
- Palindrome partitioning, IP addresses
- **Template F**

---

<a name="patterns"></a>
## 4. Complete Pattern Library

### PATTERN 1: Subsets

---

#### Pattern 1A: Subsets (No Duplicates)

**Problem:** LeetCode 78 - Generate all subsets of distinct integers

```python
def subsets(nums: list[int]) -> list[list[int]]:
    result = []

    def backtrack(start, current):
        result.append(current.copy())

        for i in range(start, len(nums)):
            current.append(nums[i])
            backtrack(i + 1, current)
            current.pop()

    backtrack(0, [])
    return result
```

**Iterative approach:**
```python
def subsets_iterative(nums: list[int]) -> list[list[int]]:
    result = [[]]
    for num in nums:
        result += [subset + [num] for subset in result]
    return result
```

---

#### Pattern 1B: Subsets II (With Duplicates)

**Problem:** LeetCode 90 - Subsets with duplicate elements

**Key:** Sort first, skip duplicates at same level

```python
def subsetsWithDup(nums: list[int]) -> list[list[int]]:
    nums.sort()  # Essential for duplicate handling
    result = []

    def backtrack(start, current):
        result.append(current.copy())

        for i in range(start, len(nums)):
            # Why `i > start` (not `i > 0`)?
            # `start` marks where this recursion level begins choosing.
            # When i == start, it's the FIRST choice at this level -- always allowed.
            # When i > start, we're making a LATER choice at the same level.
            # If nums[i] == nums[i-1] in that case, we'd generate a duplicate branch.
            #
            # Concrete example with [1, 2, 2]:
            #   At level where start=1, i=1 picks first 2 -> [1,2] (allowed, i == start)
            #   At level where start=1, i=2 picks second 2 -> [1,2] again (SKIP, i > start)
            # If we used `i > 0` instead, we'd wrongly skip [1,2,2] because when
            # building [1,2,...], the recursive call has start=2, i=2: i > 0 is true
            # and nums[2]==nums[1], so we'd skip -- but that's a DEEPER level choice,
            # not a same-level duplicate. `i > start` only skips same-level repeats.
            if i > start and nums[i] == nums[i - 1]:
                continue

            current.append(nums[i])
            backtrack(i + 1, current)
            current.pop()

    backtrack(0, [])
    return result
```

**Why sort + skip works:**
```
[1, 2, 2] sorted

Level 0: start with []
  - i=0: add 1 → [1]
  - i=1: add 2 → [2]
  - i=2: skip (nums[2] == nums[1] and i > start)

This prevents [2] from being generated twice!
```

---

### PATTERN 2: Permutations

---

#### Pattern 2A: Permutations (No Duplicates)

**Problem:** LeetCode 46 - All permutations of distinct integers

```python
def permute(nums: list[int]) -> list[list[int]]:
    result = []

    def backtrack(current, remaining):
        if not remaining:
            result.append(current.copy())
            return

        for i in range(len(remaining)):
            current.append(remaining[i])
            backtrack(current, remaining[:i] + remaining[i+1:])
            current.pop()

    backtrack([], nums)
    return result
```

**Using used array:**
```python
def permute_used(nums: list[int]) -> list[list[int]]:
    result = []
    used = [False] * len(nums)

    def backtrack(current):
        if len(current) == len(nums):
            result.append(current.copy())
            return

        for i in range(len(nums)):
            if used[i]:
                continue
            used[i] = True
            current.append(nums[i])
            backtrack(current)
            current.pop()
            used[i] = False

    backtrack([])
    return result
```

---

#### Pattern 2B: Permutations II (With Duplicates)

**Problem:** LeetCode 47 - Permutations with duplicates

```python
def permuteUnique(nums: list[int]) -> list[list[int]]:
    nums.sort()  # Essential!
    result = []
    used = [False] * len(nums)

    def backtrack(current):
        if len(current) == len(nums):
            result.append(current.copy())
            return

        for i in range(len(nums)):
            if used[i]:
                continue

            # Why `i > 0 and nums[i] == nums[i-1] and not used[i-1]`?
            # This is the trickiest duplicate-skip condition. Let's trace [1a, 1b, 2]:
            #
            # Without this check, we'd get both [1a,1b,2] AND [1b,1a,2] -- duplicates.
            # The rule: among identical values, always use them in left-to-right order.
            #
            # `not used[i-1]` means: "the previous duplicate is NOT in the current path."
            # If nums[i-1] is not used but nums[i] is the same value, it means we're
            # trying to pick the SECOND copy before the FIRST -- that's out of order.
            #
            # Trace with [1a, 1b, 2]:
            #   Pick 1a (used=[T,F,F]) -> pick 1b (used=[T,T,F]) -> pick 2. OK: [1a,1b,2]
            #   Pick 1b? (used=[F,F,F]) -> i=1, nums[1]==nums[0], used[0]=False -> SKIP.
            #     We'd be using 1b before 1a, which would create a duplicate permutation.
            #   Pick 2 (used=[F,F,T]) -> pick 1a (used=[T,F,T]) -> pick 1b. OK: [2,1a,1b]
            #     When picking 1b here, used[0]=True (1a already in path), so NOT skipped.
            if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]:
                continue

            used[i] = True
            current.append(nums[i])
            backtrack(current)
            current.pop()
            used[i] = False

    backtrack([])
    return result
```

**Why `not used[i-1]`?**
This ensures we only use the first occurrence of a duplicate at each level.
If previous duplicate is unused, we're at the same level and should skip.

---

### PATTERN 3: Combinations

---

#### Pattern 3A: Combinations (Choose K)

**Problem:** LeetCode 77 - All combinations of k numbers from 1 to n

```python
def combine(n: int, k: int) -> list[list[int]]:
    result = []

    def backtrack(start, current):
        if len(current) == k:
            result.append(current.copy())
            return

        # Pruning: need (k - len(current)) more, have (n - start + 1) left
        need = k - len(current)
        remain = n - start + 1
        if remain < need:
            return

        for i in range(start, n + 1):
            current.append(i)
            backtrack(i + 1, current)
            current.pop()

    backtrack(1, [])
    return result
```

---

#### Pattern 3B: Combination Sum (Unlimited Use)

**Problem:** LeetCode 39 - Combinations that sum to target (reuse allowed)

```python
def combinationSum(candidates: list[int], target: int) -> list[list[int]]:
    result = []

    def backtrack(start, current, remaining):
        if remaining == 0:
            result.append(current.copy())
            return
        # Why `remaining < 0`?
        # We've overshot the target -- the current combination sums to more
        # than target. No point continuing; adding more positives only makes
        # it worse. This is our "went too far" guardrail.
        if remaining < 0:
            return

        for i in range(start, len(candidates)):
            current.append(candidates[i])
            # Why `backtrack(i, ...)` instead of `backtrack(i + 1, ...)`?
            # `i` (not i+1) means "you can pick this same candidate again."
            # This is what allows unlimited reuse. If we passed i+1, we'd
            # move past this candidate and never pick it again.
            backtrack(i, current, remaining - candidates[i])
            current.pop()

    backtrack(0, [], target)
    return result
```

---

#### Pattern 3C: Combination Sum II (Each Once)

**Problem:** LeetCode 40 - Each number used at most once

```python
def combinationSum2(candidates: list[int], target: int) -> list[list[int]]:
    candidates.sort()  # For duplicate handling
    result = []

    def backtrack(start, current, remaining):
        if remaining == 0:
            result.append(current.copy())
            return
        if remaining < 0:
            return

        for i in range(start, len(candidates)):
            # Why `i > start and candidates[i] == candidates[i-1]`?
            # Same logic as Subsets II: at a given recursion level, don't
            # pick the same value twice. The first occurrence already
            # explores all combinations that include this value.
            if i > start and candidates[i] == candidates[i - 1]:
                continue

            # Why `break` instead of `continue`?
            # The array is SORTED. If candidates[i] already exceeds remaining,
            # then candidates[i+1], candidates[i+2], ... are all >= candidates[i],
            # so they'll also exceed remaining. No point checking further --
            # `break` skips them all. `continue` would wastefully check each one.
            if candidates[i] > remaining:
                break

            current.append(candidates[i])
            backtrack(i + 1, current, remaining - candidates[i])
            current.pop()

    backtrack(0, [], target)
    return result
```

---

#### Pattern 3D: Combination Sum III

**Problem:** LeetCode 216 - K numbers from 1-9 that sum to n

```python
def combinationSum3(k: int, n: int) -> list[list[int]]:
    result = []

    def backtrack(start, current, remaining):
        if len(current) == k:
            if remaining == 0:
                result.append(current.copy())
            return

        for i in range(start, 10):
            # Why `i > remaining` with `break`?
            # Numbers 1-9 are naturally sorted. If the current number `i`
            # already exceeds what's left to reach the target, every number
            # after it (i+1, i+2, ...) is even larger. Example: remaining=5,
            # i=6 -- we can't use 6 (too big), and 7,8,9 are worse. Break out.
            if i > remaining:  # Pruning
                break

            current.append(i)
            backtrack(i + 1, current, remaining - i)
            current.pop()

    backtrack(1, [], n)
    return result
```

---

### PATTERN 4: String Problems

---

#### Pattern 4A: Letter Combinations of Phone Number

**Problem:** LeetCode 17 - Generate letter combinations from digits

```python
def letterCombinations(digits: str) -> list[str]:
    if not digits:
        return []

    mapping = {
        '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
        '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
    }

    result = []

    def backtrack(index, current):
        if index == len(digits):
            result.append(''.join(current))
            return

        for char in mapping[digits[index]]:
            current.append(char)
            backtrack(index + 1, current)
            current.pop()

    backtrack(0, [])
    return result
```

---

#### Pattern 4B: Generate Parentheses

**Problem:** LeetCode 22 - Generate all valid parentheses combinations

```python
def generateParenthesis(n: int) -> list[str]:
    result = []

    def backtrack(current, open_count, close_count):
        if len(current) == 2 * n:
            result.append(''.join(current))
            return

        # Why `open_count < n`?
        # We have exactly n open parens to place. If we've already placed
        # all n, we can't add more. This is the "supply limit" for '('.
        if open_count < n:
            current.append('(')
            backtrack(current, open_count + 1, close_count)
            current.pop()

        # Why `close_count < open_count`?
        # A ')' must close a previously opened '('. If close_count == open_count,
        # every '(' so far has a matching ')' -- adding another ')' would be
        # unmatched (e.g., "())" is invalid). This single rule guarantees we
        # never produce an invalid sequence: every ')' has a '(' to its left.
        if close_count < open_count:
            current.append(')')
            backtrack(current, open_count, close_count + 1)
            current.pop()

    backtrack([], 0, 0)
    return result
```

---

#### Pattern 4C: Palindrome Partitioning

**Problem:** LeetCode 131 - Partition into palindromic substrings

```python
def partition(s: str) -> list[list[str]]:
    result = []

    def is_palindrome(string):
        return string == string[::-1]

    def backtrack(start, current):
        if start == len(s):
            result.append(current.copy())
            return

        for end in range(start + 1, len(s) + 1):
            substring = s[start:end]
            # Why check `is_palindrome(substring)` before recursing?
            # This is our pruning condition. We only want partitions where
            # EVERY piece is a palindrome. If this piece isn't, there's no
            # point exploring further splits after it -- the partition is
            # already invalid. This skips huge subtrees.
            if is_palindrome(substring):
                current.append(substring)
                backtrack(end, current)
                current.pop()

    backtrack(0, [])
    return result
```

---

#### Pattern 4D: Restore IP Addresses

**Problem:** LeetCode 93 - Insert dots to create valid IP addresses

```python
def restoreIpAddresses(s: str) -> list[str]:
    result = []

    def is_valid(segment):
        if not segment or len(segment) > 3:
            return False
        # Why `len(segment) > 1 and segment[0] == '0'`?
        # IP addresses don't allow leading zeros: "01" or "001" are invalid,
        # but "0" alone is fine. "01.01.01.01" is not a real IP address.
        # The `len > 1` check lets the single "0" through.
        if len(segment) > 1 and segment[0] == '0':
            return False
        return 0 <= int(segment) <= 255

    def backtrack(start, parts):
        # Why `len(parts) == 4` AND `start == len(s)`? Both needed!
        # An IP has exactly 4 octets, so we need exactly 4 parts.
        # But we also need every character used -- "1.2.3.4" from "12345"
        # would leave '5' unused. Only when BOTH conditions hold do we
        # have a valid, complete IP address.
        if len(parts) == 4:
            if start == len(s):
                result.append('.'.join(parts))
            return

        # Try segments of length 1, 2, 3
        for length in range(1, 4):
            if start + length > len(s):
                break

            segment = s[start:start + length]
            if is_valid(segment):
                parts.append(segment)
                backtrack(start + length, parts)
                parts.pop()

    backtrack(0, [])
    return result
```

---

### PATTERN 5: Constraint Satisfaction

---

#### Pattern 5A: N-Queens

**Problem:** LeetCode 51 - Place N queens on NxN board

```python
def solveNQueens(n: int) -> list[list[str]]:
    result = []
    board = [['.'] * n for _ in range(n)]

    # Track columns and diagonals under attack
    cols = set()
    diag1 = set()  # row - col
    diag2 = set()  # row + col

    def backtrack(row):
        if row == n:
            result.append([''.join(r) for r in board])
            return

        for col in range(n):
            # Why `row - col` for one diagonal, `row + col` for the other?
            # Picture the board. On a top-left-to-bottom-right diagonal (\),
            # as you move down-right, row increases and col increases by the
            # same amount, so `row - col` stays constant. Example:
            #   (0,0), (1,1), (2,2) all have row-col = 0
            #   (0,1), (1,2), (2,3) all have row-col = -1
            # On a top-right-to-bottom-left diagonal (/), row increases
            # while col decreases, so `row + col` stays constant. Example:
            #   (0,2), (1,1), (2,0) all have row+col = 2
            # If any queen shares the same row-col or row+col, they're on
            # the same diagonal and would attack each other.
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue

            # Place queen
            board[row][col] = 'Q'
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)

            backtrack(row + 1)

            # Remove queen
            board[row][col] = '.'
            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)

    backtrack(0)
    return result
```

---

#### Pattern 5B: N-Queens II (Count Solutions)

**Problem:** LeetCode 52 - Just count the number of solutions

```python
def totalNQueens(n: int) -> int:
    count = 0
    cols = set()
    diag1 = set()
    diag2 = set()

    def backtrack(row):
        nonlocal count
        if row == n:
            count += 1
            return

        for col in range(n):
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue

            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)

            backtrack(row + 1)

            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)

    backtrack(0)
    return count
```

---

#### Pattern 5C: Sudoku Solver

**Problem:** LeetCode 37 - Solve Sudoku puzzle

```python
def solveSudoku(board: list[list[str]]) -> None:
    def is_valid(row, col, num):
        # Check row
        if num in board[row]:
            return False

        # Check column
        if num in [board[i][col] for i in range(9)]:
            return False

        # Check 3x3 box
        # Why `3 * (row // 3)` and `3 * (col // 3)`?
        # Integer division `row // 3` tells us WHICH box (0, 1, or 2).
        # Multiplying by 3 gives the top-left corner of that box.
        # Example: row=5 -> 5//3 = 1 -> 3*1 = 3 (box starts at row 3).
        # row=7 -> 7//3 = 2 -> 3*2 = 6. This "snaps" any row to its box origin.
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                if board[i][j] == num:
                    return False

        return True

    # Why does `backtrack()` return True/False?
    # Unlike subsets/permutations where we collect ALL solutions, Sudoku
    # has exactly one solution. We need to STOP as soon as we find it.
    # Returning True propagates up the call stack: "solution found, stop searching."
    # Returning False means "this path is a dead end, try something else."
    def backtrack():
        # Find empty cell
        for i in range(9):
            for j in range(9):
                if board[i][j] == '.':
                    for num in '123456789':
                        if is_valid(i, j, num):
                            board[i][j] = num
                            if backtrack():
                                return True
                            board[i][j] = '.'
                    # Why `return False` here?
                    # We tried all 9 digits for this cell and NONE worked.
                    # This cell is unsolvable with the current board state,
                    # so the mistake was made earlier. We must backtrack --
                    # telling the caller "undo your last placement and try
                    # a different number."
                    return False  # No valid number for this cell

        return True  # All cells filled

    backtrack()
```

**Optimized with sets:**
```python
def solveSudoku_optimized(board: list[list[str]]) -> None:
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    boxes = [set() for _ in range(9)]
    empty = []

    # Initialize
    for i in range(9):
        for j in range(9):
            if board[i][j] == '.':
                empty.append((i, j))
            else:
                num = board[i][j]
                rows[i].add(num)
                cols[j].add(num)
                boxes[(i // 3) * 3 + j // 3].add(num)

    def backtrack(idx):
        if idx == len(empty):
            return True

        i, j = empty[idx]
        box_idx = (i // 3) * 3 + j // 3

        for num in '123456789':
            if num not in rows[i] and num not in cols[j] and num not in boxes[box_idx]:
                board[i][j] = num
                rows[i].add(num)
                cols[j].add(num)
                boxes[box_idx].add(num)

                if backtrack(idx + 1):
                    return True

                board[i][j] = '.'
                rows[i].remove(num)
                cols[j].remove(num)
                boxes[box_idx].remove(num)

        return False

    backtrack(0)
```

---

### PATTERN 6: Word Problems

---

#### Pattern 6A: Word Search

**Problem:** LeetCode 79 - Find word in grid

```python
def exist(board: list[list[str]], word: str) -> bool:
    rows, cols = len(board), len(board[0])

    def backtrack(r, c, index):
        if index == len(word):
            return True

        if r < 0 or r >= rows or c < 0 or c >= cols:
            return False
        # Why check `board[r][c] != word[index]`?
        # Character mismatch = prune. If this cell doesn't match the letter
        # we need, there's no point exploring any of the 4 directions from
        # here. This single check eliminates huge branches early.
        if board[r][c] != word[index]:
            return False

        # Mark as visited
        # Why mutate the board with '#' instead of using a visited set?
        # 1. Space: A set of (r,c) tuples costs O(n) extra memory; in-place
        #    marking costs O(1) extra.
        # 2. Shared state: The board IS the shared state -- marking it directly
        #    means recursive calls automatically see what's visited without
        #    passing an extra structure around.
        # 3. We restore it immediately after, so the board stays unchanged
        #    for other starting points.
        temp = board[r][c]
        board[r][c] = '#'

        # Explore neighbors
        # Why use `or` chain instead of exploring all four?
        # Short-circuit evaluation: as soon as ANY direction finds the word,
        # `or` returns True immediately without trying the remaining directions.
        # We only need one valid path, not all of them.
        found = (backtrack(r + 1, c, index + 1) or
                 backtrack(r - 1, c, index + 1) or
                 backtrack(r, c + 1, index + 1) or
                 backtrack(r, c - 1, index + 1))

        # Restore
        board[r][c] = temp
        return found

    for i in range(rows):
        for j in range(cols):
            if backtrack(i, j, 0):
                return True

    return False
```

---

#### Pattern 6B: Word Search II

**Problem:** LeetCode 212 - Find multiple words (uses Trie for efficiency)

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.word = None

def findWords(board: list[list[str]], words: list[str]) -> list[str]:
    # Build Trie
    root = TrieNode()
    for word in words:
        node = root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.word = word

    rows, cols = len(board), len(board[0])
    result = []

    def backtrack(r, c, node):
        char = board[r][c]
        if char not in node.children:
            return

        next_node = node.children[char]

        if next_node.word:
            result.append(next_node.word)
            # Why set `next_node.word = None` after finding?
            # The same word can be found starting from multiple board cells.
            # Without this, we'd add "OATH" to results every time we find it.
            # Setting to None acts like "cross it off the list" -- once found,
            # don't report it again.
            next_node.word = None  # Avoid duplicates

        # Mark visited
        board[r][c] = '#'

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] != '#':
                backtrack(nr, nc, next_node)

        # Restore
        board[r][c] = char

    for i in range(rows):
        for j in range(cols):
            backtrack(i, j, root)

    return result
```

**Why use a Trie instead of calling Word Search (6A) for each word?**
If you have `k` words and an `m x n` board, calling the single-word search for each word
costs `O(k * m * n * 4^L)` where `L` is word length. With a Trie, you do ONE traversal
of the board and simultaneously check all `k` words by walking the Trie in lockstep.
All words sharing a prefix (e.g., "oath" and "oat") share traversal work. The Trie
turns `k` separate searches into one search with shared prefix elimination.

---

<a name="post-processing"></a>
## 5. Post-Processing Reference

| Problem Type | Result Collection | Copy Needed? |
|--------------|------------------|--------------|
| **All solutions** | Append to result list | Yes (copy current) |
| **Count solutions** | Increment counter | No |
| **Any solution** | Return immediately | No |
| **Best solution** | Compare and update | Yes |

**Important:** Always copy mutable objects (lists) before appending to result!

```python
# WRONG: All results point to same list
result.append(current)

# CORRECT: Save a copy
result.append(current.copy())
# or
result.append(current[:])
# or
result.append(list(current))
```

---

<a name="pitfalls"></a>
## 6. Common Pitfalls & Solutions

### Pitfall 1: Not Copying the Result

```python
# WRONG
result.append(current)  # current will change later!
```

**Solution:** Always copy: `result.append(current.copy())`

---

### Pitfall 2: Forgetting to Backtrack

```python
# WRONG: No undo
current.append(nums[i])
backtrack(i + 1, current)
# Missing: current.pop()
```

**Solution:** Always undo after recursion

---

### Pitfall 3: Wrong Start Index

```python
# WRONG for subsets: starts from 0 each time
for i in range(len(nums)):  # Should be range(start, len(nums))
```

**Solution:** Use start parameter correctly

---

### Pitfall 4: Duplicate Handling

```python
# WRONG: Produces duplicates
for i in range(start, len(nums)):
    current.append(nums[i])
    backtrack(i + 1, current)
    current.pop()
```

**Solution:** Sort + skip duplicates
```python
if i > start and nums[i] == nums[i - 1]:
    continue
```

---

### Pitfall 5: Modifying Loop Variable

```python
# DANGEROUS: Modifying list while iterating
for item in items:
    items.remove(item)  # Don't do this!
```

**Solution:** Iterate over indices or copy

---

### Pitfall 6: Infinite Recursion

```python
# WRONG: No progress toward base case
def backtrack(current):
    # Missing base case or progress
    backtrack(current)
```

**Solution:** Ensure each recursive call makes progress

---

<a name="recognition"></a>
## 7. Problem Recognition Framework

### Step 1: Is it Backtracking?

**Backtracking indicators:**
1. "Generate all..." / "Find all..."
2. "All possible combinations/permutations"
3. Constraint satisfaction
4. Search space is exponential but prunable

**NOT backtracking if:**
- Need optimal solution only (maybe DP)
- Linear scan suffices
- Greedy works

### Step 2: What Type?

| Clue | Type |
|------|------|
| "All subsets" | Subsets pattern |
| "All arrangements" | Permutations |
| "Choose k from n" | Combinations |
| "Sum to target" | Combination Sum |
| "Place without conflict" | Constraint satisfaction |
| "Split into parts" | Partition |

### Step 3: Handle Duplicates?

| Input | Handling |
|-------|----------|
| All distinct | None needed |
| May have duplicates | Sort + skip |

### Decision Tree

```
              Generate all solutions?
                       ↓
                   ┌───┴───┐
                  Yes       No
                   ↓         ↓
             Use each      Other
             element?     approach
                   ↓
            ┌──────┴──────┐
         Yes              No
            ↓              ↓
       Permutation    Order matters?
                          ↓
                    ┌─────┴─────┐
                   No          Yes
                    ↓            ↓
              Subsets      Combinations
              (include/    (choose k)
               exclude)
```

---

<a name="checklist"></a>
## 8. Interview Preparation Checklist

### Before the Interview

**Master the fundamentals:**
- [ ] Can write subsets template from memory
- [ ] Can write permutations template from memory
- [ ] Understand when/how to skip duplicates
- [ ] Know how to prune for efficiency

**Know the patterns:**
- [ ] Subsets (with/without duplicates)
- [ ] Permutations (with/without duplicates)
- [ ] Combinations / Combination Sum
- [ ] String partitioning
- [ ] N-Queens / Sudoku

**Common problems solved:**
- [ ] LC 78: Subsets
- [ ] LC 46: Permutations
- [ ] LC 77: Combinations
- [ ] LC 39: Combination Sum
- [ ] LC 22: Generate Parentheses
- [ ] LC 51: N-Queens
- [ ] LC 79: Word Search

### During the Interview

**1. Clarify (30 seconds)**
- All solutions or just one?
- Are there duplicates in input?
- What constraints exist?

**2. Identify pattern (30 seconds)**
- Subsets, permutations, or combinations?
- Need duplicate handling?
- What's the pruning condition?

**3. Code (3-4 minutes)**
- Write the backtrack function
- Handle base case
- Make choice, recurse, undo choice

**4. Test (1-2 minutes)**
- Empty input
- Single element
- Duplicates if applicable

**5. Analyze (30 seconds)**
- Time: Usually O(k × 2^n) or O(n!)
- Space: O(n) for recursion depth

---

## 9. Quick Reference Cards

### Subsets Template
```python
def backtrack(start, current):
    result.append(current.copy())
    for i in range(start, len(nums)):
        current.append(nums[i])
        backtrack(i + 1, current)
        current.pop()
```

### Permutations Template
```python
def backtrack(current):
    if len(current) == len(nums):
        result.append(current.copy())
        return
    for i in range(len(nums)):
        if used[i]: continue
        used[i] = True
        current.append(nums[i])
        backtrack(current)
        current.pop()
        used[i] = False
```

### Skip Duplicates
```python
nums.sort()  # Required!
if i > start and nums[i] == nums[i-1]:
    continue
```

---

## 10. Complexity Reference

| Pattern | Time | Space |
|---------|------|-------|
| Subsets | O(n × 2^n) | O(n) |
| Permutations | O(n × n!) | O(n) |
| Combinations (n,k) | O(k × C(n,k)) | O(k) |
| N-Queens | O(n!) | O(n) |
| Sudoku | O(9^m) where m = empty | O(m) |

---

## Final Thoughts

**Remember:**
1. Backtracking = make choice, recurse, undo choice
2. Always copy mutable results before saving
3. Sort + skip for duplicate handling
4. Prune early for efficiency
5. The choice/recurse/undo pattern is universal

**When stuck:**
1. Draw the decision tree
2. Ask: "What choices do I have at each step?"
3. Ask: "When do I have a complete solution?"
4. Start with brute force, then optimize

---

## Appendix: Practice Problem Set

### Easy
- 401. Binary Watch

### Medium
- 17. Letter Combinations of a Phone Number
- 22. Generate Parentheses
- 39. Combination Sum
- 40. Combination Sum II
- 46. Permutations
- 47. Permutations II
- 77. Combinations
- 78. Subsets
- 79. Word Search
- 90. Subsets II
- 93. Restore IP Addresses
- 131. Palindrome Partitioning

### Hard
- 37. Sudoku Solver
- 51. N-Queens
- 52. N-Queens II
- 212. Word Search II

**Recommended Practice Order:**
1. Basic: 78 → 46 → 77
2. With duplicates: 90 → 47
3. Sum variants: 39 → 40 → 216
4. Strings: 17 → 22 → 131 → 93
5. Constraints: 51 → 37 → 79

---

## Appendix: Conditional Quick Reference

### 1. Base Cases

| Condition | Where Used | Why |
|-----------|-----------|-----|
| `index == len(nums)` | Template B (Subsets) | Decided include/exclude for every element -- subset is complete |
| `len(current) == len(nums)` | Template C (Permutations) | All elements placed -- permutation is complete |
| `len(current) == k` | Template D (Combinations) | Chosen exactly k elements -- combination is complete |
| `remaining == 0` | Combination Sum (3B, 3C) | Current selection sums exactly to target |
| `start == len(s)` | Template F, Palindrome Partition (4C) | Read cursor reached end of string -- all characters partitioned |
| `row == n` | N-Queens (5A) | Placed a queen in every row -- valid arrangement found |
| `len(parts) == 4 and start == len(s)` | IP Addresses (4D) | Exactly 4 octets AND all characters consumed -- valid IP |
| `index == len(word)` | Word Search (6A) | Matched every character in the word -- word found |
| `idx == len(empty)` | Sudoku (5C) | All empty cells filled -- puzzle solved |

### 2. Pruning Conditions

| Condition | Where Used | Why |
|-----------|-----------|-----|
| `remaining < needed` | Template D (Combinations) | Fewer elements left than slots to fill -- completion impossible |
| `remaining < 0` | Combination Sum (3B) | Overshot the target -- adding more only makes it worse |
| `candidates[i] > remaining` | Combination Sum II (3C) | Sorted array: this and all later candidates exceed target, `break` |
| `i > remaining` | Combination Sum III (3D) | Numbers 1-9 are sorted: current number exceeds remaining sum, `break` |
| `board[r][c] != word[index]` | Word Search (6A) | Character mismatch -- no point exploring neighbors from here |
| `is_palindrome(substring)` | Palindrome Partition (4C) | Only recurse on palindromic pieces -- skip invalid partitions |
| `col in cols or (row-col) in diag1 or (row+col) in diag2` | N-Queens (5A) | Queen would be attacked on column or diagonal -- invalid placement |

### 3. Duplicate Handling

| Condition | Where Used | Why |
|-----------|-----------|-----|
| `i > start and nums[i] == nums[i-1]` | Subsets II (1B), Combination Sum II (3C) | Skip same-value choice at same recursion level; `i > start` (not `i > 0`) so deeper-level picks are not wrongly skipped |
| `i > 0 and nums[i] == nums[i-1] and not used[i-1]` | Permutations II (2B) | Among identical values, enforce left-to-right usage order; `not used[i-1]` means we're trying to use a later copy before an earlier one |
| `next_node.word = None` | Word Search II (6B) | After finding a word, clear it from the Trie so it isn't added to results again |

### 4. Constraint Checks

| Condition | Where Used | Why |
|-----------|-----------|-----|
| `used[i]` | Template C (Permutations) | Element already in current permutation -- can't reuse same index |
| `open_count < n` | Generate Parentheses (4B) | Supply limit: can't place more open parens than `n` total |
| `close_count < open_count` | Generate Parentheses (4B) | Balance rule: every `)` must match a prior `(` -- prevents invalid sequences |
| `len(segment) > 1 and segment[0] == '0'` | IP Addresses (4D) | No leading zeros in IP octets: "01" is invalid, but "0" alone is fine |
| `3 * (row // 3), 3 * (col // 3)` | Sudoku (5C) | Snaps any cell to its 3x3 box's top-left corner for box constraint check |
| `backtrack()` returns `True/False` | Sudoku (5C) | Need to stop after first solution; `True` propagates "found it," `False` triggers backtracking |
| `board[r][c] = '#'` (temp mark) | Word Search (6A, 6B) | In-place visited marking: O(1) space, avoids passing a separate visited set |

Good luck with your interview preparation!
