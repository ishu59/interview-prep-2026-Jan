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
        # Every state is a valid subset
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
        # Complete when all elements used
        if len(current) == len(nums):
            result.append(current.copy())
            return

        for i in range(len(nums)):
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
        if start == len(s):
            result.append(current.copy())
            return

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
            # Skip duplicates at same level
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

            # Skip duplicates: if same as previous AND previous not used
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
        if remaining < 0:
            return

        for i in range(start, len(candidates)):
            current.append(candidates[i])
            # Pass i (not i+1) to allow reuse
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
            # Skip duplicates at same level
            if i > start and candidates[i] == candidates[i - 1]:
                continue

            # Pruning: remaining too small
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

        # Can always add open if we haven't used all
        if open_count < n:
            current.append('(')
            backtrack(current, open_count + 1, close_count)
            current.pop()

        # Can add close only if we have open to match
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
        if len(segment) > 1 and segment[0] == '0':
            return False
        return 0 <= int(segment) <= 255

    def backtrack(start, parts):
        # Have 4 parts and used all characters
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
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                if board[i][j] == num:
                    return False

        return True

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
        if board[r][c] != word[index]:
            return False

        # Mark as visited
        temp = board[r][c]
        board[r][c] = '#'

        # Explore neighbors
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

Good luck with your interview preparation!
