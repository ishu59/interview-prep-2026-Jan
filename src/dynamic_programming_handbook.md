# The Complete Dynamic Programming Handbook
> A template-based approach for mastering DP in coding interviews

**Philosophy:** Dynamic Programming is not about memorizing solutions. It's about recognizing **optimal substructure** and **overlapping subproblems**, then systematically building solutions from smaller to larger problems.

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

- **The Notebook of Solved Problems**: You're solving a big math test where question 10 requires the answer from question 7, which requires the answer from question 3. Instead of re-solving question 3 every time it's needed, you write each answer in a notebook and look it up. That notebook is your DP table.
- **The LEGO Instruction Manual**: A complex LEGO set is just small sub-assemblies combined. Step 1: figure out what sub-assemblies you need (subproblems). Step 2: figure out how to combine them (recurrence relation). Step 3: figure out what order to build them in (bottom-up) or just build on demand (top-down).

### No-Jargon Translation

- **Optimal substructure**: the best answer to the big problem is built from the best answers to smaller pieces
- **Overlapping subproblems**: the same smaller piece gets needed again and again
- **Memoization**: writing down answers so you never re-solve -- the notebook
- **Recurrence relation**: the formula that says "the answer to step N depends on steps N-1, N-2, ..."
- **Base case**: the smallest problem you can answer without help -- like knowing 0! = 1
- **State**: the information you need to describe one subproblem -- like "what's the best answer using the first i items with capacity j?"
- **Transition**: how you move from one state to the next

### Mental Model

> "Dynamic programming is solving a jigsaw puzzle by first solving each small corner, writing down the result, and reusing those solved corners whenever a bigger section needs them -- never re-solving a piece you've already figured out."

---

### What is Dynamic Programming?

DP is an optimization technique for problems with:
1. **Optimal Substructure:** Optimal solution contains optimal solutions to subproblems
2. **Overlapping Subproblems:** Same subproblems are solved multiple times

### The FAST Method (Framework for DP)

1. **F**ind the recursive relation
2. **A**nalyze the base case(s)
3. **S**tore (memoize) or build table (tabulate)
4. **T**urn around (optimize if needed)

### Top-Down vs Bottom-Up

| Approach | Method | Pros | Cons |
|----------|--------|------|------|
| **Top-Down** | Recursion + Memoization | Intuitive, only computes needed states | Recursion overhead |
| **Bottom-Up** | Iterative + Tabulation | No recursion, can optimize space | Must determine order |

### The DP Recipe

1. **Define state:** What variables describe a subproblem?
2. **Define dp[i]:** What does dp[i] represent?
3. **Find transition:** How does dp[i] relate to smaller subproblems?
4. **Identify base cases:** What are the smallest subproblems?
5. **Determine order:** In what order to fill the table?
6. **Extract answer:** Where is the final answer?

### Example: Fibonacci

```python
# 1. State: Current position n
# 2. dp[n] = n-th Fibonacci number
# 3. Transition: dp[n] = dp[n-1] + dp[n-2]
# 4. Base cases: dp[0] = 0, dp[1] = 1
# 5. Order: From 0 to n
# 6. Answer: dp[n]

def fib(n):
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]
```

---

<a name="master-templates"></a>
## 2. The Master Templates

### Template A: 1D DP (Linear)

```python
def linear_dp(nums):
    """
    Template for 1D DP problems.
    dp[i] represents answer for subproblem ending at/up to index i.
    """
    n = len(nums)
    dp = [0] * n  # or appropriate initial value

    # Base case
    dp[0] = base_value(nums[0])

    # Fill table
    for i in range(1, n):
        # Transition: dp[i] depends on previous states
        dp[i] = transition(dp, nums, i)

    # Return answer
    return answer(dp)
```

**Variants:**
- `dp[i]` = best answer **ending at** i (like max subarray)
- `dp[i]` = best answer for **first i elements** (like house robber)
- `dp[i]` = answer for subproblem of **size i** (like coin change)

---

### Template B: 2D DP (Grid/Two Sequences)

```python
def grid_dp(grid):
    """
    Template for 2D grid DP or two-sequence problems.
    dp[i][j] represents answer for subproblem at position (i, j).
    """
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]

    # Base cases: first row and first column
    dp[0][0] = grid[0][0]
    for i in range(1, m):
        dp[i][0] = transition_row(dp, grid, i, 0)
    for j in range(1, n):
        dp[0][j] = transition_col(dp, grid, 0, j)

    # Fill table
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = transition(dp, grid, i, j)

    return dp[m-1][n-1]
```

---

### Template C: 0/1 Knapsack

```python
def knapsack(weights, values, capacity):
    """
    Template for 0/1 knapsack and similar problems.
    dp[i][w] = max value using first i items with capacity w.
    """
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # Don't take item i
            dp[i][w] = dp[i-1][w]

            # Take item i (if it fits)
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w],
                              dp[i-1][w - weights[i-1]] + values[i-1])

    return dp[n][capacity]
```

**Space-optimized (1D):**
```python
def knapsack_optimized(weights, values, capacity):
    dp = [0] * (capacity + 1)

    for i in range(len(weights)):
        # Traverse backwards to avoid using updated values
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])

    return dp[capacity]
```

---

### Template D: Unbounded Knapsack

```python
def unbounded_knapsack(weights, values, capacity):
    """
    Template for unbounded knapsack (can use items multiple times).
    dp[w] = max value with capacity w.
    """
    dp = [0] * (capacity + 1)

    for w in range(1, capacity + 1):
        for i in range(len(weights)):
            if weights[i] <= w:
                dp[w] = max(dp[w], dp[w - weights[i]] + values[i])

    return dp[capacity]
```

---

### Template E: Longest Common Subsequence (LCS)

```python
def lcs(text1, text2):
    """
    Template for LCS and similar two-string problems.
    dp[i][j] = LCS of text1[:i] and text2[:j].
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]
```

---

### Template F: Interval DP

```python
def interval_dp(arr):
    """
    Template for interval DP problems.
    dp[i][j] = answer for subarray arr[i:j+1].
    """
    n = len(arr)
    dp = [[0] * n for _ in range(n)]

    # Base case: single elements
    for i in range(n):
        dp[i][i] = base_value(arr[i])

    # Fill by increasing interval length
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')  # or -inf for max

            # Try all split points
            for k in range(i, j):
                dp[i][j] = min(dp[i][j],
                              dp[i][k] + dp[k+1][j] + cost(i, j, k))

    return dp[0][n-1]
```

---

### Template G: State Machine DP

```python
def state_machine_dp(prices):
    """
    Template for state machine DP (like stock problems).
    Track multiple states at each position.
    """
    n = len(prices)
    # Define states: e.g., holding, not_holding, cooldown
    hold = [0] * n
    sold = [0] * n
    rest = [0] * n

    # Base cases
    hold[0] = -prices[0]
    sold[0] = 0
    rest[0] = 0

    for i in range(1, n):
        # Transitions based on previous states
        hold[i] = max(hold[i-1], rest[i-1] - prices[i])
        sold[i] = hold[i-1] + prices[i]
        rest[i] = max(rest[i-1], sold[i-1])

    return max(sold[n-1], rest[n-1])
```

---

### Quick Decision Matrix

| Problem Type | Template | dp[i] Represents |
|--------------|----------|------------------|
| Linear sequence | A | Answer ending at i |
| Grid paths | B | Answer at cell (i,j) |
| Subset/item selection | C | Answer using first i items |
| Unlimited use | D | Answer with capacity i |
| Two sequences | E | Answer for prefixes i,j |
| Merge/split intervals | F | Answer for interval [i,j] |
| Multiple states | G | Multiple dp arrays |

---

<a name="pattern-guide"></a>
## 3. Pattern Classification Guide

### Category 1: Linear DP
- Single array/sequence
- Answer depends on previous elements
- **Template A**
- Examples: House Robber, Max Subarray, LIS

### Category 2: Grid DP
- 2D matrix traversal
- Move right/down typically
- **Template B**
- Examples: Unique Paths, Min Path Sum

### Category 3: Knapsack Variants
- Select items with constraints
- 0/1 or unbounded
- **Template C or D**
- Examples: Partition Equal Subset, Coin Change

### Category 4: String DP
- Two strings comparison
- Edit distance, subsequences
- **Template E**
- Examples: LCS, Edit Distance

### Category 5: Interval DP
- Subarray operations
- Merge or split
- **Template F**
- Examples: Matrix Chain, Burst Balloons

### Category 6: State Machine
- Multiple states to track
- Transitions between states
- **Template G**
- Examples: Stock Problems

---

<a name="patterns"></a>
## 4. Complete Pattern Library

### PATTERN 1: Linear DP - Maximum/Minimum

---

#### Pattern 1A: Maximum Subarray (Kadane's)

**Problem:** LeetCode 53 - Find contiguous subarray with largest sum

```python
def maxSubArray(nums: list[int]) -> int:
    # dp[i] = max subarray sum ENDING at index i
    dp = [0] * len(nums)
    dp[0] = nums[0]

    for i in range(1, len(nums)):
        # Either extend previous subarray or start new
        dp[i] = max(dp[i-1] + nums[i], nums[i])

    return max(dp)
```

**Space-optimized:**
```python
def maxSubArray_optimized(nums: list[int]) -> int:
    max_sum = current = nums[0]
    for num in nums[1:]:
        current = max(current + num, num)
        max_sum = max(max_sum, current)
    return max_sum
```

---

#### Pattern 1B: House Robber

**Problem:** LeetCode 198 - Max money without robbing adjacent houses

```python
def rob(nums: list[int]) -> int:
    if len(nums) == 1:
        return nums[0]

    # dp[i] = max money from first i houses
    dp = [0] * len(nums)
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])

    for i in range(2, len(nums)):
        # Either rob house i or don't
        dp[i] = max(dp[i-1], dp[i-2] + nums[i])

    return dp[-1]
```

**Space-optimized:**
```python
def rob_optimized(nums: list[int]) -> int:
    prev2, prev1 = 0, 0
    for num in nums:
        current = max(prev1, prev2 + num)
        prev2, prev1 = prev1, current
    return prev1
```

---

#### Pattern 1C: House Robber II (Circular)

**Problem:** LeetCode 213 - Houses in a circle

```python
def rob(nums: list[int]) -> int:
    if len(nums) == 1:
        return nums[0]

    def rob_linear(houses):
        prev2, prev1 = 0, 0
        for num in houses:
            prev2, prev1 = prev1, max(prev1, prev2 + num)
        return prev1

    # Either skip first house or skip last house
    return max(rob_linear(nums[1:]), rob_linear(nums[:-1]))
```

---

#### Pattern 1D: Longest Increasing Subsequence (LIS)

**Problem:** LeetCode 300 - Length of longest increasing subsequence

**O(n²) solution:**
```python
def lengthOfLIS(nums: list[int]) -> int:
    # dp[i] = length of LIS ending at index i
    dp = [1] * len(nums)

    for i in range(1, len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)
```

**O(n log n) solution using binary search:**
```python
import bisect

def lengthOfLIS_optimized(nums: list[int]) -> int:
    # tails[i] = smallest ending element of LIS of length i+1
    tails = []

    for num in nums:
        pos = bisect.bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num

    return len(tails)
```

---

### PATTERN 2: Grid DP

---

#### Pattern 2A: Unique Paths

**Problem:** LeetCode 62 - Count paths from top-left to bottom-right

```python
def uniquePaths(m: int, n: int) -> int:
    # dp[i][j] = number of paths to cell (i, j)
    dp = [[1] * n for _ in range(m)]

    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]

    return dp[m-1][n-1]
```

**Space-optimized:**
```python
def uniquePaths_optimized(m: int, n: int) -> int:
    dp = [1] * n
    for i in range(1, m):
        for j in range(1, n):
            dp[j] += dp[j-1]
    return dp[n-1]
```

---

#### Pattern 2B: Unique Paths II (With Obstacles)

**Problem:** LeetCode 63 - Paths with obstacles

```python
def uniquePathsWithObstacles(obstacleGrid: list[list[int]]) -> int:
    m, n = len(obstacleGrid), len(obstacleGrid[0])

    if obstacleGrid[0][0] == 1:
        return 0

    dp = [[0] * n for _ in range(m)]
    dp[0][0] = 1

    # First column
    for i in range(1, m):
        dp[i][0] = dp[i-1][0] if obstacleGrid[i][0] == 0 else 0

    # First row
    for j in range(1, n):
        dp[0][j] = dp[0][j-1] if obstacleGrid[0][j] == 0 else 0

    # Fill rest
    for i in range(1, m):
        for j in range(1, n):
            if obstacleGrid[i][j] == 0:
                dp[i][j] = dp[i-1][j] + dp[i][j-1]

    return dp[m-1][n-1]
```

---

#### Pattern 2C: Minimum Path Sum

**Problem:** LeetCode 64 - Min sum path from top-left to bottom-right

```python
def minPathSum(grid: list[list[int]]) -> int:
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]

    dp[0][0] = grid[0][0]

    # First column
    for i in range(1, m):
        dp[i][0] = dp[i-1][0] + grid[i][0]

    # First row
    for j in range(1, n):
        dp[0][j] = dp[0][j-1] + grid[0][j]

    # Fill rest
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]

    return dp[m-1][n-1]
```

---

#### Pattern 2D: Triangle

**Problem:** LeetCode 120 - Minimum path sum in triangle

```python
def minimumTotal(triangle: list[list[int]]) -> int:
    # Bottom-up approach: start from bottom row
    n = len(triangle)
    dp = triangle[-1][:]  # Copy bottom row

    for i in range(n - 2, -1, -1):
        for j in range(i + 1):
            dp[j] = triangle[i][j] + min(dp[j], dp[j + 1])

    return dp[0]
```

---

### PATTERN 3: 0/1 Knapsack Variants

---

#### Pattern 3A: Partition Equal Subset Sum

**Problem:** LeetCode 416 - Can partition array into two equal subsets?

```python
def canPartition(nums: list[int]) -> bool:
    total = sum(nums)
    if total % 2:
        return False

    target = total // 2

    # dp[i] = True if sum i is achievable
    dp = [False] * (target + 1)
    dp[0] = True

    for num in nums:
        # Traverse backwards for 0/1 knapsack
        for j in range(target, num - 1, -1):
            dp[j] = dp[j] or dp[j - num]

    return dp[target]
```

---

#### Pattern 3B: Target Sum

**Problem:** LeetCode 494 - Count ways to reach target with +/-

```python
def findTargetSumWays(nums: list[int], target: int) -> int:
    total = sum(nums)

    # If target not achievable
    if (total + target) % 2 or abs(target) > total:
        return 0

    # Transform: find subset with sum = (total + target) / 2
    subset_sum = (total + target) // 2

    dp = [0] * (subset_sum + 1)
    dp[0] = 1

    for num in nums:
        for j in range(subset_sum, num - 1, -1):
            dp[j] += dp[j - num]

    return dp[subset_sum]
```

---

#### Pattern 3C: Last Stone Weight II

**Problem:** LeetCode 1049 - Minimize last stone weight

```python
def lastStoneWeightII(stones: list[int]) -> int:
    # Partition into two groups, minimize |sum1 - sum2|
    total = sum(stones)
    target = total // 2

    dp = [False] * (target + 1)
    dp[0] = True

    for stone in stones:
        for j in range(target, stone - 1, -1):
            dp[j] = dp[j] or dp[j - stone]

    # Find largest achievable sum <= target
    for j in range(target, -1, -1):
        if dp[j]:
            return total - 2 * j

    return total
```

---

### PATTERN 4: Unbounded Knapsack / Coin Change

---

#### Pattern 4A: Coin Change (Min Coins)

**Problem:** LeetCode 322 - Minimum coins to make amount

```python
def coinChange(coins: list[int], amount: int) -> int:
    # dp[i] = min coins to make amount i
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)

    return dp[amount] if dp[amount] != float('inf') else -1
```

---

#### Pattern 4B: Coin Change II (Count Ways)

**Problem:** LeetCode 518 - Count ways to make amount

```python
def change(amount: int, coins: list[int]) -> int:
    # dp[i] = number of ways to make amount i
    dp = [0] * (amount + 1)
    dp[0] = 1

    # Process coins one by one (to avoid counting permutations)
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]

    return dp[amount]
```

---

#### Pattern 4C: Perfect Squares

**Problem:** LeetCode 279 - Min perfect squares that sum to n

```python
def numSquares(n: int) -> int:
    dp = [float('inf')] * (n + 1)
    dp[0] = 0

    for i in range(1, n + 1):
        j = 1
        while j * j <= i:
            dp[i] = min(dp[i], dp[i - j * j] + 1)
            j += 1

    return dp[n]
```

---

#### Pattern 4D: Word Break

**Problem:** LeetCode 139 - Can string be segmented into words?

```python
def wordBreak(s: str, wordDict: list[str]) -> bool:
    word_set = set(wordDict)
    n = len(s)

    # dp[i] = True if s[:i] can be segmented
    dp = [False] * (n + 1)
    dp[0] = True

    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break

    return dp[n]
```

---

### PATTERN 5: String DP

---

#### Pattern 5A: Longest Common Subsequence

**Problem:** LeetCode 1143 - LCS of two strings

```python
def longestCommonSubsequence(text1: str, text2: str) -> int:
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]
```

---

#### Pattern 5B: Edit Distance

**Problem:** LeetCode 72 - Min operations to convert word1 to word2

```python
def minDistance(word1: str, word2: str) -> int:
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # Delete
                    dp[i][j-1],    # Insert
                    dp[i-1][j-1]   # Replace
                )

    return dp[m][n]
```

---

#### Pattern 5C: Longest Palindromic Subsequence

**Problem:** LeetCode 516 - Longest palindromic subsequence

```python
def longestPalindromeSubseq(s: str) -> int:
    n = len(s)
    # dp[i][j] = LPS of s[i:j+1]
    dp = [[0] * n for _ in range(n)]

    # Base case: single characters
    for i in range(n):
        dp[i][i] = 1

    # Fill by increasing length
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                dp[i][j] = dp[i+1][j-1] + 2
            else:
                dp[i][j] = max(dp[i+1][j], dp[i][j-1])

    return dp[0][n-1]
```

**Alternative: LCS(s, reverse(s))**
```python
def longestPalindromeSubseq_lcs(s: str) -> int:
    return longestCommonSubsequence(s, s[::-1])
```

---

#### Pattern 5D: Distinct Subsequences

**Problem:** LeetCode 115 - Count distinct subsequences of s that equal t

```python
def numDistinct(s: str, t: str) -> int:
    m, n = len(s), len(t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Empty t can be formed by any prefix of s
    for i in range(m + 1):
        dp[i][0] = 1

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Don't use s[i-1]
            dp[i][j] = dp[i-1][j]

            # Use s[i-1] if matches
            if s[i-1] == t[j-1]:
                dp[i][j] += dp[i-1][j-1]

    return dp[m][n]
```

---

### PATTERN 6: State Machine DP (Stock Problems)

---

#### Pattern 6A: Best Time to Buy and Sell Stock

**Problem:** LeetCode 121 - One transaction allowed

```python
def maxProfit(prices: list[int]) -> int:
    min_price = float('inf')
    max_profit = 0

    for price in prices:
        min_price = min(min_price, price)
        max_profit = max(max_profit, price - min_price)

    return max_profit
```

---

#### Pattern 6B: Best Time to Buy and Sell Stock II

**Problem:** LeetCode 122 - Unlimited transactions

```python
def maxProfit(prices: list[int]) -> int:
    profit = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            profit += prices[i] - prices[i-1]
    return profit
```

**State machine approach:**
```python
def maxProfit_dp(prices: list[int]) -> int:
    hold = -prices[0]  # Holding stock
    cash = 0           # Not holding stock

    for price in prices[1:]:
        hold = max(hold, cash - price)
        cash = max(cash, hold + price)

    return cash
```

---

#### Pattern 6C: Best Time to Buy and Sell Stock III

**Problem:** LeetCode 123 - At most 2 transactions

```python
def maxProfit(prices: list[int]) -> int:
    buy1 = buy2 = float('-inf')
    sell1 = sell2 = 0

    for price in prices:
        buy1 = max(buy1, -price)
        sell1 = max(sell1, buy1 + price)
        buy2 = max(buy2, sell1 - price)
        sell2 = max(sell2, buy2 + price)

    return sell2
```

---

#### Pattern 6D: Best Time to Buy and Sell Stock IV

**Problem:** LeetCode 188 - At most k transactions

```python
def maxProfit(k: int, prices: list[int]) -> int:
    if not prices:
        return 0

    n = len(prices)

    # If k >= n/2, unlimited transactions
    if k >= n // 2:
        return sum(max(0, prices[i] - prices[i-1]) for i in range(1, n))

    # dp[i][j] = max profit with at most i transactions on first j days
    dp = [[0] * n for _ in range(k + 1)]

    for i in range(1, k + 1):
        max_diff = -prices[0]
        for j in range(1, n):
            dp[i][j] = max(dp[i][j-1], prices[j] + max_diff)
            max_diff = max(max_diff, dp[i-1][j] - prices[j])

    return dp[k][n-1]
```

---

#### Pattern 6E: Best Time with Cooldown

**Problem:** LeetCode 309 - Must wait one day after selling

```python
def maxProfit(prices: list[int]) -> int:
    if not prices:
        return 0

    hold = -prices[0]
    sold = 0
    rest = 0

    for price in prices[1:]:
        prev_sold = sold
        sold = hold + price
        hold = max(hold, rest - price)
        rest = max(rest, prev_sold)

    return max(sold, rest)
```

---

### PATTERN 7: Interval DP

---

#### Pattern 7A: Burst Balloons

**Problem:** LeetCode 312 - Max coins from bursting balloons

```python
def maxCoins(nums: list[int]) -> int:
    # Add 1s at boundaries
    nums = [1] + nums + [1]
    n = len(nums)

    # dp[i][j] = max coins from bursting all balloons in (i, j)
    dp = [[0] * n for _ in range(n)]

    for length in range(2, n):  # length from 2 to n-1
        for i in range(n - length):
            j = i + length
            for k in range(i + 1, j):  # k is last balloon to burst
                dp[i][j] = max(dp[i][j],
                              dp[i][k] + dp[k][j] + nums[i] * nums[k] * nums[j])

    return dp[0][n-1]
```

---

#### Pattern 7B: Matrix Chain Multiplication

**Problem:** Minimum cost to multiply chain of matrices

```python
def matrixChainOrder(dims: list[int]) -> int:
    """
    dims[i] = rows of matrix i = cols of matrix i-1
    n matrices: dims has n+1 elements
    """
    n = len(dims) - 1
    dp = [[0] * n for _ in range(n)]

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            for k in range(i, j):
                cost = dp[i][k] + dp[k+1][j] + dims[i] * dims[k+1] * dims[j+1]
                dp[i][j] = min(dp[i][j], cost)

    return dp[0][n-1]
```

---

#### Pattern 7C: Palindrome Partitioning II

**Problem:** LeetCode 132 - Min cuts to partition into palindromes

```python
def minCut(s: str) -> int:
    n = len(s)

    # Precompute: is_pal[i][j] = is s[i:j+1] palindrome?
    is_pal = [[False] * n for _ in range(n)]
    for i in range(n - 1, -1, -1):
        for j in range(i, n):
            if s[i] == s[j] and (j - i < 2 or is_pal[i+1][j-1]):
                is_pal[i][j] = True

    # dp[i] = min cuts for s[0:i+1]
    dp = list(range(n))  # Worst case: all single chars

    for i in range(n):
        if is_pal[0][i]:
            dp[i] = 0
        else:
            for j in range(i):
                if is_pal[j+1][i]:
                    dp[i] = min(dp[i], dp[j] + 1)

    return dp[n-1]
```

---

### PATTERN 8: Counting DP

---

#### Pattern 8A: Climbing Stairs

**Problem:** LeetCode 70 - Ways to climb n stairs (1 or 2 steps)

```python
def climbStairs(n: int) -> int:
    if n <= 2:
        return n

    prev2, prev1 = 1, 2
    for _ in range(3, n + 1):
        prev2, prev1 = prev1, prev2 + prev1

    return prev1
```

---

#### Pattern 8B: Decode Ways

**Problem:** LeetCode 91 - Count ways to decode string

```python
def numDecodings(s: str) -> int:
    if not s or s[0] == '0':
        return 0

    n = len(s)
    dp = [0] * (n + 1)
    dp[0] = 1
    dp[1] = 1

    for i in range(2, n + 1):
        # Single digit
        if s[i-1] != '0':
            dp[i] += dp[i-1]

        # Two digits
        two_digit = int(s[i-2:i])
        if 10 <= two_digit <= 26:
            dp[i] += dp[i-2]

    return dp[n]
```

---

<a name="post-processing"></a>
## 5. Post-Processing Reference

| Problem Type | Answer Location | Notes |
|--------------|-----------------|-------|
| **Max/min ending at** | max(dp) | Like max subarray |
| **Max/min up to n** | dp[n] or dp[-1] | Like house robber |
| **Grid destination** | dp[m-1][n-1] | Bottom-right |
| **Two sequences** | dp[m][n] | Full lengths |
| **Interval** | dp[0][n-1] | Full range |
| **State machine** | max of final states | Like stock problems |

---

<a name="pitfalls"></a>
## 6. Common Pitfalls & Solutions

### Pitfall 1: Wrong Base Case

```python
# WRONG: dp[0] not properly initialized
dp = [0] * (n + 1)
for i in range(1, n + 1):
    dp[i] = dp[i-1] + 1  # dp[0] = 0, but should it be?
```

**Solution:** Carefully consider what dp[0] represents

---

### Pitfall 2: Off-by-One in Indices

```python
# WRONG: Accessing s[i] when dp is 1-indexed
for i in range(1, n + 1):
    if s[i] == 'x':  # Should be s[i-1]
```

**Solution:** Be consistent with 0-indexed vs 1-indexed

---

### Pitfall 3: Wrong Order in 0/1 Knapsack

```python
# WRONG: Forward iteration in 0/1 knapsack
for i in range(len(weights)):
    for w in range(weights[i], capacity + 1):  # Should be reversed!
        dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
```

**Solution:** Iterate backwards for 0/1 knapsack, forwards for unbounded

---

### Pitfall 4: Missing Memoization in Top-Down

```python
# WRONG: No memoization
def solve(i):
    if i <= 1:
        return i
    return solve(i-1) + solve(i-2)  # Exponential!
```

**Solution:**
```python
from functools import lru_cache

@lru_cache(maxsize=None)
def solve(i):
    if i <= 1:
        return i
    return solve(i-1) + solve(i-2)
```

---

### Pitfall 5: Integer Overflow

```python
# Can overflow in some languages
dp[i] = dp[i-1] + dp[i-2]
```

**Solution:** Use modulo if counting problems, or use Python (no overflow)

---

### Pitfall 6: Confusing Subsequence vs Substring

- **Subsequence:** Elements don't need to be contiguous
- **Substring:** Elements must be contiguous

---

<a name="recognition"></a>
## 7. Problem Recognition Framework

### Step 1: Is it DP?

**Indicators for DP:**
1. "Count the number of ways"
2. "Find the minimum/maximum"
3. "Is it possible to..."
4. Optimal solution involves optimal subproblems
5. Same subproblems solved repeatedly

**NOT DP if:**
- Need to find actual path (use backtracking)
- Greedy works (interval scheduling, some minimization)
- Graph problem (use BFS/DFS)

### Step 2: What Type of DP?

| Pattern | Clue |
|---------|------|
| Linear | Single array, sequential choices |
| Grid | 2D matrix, movement |
| Knapsack | Subset selection, capacity constraint |
| String | Two strings, matching, editing |
| Interval | Subarray operations, merge |
| State Machine | Multiple states, transitions |

### Step 3: Define the State

Ask yourself:
1. What information do I need to describe a subproblem?
2. What's the smallest valid subproblem?
3. How do larger subproblems depend on smaller ones?

### Decision Tree

```
              DP Problem
                  ↓
         ┌───────┴───────┐
      Single          Two/Multiple
      Sequence        Sequences
         ↓                ↓
    ┌────┴────┐      ┌────┴────┐
  Linear   Interval  LCS/Edit  Knapsack
    ↓         ↓        ↓         ↓
  O(n)    O(n²/n³)   O(mn)    O(nW)
```

---

<a name="checklist"></a>
## 8. Interview Preparation Checklist

### Before the Interview

**Master the fundamentals:**
- [ ] Can explain top-down vs bottom-up
- [ ] Know how to identify DP problems
- [ ] Can define state and transition for common patterns
- [ ] Understand space optimization techniques

**Know the patterns:**
- [ ] Linear DP (House Robber, LIS)
- [ ] Grid DP (Unique Paths, Min Path Sum)
- [ ] 0/1 Knapsack (Partition, Target Sum)
- [ ] Unbounded Knapsack (Coin Change)
- [ ] String DP (LCS, Edit Distance)
- [ ] State Machine (Stock problems)
- [ ] Interval DP (Matrix Chain, Burst Balloons)

**Common problems solved:**
- [ ] LC 53: Maximum Subarray
- [ ] LC 70: Climbing Stairs
- [ ] LC 198: House Robber
- [ ] LC 300: LIS
- [ ] LC 322: Coin Change
- [ ] LC 1143: LCS
- [ ] LC 72: Edit Distance
- [ ] LC 121-309: Stock problems

### During the Interview

**1. Clarify (30 seconds)**
- What exactly needs to be returned?
- Constraints on input?
- Edge cases?

**2. Identify pattern (1 minute)**
- What type of DP?
- What's the state?
- What's the transition?

**3. Explain approach (1 minute)**
- Define dp[i] clearly
- State the recurrence
- Identify base cases

**4. Code (3-4 minutes)**
- Write base cases
- Fill the table
- Return answer

**5. Optimize if needed (1 minute)**
- Can we reduce space?
- Is there a better approach?

---

## 9. Quick Reference Cards

### Linear DP
```python
dp = [0] * n
dp[0] = base
for i in range(1, n):
    dp[i] = f(dp[i-1], dp[i-2], ...)
return dp[-1]
```

### 2D/Grid DP
```python
dp = [[0] * n for _ in range(m)]
# Fill base cases
for i in range(1, m):
    for j in range(1, n):
        dp[i][j] = f(dp[i-1][j], dp[i][j-1], ...)
return dp[m-1][n-1]
```

### 0/1 Knapsack (Space Optimized)
```python
dp = [0] * (capacity + 1)
for item in items:
    for w in range(capacity, weight - 1, -1):  # Backwards!
        dp[w] = max(dp[w], dp[w - weight] + value)
```

### Unbounded Knapsack
```python
dp = [0] * (capacity + 1)
for w in range(1, capacity + 1):
    for item in items:
        if weight <= w:
            dp[w] = max(dp[w], dp[w - weight] + value)
```

---

## 10. Complexity Reference

| Pattern | Time | Space | Optimized Space |
|---------|------|-------|-----------------|
| Linear | O(n) or O(n²) | O(n) | O(1) |
| Grid | O(mn) | O(mn) | O(n) |
| 0/1 Knapsack | O(nW) | O(nW) | O(W) |
| LCS/Edit | O(mn) | O(mn) | O(n) |
| Interval | O(n²) or O(n³) | O(n²) | Usually not |
| Stock/State | O(nk) | O(nk) | O(k) |

---

## Final Thoughts

**Remember:**
1. DP is about **optimal substructure** and **overlapping subproblems**
2. Define the state clearly before coding
3. Space can often be optimized by looking at dependencies
4. Practice pattern recognition — most problems are variations of common patterns
5. Start with brute force recursion, then add memoization, then convert to tabulation

**When stuck:**
1. Draw out small examples
2. Ask: "What's the last decision I make?"
3. Try defining dp differently
4. Consider if a different pattern applies

---

## Appendix: Practice Problem Set

### Easy
- 70. Climbing Stairs
- 121. Best Time to Buy and Sell Stock
- 198. House Robber
- 303. Range Sum Query
- 746. Min Cost Climbing Stairs

### Medium
- 53. Maximum Subarray
- 62. Unique Paths
- 64. Minimum Path Sum
- 91. Decode Ways
- 139. Word Break
- 198. House Robber
- 213. House Robber II
- 300. Longest Increasing Subsequence
- 322. Coin Change
- 416. Partition Equal Subset Sum
- 518. Coin Change 2
- 1143. Longest Common Subsequence

### Hard
- 72. Edit Distance
- 123. Best Time to Buy and Sell Stock III
- 188. Best Time to Buy and Sell Stock IV
- 312. Burst Balloons
- 329. Longest Increasing Path in a Matrix

**Recommended Practice Order:**
1. Linear: 70, 198, 53, 300
2. Grid: 62, 64, 120
3. Knapsack: 416, 322, 518
4. Strings: 1143, 72
5. State Machine: 121, 122, 309
6. Interval: 516, 312

Good luck with your interview preparation!
