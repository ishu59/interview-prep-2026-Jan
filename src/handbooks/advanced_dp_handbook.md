# The Complete Advanced DP Handbook
> A template-based approach for mastering advanced dynamic programming in coding interviews

**Philosophy:** Advanced DP is not about learning new algorithms. It's about **recognizing state representations** that aren't immediately obvious — bitmasks for subsets, tree structures for hierarchies, and digit-by-digit construction for counting.

---

## Table of Contents
1. [Understanding the Core Philosophy](#core-philosophy)
2. [The Master Templates](#master-templates)
3. [Pattern Classification Guide](#pattern-guide)
4. [Complete Pattern Library](#patterns)
5. [Common Pitfalls & Solutions](#pitfalls)
6. [Problem Recognition Framework](#recognition)
7. [Interview Preparation Checklist](#checklist)

---

<a name="core-philosophy"></a>
## 1. Understanding the Core Philosophy

### First Principles

- **The Costume Closet (Bitmask DP)**: You have N costumes and need to try every possible outfit combination. A bitmask is a checklist where each bit says "wearing" or "not wearing." With 5 costumes, `10110` means "wearing costumes 2, 3, and 5." The DP state is which costumes you've already used.
- **The Tree Accountant (Tree DP)**: Each branch manager asks their sub-managers for reports, combines them with their own data, and passes the result up. The root gets the final answer. This is DP on a tree: solve leaves first, combine upward.
- **The Digit Locksmith (Digit DP)**: Building a number digit by digit, left to right, like dialing a combination lock. At each position, you choose a digit, and you track: "Am I still constrained by the upper bound?" (the `tight` flag) and "Have I started placing non-zero digits yet?" (the `started` flag).

### No-Jargon Translation

- **Bitmask**: an integer whose binary representation encodes a set -- each bit position represents "included" or "excluded"
- **Tight constraint**: whether the number you're building is still exactly at the upper limit -- once you go below, you're "free"
- **Rerooting**: computing the answer for every node as if it were the root, by cleverly adjusting the parent's answer instead of recomputing from scratch
- **State space**: all possible combinations of the DP variables -- what you're iterating over
- **Transition**: the formula for moving from one state to the next

### Mental Model

> "Advanced DP is regular DP with cleverer state representations: bitmasks encode subsets as numbers, tree DP passes answers upward like a corporate reporting chain, and digit DP constructs numbers one digit at a time like dialing a combination lock."

---

### When is DP "Advanced"?

Advanced DP problems have non-obvious state representations:

| Basic DP | Advanced DP |
|----------|-------------|
| State is index/position | State is a bitmask of used items |
| Linear/2D array | Tree structure |
| Simple transitions | Complex state machines |
| O(n²) or O(n³) | O(2^n × n) or O(n × digits) |

### The Five Advanced DP Categories

1. **Bitmask DP:** State represents a subset of items
2. **Tree DP:** DP on tree structures
3. **Digit DP:** Count numbers with digit constraints
4. **Probability DP:** Expected values and probabilities
5. **DP with Optimization:** Monotonic deque, convex hull trick

### Key Insight: State Design

The hardest part is **defining what dp[state] represents**:
- What information do I need to make future decisions?
- What's the minimum state to avoid recomputation?
- Can I encode complex states efficiently?

---

<a name="master-templates"></a>
## 2. The Master Templates

### Template A: Bitmask DP (Subset Selection)

```python
def bitmask_dp(n: int, items: list, constraint) -> int:
    """
    DP where state is which items have been selected.
    dp[mask] = best answer using items in mask
    """
    # dp[mask] = answer for subset represented by mask
    dp = [float('inf')] * (1 << n)  # or 0, -inf based on problem
    dp[0] = base_value

    for mask in range(1 << n):
        # Why `dp[mask] == float('inf'): continue`?
        # If dp[mask] is still infinity, this subset was never reached via any valid
        # transition — it's an unreachable state, so skip it to avoid propagating garbage.
        if dp[mask] == float('inf'):
            continue

        for i in range(n):
            # Why `mask & (1 << i)`?
            # Check if bit i is set in the mask, meaning item i has already been used.
            # Binary example: mask=0b1010, i=1 → 1010 & 0010 = 0010 (truthy) → item 1 is used.
            if mask & (1 << i):  # i already used
                continue

            # Can we add item i?
            if can_add(mask, i, constraint):
                # Why `mask | (1 << i)`?
                # Set bit i to 1, adding item i to the subset.
                # Binary example: mask=0b1010, i=0 → 1010 | 0001 = 1011 → item 0 now included.
                new_mask = mask | (1 << i)
                dp[new_mask] = min(dp[new_mask], dp[mask] + cost(i))

    # Why `(1 << n) - 1`?
    # This creates a bitmask with all n bits set to 1, meaning "all items selected."
    # Binary example: n=4 → (1<<4)-1 = 16-1 = 15 = 0b1111 → all 4 items included.
    return dp[(1 << n) - 1]
```

---

### Template B: Bitmask DP (Assignment Problem)

```python
def assignment_dp(n: int, cost: list[list[int]]) -> int:
    """
    Assign n workers to n jobs to minimize cost.
    dp[mask] = min cost to complete jobs in mask using first popcount(mask) workers
    """
    dp = [float('inf')] * (1 << n)
    dp[0] = 0

    for mask in range(1 << n):
        # Why `bin(mask).count('1')` gives the worker index?
        # We assign workers in order: worker 0 first, then worker 1, etc.
        # The number of 1-bits (jobs assigned so far) tells us which worker is next.
        # Example: mask=0b0101 → two 1s → workers 0,1 assigned → worker 2 is next.
        worker = bin(mask).count('1')  # Which worker we're assigning
        if worker >= n:
            continue

        for job in range(n):
            if mask & (1 << job):  # Job already assigned
                continue

            new_mask = mask | (1 << job)
            dp[new_mask] = min(dp[new_mask], dp[mask] + cost[worker][job])

    return dp[(1 << n) - 1]
```

---

### Template C: Tree DP (Bottom-Up)

```python
def tree_dp(root: TreeNode) -> int:
    """
    DP on tree, computing answers from leaves up.
    """
    result = 0

    def dfs(node) -> tuple:
        if not node:
            return (base_values)

        left_result = dfs(node.left)
        right_result = dfs(node.right)

        # Compute dp values for current node
        # Often: dp[node][state] based on children's dp values

        nonlocal result
        result = update(result, node, left_result, right_result)

        return (current_result)

    dfs(root)
    return result
```

---

### Template D: Tree DP (Rerooting)

```python
def reroot_dp(n: int, edges: list) -> list[int]:
    """
    Compute answer for each node as root.
    Two passes: down (from arbitrary root), up (reroot)
    """
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    dp = [0] * n  # Answer when subtree rooted at i
    ans = [0] * n  # Final answer when i is root

    # First DFS: compute dp[] with node 0 as root
    def dfs1(node, parent):
        for child in adj[node]:
            if child != parent:
                dfs1(child, node)
                dp[node] = combine(dp[node], dp[child])

    # Second DFS: reroot to compute ans[]
    def dfs2(node, parent, parent_contribution):
        ans[node] = combine(dp[node], parent_contribution)

        for child in adj[node]:
            if child != parent:
                # Contribution from all other children + parent
                child_contribution = remove(ans[node], dp[child])
                dfs2(child, node, child_contribution)

    dfs1(0, -1)
    dfs2(0, -1, 0)
    return ans
```

---

### Template E: Digit DP

```python
def digit_dp(num: str) -> int:
    """
    Count numbers up to `num` satisfying some property.
    State: (position, tight, property_state)
    """
    n = len(num)

    @lru_cache(maxsize=None)
    def dp(pos: int, tight: bool, state) -> int:
        if pos == n:
            # Why `is_valid(state)` at base case?
            # We've placed all digits and built a complete number. Now check if it
            # satisfies the property we're counting (e.g., all unique digits, sum constraint).
            return is_valid(state)

        # Why `int(num[pos]) if tight else 9`?
        # "tight" means we've matched the upper bound exactly so far.
        # If tight, we can only go up to the actual digit at this position.
        # Example: counting up to 347, at pos 0: if tight, limit=3 (digits 0-3);
        # if not tight (we already placed a smaller digit earlier), limit=9 (digits 0-9).
        limit = int(num[pos]) if tight else 9
        result = 0

        for digit in range(0, limit + 1):
            # Why `tight and (digit == limit)`?
            # Tight propagation: the next position stays tight ONLY if we're still
            # exactly matching the upper bound — i.e., we were tight AND chose the
            # maximum allowed digit. If we chose anything smaller, we're "free" forever.
            new_tight = tight and (digit == limit)
            new_state = update_state(state, digit)
            result += dp(pos + 1, new_tight, new_state)

        return result

    return dp(0, True, initial_state)
```

---

### Template F: Probability/Expected Value DP

```python
def expected_value_dp(states: int) -> float:
    """
    Compute expected value through state transitions.
    """
    # dp[state] = expected value/probability at state
    dp = [0.0] * states
    dp[initial] = 1.0  # or initial expected value

    for state in order_of_processing:
        for next_state, probability in transitions(state):
            dp[next_state] += dp[state] * probability

    return dp[final_state]
```

---

### Template G: DP with Monotonic Deque Optimization

```python
from collections import deque

def dp_monotonic_deque(nums: list[int], k: int) -> list[int]:
    """
    Optimize dp[i] = max(dp[j]) + cost for j in [i-k, i-1]
    """
    n = len(nums)
    dp = [0] * n
    dq = deque()  # Stores indices with decreasing dp values

    for i in range(n):
        # Remove elements outside window
        # Why `dq[0] < i - k`?
        # The front index fell out of the sliding window [i-k, i-1].
        # We can only extend from the last k elements, so expired indices must go.
        while dq and dq[0] < i - k:
            dq.popleft()

        # dp[i] uses max from window
        if dq:
            dp[i] = dp[dq[0]] + nums[i]
        else:
            dp[i] = nums[i]

        # Maintain decreasing order
        # Why `dp[dq[-1]] <= dp[i]`?
        # We maintain a decreasing invariant: the front always holds the best (max) value.
        # Any index at the back with a worse (<=) dp value than the current can never win —
        # the current index is both newer (stays in window longer) AND better, so remove them.
        while dq and dp[dq[-1]] <= dp[i]:
            dq.pop()
        dq.append(i)

    return dp
```

---

### Quick Decision Matrix

| Problem Type | Template | State Representation |
|--------------|----------|---------------------|
| Subset selection | A | Bitmask of used items |
| Assignment | B | Bitmask of assigned jobs |
| Tree answers | C | Node → subtree answer |
| All-roots tree | D | Rerooting technique |
| Digit constraints | E | (position, tight, property) |
| Probability | F | State → probability |
| Range optimization | G | Monotonic deque |

---

<a name="pattern-guide"></a>
## 3. Pattern Classification Guide

### Category 1: Bitmask DP
- Small n (≤ 20)
- Need to track which items used
- Subset problems
- **Templates A, B**

### Category 2: Tree DP
- DP on tree structure
- Answers depend on subtrees
- **Templates C, D**

### Category 3: Digit DP
- Count numbers in range
- Digit-based constraints
- **Template E**

### Category 4: Probability DP
- Expected values
- Random processes
- **Template F**

### Category 5: DP Optimizations
- Reduce from O(n²) to O(n log n) or O(n)
- Monotonic deque, segment tree, convex hull
- **Template G**

---

<a name="patterns"></a>
## 4. Complete Pattern Library

### PATTERN 1: Bitmask DP

---

#### Pattern 1A: Traveling Salesman Problem (TSP)

**Problem:** Visit all cities exactly once, return to start, minimize distance

```python
def tsp(dist: list[list[int]]) -> int:
    """
    dp[mask][i] = min cost to visit cities in mask, ending at city i
    """
    n = len(dist)
    INF = float('inf')

    # dp[mask][i] = min cost to reach city i with visited cities = mask
    dp = [[INF] * n for _ in range(1 << n)]
    # Why `dp[1][0] = 0`?
    # Start at city 0 with only city 0 visited. mask=1 means 0b0001 — only bit 0 is set.
    # Cost to be at city 0 having visited only city 0 is 0 (no travel yet).
    dp[1][0] = 0  # Start at city 0

    for mask in range(1 << n):
        for last in range(n):
            if dp[mask][last] == INF:
                continue
            # Why `not (mask & (1 << last))`?
            # Skip if city `last` is NOT actually in the visited set.
            # We're iterating all (mask, last) pairs, but only valid states have `last` in the mask.
            # Binary example: mask=0b1010, last=0 → 1010 & 0001 = 0 → city 0 not visited, skip.
            if not (mask & (1 << last)):
                continue

            for next_city in range(n):
                if mask & (1 << next_city):  # Already visited
                    continue

                new_mask = mask | (1 << next_city)
                dp[new_mask][next_city] = min(
                    dp[new_mask][next_city],
                    dp[mask][last] + dist[last][next_city]
                )

    # Return to city 0
    # Why `(1 << n) - 1`?
    # All cities visited: n bits all set to 1.
    # Binary example: n=4 → (1<<4)-1 = 0b1111 → cities 0,1,2,3 all visited.
    full_mask = (1 << n) - 1
    return min(dp[full_mask][i] + dist[i][0] for i in range(n))
```

---

#### Pattern 1B: Minimum XOR Sum of Two Arrays

**Problem:** LeetCode 1879 - Pair elements to minimize total XOR

```python
def minimumXORSum(nums1: list[int], nums2: list[int]) -> int:
    n = len(nums1)
    dp = [float('inf')] * (1 << n)
    dp[0] = 0

    for mask in range(1 << n):
        # Why `bin(mask).count('1')` tells us which element of nums1 to pair next?
        # We pair nums1 elements in order (0, 1, 2, ...). The mask tracks which nums2
        # elements are already paired. The number of 1-bits = number of pairings done
        # = index of the next nums1 element. Example: mask=0b101 → 2 bits set → nums1[2] is next.
        i = bin(mask).count('1')  # How many from nums1 we've paired
        if i >= n:
            continue

        for j in range(n):
            # Why `mask & (1 << j): continue`?
            # Skip if nums2[j] is already paired with a previous nums1 element.
            # Binary example: mask=0b101, j=2 → 101 & 100 = 100 (truthy) → nums2[2] already used.
            if mask & (1 << j):
                continue
            new_mask = mask | (1 << j)
            dp[new_mask] = min(dp[new_mask], dp[mask] + (nums1[i] ^ nums2[j]))

    return dp[(1 << n) - 1]
```

---

#### Pattern 1C: Parallel Courses II

**Problem:** LeetCode 1494 - Minimum semesters with prerequisites

```python
def minNumberOfSemesters(n: int, relations: list, k: int) -> int:
    # Compute prerequisite mask for each course
    prereq = [0] * n
    for prev, next_c in relations:
        prereq[next_c - 1] |= (1 << (prev - 1))

    # dp[mask] = min semesters to complete courses in mask
    dp = [float('inf')] * (1 << n)
    dp[0] = 0

    for mask in range(1 << n):
        if dp[mask] == float('inf'):
            continue

        # Find available courses (prerequisites satisfied)
        available = 0
        for i in range(n):
            # Why `(prereq[i] & mask) == prereq[i]`?
            # ALL prerequisites of course i must be completed (in the mask).
            # ANDing prereq[i] with mask keeps only the completed prereqs.
            # If the result equals prereq[i], then every prerequisite bit is satisfied.
            # Example: prereq[i]=0b0110, mask=0b1110 → 0110 & 1110 = 0110 == 0110 ✓ all done.
            if not (mask & (1 << i)) and (prereq[i] & mask) == prereq[i]:
                available |= (1 << i)

        # Try all subsets of available courses of size <= k
        subset = available
        while subset:
            if bin(subset).count('1') <= k:
                new_mask = mask | subset
                dp[new_mask] = min(dp[new_mask], dp[mask] + 1)
            # Why `(subset - 1) & available`?
            # Standard bitmask subset enumeration trick: iterates through ALL subsets of
            # `available` in decreasing order. Subtracting 1 flips the lowest set bit and
            # bits below it; ANDing with `available` keeps only valid bits. Terminates at 0.
            subset = (subset - 1) & available

    return dp[(1 << n) - 1]
```

---

#### Pattern 1D: Shortest Path Visiting All Nodes

**Problem:** LeetCode 847 - Visit all nodes in undirected graph

```python
from collections import deque

def shortestPathLength(graph: list[list[int]]) -> int:
    n = len(graph)
    # Why `(1 << n) - 1`?
    # All n nodes visited: a bitmask with all n bits set to 1.
    # Binary example: n=4 → (1<<4)-1 = 0b1111 → nodes 0,1,2,3 all visited.
    target = (1 << n) - 1

    # BFS: state = (mask, current_node)
    queue = deque()
    visited = set()

    for i in range(n):
        state = (1 << i, i)
        queue.append((state, 0))
        visited.add(state)

    while queue:
        (mask, node), dist = queue.popleft()

        # Why `mask == target`?
        # All bits are set — every node has been visited. We found the shortest path.
        if mask == target:
            return dist

        for neighbor in graph[node]:
            # Why `mask | (1 << neighbor)`?
            # Mark neighbor as visited by setting its bit.
            # Binary example: mask=0b1001, neighbor=1 → 1001 | 0010 = 1011 → node 1 now visited.
            new_mask = mask | (1 << neighbor)
            state = (new_mask, neighbor)

            if state not in visited:
                visited.add(state)
                queue.append((state, dist + 1))

    return -1
```

---

### PATTERN 2: Tree DP

---

#### Pattern 2A: Binary Tree Maximum Path Sum

**Problem:** LeetCode 124 - Max path sum (any start/end)

```python
def maxPathSum(root: TreeNode) -> int:
    max_sum = float('-inf')

    def dfs(node) -> int:
        """Returns max path sum starting at node going down."""
        nonlocal max_sum
        if not node:
            return 0

        # Why `max(dfs(node.left), 0)`?
        # If the child's best path sum is negative, it's better to NOT include it at all.
        # This is the "optional extension" insight: you'd rather start fresh (contribute 0)
        # than take a losing path that drags down the total.
        left = max(dfs(node.left), 0)
        right = max(dfs(node.right), 0)

        # Why `left + node.val + right` for the global update?
        # This considers a path that passes THROUGH this node using BOTH children.
        # This is the best "arch-shaped" path at this node — the global answer candidate.
        max_sum = max(max_sum, left + node.val + right)

        # Why `node.val + max(left, right)` for the RETURN value?
        # When returning to the parent, we can only go UP through ONE side — a path
        # can't fork. So we pick the better child. The global update above is the only
        # place where both sides are combined.
        return node.val + max(left, right)

    dfs(root)
    return max_sum
```

---

#### Pattern 2B: House Robber III (Tree)

**Problem:** LeetCode 337 - Rob houses in tree

```python
def rob(root: TreeNode) -> int:
    def dfs(node) -> tuple[int, int]:
        """Returns (max if rob this, max if skip this)"""
        if not node:
            return (0, 0)

        left = dfs(node.left)
        right = dfs(node.right)

        # Why `node.val + left[1] + right[1]`?
        # Rob this node → adjacent children CANNOT be robbed (house robber constraint).
        # So we take each child's "skip" value (index [1]). We're forced into their skip path.
        rob_this = node.val + left[1] + right[1]

        # Why `max(left) + max(right)`?
        # Skip this node → we're FREE to rob or skip each child independently.
        # No adjacency constraint with grandchildren, so take the best of (rob, skip) for each.
        skip_this = max(left) + max(right)

        return (rob_this, skip_this)

    return max(dfs(root))
```

---

#### Pattern 2C: Tree Diameter

**Problem:** LeetCode 1245 - Longest path in tree

```python
def treeDiameter(edges: list[list[int]]) -> int:
    from collections import defaultdict

    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    diameter = 0

    def dfs(node: int, parent: int) -> int:
        nonlocal diameter
        max1 = max2 = 0

        for child in adj[node]:
            if child == parent:
                continue
            depth = dfs(child, node) + 1

            if depth > max1:
                max2 = max1
                max1 = depth
            elif depth > max2:
                max2 = depth

        diameter = max(diameter, max1 + max2)
        return max1

    dfs(0, -1)
    return diameter
```

---

#### Pattern 2D: Sum of Distances in Tree (Rerooting)

**Problem:** LeetCode 834 - Sum of distances to all other nodes

```python
def sumOfDistancesInTree(n: int, edges: list[list[int]]) -> list[int]:
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    count = [1] * n  # Subtree size
    ans = [0] * n

    # First DFS: compute ans[0] and count[]
    def dfs1(node: int, parent: int):
        for child in adj[node]:
            if child != parent:
                dfs1(child, node)
                count[node] += count[child]
                ans[node] += ans[child] + count[child]

    # Second DFS: reroot
    def dfs2(node: int, parent: int):
        for child in adj[node]:
            if child != parent:
                # Why `ans[node] - count[child] + (n - count[child])`?
                # THE KEY REROOTING FORMULA: when we move the root from `node` to `child`:
                #   - child's subtree has count[child] nodes, each gets 1 CLOSER → subtract count[child]
                #   - the rest of the tree has (n - count[child]) nodes, each gets 1 FARTHER → add (n - count[child])
                # Net change = -count[child] + (n - count[child]) = n - 2*count[child]
                ans[child] = ans[node] - count[child] + (n - count[child])
                dfs2(child, node)

    dfs1(0, -1)
    dfs2(0, -1)
    return ans
```

---

### PATTERN 3: Digit DP

---

#### Pattern 3A: Count Numbers with Unique Digits

**Problem:** LeetCode 357

```python
def countNumbersWithUniqueDigits(n: int) -> int:
    if n == 0:
        return 1

    # For n digits: 9 * 9 * 8 * 7 * ...
    result = 10  # n=1 case
    unique_digits = 9
    available = 9

    for i in range(2, n + 1):
        unique_digits *= available
        result += unique_digits
        available -= 1

    return result
```

---

#### Pattern 3B: Numbers At Most N Given Digit Set

**Problem:** LeetCode 902

```python
def atMostNGivenDigitSet(digits: list[str], n: int) -> int:
    s = str(n)
    k = len(s)
    d = len(digits)

    # Count numbers with fewer digits
    result = sum(d ** i for i in range(1, k))

    # Count k-digit numbers <= n
    for i, char in enumerate(s):
        # Count numbers with smaller digit at position i
        smaller = sum(1 for digit in digits if digit < char)
        result += smaller * (d ** (k - i - 1))

        # If current digit not in set, stop
        if char not in digits:
            break

        # If we used all positions with matching digits
        if i == k - 1:
            result += 1

    return result
```

---

#### Pattern 3C: Count Special Integers

**Problem:** LeetCode 2376 - Count integers with all distinct digits

```python
def countSpecialNumbers(n: int) -> int:
    s = str(n)
    length = len(s)

    @lru_cache(maxsize=None)
    def dp(pos: int, mask: int, tight: bool, started: bool) -> int:
        if pos == length:
            # Why `1 if started else 0`?
            # Only count this number if we actually placed at least one non-zero digit.
            # A number like "000" (started=False) is just 0, not a valid "special integer."
            return 1 if started else 0

        limit = int(s[pos]) if tight else 9
        result = 0

        for digit in range(0, limit + 1):
            # Why `not started and digit == 0`?
            # Handle leading zeros: placing a 0 before any non-zero digit doesn't "start"
            # the number. 007 is not a 3-digit number — it's just 7. Leading zeros don't
            # consume a digit slot, so we don't update the mask.
            if not started and digit == 0:
                # Haven't started, skip leading zero
                result += dp(pos + 1, mask, False, False)
            # Why `not (mask & (1 << digit))`?
            # Check that this digit hasn't been used yet (all digits must be unique).
            # Binary example: mask=0b0100, digit=2 → 0100 & 0100 = 0100 (truthy) → digit 2 already used, skip.
            elif not (mask & (1 << digit)):
                # Digit not used yet
                new_mask = mask | (1 << digit)
                new_tight = tight and (digit == limit)
                result += dp(pos + 1, new_mask, new_tight, True)

        return result

    return dp(0, 0, True, False)
```

---

#### Pattern 3D: Numbers With Repeated Digits

**Problem:** LeetCode 1012 - Count numbers with at least one repeated digit

```python
def numDupDigitsAtMostN(n: int) -> int:
    # Count numbers WITH all unique digits, subtract from n
    s = str(n)
    length = len(s)

    # Count numbers with fewer digits (all unique)
    unique = 0
    for i in range(1, length):
        # i-digit numbers with unique digits
        unique += 9 * perm(9, i - 1)

    # Count length-digit numbers <= n with unique digits
    seen = set()
    for i, char in enumerate(s):
        digit = int(char)

        # Count numbers with smaller digit at position i
        for d in range(0 if i else 1, digit):
            if d not in seen:
                unique += perm(9 - i, length - i - 1)

        if digit in seen:
            break
        seen.add(digit)

        if i == length - 1:
            unique += 1

    return n - unique

def perm(n, k):
    result = 1
    for i in range(k):
        result *= (n - i)
    return result
```

---

### PATTERN 4: Probability DP

---

#### Pattern 4A: Knight Probability in Chessboard

**Problem:** LeetCode 688

```python
def knightProbability(n: int, k: int, row: int, column: int) -> float:
    moves = [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]

    # dp[r][c] = probability of being at (r, c)
    dp = [[0.0] * n for _ in range(n)]
    dp[row][column] = 1.0

    for _ in range(k):
        new_dp = [[0.0] * n for _ in range(n)]

        for r in range(n):
            for c in range(n):
                # Why `dp[r][c] > 0`?
                # Skip cells with zero probability — the knight can't be here, so no
                # transitions to compute. This is a pure optimization to avoid useless work.
                if dp[r][c] > 0:
                    for dr, dc in moves:
                        nr, nc = r + dr, c + dc
                        # Why `0 <= nr < n and 0 <= nc < n`?
                        # The knight must stay on the board. If it jumps off, that
                        # probability is simply lost (contributes to "fell off" outcomes).
                        if 0 <= nr < n and 0 <= nc < n:
                            # Why `dp[r][c] / 8`?
                            # A knight has exactly 8 possible moves, each equally likely.
                            # So each destination gets 1/8 of the current cell's probability.
                            new_dp[nr][nc] += dp[r][c] / 8

        dp = new_dp

    return sum(sum(row) for row in dp)
```

---

#### Pattern 4B: Soup Servings

**Problem:** LeetCode 808

```python
def soupServings(n: int) -> float:
    if n > 4800:  # Probability approaches 1
        return 1.0

    n = (n + 24) // 25  # Scale down

    @lru_cache(maxsize=None)
    def dp(a: int, b: int) -> float:
        if a <= 0 and b <= 0:
            return 0.5
        if a <= 0:
            return 1.0
        if b <= 0:
            return 0.0

        return 0.25 * (
            dp(a - 4, b) +
            dp(a - 3, b - 1) +
            dp(a - 2, b - 2) +
            dp(a - 1, b - 3)
        )

    return dp(n, n)
```

---

#### Pattern 4C: New 21 Game

**Problem:** LeetCode 837

```python
def new21Game(n: int, k: int, maxPts: int) -> float:
    if k == 0 or n >= k + maxPts:
        return 1.0

    # dp[i] = probability of getting exactly i points
    dp = [0.0] * (n + 1)
    dp[0] = 1.0
    window_sum = 1.0
    result = 0.0

    for i in range(1, n + 1):
        # Why `window_sum / maxPts`?
        # Each of the maxPts cards (1, 2, ..., maxPts) is equally likely to be drawn.
        # dp[i] is the sum of probabilities of all states that can reach i, divided by maxPts.
        dp[i] = window_sum / maxPts

        # Why `i < k`?
        # Still drawing: we haven't reached k points yet, so this state can lead to
        # future draws. Add its probability to the window for computing future dp values.
        if i < k:
            window_sum += dp[i]
        # Why `else` (i.e., i >= k)?
        # Stopped drawing: we've reached k or more points, so this is a terminal state.
        # Since i <= n, this is a winning terminal state — add to result.
        else:
            result += dp[i]

        # Why `i >= maxPts`?
        # Slide the window: dp[i - maxPts] can no longer contribute to future states
        # because the max card value is maxPts. Remove its contribution from the window.
        if i >= maxPts:
            window_sum -= dp[i - maxPts]

    return result
```

---

### PATTERN 5: DP Optimizations

---

#### Pattern 5A: Constrained Subsequence Sum (Monotonic Deque)

**Problem:** LeetCode 1425

```python
from collections import deque

def constrainedSubsetSum(nums: list[int], k: int) -> int:
    n = len(nums)
    dp = [0] * n
    dq = deque()  # Store indices with decreasing dp values

    for i in range(n):
        # Remove elements outside window
        while dq and dq[0] < i - k:
            dq.popleft()

        # dp[i] = nums[i] + max(0, dp[j]) for j in [i-k, i-1]
        dp[i] = nums[i]
        if dq:
            # Why `max(0, dp[dq[0]])`?
            # Optionally extend from the best previous element in the window.
            # max with 0 means "start fresh" if all previous dp values are negative —
            # it's better to begin a new subsequence here than extend a losing one.
            dp[i] += max(0, dp[dq[0]])

        # Maintain decreasing order
        while dq and dp[dq[-1]] <= dp[i]:
            dq.pop()
        dq.append(i)

    return max(dp)
```

---

#### Pattern 5B: Jump Game VI

**Problem:** LeetCode 1696

```python
from collections import deque

def maxResult(nums: list[int], k: int) -> int:
    n = len(nums)
    dp = [0] * n
    dp[0] = nums[0]
    dq = deque([0])

    for i in range(1, n):
        # Remove indices outside window
        while dq and dq[0] < i - k:
            dq.popleft()

        dp[i] = dp[dq[0]] + nums[i]

        # Maintain decreasing dp values
        while dq and dp[dq[-1]] <= dp[i]:
            dq.pop()
        dq.append(i)

    return dp[n - 1]
```

---

#### Pattern 5C: Max Sum of Rectangle No Larger Than K

**Problem:** LeetCode 363 - Kadane + Binary Search

```python
from sortedcontainers import SortedList

def maxSumSubmatrix(matrix: list[list[int]], k: int) -> int:
    rows, cols = len(matrix), len(matrix[0])
    result = float('-inf')

    for left in range(cols):
        row_sum = [0] * rows

        for right in range(left, cols):
            for r in range(rows):
                row_sum[r] += matrix[r][right]

            # Find max subarray sum <= k using prefix sum + binary search
            sorted_prefix = SortedList([0])
            prefix = 0

            for sum_val in row_sum:
                prefix += sum_val
                # Find smallest prefix >= prefix - k
                idx = sorted_prefix.bisect_left(prefix - k)
                if idx < len(sorted_prefix):
                    result = max(result, prefix - sorted_prefix[idx])
                sorted_prefix.add(prefix)

    return result
```

---

<a name="pitfalls"></a>
## 5. Common Pitfalls & Solutions

### Pitfall 1: Bitmask Size

```python
# For n > 20, 2^n becomes too large
# Check constraints: n <= 20 for bitmask DP
```

### Pitfall 2: Digit DP Leading Zeros

```python
# Must handle leading zeros separately
# "007" is not a valid 3-digit number
```

### Pitfall 3: Tree DP Direction

```python
# Make sure you're computing in the right order
# Bottom-up: process children before parent
# Rerooting: two passes required
```

### Pitfall 4: Probability Precision

```python
# Use float carefully, consider precision issues
# For large values, probabilities may approach 0 or 1
```

---

<a name="recognition"></a>
## 6. Problem Recognition Framework

### When to Use Each Pattern

| Clue | Pattern |
|------|---------|
| n ≤ 20, "all subsets" | Bitmask DP |
| Tree structure | Tree DP |
| "Count numbers in range" | Digit DP |
| "Expected value", "probability" | Probability DP |
| "Sliding window max/min in DP" | Monotonic Deque |

### Complexity Guide

| Pattern | Time | Space |
|---------|------|-------|
| Bitmask DP | O(2^n × n) | O(2^n) |
| Tree DP | O(n) | O(n) |
| Digit DP | O(digits × states) | O(digits × states) |
| Monotonic Deque | O(n) | O(k) |

---

<a name="checklist"></a>
## 7. Interview Preparation Checklist

### Before the Interview

**Master the fundamentals:**
- [ ] Can implement bitmask DP
- [ ] Understand subset enumeration
- [ ] Can do tree DP (bottom-up)
- [ ] Know digit DP template
- [ ] Understand monotonic deque optimization

**Know the patterns:**
- [ ] TSP-like problems
- [ ] Assignment problems
- [ ] Tree path problems
- [ ] Counting digit problems
- [ ] Probability/expected value

**Common problems solved:**
- [ ] LC 847: Shortest Path Visiting All Nodes
- [ ] LC 1879: Minimum XOR Sum
- [ ] LC 124: Binary Tree Max Path Sum
- [ ] LC 834: Sum of Distances in Tree
- [ ] LC 1425: Constrained Subsequence Sum

### During the Interview

**1. Identify the pattern (30 seconds)**
- Check n value for bitmask feasibility
- Look for tree structure
- Check for digit/counting constraints

**2. Define state carefully (1 minute)**
- What information encodes a subproblem?
- What transitions are possible?

**3. Code (4-5 minutes)**
- Set up state representation
- Handle base cases
- Implement transitions

**4. Test (1-2 minutes)**
- Small examples
- Edge cases
- Verify complexity

---

## 8. Quick Reference Cards

### Bitmask DP
```python
for mask in range(1 << n):
    for i in range(n):
        if not (mask & (1 << i)):  # i not in mask
            new_mask = mask | (1 << i)
            dp[new_mask] = min(dp[new_mask], dp[mask] + cost[i])
```

### Digit DP
```python
def dp(pos, tight, state):
    if pos == n: return is_valid(state)
    limit = int(s[pos]) if tight else 9
    for d in range(limit + 1):
        result += dp(pos+1, tight and d==limit, update(state,d))
```

### Tree DP
```python
def dfs(node):
    for child in children[node]:
        dfs(child)
        dp[node] = combine(dp[node], dp[child])
```

---

## 9. Complexity Reference

| Pattern | Time | Space |
|---------|------|-------|
| Bitmask DP | O(2^n × n) | O(2^n) |
| Tree DP | O(n) | O(n) |
| Rerooting | O(n) | O(n) |
| Digit DP | O(D × S) | O(D × S) |
| Probability DP | O(states × transitions) | O(states) |

---

## Final Thoughts

**Remember:**
1. Bitmask DP only for n ≤ 20
2. Tree DP processes bottom-up or with rerooting
3. Digit DP uses (position, tight, state)
4. Probability DP: sum of probabilities = 1
5. Optimization reduces complexity significantly

**When stuck:**
1. Draw the state transition diagram
2. Start with brute force, then optimize
3. Check if problem fits known pattern
4. Consider if state can be compressed

---

## Appendix: Practice Problem Set

### Bitmask DP
- 847. Shortest Path Visiting All Nodes
- 1125. Smallest Sufficient Team
- 1494. Parallel Courses II
- 1879. Minimum XOR Sum of Two Arrays
- 943. Find the Shortest Superstring

### Tree DP
- 124. Binary Tree Maximum Path Sum
- 337. House Robber III
- 543. Diameter of Binary Tree
- 834. Sum of Distances in Tree
- 968. Binary Tree Cameras

### Digit DP
- 233. Number of Digit One
- 357. Count Numbers with Unique Digits
- 902. Numbers At Most N Given Digit Set
- 1012. Numbers With Repeated Digits
- 2376. Count Special Integers

### Probability DP
- 688. Knight Probability in Chessboard
- 808. Soup Servings
- 837. New 21 Game

### DP Optimization
- 1425. Constrained Subsequence Sum
- 1696. Jump Game VI
- 363. Max Sum of Rectangle No Larger Than K

**Recommended Practice Order:**
1. Bitmask: 847 → 1879
2. Tree: 337 → 124 → 834
3. Digit: 357 → 902 → 2376
4. Optimization: 1696 → 1425

---

## Appendix: Conditional Quick Reference

### Bitmask Operations Quick Reference

| Operation | Code | Purpose | Binary Example (n=4) |
|-----------|------|---------|----------------------|
| Check if bit i is set | `mask & (1 << i)` | Is item i in the subset? | `mask=0b1010, i=1 → 1010 & 0010 = 0010` (truthy: item 1 is in set) |
| Set bit i | `mask \| (1 << i)` | Add item i to the subset | `mask=0b1010, i=0 → 1010 \| 0001 = 1011` (item 0 added) |
| All items selected | `(1 << n) - 1` | Bitmask with all n bits set | `n=4 → 10000 - 1 = 0b1111` (all 4 items) |
| Popcount (count 1s) | `bin(mask).count('1')` | How many items are in the subset | `mask=0b1010 → 2 bits set → 2 items selected` |
| Enumerate subsets of mask | `sub = mask; while sub: ...; sub = (sub-1) & mask` | Iterate all subsets of a bitmask | `mask=0b110 → subsets: 110, 100, 010` (then 0 terminates) |
| Clear bit i | `mask & ~(1 << i)` | Remove item i from subset | `mask=0b1010, i=3 → 1010 & 0111 = 0010` (item 3 removed) |
| Toggle bit i | `mask ^ (1 << i)` | Flip item i in/out of subset | `mask=0b1010, i=1 → 1010 ^ 0010 = 1000` (item 1 toggled off) |

### Key DP Conditionals

| Conditional | Pattern | Why It Works |
|-------------|---------|-------------|
| `int(num[pos]) if tight else 9` | Digit DP | **Tight constraint:** if still matching the upper bound exactly, we can only use digits up to the actual digit at this position. Once we go below, we're "free" to use 0-9. Example: for 347, at pos 0 tight means limit=3; not tight means limit=9. |
| `tight and (digit == limit)` | Digit DP | **Tight propagation:** the next position stays tight ONLY if we chose the exact upper-bound digit. Choosing anything smaller means all future positions are unconstrained. |
| `not started and digit == 0` | Digit DP | **Leading zeros:** a zero before any non-zero digit doesn't count as a placed digit. 007 is just 7, not a 3-digit number. Don't update the digit-tracking mask for leading zeros. |
| `1 if started else 0` | Digit DP | **Base case validity:** only count the number if at least one non-zero digit was placed. |
| `max(dfs(child), 0)` | Tree DP (Path Sum) | **Optional extension:** if a child's path sum is negative, don't include it. Starting fresh (0) is better than a losing path. |
| `left + val + right` vs `val + max(left, right)` | Tree DP (Path Sum) | **Global vs return:** the global update uses BOTH children (path through node). The return value uses only ONE child (path can't fork going upward). |
| `node.val + left[1] + right[1]` | Tree DP (House Robber) | **Rob this node:** must skip both children (adjacency constraint), so take their "skip" values `[1]`. |
| `max(left) + max(right)` | Tree DP (House Robber) | **Skip this node:** free to rob or skip each child independently — take the better option for each. |
| `ans[node] - count[child] + (n - count[child])` | Tree DP (Rerooting) | **Reroot formula:** moving root to child makes child's subtree (count[child] nodes) 1 closer and the rest (n - count[child] nodes) 1 farther. |
| `dp[r][c] / 8` | Probability DP | **Uniform distribution:** knight has 8 equally likely moves, each gets 1/8 of current probability. |
| `0 <= nr < n` | Probability DP | **Board bounds:** if knight jumps off board, that probability is lost (not added anywhere). |
| `i < k` (add to window) vs `i >= k` (add to result) | Probability DP (21 Game) | **Drawing vs terminal:** below k points, still drawing (contribute to future states). At k+, stopped drawing (terminal state, count toward result). |
| `max(0, dp[dq[0]])` | DP + Monotonic Deque | **Optional start:** extend from the best previous value, or start a new subsequence (0) if all previous values are negative. |

Good luck with your interview preparation!
