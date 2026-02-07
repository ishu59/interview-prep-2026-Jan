# The Complete Union-Find (Disjoint Set) Handbook
> A template-based approach for mastering Union-Find in coding interviews

**Philosophy:** Union-Find is not just a data structure. It's about **efficiently tracking and merging connected components** — answering "are these two elements in the same group?" and "merge these two groups" in nearly constant time.

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

### What is Union-Find?

Union-Find (also called Disjoint Set Union or DSU) is a data structure that tracks a set of elements partitioned into disjoint (non-overlapping) subsets. It supports two operations:

1. **Find(x):** Which set does element x belong to? (Returns the representative/root)
2. **Union(x, y):** Merge the sets containing x and y

### The Mental Model

Think of it as a **forest of trees**:
- Each set is a tree
- Each element points to its parent
- The root is the **representative** of the set
- Two elements are in the same set if they have the same root

```
Initial: Each element is its own tree
    0    1    2    3    4

After Union(0, 1):
    0    2    3    4
    |
    1

After Union(2, 3):
    0    2    4
    |    |
    1    3

After Union(0, 2):
      0      4
     /|
    1 2
      |
      3

Find(3) = 0 (follow parent pointers to root)
Find(1) = 0
Therefore, 1 and 3 are in the same set!
```

### Why Union-Find Over DFS/BFS?

| Approach | Time per Query | Total for Q Queries |
|----------|---------------|---------------------|
| DFS/BFS | O(n) | O(Q × n) |
| Union-Find | O(α(n)) ≈ O(1) | O(Q × α(n)) ≈ O(Q) |

**Key Insight:** When you need to repeatedly check connectivity or merge groups, Union-Find is dramatically faster than running DFS/BFS each time.

### The Two Optimizations

#### 1. Path Compression
When finding the root, make every node on the path point directly to the root.

```
Before Find(4):     After Find(4):
    0                   0
    |                 / | \
    1               1   2  4
    |                   |
    2                   3
    |
    3
    |
    4
```

#### 2. Union by Rank/Size
When merging two trees, attach the smaller tree under the larger one.

```
Without rank:       With rank:
    0                   0
    |                 / | \
    1               1   2  5
    |                   |
    2                   3
    |                   |
    3                   4
    |
    4
    |
    5
```

**Combined:** These optimizations give O(α(n)) time per operation, where α is the inverse Ackermann function (effectively constant, ≤ 4 for any practical input size).

---

<a name="master-templates"></a>
## 2. The Master Templates

### Template A: Basic Union-Find

```python
class UnionFind:
    """
    Basic Union-Find with path compression and union by rank.
    """
    def __init__(self, n: int):
        self.parent = list(range(n))  # Each element is its own parent
        self.rank = [0] * n            # Rank for union by rank

    def find(self, x: int) -> int:
        """Find root with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """
        Union two sets. Returns True if they were in different sets.
        """
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False  # Already in same set

        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        return True

    def connected(self, x: int, y: int) -> bool:
        """Check if two elements are in the same set."""
        return self.find(x) == self.find(y)
```

---

### Template B: Union-Find with Size Tracking

```python
class UnionFindWithSize:
    """
    Union-Find that tracks size of each component.
    Useful for problems asking about component sizes.
    """
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n  # Size of each component
        self.count = n       # Number of components

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False

        # Union by size (attach smaller to larger)
        if self.size[root_x] < self.size[root_y]:
            root_x, root_y = root_y, root_x

        self.parent[root_y] = root_x
        self.size[root_x] += self.size[root_y]
        self.count -= 1

        return True

    def get_size(self, x: int) -> int:
        """Get size of the component containing x."""
        return self.size[self.find(x)]

    def get_count(self) -> int:
        """Get number of disjoint components."""
        return self.count
```

---

### Template C: Union-Find with Weighted Edges

```python
class WeightedUnionFind:
    """
    Union-Find where edges have weights.
    Maintains relative weights between elements.
    weight[x] = weight of edge from x to parent[x]
    """
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.weight = [0.0] * n  # Weight from node to parent

    def find(self, x: int) -> tuple:
        """
        Returns (root, weight_to_root).
        """
        if self.parent[x] != x:
            root, weight_to_root = self.find(self.parent[x])
            self.parent[x] = root
            self.weight[x] += weight_to_root  # Path compression with weight update
        return self.parent[x], self.weight[x]

    def union(self, x: int, y: int, w: float) -> bool:
        """
        Union x and y with weight w, meaning weight[x]/weight[y] = w.
        """
        root_x, weight_x = self.find(x)
        root_y, weight_y = self.find(y)

        if root_x == root_y:
            return False

        # Attach root_x to root_y
        self.parent[root_x] = root_y
        # weight_x + new_weight = w + weight_y
        # new_weight = w + weight_y - weight_x
        self.weight[root_x] = w + weight_y - weight_x

        return True

    def query(self, x: int, y: int) -> float:
        """
        Returns weight[x]/weight[y] if in same set, else -1.
        """
        root_x, weight_x = self.find(x)
        root_y, weight_y = self.find(y)

        if root_x != root_y:
            return -1.0

        return weight_x - weight_y  # For division: weight_x / weight_y
```

---

### Template D: Union-Find for 2D Grid

```python
class GridUnionFind:
    """
    Union-Find for 2D grid problems.
    Converts (row, col) to linear index.
    """
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        n = rows * cols
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = 0  # Active components

    def _index(self, r: int, c: int) -> int:
        """Convert (row, col) to linear index."""
        return r * self.cols + c

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, r1: int, c1: int, r2: int, c2: int) -> bool:
        """Union two cells."""
        x = self._index(r1, c1)
        y = self._index(r2, c2)

        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False

        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        self.count -= 1
        return True

    def add(self, r: int, c: int):
        """Mark cell as active (for dynamic problems)."""
        self.count += 1

    def connected(self, r1: int, c1: int, r2: int, c2: int) -> bool:
        x = self._index(r1, c1)
        y = self._index(r2, c2)
        return self.find(x) == self.find(y)
```

---

### Template E: Union-Find with Dictionary (Dynamic Nodes)

```python
class DynamicUnionFind:
    """
    Union-Find with dynamic node creation.
    Useful when nodes aren't numbered 0 to n-1.
    """
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x) -> any:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x

        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y) -> bool:
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False

        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        return True

    def connected(self, x, y) -> bool:
        return self.find(x) == self.find(y)
```

---

### Quick Decision Matrix

| Problem Type | Template | Key Feature |
|--------------|----------|-------------|
| Basic connectivity | A | Rank optimization |
| Count components | B | Size tracking |
| Component sizes | B | Size array |
| Weighted relations | C | Weight tracking |
| Grid problems | D | 2D to 1D mapping |
| Dynamic nodes | E | Dictionary-based |

---

<a name="pattern-guide"></a>
## 3. Pattern Classification Guide

### Category 1: Connected Components
- Count number of groups
- Check if two elements are connected
- **Template A or B**

### Category 2: Dynamic Connectivity
- Process edges one by one
- Answer connectivity queries over time
- **Template A or B**

### Category 3: Component Size Queries
- Track size of each component
- Find largest component
- **Template B**

### Category 4: Grid Connectivity
- Islands, regions in 2D grid
- Often combined with BFS/DFS
- **Template D**

### Category 5: Weighted/Ratio Problems
- Equation evaluation
- Relative relationships
- **Template C**

### Category 6: Cycle Detection
- Detect if adding edge creates cycle
- MST (Kruskal's algorithm)
- **Template A**

---

<a name="patterns"></a>
## 4. Complete Pattern Library

### PATTERN 1: Connected Components

---

#### Pattern 1A: Number of Connected Components

**Problem:** LeetCode 323 - Count connected components

```python
def countComponents(n: int, edges: list[list[int]]) -> int:
    uf = UnionFind(n)

    for u, v in edges:
        uf.union(u, v)

    # Count unique roots
    return len(set(uf.find(i) for i in range(n)))
```

**Using size-tracking template:**
```python
def countComponents_v2(n: int, edges: list[list[int]]) -> int:
    uf = UnionFindWithSize(n)

    for u, v in edges:
        uf.union(u, v)

    return uf.get_count()
```

---

#### Pattern 1B: Friend Circles / Number of Provinces

**Problem:** LeetCode 547 - Count friend groups from adjacency matrix

```python
def findCircleNum(isConnected: list[list[int]]) -> int:
    n = len(isConnected)
    uf = UnionFindWithSize(n)

    for i in range(n):
        for j in range(i + 1, n):
            if isConnected[i][j] == 1:
                uf.union(i, j)

    return uf.get_count()
```

---

#### Pattern 1C: Earliest Time When Everyone Becomes Friends

**Problem:** LeetCode 1101 - Find when all friends are connected

```python
def earliestAcq(logs: list[list[int]], n: int) -> int:
    # Sort by timestamp
    logs.sort(key=lambda x: x[0])

    uf = UnionFindWithSize(n)

    for timestamp, x, y in logs:
        uf.union(x, y)
        if uf.get_count() == 1:
            return timestamp

    return -1
```

---

### PATTERN 2: Number of Islands Variants

---

#### Pattern 2A: Number of Islands II (Dynamic)

**Problem:** LeetCode 305 - Count islands after each addLand operation

```python
def numIslands2(m: int, n: int, positions: list[list[int]]) -> list[int]:
    uf = GridUnionFind(m, n)
    grid = [[0] * n for _ in range(m)]
    result = []
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    for r, c in positions:
        if grid[r][c] == 1:  # Already land
            result.append(uf.count)
            continue

        grid[r][c] = 1
        uf.add(r, c)  # New island

        # Try to connect with adjacent land
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n and grid[nr][nc] == 1:
                uf.union(r, c, nr, nc)

        result.append(uf.count)

    return result
```

---

#### Pattern 2B: Making A Large Island

**Problem:** LeetCode 827 - Flip one 0 to maximize island size

```python
def largestIsland(grid: list[list[int]]) -> int:
    n = len(grid)
    uf = UnionFindWithSize(n * n)

    def index(r, c):
        return r * n + c

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    # Union all existing land cells
    for r in range(n):
        for c in range(n):
            if grid[r][c] == 1:
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] == 1:
                        uf.union(index(r, c), index(nr, nc))

    # Try flipping each 0
    max_size = 0

    for r in range(n):
        for c in range(n):
            if grid[r][c] == 0:
                # Find unique adjacent islands
                adjacent_roots = set()
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] == 1:
                        adjacent_roots.add(uf.find(index(nr, nc)))

                # Size = 1 (flipped cell) + sum of adjacent island sizes
                size = 1 + sum(uf.get_size(root) for root in adjacent_roots)
                max_size = max(max_size, size)

    # Edge case: entire grid is 1s
    if max_size == 0:
        max_size = n * n

    return max_size
```

---

### PATTERN 3: Graph Validity / Cycle Detection

---

#### Pattern 3A: Graph Valid Tree

**Problem:** LeetCode 261 - Check if edges form a valid tree

```python
def validTree(n: int, edges: list[list[int]]) -> bool:
    # Tree has exactly n-1 edges and is connected
    if len(edges) != n - 1:
        return False

    uf = UnionFind(n)

    for u, v in edges:
        if not uf.union(u, v):  # Cycle detected
            return False

    return True
```

---

#### Pattern 3B: Redundant Connection

**Problem:** LeetCode 684 - Find edge that creates cycle

```python
def findRedundantConnection(edges: list[list[int]]) -> list[int]:
    n = len(edges)
    uf = UnionFind(n + 1)  # 1-indexed

    for u, v in edges:
        if not uf.union(u, v):
            return [u, v]

    return []
```

---

#### Pattern 3C: Redundant Connection II (Directed)

**Problem:** LeetCode 685 - Find redundant directed edge

```python
def findRedundantDirectedConnection(edges: list[list[int]]) -> list[int]:
    n = len(edges)
    parent = [0] * (n + 1)
    candidate1 = candidate2 = None

    # Find node with two parents
    for u, v in edges:
        if parent[v] != 0:
            candidate1 = [parent[v], v]
            candidate2 = [u, v]
        else:
            parent[v] = u

    uf = UnionFind(n + 1)

    for u, v in edges:
        if [u, v] == candidate2:
            continue  # Skip second candidate

        if not uf.union(u, v):
            # Found cycle
            if candidate1:
                return candidate1
            return [u, v]

    return candidate2
```

---

### PATTERN 4: Accounts Merge / Grouping

---

#### Pattern 4A: Accounts Merge

**Problem:** LeetCode 721 - Merge accounts by common emails

```python
def accountsMerge(accounts: list[list[str]]) -> list[list[str]]:
    email_to_id = {}
    email_to_name = {}
    uf = DynamicUnionFind()

    # Map emails to account IDs
    for i, account in enumerate(accounts):
        name = account[0]
        for email in account[1:]:
            if email not in email_to_id:
                email_to_id[email] = i
            email_to_name[email] = name
            uf.union(email, account[1])  # Union all emails in account

    # Group emails by root
    groups = defaultdict(list)
    for email in email_to_id:
        root = uf.find(email)
        groups[root].append(email)

    # Build result
    result = []
    for root, emails in groups.items():
        name = email_to_name[root]
        result.append([name] + sorted(emails))

    return result
```

---

#### Pattern 4B: Sentence Similarity II

**Problem:** LeetCode 737 - Check similarity with transitive pairs

```python
def areSentencesSimilarTwo(
    sentence1: list[str], sentence2: list[str], similarPairs: list[list[str]]
) -> bool:
    if len(sentence1) != len(sentence2):
        return False

    uf = DynamicUnionFind()

    for w1, w2 in similarPairs:
        uf.union(w1, w2)

    for w1, w2 in zip(sentence1, sentence2):
        if w1 != w2 and not uf.connected(w1, w2):
            return False

    return True
```

---

### PATTERN 5: Weighted Union-Find (Ratios/Equations)

---

#### Pattern 5A: Evaluate Division

**Problem:** LeetCode 399 - Evaluate a/b given equations a/b = value

```python
def calcEquation(
    equations: list[list[str]],
    values: list[float],
    queries: list[list[str]]
) -> list[float]:
    uf = {}

    def find(x):
        if x not in uf:
            uf[x] = (x, 1.0)  # (parent, weight)
        if uf[x][0] != x:
            parent, weight = find(uf[x][0])
            uf[x] = (parent, uf[x][1] * weight)
        return uf[x]

    def union(x, y, value):
        px, wx = find(x)
        py, wy = find(y)
        if px != py:
            # px to py: wx * w = value * wy
            # w = value * wy / wx
            uf[px] = (py, value * wy / wx)

    # Build union-find
    for (a, b), value in zip(equations, values):
        union(a, b, value)

    # Answer queries
    result = []
    for a, b in queries:
        if a not in uf or b not in uf:
            result.append(-1.0)
        else:
            pa, wa = find(a)
            pb, wb = find(b)
            if pa != pb:
                result.append(-1.0)
            else:
                result.append(wa / wb)

    return result
```

---

### PATTERN 6: MST and Connectivity with Edges

---

#### Pattern 6A: Minimum Spanning Tree (Kruskal's)

**Problem:** Build MST using Union-Find

```python
def kruskal(n: int, edges: list[list[int]]) -> int:
    """
    edges[i] = [u, v, weight]
    Returns total weight of MST.
    """
    # Sort edges by weight
    edges.sort(key=lambda x: x[2])

    uf = UnionFind(n)
    mst_weight = 0
    edges_used = 0

    for u, v, weight in edges:
        if uf.union(u, v):
            mst_weight += weight
            edges_used += 1
            if edges_used == n - 1:
                break

    return mst_weight if edges_used == n - 1 else -1
```

---

#### Pattern 6B: Min Cost to Connect All Points

**Problem:** LeetCode 1584 - Connect all points with minimum cost

```python
def minCostConnectPoints(points: list[list[int]]) -> int:
    n = len(points)
    edges = []

    # Generate all edges
    for i in range(n):
        for j in range(i + 1, n):
            dist = abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1])
            edges.append((dist, i, j))

    # Sort by distance
    edges.sort()

    uf = UnionFind(n)
    total_cost = 0
    edges_used = 0

    for dist, u, v in edges:
        if uf.union(u, v):
            total_cost += dist
            edges_used += 1
            if edges_used == n - 1:
                break

    return total_cost
```

---

#### Pattern 6C: Connecting Cities With Minimum Cost

**Problem:** LeetCode 1135 - Minimum cost to connect all cities

```python
def minimumCost(n: int, connections: list[list[int]]) -> int:
    # Sort by cost
    connections.sort(key=lambda x: x[2])

    uf = UnionFindWithSize(n + 1)  # 1-indexed

    total_cost = 0
    edges_used = 0

    for u, v, cost in connections:
        if uf.union(u, v):
            total_cost += cost
            edges_used += 1
            if edges_used == n - 1:
                return total_cost

    return -1  # Not all cities connected
```

---

### PATTERN 7: Special Applications

---

#### Pattern 7A: Longest Consecutive Sequence

**Problem:** LeetCode 128 - Find longest consecutive sequence

```python
def longestConsecutive(nums: list[int]) -> int:
    if not nums:
        return 0

    uf = DynamicUnionFind()
    num_set = set(nums)

    for num in nums:
        uf.find(num)  # Initialize
        if num + 1 in num_set:
            uf.union(num, num + 1)

    # Count sizes
    groups = defaultdict(int)
    for num in nums:
        root = uf.find(num)
        groups[root] += 1

    return max(groups.values())
```

**Alternative (simpler, without UF):**
```python
def longestConsecutive_simple(nums: list[int]) -> int:
    num_set = set(nums)
    max_length = 0

    for num in num_set:
        if num - 1 not in num_set:  # Start of sequence
            length = 1
            while num + length in num_set:
                length += 1
            max_length = max(max_length, length)

    return max_length
```

---

#### Pattern 7B: Satisfiability of Equality Equations

**Problem:** LeetCode 990 - Check if equations are satisfiable

```python
def equationsPossible(equations: list[str]) -> bool:
    uf = UnionFind(26)

    # Process equalities first
    for eq in equations:
        if eq[1] == '=':
            x = ord(eq[0]) - ord('a')
            y = ord(eq[3]) - ord('a')
            uf.union(x, y)

    # Check inequalities
    for eq in equations:
        if eq[1] == '!':
            x = ord(eq[0]) - ord('a')
            y = ord(eq[3]) - ord('a')
            if uf.connected(x, y):
                return False

    return True
```

---

#### Pattern 7C: Regions Cut By Slashes

**Problem:** LeetCode 959 - Count regions in grid with slashes

```python
def regionsBySlashes(grid: list[str]) -> int:
    n = len(grid)
    # Each cell is split into 4 triangles: top, right, bottom, left (0,1,2,3)
    uf = UnionFind(n * n * 4)

    def index(r, c, k):
        return (r * n + c) * 4 + k

    for r in range(n):
        for c in range(n):
            # Connect to adjacent cells
            # Right neighbor
            if c + 1 < n:
                uf.union(index(r, c, 1), index(r, c + 1, 3))
            # Bottom neighbor
            if r + 1 < n:
                uf.union(index(r, c, 2), index(r + 1, c, 0))

            # Connect within cell based on character
            if grid[r][c] == '/':
                uf.union(index(r, c, 0), index(r, c, 3))
                uf.union(index(r, c, 1), index(r, c, 2))
            elif grid[r][c] == '\\':
                uf.union(index(r, c, 0), index(r, c, 1))
                uf.union(index(r, c, 2), index(r, c, 3))
            else:  # space
                uf.union(index(r, c, 0), index(r, c, 1))
                uf.union(index(r, c, 1), index(r, c, 2))
                uf.union(index(r, c, 2), index(r, c, 3))

    return len(set(uf.find(i) for i in range(n * n * 4)))
```

---

<a name="post-processing"></a>
## 5. Post-Processing Reference

| Problem Type | Return Value | Notes |
|--------------|--------------|-------|
| **Component count** | Number of unique roots | Use set or count variable |
| **Connected check** | Boolean | Compare roots |
| **Component sizes** | Size array/list | Track in size array |
| **Cycle detection** | Boolean | Union returns False |
| **Weighted query** | Float | Compute from weights |

---

<a name="pitfalls"></a>
## 6. Common Pitfalls & Solutions

### Pitfall 1: Forgetting Path Compression

```python
# WRONG: No path compression, O(n) per find
def find(x):
    while parent[x] != x:
        x = parent[x]
    return x
```

**Solution:**
```python
def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])  # Path compression
    return parent[x]
```

---

### Pitfall 2: Union Without Finding Roots

```python
# WRONG: Connecting nodes, not roots
def union(x, y):
    parent[x] = y  # Should be parent[root_x] = root_y
```

**Solution:**
```python
def union(x, y):
    root_x = find(x)
    root_y = find(y)
    if root_x != root_y:
        parent[root_x] = root_y
```

---

### Pitfall 3: 1-Indexed vs 0-Indexed

```python
# WRONG: Edges are 1-indexed but array is 0-indexed
uf = UnionFind(n)
for u, v in edges:  # u, v are 1 to n
    uf.union(u, v)  # Index out of bounds!
```

**Solution:**
```python
uf = UnionFind(n + 1)  # Extra space for 1-indexing
# Or subtract 1:
uf.union(u - 1, v - 1)
```

---

### Pitfall 4: Not Handling Disconnected Components

```python
# WRONG: Assumes all nodes connected
components = 1
for u, v in edges:
    uf.union(u, v)
```

**Solution:** Track count explicitly or count unique roots

---

### Pitfall 5: Modifying During Iteration

```python
# WRONG: find() modifies parent during iteration
for i in range(n):
    if parent[i] == i:  # This changes as find() runs
        count += 1
```

**Solution:**
```python
roots = set(find(i) for i in range(n))
count = len(roots)
```

---

### Pitfall 6: Weighted UF Weight Calculation Errors

**Problem:** Getting the weight formula wrong in weighted union-find

**Solution:** Draw it out:
```
If x has weight wx to root_x
And y has weight wy to root_y
And we want x/y = value

After union: root_x points to root_y with weight w
x's weight to root_y = wx + w = value * wy
Therefore: w = value * wy - wx (for division)
          w = value + wy - wx (for subtraction)
```

---

<a name="recognition"></a>
## 7. Problem Recognition Framework

### Step 1: Is Union-Find Applicable?

**Good indicators:**
1. "Connected" or "connectivity" mentioned
2. Grouping elements together
3. "Same group/set/component"
4. Dynamic edges (adding edges over time)
5. Need to answer many connectivity queries

**Red flags (might not be UF):**
1. Need shortest path (use BFS/Dijkstra)
2. Need actual path (use DFS/BFS)
3. Directed graph with complex dependencies (use DFS)
4. Need to split groups (UF can't undo)

### Step 2: What Additional Features Needed?

| Feature | Template |
|---------|----------|
| Just connectivity | A (basic) |
| Component count | B (with count) |
| Component sizes | B (with size) |
| Ratios/equations | C (weighted) |
| 2D grid | D (grid) |
| Non-integer nodes | E (dynamic) |

### Step 3: Can We Use Simpler Approach?

For some problems, DFS/BFS might be simpler:
- If edges are processed once (not dynamically)
- If we need actual paths
- If graph is small

### Decision Tree

```
              Need repeated connectivity queries?
                          ↓
                    ┌─────┴─────┐
                   Yes          No
                    ↓            ↓
              Union-Find    DFS/BFS might
                    ↓        be simpler
              ┌─────┴─────┐
         Need sizes?    Need weights?
              ↓              ↓
        Template B      Template C
```

---

<a name="checklist"></a>
## 8. Interview Preparation Checklist

### Before the Interview

**Master the fundamentals:**
- [ ] Can write basic Union-Find from memory
- [ ] Understand path compression
- [ ] Understand union by rank/size
- [ ] Know time complexity (O(α(n)) ≈ O(1))

**Practice pattern recognition:**
- [ ] Can identify Union-Find problems
- [ ] Know when UF is better than DFS/BFS
- [ ] Can choose right template variant

**Know the patterns:**
- [ ] Connected components
- [ ] Cycle detection
- [ ] Dynamic connectivity
- [ ] Weighted equations
- [ ] MST (Kruskal's)

**Common problems solved:**
- [ ] LC 200: Number of Islands (compare with DFS)
- [ ] LC 323: Number of Connected Components
- [ ] LC 684: Redundant Connection
- [ ] LC 721: Accounts Merge
- [ ] LC 399: Evaluate Division
- [ ] LC 1584: Min Cost to Connect Points

### During the Interview

**1. Clarify (30 seconds)**
- How are nodes numbered?
- Dynamic or static edges?
- Need sizes or just connectivity?

**2. Identify pattern (30 seconds)**
- Is this a grouping problem?
- Do I need to track sizes?
- Is there a weighted relationship?

**3. Code (3-4 minutes)**
- Write Union-Find class
- Implement find with path compression
- Implement union with rank
- Process edges/queries

**4. Test (1-2 minutes)**
- Single node
- Already connected
- All separate
- Cycle case

**5. Analyze (30 seconds)**
- Time: O(α(n)) per operation
- Space: O(n)

---

## 9. Quick Reference Cards

### Basic Union-Find
```python
class UF:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py: return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True
```

### With Size Tracking
```python
# In union:
self.size[root_x] += self.size[root_y]
self.count -= 1
```

---

## 10. Complexity Reference

| Operation | Time | Notes |
|-----------|------|-------|
| find | O(α(n)) | With path compression |
| union | O(α(n)) | With rank/size |
| Build from n edges | O(n × α(n)) | ≈ O(n) |

α(n) = inverse Ackermann function, practically ≤ 4 for any n

| Comparison | Union-Find | DFS/BFS |
|------------|-----------|---------|
| Q connectivity queries | O(Q × α(n)) | O(Q × n) |
| Single traversal | Same | Simpler to code |
| Need actual path | Can't | Can |

---

## Final Thoughts

**Remember:**
1. Union-Find is for **dynamic connectivity** problems
2. Always use **path compression** and **union by rank**
3. Time is **essentially O(1)** per operation
4. Can't handle **deletions** (only unions)
5. Perfect for **Kruskal's MST** algorithm

**When stuck:**
1. Ask: "Am I repeatedly checking if things are connected?"
2. Consider: "Are edges added dynamically?"
3. Think: "Do I need actual paths or just connectivity?"
4. Remember: Union-Find can't undo unions

---

## Appendix: Practice Problem Set

### Easy
- 547. Number of Provinces
- 990. Satisfiability of Equality Equations

### Medium
- 128. Longest Consecutive Sequence
- 200. Number of Islands (compare approaches)
- 261. Graph Valid Tree
- 323. Number of Connected Components
- 399. Evaluate Division
- 684. Redundant Connection
- 721. Accounts Merge
- 737. Sentence Similarity II
- 959. Regions Cut By Slashes
- 1101. The Earliest Moment When Everyone Become Friends

### Hard
- 305. Number of Islands II
- 685. Redundant Connection II
- 803. Bricks Falling When Hit
- 827. Making A Large Island
- 1584. Min Cost to Connect All Points

**Recommended Practice Order:**
1. Start with LC 547, 323 (basic connectivity)
2. Practice LC 684 (cycle detection)
3. Do LC 721 (grouping with strings)
4. Master LC 399 (weighted UF)
5. Try LC 305 (dynamic connectivity)

Good luck with your interview preparation!
