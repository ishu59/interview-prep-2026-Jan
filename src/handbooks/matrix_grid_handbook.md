# The Complete Matrix & Grid Handbook
> A template-based approach for mastering matrix and grid problems in coding interviews

**Philosophy:** Matrix/grid problems are not a separate category -- they're graph, DP, and simulation problems wearing a 2D costume. The key insight is that a grid is just a graph where each cell has up to 4 neighbors.

---

## Table of Contents
1. [Understanding the Core Philosophy](#core-philosophy)
2. [The 4 Master Templates](#master-templates)
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

- **The Chess Board**: Every cell has an address (row, col) and neighbors (up, down, left, right). Moving through a grid is walking on a chess board -- you just need to know the rules of movement.
- **The Flood Fill (Paint Bucket)**: Click on a pixel and all connected same-color pixels change. This is BFS/DFS on a grid. "Connected" means sharing an edge (4-directional) or corner (8-directional).

### No-Jargon Translation

- **Grid/Matrix**: a 2D array accessed by (row, col) -- row is which horizontal line, col is which vertical position
- **4-directional neighbors**: up, down, left, right -- cells sharing an edge
- **8-directional neighbors**: also includes diagonals -- cells sharing an edge or corner
- **In-bounds check**: verifying (row, col) is inside the grid before accessing it
- **Visited set**: tracking which cells you've already processed to avoid infinite loops

### Mental Model

> "A grid is a city map -- each intersection is a cell, streets connect to 4 neighbors, and solving grid problems is just navigating the city with different objectives (shortest path, flood zone, counting neighborhoods)."

---

### The Key Insight: Grid as Graph

Every grid problem is secretly a graph problem. Each cell `(r, c)` is a node, and edges connect it to its neighbors.

```
Grid:                    Implicit Graph:
+---+---+---+           (0,0) -- (0,1) -- (0,2)
| 1 | 1 | 0 |             |        |        |
+---+---+---+           (1,0) -- (1,1) -- (1,2)
| 1 | 1 | 0 |             |        |        |
+---+---+---+           (2,0) -- (2,1) -- (2,2)
| 0 | 0 | 1 |
+---+---+---+
```

This means every technique you know for graphs (BFS, DFS, Dijkstra) works on grids -- you just need to translate "neighbors" into direction offsets.

### The Directions Array: Your Most Important Tool

```python
# 4-directional (most common)
directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
#             right    left     down     up

# 8-directional (includes diagonals)
directions = [(0, 1), (0, -1), (1, 0), (-1, 0),
              (1, 1), (1, -1), (-1, 1), (-1, -1)]
```

### The Universal Neighbor Loop

```python
rows, cols = len(grid), len(grid[0])
for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
    nr, nc = r + dr, c + dc
    # Why chained comparison `0 <= nr < rows`?
    # Python allows this syntax as shorthand for `0 <= nr and nr < rows`.
    # It checks BOTH lower bound (not negative) and upper bound (within grid).
    # Why `<` and not `<=`? Because rows/cols are LENGTHS, and valid indices
    # go from 0 to length-1. Index == length is already out of bounds.
    if 0 <= nr < rows and 0 <= nc < cols:
        # (nr, nc) is a valid neighbor -- process it
        pass
```

This pattern appears in almost every grid problem. Memorize it.

### Visual Understanding: Coordinate System

```
       col 0   col 1   col 2
        ↓       ↓       ↓
row 0 → (0,0)  (0,1)  (0,2)
row 1 → (1,0)  (1,1)  (1,2)
row 2 → (2,0)  (2,1)  (2,2)

Neighbors of (1,1):
  Up:    (0,1)  →  (r-1, c)
  Down:  (2,1)  →  (r+1, c)
  Left:  (1,0)  →  (r, c-1)
  Right: (1,2)  →  (r, c+1)
```

### Grid vs Graph: Key Differences

| Aspect | General Graph | Grid |
|--------|--------------|------|
| Neighbors | Adjacency list | Direction offsets |
| Visited | Set of node IDs | Set of (r,c) or modify grid |
| Bounds | N/A | Must check 0 <= r < rows, 0 <= c < cols |
| Edge weight | Explicit | Usually uniform (1) |
| Node count | V | rows * cols |
| Edge count | E | ~4 * rows * cols |

---

<a name="master-templates"></a>
## 2. The 4 Master Templates

### Template Decision Matrix

| Situation | Template | Data Structure | Time |
|-----------|----------|---------------|------|
| Shortest path / multi-source / level-by-level | Grid BFS | deque | O(R*C) |
| Flood fill / connected components / reachability | Grid DFS | recursion/stack | O(R*C) |
| Spiral / diagonal / zigzag traversal | Traversal Patterns | indices | O(R*C) |
| Rotate / transpose / reflect in place | In-place Transform | swaps | O(R*C) |

### Template A: Grid BFS (Shortest Path / Multi-Source)

```python
from collections import deque

def grid_bfs(grid: list[list[int]], starts: list[tuple[int, int]]) -> int:
    """
    BFS on a grid from one or more starting cells.
    Returns shortest distance or processes cells level by level.
    """
    rows, cols = len(grid), len(grid[0])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    visited = set()
    queue = deque()

    # Initialize: add all starting cells
    for r, c in starts:
        visited.add((r, c))
        queue.append((r, c, 0))  # (row, col, distance)

    while queue:
        r, c, dist = queue.popleft()

        # Process cell (r, c) at distance dist
        # Check if target reached, update result, etc.

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            # Why check ALL THREE conditions (bounds, visited, wall)?
            # 1. Bounds: accessing grid[-1][0] would wrap or crash
            # 2. Not visited: prevents revisiting and infinite loops
            # 3. Not a wall: respects the grid's traversal rules
            # Order matters for short-circuit: bounds check FIRST prevents
            # IndexError, then visited check avoids redundant work.
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                if grid[nr][nc] != WALL:  # Check cell is valid
                    # Mark visited NOW, not when popping. If we wait until
                    # popping, the same cell gets added to the queue by
                    # multiple neighbors, wasting time and memory.
                    visited.add((nr, nc))
                    queue.append((nr, nc, dist + 1))

    return -1  # Target not reached
```

**When to use:** Shortest path on grid, rotting oranges, 01 matrix, any "minimum steps" problem.

### Template B: Grid DFS (Flood Fill / Connected Components)

```python
def grid_dfs(grid: list[list[int]], r: int, c: int, visited: set) -> int:
    """
    DFS on a grid from a single starting cell.
    Returns size of connected component or modifies grid in place.
    """
    rows, cols = len(grid), len(grid[0])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    # Base cases: Why two separate checks?
    # First: boundary check -- DFS may step outside the grid since we
    # recurse with r+dr BEFORE checking. Must reject out-of-bounds.
    if r < 0 or r >= rows or c < 0 or c >= cols:
        return 0
    # Second: skip if already visited OR cell is water (0).
    # Why check BOTH? visited prevents infinite loops (A visits B,
    # B visits A); grid[r][c]==0 stops at water boundaries.
    if (r, c) in visited or grid[r][c] == 0:
        return 0

    visited.add((r, c))
    area = 1  # Count this cell

    for dr, dc in directions:
        area += grid_dfs(grid, r + dr, c + dc, visited)

    return area
```

**When to use:** Number of islands, max area of island, flood fill, surrounded regions.

### Template C: Matrix Traversal Patterns (Spiral / Diagonal / Zigzag)

```python
def spiral_order(matrix: list[list[int]]) -> list[int]:
    """
    Traverse matrix in spiral order using boundary pointers.
    """
    if not matrix:
        return []

    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1

    while top <= bottom and left <= right:
        # Traverse right across top row
        for c in range(left, right + 1):
            result.append(matrix[top][c])
        top += 1

        # Traverse down right column
        for r in range(top, bottom + 1):
            result.append(matrix[r][right])
        right -= 1

        # Traverse left across bottom row
        if top <= bottom:
            for c in range(right, left - 1, -1):
                result.append(matrix[bottom][c])
            bottom -= 1

        # Traverse up left column
        if left <= right:
            for r in range(bottom, top - 1, -1):
                result.append(matrix[r][left])
            left += 1

    return result
```

**When to use:** Spiral matrix, diagonal traversal, zigzag order.

### Template D: In-Place Matrix Operations (Rotation / Transformation)

```python
def rotate_matrix(matrix: list[list[int]]) -> None:
    """
    Rotate matrix 90 degrees clockwise in place.
    Step 1: Transpose (swap rows and columns)
    Step 2: Reverse each row
    """
    n = len(matrix)

    # Step 1: Transpose
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

    # Step 2: Reverse each row
    for i in range(n):
        matrix[i].reverse()
```

**When to use:** Rotate image, set matrix zeroes, game of life.

---

<a name="pattern-guide"></a>
## 3. Pattern Classification Guide

### Category 1: Island Problems (Connected Components)
Grid cells form "islands" of connected values. Count them, measure them, classify them.
- **Template:** DFS or BFS
- **Key:** Mark visited cells, iterate over all cells
- **Problems:** LC 200, 695, 463, 694

### Category 2: Matrix BFS / Shortest Path
Find minimum steps or propagate from sources level by level.
- **Template:** Grid BFS (often multi-source)
- **Key:** All sources start in queue simultaneously
- **Problems:** LC 994, 1091, 542

### Category 3: Matrix DFS / Flood Fill
Explore connected regions, mark or transform cells based on reachability.
- **Template:** Grid DFS
- **Key:** Start from boundary or specific cells, propagate inward
- **Problems:** LC 733, 130, 417

### Category 4: Matrix Rotation / Transformation
Rearrange elements following geometric patterns.
- **Template:** In-place operations with index math
- **Key:** Decompose into transpose + reverse, or use boundary pointers
- **Problems:** LC 48, 54, 59

### Category 5: Matrix DP
Optimal path or counting problems where each cell depends on previous cells.
- **Template:** Standard DP with 2D table
- **Key:** Fill order (top-left to bottom-right usually)
- **Problems:** LC 62, 63, 64, 221

### Category 6: Search in Sorted Matrix
Matrix has sorted properties -- exploit them for efficient search.
- **Template:** Binary search or staircase search
- **Key:** Identify the sorted invariant
- **Problems:** LC 74, 240

### Category 7: Simulation / State Encoding
Simulate step-by-step processes on the grid, often requiring careful state management.
- **Template:** In-place with state encoding
- **Key:** Encode old and new state in same cell
- **Problems:** LC 289, 36, 73

---

<a name="patterns"></a>
## 4. Complete Pattern Library

---

### PATTERN 1: Island Problems (Connected Components on Grid)

---

#### Pattern 1A: Number of Islands

**Problem:** LeetCode 200 - Given an m x n 2D binary grid where '1' represents land and '0' represents water, count the number of islands. An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically.

**Example:**
```
Input:
  grid = [
    ["1","1","0","0","0"],
    ["1","1","0","0","0"],
    ["0","0","1","0","0"],
    ["0","0","0","1","1"]
  ]
Output: 3

Visual:
  1 1 0 0 0      A A . . .
  1 1 0 0 0  →   A A . . .    (3 islands: A, B, C)
  0 0 1 0 0      . . B . .
  0 0 0 1 1      . . . C C
```

**Key Insight:** Each unvisited '1' is the start of a new island. DFS/BFS from it to mark all connected '1's as visited. Count how many times you start a new search.

**Visual Trace:**
```
Step 1: Find (0,0)='1', start DFS → marks island A
  V V 0 0 0      V = visited
  V V 0 0 0      count = 1
  0 0 1 0 0
  0 0 0 1 1

Step 2: Skip (0,2), (0,3), (0,4) — all '0'
        Skip (1,0), (1,1) — already visited

Step 3: Find (2,2)='1', start DFS → marks island B
  V V 0 0 0
  V V 0 0 0      count = 2
  0 0 V 0 0
  0 0 0 1 1

Step 4: Find (3,3)='1', start DFS → marks island C
  V V 0 0 0
  V V 0 0 0      count = 3
  0 0 V 0 0
  0 0 0 V V
```

```python
def numIslands(grid: list[list[str]]) -> int:
    if not grid:
        return 0

    rows, cols = len(grid), len(grid[0])
    count = 0

    def dfs(r, c):
        # Out of bounds -- recursive DFS explores blindly then rejects
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return
        # Not land (either water '0' or already "sunk")
        if grid[r][c] != '1':
            return

        # Why `grid[r][c] = '0'` instead of a visited set?
        # In-place marking ("sinking the island") saves O(R*C) space.
        # Tradeoff: this MUTATES the input. If you need to preserve
        # the original grid, use a visited set instead.
        grid[r][c] = '0'  # Mark as visited by sinking

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            dfs(r + dr, c + dc)

    # Scan every cell: each unvisited '1' is a NEW island's starting point
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                dfs(r, c)    # Sink the entire island
                count += 1   # Count it

    return count
```

**Complexity:** Time O(R * C), Space O(R * C) worst case for recursion stack

---

#### Pattern 1B: Max Area of Island

**Problem:** LeetCode 695 - Given a binary grid, find the maximum area of an island. An island's area is the number of 1's in the connected component.

**Example:**
```
Input:
  grid = [
    [0,0,1,0,0,0,0,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,1,1,0,0,0],
    [0,1,1,0,1,0,0,0,0,0,0,0,0],
    [0,1,0,0,1,1,0,0,1,0,1,0,0],
    [0,1,0,0,1,1,0,0,1,1,1,0,0],
    [0,0,0,0,0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,1,1,1,0,0,0],
    [0,0,0,0,0,0,0,1,1,0,0,0,0]
  ]
Output: 6
```

**Key Insight:** Same as Number of Islands, but DFS returns the count of cells visited. Track the maximum across all DFS calls.

```python
def maxAreaOfIsland(grid: list[list[int]]) -> int:
    rows, cols = len(grid), len(grid[0])
    max_area = 0

    def dfs(r, c) -> int:
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return 0
        if grid[r][c] != 1:
            return 0

        grid[r][c] = 0  # Mark visited
        area = 1

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            area += dfs(r + dr, c + dc)

        return area

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                max_area = max(max_area, dfs(r, c))

    return max_area
```

**Complexity:** Time O(R * C), Space O(R * C)

---

#### Pattern 1C: Island Perimeter

**Problem:** LeetCode 463 - Given a grid where 1 represents land and 0 represents water, find the perimeter of the island (there is exactly one island).

**Example:**
```
Input:
  grid = [
    [0,1,0,0],
    [1,1,1,0],
    [0,1,0,0],
    [1,1,0,0]
  ]
Output: 16

Visual (each land cell contributes edges not shared with another land cell):
  . L .  .       L has 2 exposed edges (top, right)
  L L L  .       Each edge touching water or boundary = +1 to perimeter
  . L .  .
  L L .  .
```

**Key Insight:** For each land cell, count how many of its 4 sides are either at the boundary or adjacent to water. No DFS needed -- a single pass suffices.

```python
def islandPerimeter(grid: list[list[int]]) -> int:
    rows, cols = len(grid), len(grid[0])
    perimeter = 0

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                # Why start with 4? Every land cell is a square with 4 sides.
                # Each side touching another land cell is NOT perimeter (shared edge).
                # So: perimeter contribution = 4 - (number of land neighbors).
                perimeter += 4

                # Subtract shared edges with neighbors
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    # Why subtract 1 for each land neighbor?
                    # A shared edge between two land cells is interior, not perimeter.
                    # Boundary cells or water neighbors keep that side as perimeter.
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                        perimeter -= 1

    return perimeter
```

**Complexity:** Time O(R * C), Space O(1)

---

#### Pattern 1D: Number of Distinct Islands

**Problem:** LeetCode 694 - Count the number of distinct islands. Two islands are considered the same if one can be translated (not rotated/reflected) to match the other.

**Example:**
```
Input:
  grid = [
    [1,1,0,0,0],
    [1,1,0,0,0],
    [0,0,0,1,1],
    [0,0,0,1,1]
  ]
Output: 1  (both islands have the same shape)

Input:
  grid = [
    [1,1,0,1,1],
    [1,0,0,0,0],
    [0,0,0,0,1],
    [1,1,0,1,1]
  ]
Output: 3  (three distinct shapes)
```

**Key Insight:** Encode each island's shape as a path signature relative to its starting cell. Use DFS and record directions taken. Store signatures in a set to count distinct shapes.

```python
def numDistinctIslands(grid: list[list[int]]) -> int:
    rows, cols = len(grid), len(grid[0])
    distinct = set()

    def dfs(r, c, path, direction):
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return
        if grid[r][c] != 1:
            return

        grid[r][c] = 0  # Mark visited
        path.append(direction)

        dfs(r + 1, c, path, 'D')  # Down
        dfs(r - 1, c, path, 'U')  # Up
        dfs(r, c + 1, path, 'R')  # Right
        dfs(r, c - 1, path, 'L')  # Left

        # Why append a backtrack marker 'B'?
        # Without it, different shapes can produce the same direction sequence.
        # Example: an L-shape "DRB" vs a straight line "DR" — the 'B' records
        # WHERE we backtracked, which encodes the shape's structure.
        # Think of it like parentheses in math: (D(R)B) vs (DR) are different.
        path.append('B')

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                path = []
                dfs(r, c, path, 'S')  # S = start
                distinct.add(tuple(path))

    return len(distinct)
```

**Complexity:** Time O(R * C), Space O(R * C)

---

### PATTERN 2: Matrix BFS / Shortest Path

---

#### Pattern 2A: Rotting Oranges

**Problem:** LeetCode 994 - In a grid, 0 = empty, 1 = fresh orange, 2 = rotten orange. Every minute, fresh oranges adjacent (4-directionally) to rotten ones become rotten. Return the minimum minutes until no fresh oranges remain, or -1 if impossible.

**Example:**
```
Input:
  grid = [
    [2,1,1],
    [1,1,0],
    [0,1,1]
  ]
Output: 4

Visual Trace:
  Minute 0:    Minute 1:    Minute 2:    Minute 3:    Minute 4:
  2 1 1        2 2 1        2 2 2        2 2 2        2 2 2
  1 1 0        2 1 0        2 2 0        2 2 0        2 2 0
  0 1 1        0 1 1        0 1 1        0 2 1        0 2 2
```

**Key Insight:** This is **multi-source BFS**. All rotten oranges start in the queue simultaneously. Each BFS level = one minute. The answer is the number of levels needed to rot all fresh oranges.

```python
from collections import deque

def orangesRotting(grid: list[list[int]]) -> int:
    rows, cols = len(grid), len(grid[0])
    queue = deque()
    fresh = 0

    # Find all rotten oranges and count fresh
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                queue.append((r, c))
            elif grid[r][c] == 1:
                fresh += 1

    if fresh == 0:
        return 0

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    minutes = 0

    # Why `fresh > 0` in addition to `while queue`?
    # Optimization: once all fresh oranges are rotten, stop immediately.
    # Without it, BFS would keep running through remaining queued rotten cells.
    while queue and fresh > 0:
        minutes += 1
        # Why `range(len(queue))`? This is the "level snapshot" trick:
        # Process exactly the cells at the CURRENT minute, not the newly
        # rotten ones we're adding. Each full level = one minute passing.
        for _ in range(len(queue)):
            r, c = queue.popleft()
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                # Why `grid[nr][nc] == 1`? Only fresh (1) oranges can rot.
                # Rotten (2) and empty (0) are skipped.
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                    grid[nr][nc] = 2  # Rot it (also marks as visited)
                    fresh -= 1
                    queue.append((nr, nc))

    # Why check `fresh == 0`? If any fresh orange is unreachable from
    # all rotten sources (isolated by empty cells), return -1.
    return minutes if fresh == 0 else -1
```

**Complexity:** Time O(R * C), Space O(R * C)

---

#### Pattern 2B: Shortest Path in Binary Matrix

**Problem:** LeetCode 1091 - Given an n x n binary matrix, return the length of the shortest clear path from top-left to bottom-right. A clear path consists of 0's, and you can move in 8 directions.

**Example:**
```
Input:
  grid = [
    [0,0,0],
    [1,1,0],
    [1,1,0]
  ]
Output: 4

Path: (0,0) → (0,1) → (0,2) → (1,2) → (2,2)

Visual:
  S → → .
  X X ↓ .
  X X E .
  S = start, E = end, path length = 4
```

**Key Insight:** Standard BFS from (0,0) to (n-1,n-1). Use **8-directional** movement. BFS guarantees the first time we reach the target is the shortest path.

```python
from collections import deque

def shortestPathBinaryMatrix(grid: list[list[int]]) -> int:
    n = len(grid)
    # Why check both corners? If START or END is blocked (1),
    # no path can exist — fail fast before BFS.
    if grid[0][0] != 0 or grid[n - 1][n - 1] != 0:
        return -1

    # 8 directions (including diagonals)
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0),
                  (1, 1), (1, -1), (-1, 1), (-1, -1)]

    queue = deque([(0, 0, 1)])  # (row, col, path_length)
    visited = {(0, 0)}

    while queue:
        r, c, length = queue.popleft()

        if r == n - 1 and c == n - 1:
            return length

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            # Why FOUR conditions? Each filters a different invalid case:
            # 1. Bounds: don't step off the grid
            # 2. Not visited: don't revisit (infinite loop / wasted work)
            # 3. grid == 0: only open cells are passable
            # Early marking (add to visited NOW) ensures each cell is
            # queued at most once — same principle as BFS Template A.
            if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in visited and grid[nr][nc] == 0:
                visited.add((nr, nc))
                queue.append((nr, nc, length + 1))

    return -1
```

**Complexity:** Time O(N^2), Space O(N^2)

---

#### Pattern 2C: 01 Matrix (Multi-Source BFS)

**Problem:** LeetCode 542 - Given an m x n binary matrix, find the distance of the nearest 0 for each cell.

**Example:**
```
Input:                 Output:
  0 0 0                  0 0 0
  0 1 0        →         0 1 0
  1 1 1                  1 2 1

Visual Trace (multi-source BFS from all 0's):
  Level 0 (dist=0):  0 cells form the starting frontier
  Level 1 (dist=1):  1 cells adjacent to 0 cells get distance 1
  Level 2 (dist=2):  remaining cells get distance 2
```

**Key Insight:** **Reverse the thinking.** Instead of BFS from each 1 to find nearest 0, do multi-source BFS from ALL 0's simultaneously. Each 0 starts in the queue with distance 0.

```python
from collections import deque

def updateMatrix(mat: list[list[int]]) -> list[list[int]]:
    rows, cols = len(mat), len(mat[0])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    queue = deque()
    # Initialize: all 0-cells start with distance 0
    for r in range(rows):
        for c in range(cols):
            if mat[r][c] == 0:
                queue.append((r, c))
            else:
                mat[r][c] = float('inf')  # Not yet reached by BFS

    while queue:
        r, c = queue.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                # Why `> mat[r][c] + 1` instead of a visited set?
                # This serves DOUBLE duty:
                # 1. "Visited" check: if distance is already optimal (≤ current+1),
                #    the condition is false → we skip it.
                # 2. Distance update: if we found a shorter path, update and re-queue.
                # For standard BFS on unweighted grids, each cell is updated at most
                # once, so this is equivalent to a visited set but simpler.
                if mat[nr][nc] > mat[r][c] + 1:
                    mat[nr][nc] = mat[r][c] + 1
                    queue.append((nr, nc))

    return mat
```

**Complexity:** Time O(R * C), Space O(R * C)

---

### PATTERN 3: Matrix DFS / Flood Fill

---

#### Pattern 3A: Flood Fill

**Problem:** LeetCode 733 - Given an image (grid of ints), a starting pixel (sr, sc), and a new color, perform a flood fill: change the starting pixel and all 4-directionally connected pixels of the same original color to the new color.

**Example:**
```
Input:
  image = [
    [1,1,1],
    [1,1,0],
    [1,0,1]
  ]
  sr = 1, sc = 1, color = 2

Output:
  [
    [2,2,2],
    [2,2,0],
    [2,0,1]
  ]

Visual:
  1 1 1      2 2 2
  1 1 0  →   2 2 0    (all 1's connected to (1,1) become 2)
  1 0 1      2 0 1
```

**Key Insight:** Classic DFS/BFS from the starting pixel. Only visit cells with the original color. Watch out for the edge case where new color equals original color (infinite loop if not handled).

**Visual Trace:**
```
Start at (1,1), original color = 1, new color = 2:
  DFS(1,1): color=1 → set to 2, explore neighbors
    DFS(0,1): color=1 → set to 2
      DFS(0,0): color=1 → set to 2 (no more 1-neighbors)
      DFS(0,2): color=1 → set to 2 (no more 1-neighbors)
    DFS(2,1): color=0 → skip
    DFS(1,0): color=1 → set to 2
      DFS(2,0): color=1 → set to 2 (no more 1-neighbors)
    DFS(1,2): color=0 → skip
```

```python
def floodFill(image: list[list[int]], sr: int, sc: int, color: int) -> list[list[int]]:
    original = image[sr][sc]
    # Why this early return? If new color == original color, every cell we
    # visit is ALREADY the target color. We'd never stop exploring because
    # `image[r][c] != original` would never be true after coloring.
    # Result: infinite recursion / stack overflow. This guard prevents it.
    if original == color:
        return image

    rows, cols = len(image), len(image[0])

    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return
        if image[r][c] != original:
            return

        image[r][c] = color

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            dfs(r + dr, c + dc)

    dfs(sr, sc)
    return image
```

**Complexity:** Time O(R * C), Space O(R * C)

---

#### Pattern 3B: Surrounded Regions

**Problem:** LeetCode 130 - Given an m x n board containing 'X' and 'O', capture all regions surrounded by 'X'. A region is surrounded if it is not connected to any 'O' on the border of the board.

**Example:**
```
Input:
  X X X X        X X X X
  X O O X   →    X X X X
  X X O X        X X X X
  X O X X        X O X X

The O at (3,1) is connected to the border → not captured.
The O's at (1,1), (1,2), (2,2) are fully surrounded → captured.
```

**Key Insight:** **Reverse thinking.** Instead of finding surrounded regions, find UN-surrounded ones. DFS from every 'O' on the border to mark safe cells. Everything else gets captured.

```python
def solve(board: list[list[str]]) -> None:
    if not board:
        return

    rows, cols = len(board), len(board[0])

    # KEY INSIGHT: "Find surrounded O's" is hard. Instead, flip the question:
    # "Find UN-surrounded O's" — those touching the border. Mark them safe,
    # then everything else must be surrounded. Like finding dry land by
    # marking everything the ocean touches, then flooding the rest.
    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return
        # Why `!= 'O'`? Stop at 'X' (walls) AND 'E' (already marked safe).
        # Without the 'E' check, we'd revisit safe cells infinitely.
        if board[r][c] != 'O':
            return

        board[r][c] = 'E'  # 'E' = Escaped (connected to border = safe)

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            dfs(r + dr, c + dc)

    # Step 1: DFS from ALL border cells. Any 'O' reachable from
    # the border is NOT surrounded — mark it 'E' (escaped).
    for r in range(rows):
        dfs(r, 0)          # Left edge
        dfs(r, cols - 1)   # Right edge
    for c in range(cols):
        dfs(0, c)          # Top edge
        dfs(rows - 1, c)   # Bottom edge

    # Step 2: Three-way conversion:
    # 'O' → 'X': NOT connected to border = surrounded = captured
    # 'E' → 'O': WAS connected to border = safe = restore
    # 'X' stays 'X': was always a wall
    for r in range(rows):
        for c in range(cols):
            if board[r][c] == 'O':
                board[r][c] = 'X'  # Captured
            elif board[r][c] == 'E':
                board[r][c] = 'O'  # Restored
```

**Complexity:** Time O(R * C), Space O(R * C)

---

#### Pattern 3C: Pacific Atlantic Water Flow

**Problem:** LeetCode 417 - Given an m x n matrix of heights, find all cells where water can flow to both the Pacific (top/left edges) and Atlantic (bottom/right edges) oceans. Water flows from a cell to a neighbor with equal or lower height.

**Example:**
```
Input:
  heights = [
    [1,2,2,3,5],
    [3,2,3,4,4],
    [2,4,5,3,1],
    [6,7,1,4,5],
    [5,1,1,2,4]
  ]
Output: [[0,4],[1,3],[1,4],[2,2],[3,0],[3,1],[4,0]]

  Pacific ~   ~   ~   ~   ~
       ~  1   2   2   3  (5) *
       ~  3   2   3  (4) (4) *
       ~  2   4  (5)  3   1  *
       ~ (6) (7)  1   4   5  *
       ~ (5)  1   1   2   4  *
          *   *   *   *   * Atlantic
  Cells in () can reach both oceans.
```

**Key Insight:** **Reverse the flow.** Instead of checking if each cell can flow to both oceans, start from the oceans and flow uphill. Find cells reachable from Pacific (DFS from top/left edges) and cells reachable from Atlantic (DFS from bottom/right edges). The answer is the intersection.

```python
def pacificAtlantic(heights: list[list[int]]) -> list[list[int]]:
    if not heights:
        return []

    rows, cols = len(heights), len(heights[0])
    pacific = set()
    atlantic = set()

    # KEY TRICK: Checking "can water from cell X reach the ocean?" for every
    # cell is expensive. Instead, reverse the flow: start FROM each ocean
    # and flow UPHILL. Any cell reachable by uphill flow can drain to that ocean.
    def dfs(r, c, visited, prev_height):
        if (r, c) in visited:
            return
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return
        # Why `< prev_height`? We're flowing UPHILL (reverse of water flow).
        # Water flows high→low, so reverse = we can only step to cells that
        # are >= our current height. If the neighbor is LOWER, water couldn't
        # have flowed FROM there to us, so we can't reach it going backward.
        if heights[r][c] < prev_height:
            return

        visited.add((r, c))

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            dfs(r + dr, c + dc, visited, heights[r][c])

    # Why start with prev_height=0? Ocean level is 0, and any land cell
    # height ≥ 0, so the first step from the edge always succeeds.
    # DFS from Pacific edges (top row and left column)
    for c in range(cols):
        dfs(0, c, pacific, 0)
    for r in range(rows):
        dfs(r, 0, pacific, 0)

    # DFS from Atlantic edges (bottom row and right column)
    for c in range(cols):
        dfs(rows - 1, c, atlantic, 0)
    for r in range(rows):
        dfs(r, cols - 1, atlantic, 0)

    # Intersection: cells reachable from BOTH oceans = can drain to both.
    return [[r, c] for r, c in pacific & atlantic]
```

**Complexity:** Time O(R * C), Space O(R * C)

---

### PATTERN 4: Matrix Rotation / Transformation

---

#### Pattern 4A: Rotate Image

**Problem:** LeetCode 48 - Rotate an n x n matrix 90 degrees clockwise in place.

**Example:**
```
Input:                  Output:
  1  2  3               7  4  1
  4  5  6       →       8  5  2
  7  8  9               9  6  3

Transformation:
  Step 1 - Transpose:   Step 2 - Reverse rows:
  1  4  7               7  4  1
  2  5  8       →       8  5  2
  3  6  9               9  6  3
```

**Key Insight:** 90-degree clockwise rotation = Transpose + Reverse each row. This decomposition makes the in-place operation simple.

**Visual Trace:**
```
Original:       Transpose (swap i,j with j,i):    Reverse each row:
1 2 3           1 4 7                              7 4 1
4 5 6     →     2 5 8                        →     8 5 2
7 8 9           3 6 9                              9 6 3
```

```python
def rotate(matrix: list[list[int]]) -> None:
    n = len(matrix)

    # Step 1: Transpose (swap across main diagonal)
    # Why `j in range(i + 1, n)` and not `range(n)`?
    # We only swap ABOVE the diagonal. If we iterated all (i,j),
    # we'd swap each pair twice → back to the original! Starting
    # at i+1 means each (i,j)↔(j,i) pair is swapped exactly once.
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

    # Step 2: Reverse each row
    # Why does transpose + reverse = 90° clockwise rotation?
    # Transpose moves element (r,c) to (c,r). Reversing the row then
    # moves it to (c, n-1-r). Combined: (r,c) → (c, n-1-r), which
    # IS the formula for 90° clockwise rotation.
    for i in range(n):
        matrix[i].reverse()
```

**Rotation cheat sheet:**
- **90 clockwise:** transpose + reverse rows
- **90 counter-clockwise:** transpose + reverse columns (or reverse rows + transpose)
- **180 degrees:** reverse rows + reverse columns

**Complexity:** Time O(N^2), Space O(1)

---

#### Pattern 4B: Spiral Matrix

**Problem:** LeetCode 54 - Given an m x n matrix, return all elements in spiral order.

**Example:**
```
Input:
  1  2  3
  4  5  6
  7  8  9

Output: [1,2,3,6,9,8,7,4,5]

Traversal order:
  → → →
  ↑     ↓
  ← ← ←
  (then inner: →)
  Result: 1→2→3→6→9→8→7→4→5
```

**Key Insight:** Use four boundary pointers (top, bottom, left, right) and shrink them inward after each traversal direction. Handle edge cases where only one row or column remains.

```python
def spiralOrder(matrix: list[list[int]]) -> list[int]:
    if not matrix:
        return []

    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1

    # Why `top <= bottom AND left <= right`?
    # We peel off one layer of the spiral each iteration. When the
    # boundaries cross (top > bottom or left > right), the entire
    # matrix has been consumed. Both conditions needed because the
    # matrix may be non-square (e.g., 1×4 runs out of rows first).
    while top <= bottom and left <= right:
        # Traverse right along top row
        for c in range(left, right + 1):
            result.append(matrix[top][c])
        top += 1  # Top row consumed, shrink inward

        # Traverse down along right column
        for r in range(top, bottom + 1):
            result.append(matrix[r][right])
        right -= 1  # Right column consumed, shrink inward

        # Traverse left along bottom row
        # Why `if top <= bottom`? After moving top down, the top and bottom
        # might have crossed — meaning there's no bottom row left to traverse.
        # Without this check, a single-row matrix would double-count.
        if top <= bottom:
            for c in range(right, left - 1, -1):
                result.append(matrix[bottom][c])
            bottom -= 1

        # Traverse up along left column
        # Why `if left <= right`? Same idea — after moving right inward,
        # there might be no left column remaining. A single-column matrix
        # would double-count without this guard.
        if left <= right:
            for r in range(bottom, top - 1, -1):
                result.append(matrix[r][left])
            left += 1

    return result
```

**Complexity:** Time O(R * C), Space O(1) excluding output

---

#### Pattern 4C: Spiral Matrix II

**Problem:** LeetCode 59 - Given an integer n, generate an n x n matrix filled with elements from 1 to n^2 in spiral order.

**Example:**
```
Input: n = 3
Output:
  1  2  3
  8  9  4
  7  6  5
```

**Key Insight:** Same boundary pointer technique as Spiral Matrix, but instead of reading, we write values 1 through n^2.

```python
def generateMatrix(n: int) -> list[list[int]]:
    matrix = [[0] * n for _ in range(n)]
    top, bottom = 0, n - 1
    left, right = 0, n - 1
    num = 1

    while top <= bottom and left <= right:
        # Fill right along top row
        for c in range(left, right + 1):
            matrix[top][c] = num
            num += 1
        top += 1

        # Fill down along right column
        for r in range(top, bottom + 1):
            matrix[r][right] = num
            num += 1
        right -= 1

        # Fill left along bottom row
        if top <= bottom:
            for c in range(right, left - 1, -1):
                matrix[bottom][c] = num
                num += 1
            bottom -= 1

        # Fill up along left column
        if left <= right:
            for r in range(bottom, top - 1, -1):
                matrix[r][left] = num
                num += 1
            left += 1

    return matrix
```

**Complexity:** Time O(N^2), Space O(N^2)

---

### PATTERN 5: Matrix DP

---

#### Pattern 5A: Unique Paths

**Problem:** LeetCode 62 - A robot is on an m x n grid at the top-left corner. It can only move right or down. How many unique paths exist to reach the bottom-right corner?

**Example:**
```
Input: m = 3, n = 3
Output: 6

Visual (all paths from S to E):
  S → →
  ↓   ↓
  ↓ → E

dp table:
  1  1  1
  1  2  3
  1  3  6   ← answer is dp[2][2] = 6
```

**Key Insight:** Each cell can only be reached from the cell above or the cell to the left. So `dp[r][c] = dp[r-1][c] + dp[r][c-1]`. First row and first column are all 1 (only one way to reach them).

**Visual Trace:**
```
Initialize first row and column to 1:
  1  1  1
  1  .  .
  1  .  .

Fill remaining cells: dp[r][c] = dp[r-1][c] + dp[r][c-1]
  dp[1][1] = dp[0][1] + dp[1][0] = 1 + 1 = 2
  dp[1][2] = dp[0][2] + dp[1][1] = 1 + 2 = 3
  dp[2][1] = dp[1][1] + dp[2][0] = 2 + 1 = 3
  dp[2][2] = dp[1][2] + dp[2][1] = 3 + 3 = 6

Final:
  1  1  1
  1  2  3
  1  3  6
```

```python
def uniquePaths(m: int, n: int) -> int:
    # Why initialize entire grid to 1? The first row and first column
    # each have exactly ONE path (all-right or all-down). Setting them
    # to 1 is the base case. We only need to compute the interior cells.
    dp = [[1] * n for _ in range(m)]

    # Why start at (1,1)? Row 0 and col 0 are already correct (all 1s).
    for r in range(1, m):
        for c in range(1, n):
            # Why `dp[r-1][c] + dp[r][c-1]`?
            # You can only arrive from ABOVE or from the LEFT (no diagonal,
            # no going up/left). So total paths to (r,c) = paths that came
            # from above + paths that came from the left.
            dp[r][c] = dp[r - 1][c] + dp[r][c - 1]

    return dp[m - 1][n - 1]
```

**Space-optimized version (single row):**

```python
def uniquePaths_optimized(m: int, n: int) -> int:
    row = [1] * n

    for r in range(1, m):
        for c in range(1, n):
            row[c] += row[c - 1]

    return row[n - 1]
```

**Complexity:** Time O(m * n), Space O(n) optimized

---

#### Pattern 5B: Unique Paths II

**Problem:** LeetCode 63 - Same as Unique Paths, but some cells have obstacles (marked as 1). Find the number of unique paths avoiding obstacles.

**Example:**
```
Input:
  grid = [
    [0,0,0],
    [0,1,0],
    [0,0,0]
  ]
Output: 2

Visual:
  S . .        Two paths:
  . X .        S→→↓↓→E  and  S↓↓→→↑→↓E  → actually:
  . . E        S→→↓↓ and S↓↓→→ (both go around the obstacle)

dp table:
  1  1  1
  1  0  1
  1  1  2
```

**Key Insight:** Same DP as Unique Paths, but any cell with an obstacle has dp value 0 (unreachable). Also check if the start or end cell is blocked.

```python
def uniquePathsWithObstacles(obstacleGrid: list[list[int]]) -> int:
    m, n = len(obstacleGrid), len(obstacleGrid[0])

    # If start or end is blocked, no path exists.
    if obstacleGrid[0][0] == 1 or obstacleGrid[m - 1][n - 1] == 1:
        return 0

    dp = [[0] * n for _ in range(m)]
    dp[0][0] = 1

    # First column: each cell reachable only from above.
    # Why `dp[r-1][0]` (not just 1)? Once you hit an obstacle,
    # all cells BELOW it in the first column become unreachable (0).
    # dp propagates: 1, 1, 1, 0(obstacle), 0, 0, 0...
    for r in range(1, m):
        dp[r][0] = dp[r - 1][0] if obstacleGrid[r][0] == 0 else 0

    # First row: same logic, but obstacles block everything to the RIGHT.
    for c in range(1, n):
        dp[0][c] = dp[0][c - 1] if obstacleGrid[0][c] == 0 else 0

    # Fill rest: same as Unique Paths, but obstacles get dp = 0.
    for r in range(1, m):
        for c in range(1, n):
            if obstacleGrid[r][c] == 0:
                dp[r][c] = dp[r - 1][c] + dp[r][c - 1]
            else:
                dp[r][c] = 0  # Obstacle: no paths go through here

    return dp[m - 1][n - 1]
```

**Complexity:** Time O(m * n), Space O(m * n)

---

#### Pattern 5C: Minimum Path Sum

**Problem:** LeetCode 64 - Given an m x n grid filled with non-negative numbers, find a path from top-left to bottom-right that minimizes the sum. You can only move right or down.

**Example:**
```
Input:
  grid = [
    [1,3,1],
    [1,5,1],
    [4,2,1]
  ]
Output: 7

Path: 1 → 3 → 1 → 1 → 1 = 7

dp table:
  1   4   5
  2   7   6
  6   8   7  ← answer
```

**Key Insight:** `dp[r][c] = grid[r][c] + min(dp[r-1][c], dp[r][c-1])`. The minimum cost to reach a cell is its own value plus the cheaper of coming from above or from the left.

```python
def minPathSum(grid: list[list[int]]) -> int:
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = grid[0][0]

    # Why accumulate prefix sums for first row/col?
    # First row cells are only reachable from the left (can't come from above).
    # First col cells are only reachable from above (can't come from the left).
    # No choice = just add: dp[0][c] = dp[0][c-1] + grid[0][c].
    for c in range(1, n):
        dp[0][c] = dp[0][c - 1] + grid[0][c]

    for r in range(1, m):
        dp[r][0] = dp[r - 1][0] + grid[r][0]

    # Why `grid[r][c] + min(dp[r-1][c], dp[r][c-1])`?
    # You can only arrive from ABOVE (dp[r-1][c]) or from the LEFT (dp[r][c-1]).
    # Take the cheaper route, then add the current cell's cost.
    # It's like GPS: always enter this intersection via the cheaper road.
    for r in range(1, m):
        for c in range(1, n):
            dp[r][c] = grid[r][c] + min(dp[r - 1][c], dp[r][c - 1])

    return dp[m - 1][n - 1]
```

**Complexity:** Time O(m * n), Space O(m * n) (can optimize to O(n))

---

#### Pattern 5D: Maximal Square

**Problem:** LeetCode 221 - Given an m x n binary matrix, find the largest square containing only 1's and return its area.

**Example:**
```
Input:
  matrix = [
    ["1","0","1","0","0"],
    ["1","0","1","1","1"],
    ["1","1","1","1","1"],
    ["1","0","0","1","0"]
  ]
Output: 4  (the 2x2 square)

dp table (dp[r][c] = side length of largest square ending at (r,c)):
  1  0  1  0  0
  1  0  1  1  1
  1  1  1  2  2
  1  0  0  1  0
  Max value = 2, area = 2*2 = 4
```

**Key Insight:** `dp[r][c] = min(dp[r-1][c], dp[r][c-1], dp[r-1][c-1]) + 1` if the cell is '1'. The square size at (r,c) is limited by the smallest square among its three neighbors (top, left, top-left diagonal).

**Visual Trace:**
```
Why min of three neighbors?
  To form a square of side k at (r,c), you need:
  - A square of side k-1 ending at (r-1,c)      ← top
  - A square of side k-1 ending at (r,c-1)      ← left
  - A square of side k-1 ending at (r-1,c-1)    ← diagonal

  If any of these is smaller, the square at (r,c) is constrained.
  Example at (2,4): min(1, 2, 1) + 1 = 2
    dp[1][4]=1  dp[1][3]=1
    dp[2][3]=2  dp[2][4]=?  → min(1, 2, 1) + 1 = 2
```

```python
def maximalSquare(matrix: list[list[str]]) -> int:
    if not matrix:
        return 0

    m, n = len(matrix), len(matrix[0])
    dp = [[0] * n for _ in range(m)]
    max_side = 0

    for r in range(m):
        for c in range(n):
            if matrix[r][c] == '1':
                # Why separate first row/column? They can only form 1×1 squares
                # (no room for a larger square above or to the left).
                if r == 0 or c == 0:
                    dp[r][c] = 1
                else:
                    # Why min of THREE neighbors + 1?
                    # To form a k×k square ending at (r,c), you need:
                    #   - A (k-1)×(k-1) square ending at (r-1,c)   [top]
                    #   - A (k-1)×(k-1) square ending at (r,c-1)   [left]
                    #   - A (k-1)×(k-1) square ending at (r-1,c-1) [diagonal]
                    # The SMALLEST of these three limits your square size.
                    # Like building a box: the shortest wall determines the height.
                    dp[r][c] = min(dp[r - 1][c], dp[r][c - 1], dp[r - 1][c - 1]) + 1
                max_side = max(max_side, dp[r][c])

    return max_side * max_side
```

**Complexity:** Time O(m * n), Space O(m * n)

---

### PATTERN 6: Search in Sorted Matrix

---

#### Pattern 6A: Search a 2D Matrix

**Problem:** LeetCode 74 - Write an efficient algorithm to search for a value in an m x n matrix. Integers in each row are sorted left to right. The first integer of each row is greater than the last integer of the previous row.

**Example:**
```
Input:
  matrix = [
    [ 1, 3, 5, 7],
    [10,11,16,20],
    [23,30,34,60]
  ]
  target = 3
Output: True

The matrix is essentially one sorted list:
  1, 3, 5, 7, 10, 11, 16, 20, 23, 30, 34, 60
  Binary search index 1 → row 0, col 1 → value 3
```

**Key Insight:** Treat the entire matrix as a single sorted array of length m*n. Use standard binary search. Convert 1D index to 2D: `row = mid // cols`, `col = mid % cols`.

```python
def searchMatrix(matrix: list[list[int]], target: int) -> bool:
    if not matrix:
        return False

    rows, cols = len(matrix), len(matrix[0])
    lo, hi = 0, rows * cols - 1

    while lo <= hi:
        mid = (lo + hi) // 2
        # Why `mid // cols` and `mid % cols`?
        # Imagine unrolling the matrix into one long array of length rows*cols.
        # Index `mid` in that array maps to: row = mid ÷ cols (which "row chunk"),
        # col = mid mod cols (position within that row).
        # Example: cols=4, mid=6 → row=6//4=1, col=6%4=2 → matrix[1][2].
        r, c = mid // cols, mid % cols
        val = matrix[r][c]

        # Standard binary search: adjust bounds based on comparison.
        if val == target:
            return True
        elif val < target:
            lo = mid + 1
        else:
            hi = mid - 1

    return False
```

**Complexity:** Time O(log(R * C)), Space O(1)

---

#### Pattern 6B: Search a 2D Matrix II

**Problem:** LeetCode 240 - Search for a value in an m x n matrix where each row is sorted left to right AND each column is sorted top to bottom (but first element of next row is NOT necessarily greater than last element of previous row).

**Example:**
```
Input:
  matrix = [
    [ 1, 4, 7,11,15],
    [ 2, 5, 8,12,19],
    [ 3, 6, 9,16,22],
    [10,13,14,17,24],
    [18,21,23,26,30]
  ]
  target = 5
Output: True
```

**Key Insight:** **Staircase search.** Start from top-right (or bottom-left). If current value is too large, move left (eliminate column). If too small, move down (eliminate row). Each step eliminates an entire row or column.

**Visual Trace:**
```
Search for 5, start at top-right (0,4)=15:
  15 > 5 → move left → (0,3)=11
  11 > 5 → move left → (0,2)=7
   7 > 5 → move left → (0,1)=4
   4 < 5 → move down → (1,1)=5
   5 == 5 → found!
```

```python
def searchMatrix(matrix: list[list[int]], target: int) -> bool:
    if not matrix:
        return False

    rows, cols = len(matrix), len(matrix[0])
    # Why start at TOP-RIGHT (not top-left or bottom-right)?
    # Top-right is the only corner where ONE move eliminates an entire row or column:
    # - current > target → move LEFT (every cell below this column is also > target: all eliminated)
    # - current < target → move DOWN (every cell left in this row is also < target: all eliminated)
    # Top-left has two "greater" directions (right and down) — no clear elimination.
    # Bottom-right has two "smaller" directions (left and up) — same problem.
    r, c = 0, cols - 1

    # Why `r < rows AND c >= 0`?
    # r < rows: we haven't fallen off the bottom. c >= 0: we haven't fallen off the left.
    # Either exit means we've searched all reachable cells.
    while r < rows and c >= 0:
        if matrix[r][c] == target:
            return True
        elif matrix[r][c] > target:
            c -= 1  # Too big → move left (eliminate this entire column below us)
        else:
            r += 1  # Too small → move down (eliminate this entire row to our left)

    return False
```

**Complexity:** Time O(R + C), Space O(1)

---

### PATTERN 7: Game of Life / Simulation

---

#### Pattern 7A: Game of Life

**Problem:** LeetCode 289 - Implement the Game of Life. Each cell is alive (1) or dead (0). Apply rules simultaneously:
1. Live cell with < 2 live neighbors dies (underpopulation)
2. Live cell with 2-3 live neighbors lives
3. Live cell with > 3 live neighbors dies (overpopulation)
4. Dead cell with exactly 3 live neighbors becomes alive (reproduction)

Update the board in place.

**Example:**
```
Input:           Output:
  0 1 0          0 0 0
  0 0 1    →     1 0 1
  1 1 1          0 1 1
  0 0 0          0 1 0
```

**Key Insight:** The challenge is simultaneous updates -- you need the original state to compute all cells, but you are modifying in place. **Encode both states in the same cell:** use 2 to mean "was alive, now dead" and 3 to mean "was dead, now alive". Then do a final pass to convert 2 -> 0 and 3 -> 1.

**Visual Trace:**
```
State encoding:
  0 = was dead, stays dead
  1 = was alive, stays alive
  2 = was alive, now dead     (original state: alive → old & 1 == 1)
  3 = was dead, now alive     (original state: dead  → old & 1 == 0)

When counting neighbors, use (val & 1) to get original state.
Final pass: new_state = val >> 1 ... or simpler: 2→0, 3→1, else keep.
Actually simpler: 0→0, 1→1, 2→0, 3→1. So new = val % 2 ... no.
Encode: 2 means was-alive-now-dead. 3 means was-dead-now-alive.
Original state = val % 2 ... for 0→0, 1→1, 2→0, 3→1. Yes!
New state: 0→0, 1→1, 2→0, 3→1. So final = 0 if val in {0,2}, 1 if val in {1,3}.
Use: original = board[r][c] & 1 (bit 0 = original state)
     Set bit 1 for new state: board[r][c] |= (new_state << 1)
     Final: board[r][c] >>= 1
```

```python
def gameOfLife(board: list[list[int]]) -> None:
    rows, cols = len(board), len(board[0])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0),
                  (1, 1), (1, -1), (-1, 1), (-1, -1)]

    for r in range(rows):
        for c in range(cols):
            live_neighbors = 0
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    # Why `board[nr][nc] & 1` instead of just `board[nr][nc]`?
                    # We use bit 0 to store the ORIGINAL state. Cells already processed
                    # may have had bit 1 set (new state), making them 2 or 3.
                    # `& 1` masks away bit 1, recovering the original 0 or 1.
                    # Without this, a neighbor value of 2 (was alive, now dead) would
                    # be counted as 2 live neighbors — completely wrong!
                    live_neighbors += board[nr][nc] & 1

            # Why `board[r][c] & 1` to check current state?
            # Same reason: bit 0 is original alive/dead. We haven't touched this cell
            # yet in this pass, but we use & 1 for consistency and safety.
            if board[r][c] & 1:  # Originally alive
                if live_neighbors in (2, 3):
                    # Why `|= 2`? We're setting bit 1 (value 2) to record "new state = alive".
                    # Bit 0 still holds the original state. The cell is now 1|2 = 3.
                    board[r][c] |= 2  # Stays alive → set bit 1
                # else: underpopulation or overpopulation → bit 1 stays 0 (dies)
            else:  # Originally dead
                if live_neighbors == 3:
                    board[r][c] |= 2  # Becomes alive → bit 1 = 1, cell becomes 0|2 = 2

    # Why `>>= 1`? This shifts all bits right by 1, discarding bit 0 (old state)
    # and moving bit 1 (new state) into bit 0. Result: 0→0, 1→0, 2→1, 3→1.
    # Old cells that died (were 1, bit1=0) become 0. New cells (bit1=1) become 1.
    for r in range(rows):
        for c in range(cols):
            board[r][c] >>= 1
```

**Complexity:** Time O(R * C), Space O(1)

---

#### Pattern 7B: Valid Sudoku

**Problem:** LeetCode 36 - Determine if a 9x9 Sudoku board is valid. Only filled cells need to be validated: each row, each column, and each 3x3 box must contain digits 1-9 without repetition.

**Example:**
```
Input: A standard 9x9 Sudoku board with some cells filled
Output: True or False

Key: the 9 boxes are indexed as:
  box 0 | box 1 | box 2
  box 3 | box 4 | box 5
  box 6 | box 7 | box 8

  box_index = (r // 3) * 3 + (c // 3)
```

**Key Insight:** Use three sets of hash sets: one per row, one per column, one per 3x3 box. For each filled cell, check if the digit already exists in its row, column, or box.

```python
def isValidSudoku(board: list[list[str]]) -> bool:
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    boxes = [set() for _ in range(9)]

    for r in range(9):
        for c in range(9):
            val = board[r][c]
            if val == '.':
                continue

            # Why `(r // 3) * 3 + (c // 3)`?
            # The 9×9 board has 9 boxes arranged in a 3×3 grid of boxes.
            # `r // 3` gives the BOX ROW (0, 1, or 2). Multiply by 3 to get
            # the starting index for that box row (0, 3, or 6).
            # `c // 3` gives the BOX COLUMN (0, 1, or 2) — add it as offset.
            # Example: cell (4, 7) → box row = 4//3=1, box col = 7//3=2 → box 1*3+2 = 5.
            box_idx = (r // 3) * 3 + (c // 3)

            # Why check all THREE (row, col, box) before adding?
            # A digit can violate the row constraint, the col constraint, OR the box
            # constraint independently. One check failing is enough to return False.
            if val in rows[r] or val in cols[c] or val in boxes[box_idx]:
                return False

            rows[r].add(val)
            cols[c].add(val)
            boxes[box_idx].add(val)

    return True
```

**Complexity:** Time O(1) (fixed 81 cells), Space O(1) (fixed size sets)

---

#### Pattern 7C: Set Matrix Zeroes

**Problem:** LeetCode 73 - Given an m x n matrix, if an element is 0, set its entire row and column to 0. Do it in place.

**Example:**
```
Input:           Output:
  1 1 1          1 0 1
  1 0 1    →     0 0 0
  1 1 1          1 0 1
```

**Key Insight:** Use the first row and first column as markers. If cell (r,c) is 0, mark matrix[r][0] = 0 and matrix[0][c] = 0. Then iterate again and zero out cells based on markers. Handle the first row and column separately with boolean flags.

**Visual Trace:**
```
Step 1: Scan and mark
  matrix[1][1] == 0, so mark matrix[1][0] = 0 and matrix[0][1] = 0

  First row markers: 1  0  1  (col 1 marked)
  First col markers: 1         (row 0 not marked)
                     0         (row 1 marked)
                     1         (row 2 not marked)

Step 2: Zero out based on markers (skip first row/col)
  For (1,0): matrix[1][0]==0 → zero entire row 1
  For (0,1): matrix[0][1]==0 → zero entire col 1

Step 3: Handle first row/col using flags
```

```python
def setZeroes(matrix: list[list[int]]) -> None:
    rows, cols = len(matrix), len(matrix[0])
    first_row_zero = False
    first_col_zero = False

    # Why save first row/col state BEFORE using them as markers?
    # We're about to overwrite matrix[r][0] and matrix[0][c] with 0s as "flags".
    # If row 0 or col 0 already had a 0, we need to know that BEFORE the marking
    # phase corrupts that info. Record it now, apply it last.
    for c in range(cols):
        if matrix[0][c] == 0:
            first_row_zero = True
            break

    for r in range(rows):
        if matrix[r][0] == 0:
            first_col_zero = True
            break

    # Why start at range(1, rows) and range(1, cols)?
    # We're USING row 0 and col 0 as our marker storage. If we started at 0,
    # we'd be both reading and writing the markers simultaneously — corrupting them.
    # Skip first row and first column; handle them separately with flags above.
    for r in range(1, rows):
        for c in range(1, cols):
            if matrix[r][c] == 0:
                # Mark: row r needs zeroing (set its first-col marker).
                # Mark: col c needs zeroing (set its first-row marker).
                matrix[r][0] = 0
                matrix[0][c] = 0

    # Why `matrix[r][0] == 0 OR matrix[0][c] == 0`?
    # A cell (r, c) should be zeroed if EITHER its row was marked (by a zero anywhere
    # in that row) OR its column was marked (by a zero anywhere in that column).
    for r in range(1, rows):
        for c in range(1, cols):
            if matrix[r][0] == 0 or matrix[0][c] == 0:
                matrix[r][c] = 0

    # Why handle first row/col LAST?
    # If we zeroed them first, their cells would become 0 and incorrectly mark
    # other rows/cols. Apply the saved flags only after all other cells are done.
    if first_row_zero:
        for c in range(cols):
            matrix[0][c] = 0

    if first_col_zero:
        for r in range(rows):
            matrix[r][0] = 0
```

**Complexity:** Time O(R * C), Space O(1)

---

<a name="post-processing"></a>
## 5. Post-Processing Reference

| Problem Type | Return Value | Edge Cases |
|--------------|--------------|------------|
| **Count islands** | Integer count | Empty grid, all water, all land |
| **Shortest path** | Distance or -1 | Start == end, blocked path |
| **Flood fill** | Modified grid | New color == old color |
| **Matrix rotation** | In-place modification | 1x1 matrix, non-square for spiral |
| **Matrix DP** | Single value (count, min, max) | Single cell grid, first row/col init |
| **Search in matrix** | Boolean or index | Empty matrix, target not present |
| **Simulation** | In-place modification | Empty board, all same state |
| **Connected components** | Count or max size | No components, single cell |
| **Perimeter** | Integer | Single cell island |
| **Multi-source BFS** | Distance matrix or time | No sources, all sources |

---

<a name="pitfalls"></a>
## 6. Common Pitfalls & Solutions

### Pitfall 1: Off-by-One in Bounds Checking

**Problem:** Using wrong comparison operators for grid boundaries

```python
# WRONG: allows index equal to rows/cols (out of bounds!)
if r >= 0 and r <= rows and c >= 0 and c <= cols:
    process(grid[r][c])  # IndexError when r == rows!
```

**Solution:** Use strict less-than for upper bounds
```python
# CORRECT
if 0 <= r < rows and 0 <= c < cols:
    process(grid[r][c])
```

---

### Pitfall 2: Forgetting to Mark Visited (Infinite Loop)

**Problem:** Not marking cells as visited causes infinite recursion between neighbors

```python
# WRONG: cells are never marked, DFS revisits endlessly
def dfs(r, c):
    if grid[r][c] != 1:
        return
    # Forgot to mark grid[r][c] = 0 or add to visited!
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        dfs(r + dr, c + dc)
```

**Solution:** Mark visited BEFORE exploring neighbors
```python
# CORRECT
def dfs(r, c):
    if grid[r][c] != 1:
        return
    grid[r][c] = 0  # Mark visited first!
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            dfs(nr, nc)
```

---

### Pitfall 3: BFS -- Marking Visited When Popping Instead of When Adding

**Problem:** Adding a cell to the queue multiple times because you check visited only when processing

```python
# WRONG: cell (nr, nc) can be added to queue by multiple neighbors
while queue:
    r, c = queue.popleft()
    if (r, c) in visited:  # Too late! Already in queue many times
        continue
    visited.add((r, c))
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        queue.append((nr, nc))
```

**Solution:** Mark visited immediately when adding to queue
```python
# CORRECT
while queue:
    r, c = queue.popleft()
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
            visited.add((nr, nc))  # Mark NOW, not when popping
            queue.append((nr, nc))
```

---

### Pitfall 4: Modifying Grid While Reading Neighbors

**Problem:** In simulation problems (Game of Life), changing a cell affects the neighbor count for other cells in the same pass

```python
# WRONG: changing board[r][c] mid-iteration corrupts neighbor counts
for r in range(rows):
    for c in range(cols):
        neighbors = count_live_neighbors(r, c)
        if board[r][c] == 1 and neighbors < 2:
            board[r][c] = 0  # This affects neighbor counts of (r+1, c) etc.!
```

**Solution:** Encode old and new state in the same cell, or use a copy
```python
# CORRECT: encode both states using bit manipulation
for r in range(rows):
    for c in range(cols):
        neighbors = count_live_neighbors(r, c)  # Uses board[nr][nc] & 1
        if should_live(board[r][c] & 1, neighbors):
            board[r][c] |= 2  # Set bit 1 for new state

# Final pass
for r in range(rows):
    for c in range(cols):
        board[r][c] >>= 1
```

---

### Pitfall 5: Confusing Row/Column with X/Y Coordinates

**Problem:** Mixing up (row, col) with (x, y) leads to transposed grids and wrong answers

```python
# WRONG: treating row as x (horizontal) and col as y (vertical)
x, y = 2, 3
grid[x][y]  # This is row 2, col 3 -- but is that what you meant?

# In many problems, directions use (dx, dy) but grid uses (row, col)
# row increases DOWNWARD, y increases UPWARD in math
```

**Solution:** Always use (row, col) naming for grids. Row is the first index, col is the second.
```python
# CORRECT: consistent naming
r, c = 2, 3
grid[r][c]  # Row 2, column 3

# Directions: (delta_row, delta_col)
for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
    nr, nc = r + dr, c + dc
```

---

### Pitfall 6: Stack Overflow on Large Grids

**Problem:** Recursive DFS on a 300x300 grid can exceed Python's default recursion limit (1000)

```python
# CRASHES on large grids
def dfs(r, c):
    grid[r][c] = 0
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if valid(nr, nc) and grid[nr][nc] == 1:
            dfs(nr, nc)  # RecursionError: maximum recursion depth exceeded
```

**Solution:** Use iterative DFS with an explicit stack, or increase recursion limit
```python
# CORRECT: iterative DFS
def dfs(start_r, start_c):
    stack = [(start_r, start_c)]
    grid[start_r][start_c] = 0

    while stack:
        r, c = stack.pop()
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                grid[nr][nc] = 0
                stack.append((nr, nc))
```

---

### Pitfall 7: Spiral Traversal Missing Inner Elements

**Problem:** Not checking `top <= bottom` and `left <= right` after advancing boundaries causes duplicate visits or missed elements

```python
# WRONG: for a single-row or single-column matrix, this double-counts
while top <= bottom and left <= right:
    for c in range(left, right + 1):
        result.append(matrix[top][c])
    top += 1
    for r in range(top, bottom + 1):
        result.append(matrix[r][right])
    right -= 1
    for c in range(right, left - 1, -1):      # No check!
        result.append(matrix[bottom][c])        # Might revisit row
    bottom -= 1
    for r in range(bottom, top - 1, -1):       # No check!
        result.append(matrix[r][left])          # Might revisit col
    left += 1
```

**Solution:** Add boundary checks before the third and fourth traversals
```python
# CORRECT
if top <= bottom:
    for c in range(right, left - 1, -1):
        result.append(matrix[bottom][c])
    bottom -= 1

if left <= right:
    for r in range(bottom, top - 1, -1):
        result.append(matrix[r][left])
    left += 1
```

---

<a name="recognition"></a>
## 7. Problem Recognition Framework

### Step 1: Is It a Grid/Matrix Problem?

**Grid indicators:**
- Input is a 2D array/list of lists
- Problem mentions "grid", "matrix", "board", "map"
- Cells have values (0/1, colors, heights, obstacles)
- Movement between adjacent cells

### Step 2: What Type of Grid Problem?

| Clue in Problem | Category |
|-----------------|----------|
| "Count islands", "connected regions" | Island / Connected Components |
| "Minimum steps", "shortest path", "nearest" | BFS Shortest Path |
| "Fill", "capture", "flow to both" | DFS Flood Fill |
| "Rotate", "spiral order", "transpose" | Matrix Transformation |
| "Number of paths", "minimum cost path" | Matrix DP |
| "Search for target", "sorted rows/cols" | Sorted Matrix Search |
| "Simultaneous update", "next state", "in-place" | Simulation |

### Step 3: Decision Tree

```
                     Is it a 2D grid/matrix problem?
                                 |
                    Yes ─────────┴─────────── No
                     |                        |
              What's being asked?        Not this handbook
                     |
     ┌───────┬───────┼───────┬───────┬───────┬───────┐
     |       |       |       |       |       |       |
  Count/   Shortest  Fill/  Transform  Path   Search  Simulate
  measure  distance  mark   rotate    count  in sorted  state
  regions            cells  elements  /cost  matrix    change
     |       |       |       |       |       |        |
   DFS      BFS     DFS   Index    DP     Binary   Encode
   mark   (multi-  from   math            search   states
   visit   source) boundary or            or       in-place
     |       |       |    boundary       staircase    |
  Pattern  Pattern Pattern pointers      search   Pattern 7
    1        2       3       |          |        |
                          Pattern 4  Pattern 5  Pattern 6
```

### Quick Pattern Matching Cheat Sheet

```
"Number of islands"        → DFS/BFS flood fill, count starts
"Rotting oranges"          → Multi-source BFS
"Shortest path in grid"    → BFS (unweighted) or Dijkstra (weighted)
"Flood fill"               → DFS from starting cell
"Surrounded regions"       → DFS from border, then invert
"Water flow"               → Reverse DFS from edges
"Rotate matrix"            → Transpose + reverse
"Spiral order"             → Boundary pointers
"Unique paths"             → DP: dp[r][c] = dp[r-1][c] + dp[r][c-1]
"Minimum path sum"         → DP: dp[r][c] = val + min(top, left)
"Maximal square"           → DP: dp[r][c] = min(top, left, diag) + 1
"Search sorted matrix"     → Binary search or staircase
"Game of Life"             → Bit encoding for simultaneous update
"Set matrix zeroes"        → Use first row/col as markers
```

---

<a name="checklist"></a>
## 8. Interview Preparation Checklist

### Before the Interview

**Master the fundamentals:**
- [ ] Can write the 4-directional neighbor loop from memory
- [ ] Can write grid BFS template from memory
- [ ] Can write grid DFS template from memory
- [ ] Understand when to use BFS vs DFS on grids
- [ ] Know the directions array: `[(0,1),(0,-1),(1,0),(-1,0)]`

**Practice pattern recognition:**
- [ ] Can identify island problems immediately
- [ ] Know when multi-source BFS applies
- [ ] Understand the "reverse flow from boundary" trick
- [ ] Can decompose matrix rotation into transpose + reverse
- [ ] Know the spiral boundary pointer technique

**Know the patterns:**
- [ ] Island counting and measurement (DFS)
- [ ] Multi-source BFS (rotting oranges, 01 matrix)
- [ ] Boundary DFS (surrounded regions, water flow)
- [ ] Matrix transformation (rotate, spiral)
- [ ] Matrix DP (unique paths, min path sum, maximal square)
- [ ] Sorted matrix search (binary search, staircase)
- [ ] Simulation with state encoding (game of life)

**Core problems solved:**
- [ ] LC 200: Number of Islands
- [ ] LC 994: Rotting Oranges
- [ ] LC 733: Flood Fill
- [ ] LC 48: Rotate Image
- [ ] LC 54: Spiral Matrix
- [ ] LC 62: Unique Paths
- [ ] LC 221: Maximal Square
- [ ] LC 74: Search a 2D Matrix
- [ ] LC 289: Game of Life
- [ ] LC 73: Set Matrix Zeroes

### During the Interview

**1. Clarify (30 seconds)**
- Grid dimensions? Can it be empty?
- What do the cell values represent?
- 4-directional or 8-directional movement?
- Modify in place or return new grid?
- What to return if no valid answer?

**2. Identify pattern (30 seconds)**
- Connected components? → DFS/BFS flood fill
- Shortest distance? → BFS
- Path counting/optimization? → DP
- Transformation? → Index math
- Sorted? → Binary search / staircase

**3. Code (3-4 minutes)**
- Set up `rows, cols = len(grid), len(grid[0])`
- Define `directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]`
- Write bounds check: `0 <= nr < rows and 0 <= nc < cols`
- Initialize visited set or modify grid in place
- Implement the core logic from the template

**4. Test (1-2 minutes)**
- Empty grid
- Single cell grid
- All same values
- Edge cases specific to the pattern

**5. Analyze (30 seconds)**
- Time: usually O(R * C)
- Space: O(R * C) for BFS/DFS visited, O(1) for in-place

---

## 9. Quick Reference Cards

### Grid BFS Template
```python
from collections import deque
directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
queue = deque([(start_r, start_c, 0)])
visited = {(start_r, start_c)}
while queue:
    r, c, dist = queue.popleft()
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and grid[nr][nc] != WALL:
            visited.add((nr, nc))
            queue.append((nr, nc, dist + 1))
```

### Grid DFS Template
```python
def dfs(r, c):
    if r < 0 or r >= rows or c < 0 or c >= cols:
        return
    if grid[r][c] != TARGET:
        return
    grid[r][c] = VISITED  # Mark visited
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        dfs(r + dr, c + dc)
```

### Multi-Source BFS Template
```python
queue = deque()
for r in range(rows):
    for c in range(cols):
        if grid[r][c] == SOURCE:
            queue.append((r, c, 0))
            visited.add((r, c))
# Then standard BFS loop
```

### Matrix DP Template
```python
dp = [[0] * cols for _ in range(rows)]
dp[0][0] = base_case
for r in range(rows):
    for c in range(cols):
        if r > 0: dp[r][c] += dp[r-1][c]  # from above
        if c > 0: dp[r][c] += dp[r][c-1]  # from left
```

### Spiral Traversal Template
```python
top, bottom, left, right = 0, rows - 1, 0, cols - 1
while top <= bottom and left <= right:
    for c in range(left, right + 1): visit(top, c)    # right
    top += 1
    for r in range(top, bottom + 1): visit(r, right)  # down
    right -= 1
    if top <= bottom:
        for c in range(right, left - 1, -1): visit(bottom, c)  # left
        bottom -= 1
    if left <= right:
        for r in range(bottom, top - 1, -1): visit(r, left)    # up
        left += 1
```

### In-Place State Encoding Template
```python
# Encode: use bit 0 for old state, bit 1 for new state
for r in range(rows):
    for c in range(cols):
        old_state = board[r][c] & 1
        new_state = compute_new_state(r, c)
        if new_state:
            board[r][c] |= 2
# Decode
for r in range(rows):
    for c in range(cols):
        board[r][c] >>= 1
```

---

## 10. Complexity Reference

| Pattern | Time | Space | Notes |
|---------|------|-------|-------|
| Island counting (DFS/BFS) | O(R * C) | O(R * C) | Visit every cell once |
| Multi-source BFS | O(R * C) | O(R * C) | All sources start together |
| Flood fill | O(R * C) | O(R * C) | Recursion stack / queue |
| Boundary DFS | O(R * C) | O(R * C) | Start from edges |
| Spiral traversal | O(R * C) | O(1) | Excluding output |
| Matrix rotation | O(N^2) | O(1) | In-place transpose + reverse |
| Matrix DP | O(R * C) | O(R * C) or O(C) | Can optimize to single row |
| Binary search in matrix | O(log(R * C)) | O(1) | Fully sorted matrix |
| Staircase search | O(R + C) | O(1) | Row-sorted + col-sorted |
| Game of Life simulation | O(R * C) | O(1) | Bit encoding in-place |
| Set matrix zeroes | O(R * C) | O(1) | First row/col as markers |
| Island perimeter | O(R * C) | O(1) | Count exposed edges |
| Maximal square DP | O(R * C) | O(R * C) | min(top, left, diag) + 1 |
| Distinct islands | O(R * C) | O(R * C) | Path signature hashing |

Where R = rows, C = columns, N = side length (for square matrices).

---

## Final Thoughts

**Remember:**
1. A grid is a graph -- `[(0,1),(0,-1),(1,0),(-1,0)]` is your adjacency list
2. BFS for shortest path, DFS for connected components and flood fill
3. Multi-source BFS: add ALL sources to queue before starting
4. "Reverse thinking" solves many problems: flow from boundary, BFS from targets
5. In-place updates need state encoding (bit tricks) or a copy
6. Matrix DP fills top-to-bottom, left-to-right (usually)
7. Spiral traversal uses boundary pointers; always check before 3rd and 4th directions

**When stuck:**
1. Draw the grid on paper and trace through your algorithm cell by cell
2. Ask: "Is this a graph traversal (BFS/DFS) or a DP problem?"
3. Ask: "Can I reverse the direction of search?" (boundary DFS, BFS from targets)
4. Ask: "Do I need to encode state in place or can I use extra space?"
5. Check: "Am I handling all edge cases?" (empty grid, single cell, all same values)
6. Verify: "Am I marking visited BEFORE adding to the queue, not after popping?"

---

## Appendix: Practice Problem Set

### Easy
- 463. Island Perimeter
- 733. Flood Fill
- 867. Transpose Matrix
- 1572. Matrix Diagonal Sum

### Medium (Core Interview Level)
- 48. Rotate Image
- 54. Spiral Matrix
- 59. Spiral Matrix II
- 62. Unique Paths
- 63. Unique Paths II
- 64. Minimum Path Sum
- 73. Set Matrix Zeroes
- 74. Search a 2D Matrix
- 130. Surrounded Regions
- 200. Number of Islands
- 221. Maximal Square
- 240. Search a 2D Matrix II
- 289. Game of Life
- 417. Pacific Atlantic Water Flow
- 542. 01 Matrix
- 694. Number of Distinct Islands
- 695. Max Area of Island
- 994. Rotting Oranges
- 1091. Shortest Path in Binary Matrix

### Hard
- 36. Valid Sudoku (actually Medium but good practice)
- 37. Sudoku Solver
- 329. Longest Increasing Path in a Matrix
- 407. Trapping Rain Water II
- 778. Swim in Rising Water
- 827. Making A Large Island

**Recommended Practice Order:**
1. Start with LC 733 (flood fill -- simplest grid DFS)
2. Master LC 200 (number of islands -- core pattern)
3. Do LC 994 and 542 (multi-source BFS)
4. Practice LC 130 and 417 (boundary DFS tricks)
5. Work through LC 62, 64, 221 (matrix DP)
6. Do LC 48 and 54 (matrix transformations)
7. Practice LC 74 and 240 (sorted matrix search)
8. Attempt LC 289 and 73 (simulation / in-place tricks)
9. Try LC 329 (combines DFS + memoization on grid -- hard)

Good luck with your interview preparation!

---

## Appendix: Conditional Quick Reference

This table lists every key condition used in this handbook, its plain-English meaning, and the intuition behind it.

### A. Bounds & Traversal Conditions

| Condition | Plain English | Why it works |
|-----------|---------------|--------------|
| `0 <= nr < rows and 0 <= nc < cols` | "This cell is inside the grid" | Lower bound prevents negative indices; upper bound (strict `<`) prevents index == length (off-by-one) |
| `while lo <= hi` (binary search) | "Search range is non-empty" | `lo > hi` means the range collapsed — target not found |
| `while r < rows and c >= 0` (staircase) | "Still inside the top-right search zone" | r falls off bottom when we exhaust rows; c falls off left when we exhaust cols |
| `while top <= bottom and left <= right` (spiral) | "Boundary still has cells to visit" | When top > bottom or left > right, the layers have met — done |

### B. BFS / DFS Visit Conditions

| Condition | Plain English | Why it works |
|-----------|---------------|--------------|
| `(nr, nc) not in visited` | "Haven't been here before" | Prevents infinite loops between neighbors; mark BEFORE enqueuing |
| `grid[nr][nc] != WALL` | "This cell is passable" | Only enqueue cells we can actually traverse |
| `if original == color` (flood fill) | "Starting cell is same as fill color" | If already the target color, doing fill would overwrite then re-fill forever |
| `if r < 0 or r >= rows or c < 0 or c >= cols: return` (DFS base) | "Stepped outside grid — stop" | Early exit at boundary avoids index errors |
| `if grid[r][c] != TARGET: return` (DFS base) | "Wrong cell type — don't explore" | Only continues DFS into cells that belong to the current region |

### C. BFS Level / Multi-Source Conditions

| Condition | Plain English | Why it works |
|-----------|---------------|--------------|
| `while queue and fresh > 0` (rotting oranges) | "Still oranges to rot AND unrotted oranges remain" | Short-circuits once all fresh oranges are already rotted — avoids extra empty iterations |
| `for _ in range(len(queue))` (level BFS) | "Process exactly this layer" | Snapshot of current layer size; new cells added don't get processed until next minute |
| `dist[nr][nc] > dist[r][c] + 1` (01 matrix) | "Found a shorter route to this cell" | Standard relaxation: only update if new path is strictly shorter |

### D. DFS Boundary Flood (Surrounded Regions / Pacific Atlantic)

| Condition | Plain English | Why it works |
|-----------|---------------|--------------|
| `r == 0 or r == rows-1 or c == 0 or c == cols-1` | "This cell is on the border" | Only border cells can "escape" — start DFS from them to mark safe regions |
| `height[nr][nc] >= prev_height` (Pacific/Atlantic) | "Water can flow uphill in reverse" | We're doing reverse-flow: working from ocean back into land. Reverse-flow goes from low→high |
| `if grid[r][c] == 'O': grid[r][c] = 'X'` | "Capture this unprotected region" | Any 'O' not reached by border DFS is surrounded; convert it now |

### E. Matrix Transformation Conditions

| Condition | Plain English | Why it works |
|-----------|---------------|--------------|
| `for j in range(i+1, n)` (transpose) | "Swap each pair exactly once" | Starting from `i+1` prevents double-swapping — if we swapped (i,j) and (j,i) both, they'd cancel out |
| `if top <= bottom` before bottom row (spiral) | "Bottom row hasn't merged with top row" | A single-row grid: after traversing top row, top > bottom. No bottom row to traverse |
| `if left <= right` before left col (spiral) | "Left col hasn't merged with right col" | A single-col grid: after traversing right col, right < left. No left col to traverse |

### F. Matrix DP Conditions

| Condition | Plain English | Why it works |
|-----------|---------------|--------------|
| `dp = [[1]*n for _ in range(m)]` (unique paths init) | "Every cell starts with 1 path" | First row and col each have exactly 1 path (only right or only down). Pre-filling avoids separate init loops |
| `for r in range(1, m)` (DP main loop) | "Skip row 0 — already initialized" | Row 0 values are base cases; computing them again would incorrectly add 0 from above |
| `r == 0 or c == 0` (maximal square edge) | "On the border — can only be a 1×1 square" | No cells above/left exist to form a larger square from; safe to cap at 1 |
| `min(dp[r-1][c], dp[r][c-1], dp[r-1][c-1]) + 1` | "Size limited by shortest neighboring square" | To grow a k×k square, all three L-shaped neighbors must already have (k-1)×(k-1) squares |

### G. Simulation / In-Place Conditions

| Condition | Plain English | Why it works |
|-----------|---------------|--------------|
| `board[nr][nc] & 1` | "Original alive/dead (ignore new state)" | Bit 0 = original state. `& 1` masks away bit 1 (new state) so already-updated cells don't corrupt neighbor counts |
| `board[r][c] & 1` | "This cell was originally alive" | Same mask: read only the original state when deciding what rule applies |
| `board[r][c] \|= 2` | "Record: new state = alive" | Sets bit 1 without touching bit 0 (original state). After the pass, bit 1 holds the new state |
| `board[r][c] >>= 1` | "Commit: shift new state into bit 0" | Discards original state (bit 0), moves new state (bit 1) down. Values: 0→0, 1→0, 2→1, 3→1 |
| `matrix[r][0] == 0 or matrix[0][c] == 0` (set zeroes) | "Row r or col c was flagged for zeroing" | Either marker being 0 means some cell in that row or col was originally 0 |
| `(r // 3) * 3 + (c // 3)` (Sudoku box) | "Which of the 9 boxes does this cell belong to?" | Box row = `r//3` (0,1,2); box col = `c//3` (0,1,2); linear index = box_row * 3 + box_col |
