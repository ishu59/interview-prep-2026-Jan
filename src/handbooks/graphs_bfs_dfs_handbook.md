# The Complete BFS and DFS (Graphs) Handbook
> A template-based approach for mastering graph traversal in coding interviews

**Philosophy:** Graph problems are not about memorizing algorithms. It's about recognizing **which traversal pattern fits the problem structure** — BFS for shortest paths in unweighted graphs, DFS for exploring all possibilities, and knowing when special algorithms like Dijkstra or topological sort apply.

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

- **The City Map**: Nodes are intersections, edges are roads. Some roads are one-way (directed), some are two-way (undirected). Some roads have tolls (weighted). Every graph problem is a question about this city: "Can I get from A to B?", "What's the cheapest route?", "Are there any dead ends?"
- **Two Ways to Explore**: Imagine you're exploring a hedge maze. BFS = you send clones of yourself down every path simultaneously, one step at a time (you'll find the shortest path first). DFS = you personally walk one path all the way to its dead end, then backtrack and try the next (you'll explore everything but in no particular order of distance).

### No-Jargon Translation

- **Graph**: a collection of things connected by relationships
- **Node/vertex**: one thing
- **Edge**: a connection between two things
- **Adjacency list**: for each intersection, a list of which roads leave it
- **Visited set**: intersections you've already been to -- so you don't walk in circles
- **BFS**: explore level by level, like ripples in a pond
- **DFS**: explore path by path, going as deep as possible first
- **Topological sort**: ordering tasks so that every prerequisite comes before the thing that needs it
- **Cycle**: a path that leads back to where you started

### Mental Model

> "A graph is a city map, BFS explores it like a spreading wildfire (nearest first), and DFS explores it like a rat in a maze (one tunnel at a time, backtracking at dead ends)."

---

### Graph Representation

**Adjacency List (Most Common):**
```python
# List of lists
graph = [
    [1, 2],     # Node 0 connects to 1, 2
    [0, 3],     # Node 1 connects to 0, 3
    [0, 3],     # Node 2 connects to 0, 3
    [1, 2]      # Node 3 connects to 1, 2
]

# Dictionary (for sparse graphs or non-integer nodes)
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D'],
    'C': ['A', 'D'],
    'D': ['B', 'C']
}

# Building from edge list
edges = [[0, 1], [0, 2], [1, 3], [2, 3]]
graph = defaultdict(list)
for u, v in edges:
    graph[u].append(v)
    graph[v].append(u)  # For undirected
```

**Adjacency Matrix (Dense graphs):**
```python
# matrix[i][j] = 1 if edge from i to j
matrix = [
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 0]
]
```

### BFS vs DFS: The Key Differences

| Aspect | BFS | DFS |
|--------|-----|-----|
| Data Structure | Queue (FIFO) | Stack/Recursion (LIFO) |
| Explores | Level by level | As deep as possible |
| Shortest Path | Yes (unweighted) | No |
| Memory | O(width) | O(depth) |
| Best For | Shortest path, level order | Connectivity, cycles, backtracking |

### Visual Comparison

```
Graph:
    1 --- 2
    |     |
    3 --- 4 --- 5

BFS from 1:         DFS from 1:
Level 0: [1]        Visit: 1
Level 1: [2, 3]     Visit: 2 (go deep)
Level 2: [4]        Visit: 4 (go deep)
Level 3: [5]        Visit: 3 (backtrack, go other way)
                    Visit: 5 (go deep from 4)

BFS visits level by level
DFS goes as deep as possible before backtracking
```

### When to Use Which

**Use BFS when:**
- Finding **shortest path** in unweighted graph
- Need to process nodes **level by level**
- Finding nodes **within k distance**
- **Minimum** steps/moves problems

**Use DFS when:**
- Checking **connectivity** (is path possible?)
- Finding **all paths** or **counting paths**
- **Cycle detection**
- **Topological sort**
- Problems requiring **backtracking**
- Tree traversals

---

<a name="master-templates"></a>
## 2. The Master Templates

### Template A: BFS (Level-Order Traversal)

```python
from collections import deque

def bfs(graph: dict, start: int) -> list:
    """
    Standard BFS template.
    Returns nodes in BFS order.
    """
    visited = {start}
    queue = deque([start])
    result = []

    # Why does `while queue` terminate?
    # Each node enters the queue at most once (we check `visited` before adding).
    # So after at most V iterations, the queue is empty.
    while queue:
        node = queue.popleft()
        result.append(node)

        for neighbor in graph[node]:
            # Why check `visited` BEFORE adding to queue (not after popping)?
            # This is called "early marking." If we waited until popping,
            # the same node could be added to the queue by MULTIPLE neighbors,
            # wasting memory and time. Early marking ensures each node is
            # queued exactly once.
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return result
```

**BFS with Level Tracking:**
```python
def bfs_levels(graph: dict, start: int) -> list[list]:
    """
    BFS with explicit level tracking.
    Returns nodes grouped by level.
    """
    visited = {start}
    queue = deque([start])
    levels = []

    while queue:
        # Why snapshot len(queue) here?
        # The queue currently holds ALL nodes at the current level.
        # As we process them, we'll add next-level nodes to the queue.
        # Without the snapshot, we'd mix levels together.
        # The snapshot tells us "process exactly this many, then stop."
        level_size = len(queue)
        level = []

        for _ in range(level_size):
            node = queue.popleft()
            level.append(node)

            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        levels.append(level)

    return levels
```

---

### Template B: BFS Shortest Path

```python
from collections import deque

def bfs_shortest_path(graph: dict, start: int, end: int) -> int:
    """
    Find shortest path in unweighted graph.
    Returns distance (-1 if no path).
    """
    if start == end:
        return 0

    visited = {start}
    queue = deque([(start, 0)])  # (node, distance)

    while queue:
        node, dist = queue.popleft()

        for neighbor in graph[node]:
            if neighbor == end:
                return dist + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    return -1  # No path found
```

**BFS with Path Reconstruction:**
```python
def bfs_path(graph: dict, start: int, end: int) -> list:
    """
    Find shortest path and return the actual path.
    """
    if start == end:
        return [start]

    visited = {start}
    queue = deque([(start, [start])])  # (node, path)

    while queue:
        node, path = queue.popleft()

        for neighbor in graph[node]:
            if neighbor == end:
                return path + [neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return []  # No path
```

---

### Template C: DFS (Recursive)

```python
def dfs_recursive(graph: dict, start: int) -> list:
    """
    Standard recursive DFS.
    """
    visited = set()
    result = []

    def dfs(node):
        visited.add(node)
        result.append(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)

    dfs(start)
    return result
```

---

### Template D: DFS (Iterative)

```python
def dfs_iterative(graph: dict, start: int) -> list:
    """
    Iterative DFS using explicit stack.
    """
    visited = set()
    stack = [start]
    result = []

    while stack:
        node = stack.pop()
        # Why check AFTER popping (not before adding like BFS)?
        # With a stack, the same node can be pushed multiple times by
        # different neighbors. Checking at pop time is simpler and correct:
        # "Did someone already process me? If so, skip."
        # BFS can use early marking because FIFO order guarantees the first
        # enqueue finds the shortest path. DFS has no such guarantee.
        if node in visited:
            continue

        visited.add(node)
        result.append(node)

        # Add neighbors (reverse for same order as recursive)
        for neighbor in reversed(graph[node]):
            if neighbor not in visited:
                stack.append(neighbor)

    return result
```

---

### Template E: DFS for All Paths

```python
def all_paths(graph: dict, start: int, end: int) -> list[list]:
    """
    Find all paths from start to end.
    Uses backtracking.
    """
    result = []
    path = [start]

    def dfs(node):
        if node == end:
            result.append(path.copy())
            return

        for neighbor in graph[node]:
            # Why `not in path` instead of `not in visited`?
            # A visited set would permanently exclude nodes. But for
            # ALL-PATHS problems, we need to revisit nodes on DIFFERENT
            # paths. Using the current `path` means: "don't revisit on
            # THIS path (avoid cycles), but allow it on other paths."
            if neighbor not in path:
                path.append(neighbor)
                dfs(neighbor)
                path.pop()  # Backtrack: undo choice, try next neighbor

    dfs(start)
    return result
```

---

### Template F: Grid BFS (Multi-Source)

```python
from collections import deque

def grid_bfs(grid: list[list[int]], sources: list[tuple]) -> list[list[int]]:
    """
    Multi-source BFS on grid.
    Returns distance from nearest source for each cell.
    """
    rows, cols = len(grid), len(grid[0])
    distances = [[float('inf')] * cols for _ in range(rows)]
    queue = deque()

    # Initialize sources
    for r, c in sources:
        distances[r][c] = 0
        queue.append((r, c))

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    while queue:
        r, c = queue.popleft()

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                # Why `> distances[r][c] + 1` instead of a visited set?
                # Multi-source BFS may reach the same cell from different sources.
                # We only want to process it if we found a SHORTER path.
                # This also serves as the "visited" check — if distance is
                # already optimal, the condition is false and we skip it.
                if distances[nr][nc] > distances[r][c] + 1:
                    distances[nr][nc] = distances[r][c] + 1
                    queue.append((nr, nc))

    return distances
```

---

### Template G: Topological Sort (Kahn's Algorithm - BFS)

```python
from collections import deque, defaultdict

def topological_sort(num_nodes: int, edges: list[list[int]]) -> list:
    """
    Topological sort using BFS (Kahn's algorithm).
    edges[i] = [a, b] means a -> b (a must come before b).
    Returns sorted order, or empty list if cycle exists.
    """
    # Build graph and in-degree count
    graph = defaultdict(list)
    in_degree = [0] * num_nodes

    for a, b in edges:
        graph[a].append(b)
        in_degree[b] += 1

    # Start with nodes having no prerequisites (in-degree 0).
    # These are "ready" — nothing blocks them.
    # Analogy: courses with no prerequisites can be taken immediately.
    queue = deque([i for i in range(num_nodes) if in_degree[i] == 0])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)  # "Take" this course

        for neighbor in graph[node]:
            # "Completing" this node satisfies one prerequisite for neighbor.
            in_degree[neighbor] -= 1
            # When ALL prerequisites are satisfied (in-degree reaches 0),
            # the neighbor becomes "ready" and can enter the queue.
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Why does len(result) != num_nodes mean a cycle?
    # If there's a cycle, nodes in it can NEVER reach in-degree 0
    # (they're all waiting on each other). So they never enter the queue
    # and never appear in the result. Missing nodes = cycle exists.
    return result if len(result) == num_nodes else []
```

---

### Template H: Topological Sort (DFS)

```python
def topological_sort_dfs(num_nodes: int, edges: list[list[int]]) -> list:
    """
    Topological sort using DFS.
    Uses post-order traversal, then reverse.
    """
    graph = defaultdict(list)
    for a, b in edges:
        graph[a].append(b)

    # The 3-color system:
    # WHITE = unvisited (haven't seen this node yet)
    # GRAY  = in progress (currently in the DFS call stack — we started
    #         exploring it but haven't finished all its descendants)
    # BLACK = fully processed (all descendants explored)
    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * num_nodes
    result = []
    has_cycle = False

    def dfs(node):
        nonlocal has_cycle
        if has_cycle:
            return

        color[node] = GRAY  # "I'm exploring this node's subtree now"

        for neighbor in graph[node]:
            if color[neighbor] == GRAY:
                # GRAY neighbor = back edge = CYCLE!
                # This neighbor is an ANCESTOR in our current DFS path.
                # We left it to explore deeper, and now we've circled back.
                # Like finding footprints ahead of you on a one-way trail.
                has_cycle = True
                return
            if color[neighbor] == WHITE:  # Unvisited: explore it
                dfs(neighbor)

        color[node] = BLACK  # "Done with this node and all its descendants"
        result.append(node)  # Post-order: add after all descendants processed

    for i in range(num_nodes):
        if color[i] == WHITE:
            dfs(i)

    return result[::-1] if not has_cycle else []
```

---

### Template I: Dijkstra's Algorithm (Weighted Shortest Path)

```python
import heapq
from collections import defaultdict

def dijkstra(graph: dict, start: int, end: int) -> int:
    """
    Shortest path in weighted graph with non-negative weights.
    graph[node] = [(neighbor, weight), ...]
    """
    distances = defaultdict(lambda: float('inf'))
    distances[start] = 0
    heap = [(0, start)]  # (distance, node)
    visited = set()

    while heap:
        dist, node = heapq.heappop(heap)

        # Why skip if already visited?
        # The heap may contain STALE entries for this node (older, longer paths).
        # Since we process shortest-first, the FIRST time we pop a node is
        # guaranteed to be its shortest distance. All later pops are outdated.
        if node in visited:
            continue
        visited.add(node)

        if node == end:
            return dist

        for neighbor, weight in graph[node]:
            if neighbor not in visited:
                new_dist = dist + weight
                # Relaxation: "Is this new path shorter than the best known?"
                # If yes, update and push to heap. We don't remove the old
                # heap entry (expensive), instead we rely on the visited check
                # above to skip it when it's eventually popped.
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    heapq.heappush(heap, (new_dist, neighbor))

    return -1  # No path
```

---

### Quick Decision Matrix

| Problem Type | Algorithm | Template |
|--------------|-----------|----------|
| Shortest path (unweighted) | BFS | B |
| Shortest path (weighted) | Dijkstra | I |
| Connected components | DFS/BFS | C or A |
| Cycle detection (directed) | DFS with colors | H |
| Cycle detection (undirected) | DFS with parent | Custom |
| Topological sort | BFS (Kahn's) or DFS | G or H |
| All paths | DFS backtracking | E |
| Level-order processing | BFS | A |
| Grid traversal | BFS/DFS | F |

---

<a name="pattern-guide"></a>
## 3. Pattern Classification Guide

### Category 1: Traversal and Connectivity
- Visit all nodes, check if connected
- Count connected components
- **Templates A, C, D**

### Category 2: Shortest Path
- Minimum moves/steps in unweighted graph
- Distance to target
- **Templates B, I (for weighted)**

### Category 3: Cycle Detection
- Detect cycles in directed/undirected graphs
- **Template H (DFS with colors)**

### Category 4: Topological Ordering
- Task scheduling with prerequisites
- Course ordering
- **Templates G, H**

### Category 5: Grid Problems
- Matrix traversal
- Island problems
- Rotting oranges type
- **Template F**

### Category 6: Path Finding
- All paths from A to B
- Path with constraints
- **Template E**

---

<a name="patterns"></a>
## 4. Complete Pattern Library

### PATTERN 1: Connected Components

---

#### Pattern 1A: Number of Islands

**Problem:** LeetCode 200 - Count islands in grid

```python
def numIslands(grid: list[list[str]]) -> int:
    if not grid:
        return 0

    rows, cols = len(grid), len(grid[0])
    count = 0

    def dfs(r, c):
        # Boundary check: gone off the grid edges.
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return
        # Not land (either water '0' or already-visited-land marked as '0').
        if grid[r][c] != '1':
            return

        # Mark visited IN-PLACE by changing '1' to '0'.
        # Why not use a separate visited set? Saves O(m*n) space.
        # Tradeoff: mutates the input. If that's not allowed, use visited set.
        grid[r][c] = '0'

        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                # Found an unvisited land cell = new island!
                # Each DFS call "sinks" the entire connected island,
                # so next time we find a '1', it must be a DIFFERENT island.
                count += 1
                dfs(r, c)

    return count
```

**BFS version:**
```python
from collections import deque

def numIslands_bfs(grid: list[list[str]]) -> int:
    if not grid:
        return 0

    rows, cols = len(grid), len(grid[0])
    count = 0

    def bfs(start_r, start_c):
        queue = deque([(start_r, start_c)])
        grid[start_r][start_c] = '0'

        while queue:
            r, c = queue.popleft()
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == '1':
                    grid[nr][nc] = '0'
                    queue.append((nr, nc))

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                count += 1
                bfs(r, c)

    return count
```

---

#### Pattern 1B: Number of Connected Components

**Problem:** LeetCode 323 - Count connected components in undirected graph

```python
def countComponents(n: int, edges: list[list[int]]) -> int:
    # Build adjacency list
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    visited = set()
    count = 0

    def dfs(node):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)

    for i in range(n):
        if i not in visited:
            count += 1
            dfs(i)

    return count
```

---

#### Pattern 1C: Max Area of Island

**Problem:** LeetCode 695 - Find largest island area

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

        area += dfs(r + 1, c)
        area += dfs(r - 1, c)
        area += dfs(r, c + 1)
        area += dfs(r, c - 1)

        return area

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                max_area = max(max_area, dfs(r, c))

    return max_area
```

---

### PATTERN 2: Shortest Path (BFS)

---

#### Pattern 2A: Shortest Path in Binary Matrix

**Problem:** LeetCode 1091 - Shortest path from top-left to bottom-right

```python
from collections import deque

def shortestPathBinaryMatrix(grid: list[list[int]]) -> int:
    n = len(grid)
    # Why check both corners? If START or END is blocked (1),
    # no path can possibly exist — fail fast before BFS.
    if grid[0][0] == 1 or grid[n-1][n-1] == 1:
        return -1

    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]

    queue = deque([(0, 0, 1)])  # (row, col, path_length)
    grid[0][0] = 1  # Mark visited

    while queue:
        r, c, length = queue.popleft()

        if r == n - 1 and c == n - 1:
            return length

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            # Why `grid[nr][nc] == 0`? Only unvisited open cells.
            # We mark visited by setting to 1 (same as blocked),
            # so this check handles BOTH "is it open?" and "is it unvisited?"
            if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] == 0:
                grid[nr][nc] = 1  # Mark visited (early marking, like BFS Template A)
                queue.append((nr, nc, length + 1))

    return -1
```

---

#### Pattern 2B: Word Ladder

**Problem:** LeetCode 127 - Transform word to target one letter at a time

```python
from collections import deque, defaultdict

def ladderLength(beginWord: str, endWord: str, wordList: list[str]) -> int:
    word_set = set(wordList)
    if endWord not in word_set:
        return 0

    # BFS
    queue = deque([(beginWord, 1)])
    visited = {beginWord}

    while queue:
        word, length = queue.popleft()

        if word == endWord:
            return length

        # Try all single-letter transformations
        # Why iterate 26 letters × word length, instead of checking all pairs?
        # Checking all pairs is O(n²). This is O(26 × L × n) which is better
        # when word list is large. Each transformation is O(L) string ops.
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                new_word = word[:i] + c + word[i+1:]
                # Why check `word_set` AND `visited`?
                # word_set = valid dictionary words. visited = already queued.
                # Both must pass: word must exist AND not already processed.
                if new_word in word_set and new_word not in visited:
                    visited.add(new_word)
                    queue.append((new_word, length + 1))

    return 0
```

**Optimized with pattern matching:**
```python
def ladderLength_optimized(beginWord: str, endWord: str, wordList: list[str]) -> int:
    if endWord not in wordList:
        return 0

    # Build pattern graph: h*t -> [hat, hot, hit]
    patterns = defaultdict(list)
    for word in wordList:
        for i in range(len(word)):
            pattern = word[:i] + '*' + word[i+1:]
            patterns[pattern].append(word)

    queue = deque([(beginWord, 1)])
    visited = {beginWord}

    while queue:
        word, length = queue.popleft()

        for i in range(len(word)):
            pattern = word[:i] + '*' + word[i+1:]
            for neighbor in patterns[pattern]:
                if neighbor == endWord:
                    return length + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, length + 1))

    return 0
```

---

#### Pattern 2C: Rotting Oranges (Multi-Source BFS)

**Problem:** LeetCode 994 - Minimum time for all oranges to rot

```python
from collections import deque

def orangesRotting(grid: list[list[int]]) -> int:
    rows, cols = len(grid), len(grid[0])
    queue = deque()
    fresh = 0

    # Initialize: find all rotten oranges and count fresh
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                queue.append((r, c, 0))
            elif grid[r][c] == 1:
                fresh += 1

    # Why return 0 if no fresh oranges? Nothing to rot → 0 minutes.
    # Edge case: grid might be all empty or all already rotten.
    if fresh == 0:
        return 0

    max_time = 0
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    while queue:
        r, c, time = queue.popleft()
        max_time = max(max_time, time)

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            # Why `grid[nr][nc] == 1`? Only fresh oranges can be rotted.
            # Rotten (2) or empty (0) cells are skipped.
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                grid[nr][nc] = 2  # Rot it (also serves as "visited" mark)
                fresh -= 1
                queue.append((nr, nc, time + 1))

    # Why `fresh == 0`? If any fresh orange remains unreachable
    # from all rotten sources, it's impossible → return -1.
    return max_time if fresh == 0 else -1
```

---

#### Pattern 2D: 01 Matrix (Distance to Nearest 0)

**Problem:** LeetCode 542 - Distance from each cell to nearest 0

```python
from collections import deque

def updateMatrix(mat: list[list[int]]) -> list[list[int]]:
    rows, cols = len(mat), len(mat[0])
    distances = [[float('inf')] * cols for _ in range(rows)]
    queue = deque()

    # Initialize: all 0s have distance 0
    for r in range(rows):
        for c in range(cols):
            if mat[r][c] == 0:
                distances[r][c] = 0
                queue.append((r, c))

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    while queue:
        r, c = queue.popleft()

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if distances[nr][nc] > distances[r][c] + 1:
                    distances[nr][nc] = distances[r][c] + 1
                    queue.append((nr, nc))

    return distances
```

---

### PATTERN 3: Cycle Detection

---

#### Pattern 3A: Course Schedule (Detect Cycle in Directed Graph)

**Problem:** LeetCode 207 - Can finish all courses?

```python
def canFinish(numCourses: int, prerequisites: list[list[int]]) -> bool:
    graph = defaultdict(list)
    for course, prereq in prerequisites:
        graph[prereq].append(course)

    # 3-state system (same idea as WHITE/GRAY/BLACK above):
    # 0 = unvisited, 1 = visiting (in current DFS path), 2 = fully visited
    state = [0] * numCourses

    def has_cycle(course):
        if state[course] == 1:
            # Currently in our DFS path — we've circled back! Cycle!
            return True
        if state[course] == 2:
            # Already fully explored in a previous DFS — no cycle from here.
            # This is a KEY optimization: without it, we'd re-explore nodes.
            return False

        state[course] = 1  # Mark as "currently exploring"

        for next_course in graph[course]:
            if has_cycle(next_course):
                return True

        state[course] = 2  # Mark as visited
        return False

    for course in range(numCourses):
        if has_cycle(course):
            return False

    return True
```

**Using Kahn's algorithm (BFS):**
```python
def canFinish_bfs(numCourses: int, prerequisites: list[list[int]]) -> bool:
    graph = defaultdict(list)
    in_degree = [0] * numCourses

    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1

    queue = deque([i for i in range(numCourses) if in_degree[i] == 0])
    count = 0

    while queue:
        course = queue.popleft()
        count += 1

        for next_course in graph[course]:
            in_degree[next_course] -= 1
            if in_degree[next_course] == 0:
                queue.append(next_course)

    return count == numCourses
```

---

#### Pattern 3B: Cycle in Undirected Graph

**Problem:** Detect cycle in undirected graph

```python
def hasCycle(n: int, edges: list[list[int]]) -> bool:
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    visited = set()

    def dfs(node, parent):
        visited.add(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                if dfs(neighbor, node):
                    return True
            elif neighbor != parent:
                # Why `!= parent`? In an undirected graph, edge (A, B)
                # appears in BOTH adjacency lists. When we DFS from A to B,
                # B's neighbor list includes A. That's not a cycle — it's just
                # the edge we came from. We only have a cycle if we reach a
                # VISITED node that ISN'T our immediate parent.
                return True

        return False

    for i in range(n):
        if i not in visited:
            if dfs(i, -1):
                return True

    return False
```

---

### PATTERN 4: Topological Sort

---

#### Pattern 4A: Course Schedule II (Find Order)

**Problem:** LeetCode 210 - Return course order

```python
from collections import deque, defaultdict

def findOrder(numCourses: int, prerequisites: list[list[int]]) -> list[int]:
    graph = defaultdict(list)
    in_degree = [0] * numCourses

    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1

    queue = deque([i for i in range(numCourses) if in_degree[i] == 0])
    order = []

    while queue:
        course = queue.popleft()
        order.append(course)

        for next_course in graph[course]:
            in_degree[next_course] -= 1
            if in_degree[next_course] == 0:
                queue.append(next_course)

    return order if len(order) == numCourses else []
```

---

#### Pattern 4B: Alien Dictionary

**Problem:** LeetCode 269 - Determine order of letters in alien language

```python
from collections import deque, defaultdict

def alienOrder(words: list[str]) -> str:
    # Build graph from word comparisons
    graph = defaultdict(set)
    in_degree = {c: 0 for word in words for c in word}

    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]

        # Invalid: prefix comes after longer word
        if len(w1) > len(w2) and w1.startswith(w2):
            return ""

        # Find first differing character
        for c1, c2 in zip(w1, w2):
            if c1 != c2:
                if c2 not in graph[c1]:
                    graph[c1].add(c2)
                    in_degree[c2] += 1
                break

    # Topological sort
    queue = deque([c for c in in_degree if in_degree[c] == 0])
    result = []

    while queue:
        c = queue.popleft()
        result.append(c)

        for neighbor in graph[c]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return ''.join(result) if len(result) == len(in_degree) else ""
```

---

### PATTERN 5: Graph Coloring / Bipartite

---

#### Pattern 5A: Is Graph Bipartite?

**Problem:** LeetCode 785 - Check if graph is bipartite

```python
def isBipartite(graph: list[list[int]]) -> bool:
    n = len(graph)
    # 0 = uncolored, 1 or -1 = two "team" colors.
    # Bipartite = can split all nodes into two groups where
    # every edge connects different groups (like a two-team game).
    color = [0] * n

    def bfs(start):
        queue = deque([start])
        color[start] = 1  # Assign first team

        while queue:
            node = queue.popleft()

            for neighbor in graph[node]:
                if color[neighbor] == 0:
                    # Uncolored: assign OPPOSITE color (other team)
                    color[neighbor] = -color[node]
                    queue.append(neighbor)
                elif color[neighbor] == color[node]:
                    # SAME color as me = conflict! Two connected nodes
                    # on the same team → not bipartite.
                    return False

        return True

    # Why loop over all nodes? Graph may be disconnected.
    # Each component must be independently bipartite.
    for i in range(n):
        if color[i] == 0:
            if not bfs(i):
                return False

    return True
```

**DFS version:**
```python
def isBipartite_dfs(graph: list[list[int]]) -> bool:
    n = len(graph)
    color = [0] * n

    def dfs(node, c):
        color[node] = c

        for neighbor in graph[node]:
            if color[neighbor] == c:
                # Same color = conflict → not bipartite
                return False
            if color[neighbor] == 0 and not dfs(neighbor, -c):
                # Uncolored: try assigning opposite color (-c).
                # If that fails (returns False), conflict found deeper.
                return False

        return True

    for i in range(n):
        if color[i] == 0:
            if not dfs(i, 1):
                return False

    return True
```

---

### PATTERN 6: Shortest Path with Weights

---

#### Pattern 6A: Network Delay Time (Dijkstra)

**Problem:** LeetCode 743 - Time for signal to reach all nodes

```python
import heapq
from collections import defaultdict

def networkDelayTime(times: list[list[int]], n: int, k: int) -> int:
    graph = defaultdict(list)
    for u, v, w in times:
        graph[u].append((v, w))

    distances = {i: float('inf') for i in range(1, n + 1)}
    distances[k] = 0
    heap = [(0, k)]

    while heap:
        dist, node = heapq.heappop(heap)

        # Why `dist > distances[node]`? Skip stale heap entries.
        # We may have pushed this node multiple times with decreasing
        # distances. Only the first pop (smallest dist) matters.
        if dist > distances[node]:
            continue

        for neighbor, weight in graph[node]:
            new_dist = dist + weight
            # Relaxation: found a shorter path to neighbor?
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heapq.heappush(heap, (new_dist, neighbor))

    max_dist = max(distances.values())
    # Why check for inf? If any node is unreachable (distance = inf),
    # the signal can't reach all nodes → return -1.
    return max_dist if max_dist < float('inf') else -1
```

---

#### Pattern 6B: Cheapest Flights Within K Stops

**Problem:** LeetCode 787 - Cheapest flight with at most k stops

```python
import heapq
from collections import defaultdict

def findCheapestPrice(n: int, flights: list[list[int]], src: int, dst: int, k: int) -> int:
    graph = defaultdict(list)
    for u, v, price in flights:
        graph[u].append((v, price))

    # (cost, stops, node)
    heap = [(0, 0, src)]
    # Track minimum stops to reach each node
    best = {}

    while heap:
        cost, stops, node = heapq.heappop(heap)

        if node == dst:
            return cost

        # Why `stops > k`? We've used more intermediate stops than allowed.
        # k stops = k+1 flights. This prunes paths that are too long.
        if stops > k:
            continue

        # Why track `best` stops instead of just visited?
        # Unlike standard Dijkstra, a longer path with fewer stops might
        # lead to a cheaper overall route. We only skip if we've already
        # reached this node with FEWER stops (strictly better).
        if node in best and best[node] <= stops:
            continue
        best[node] = stops

        for neighbor, price in graph[node]:
            heapq.heappush(heap, (cost + price, stops + 1, neighbor))

    return -1
```

---

### PATTERN 7: Special Graph Problems

---

#### Pattern 7A: Clone Graph

**Problem:** LeetCode 133 - Deep copy of graph

```python
class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors else []

def cloneGraph(node: Node) -> Node:
    if not node:
        return None

    # Map: original node → its clone. Serves double duty:
    # 1. Prevents infinite loops (acts as "visited")
    # 2. Lets us reuse already-cloned nodes for neighbor links
    cloned = {}

    def dfs(original):
        # Why check `original in cloned`?
        # If already cloned, return the existing copy.
        # Without this, cycles in the graph cause infinite recursion.
        if original in cloned:
            return cloned[original]

        copy = Node(original.val)
        # MUST register clone BEFORE recursing into neighbors.
        # If we wait until after, a cycle would recurse back here
        # and not find the clone, causing infinite recursion.
        cloned[original] = copy

        for neighbor in original.neighbors:
            copy.neighbors.append(dfs(neighbor))

        return copy

    return dfs(node)
```

---

#### Pattern 7B: Pacific Atlantic Water Flow

**Problem:** LeetCode 417 - Cells that can flow to both oceans

```python
def pacificAtlantic(heights: list[list[int]]) -> list[list[int]]:
    if not heights:
        return []

    rows, cols = len(heights), len(heights[0])
    pacific = set()
    atlantic = set()

    # KEY TRICK: Instead of "from each cell, can water reach the ocean?"
    # (expensive), we reverse it: "from each ocean, which cells can flow TO it?"
    # Water flows downhill, so reverse-flow = go UPHILL from ocean edges.
    def dfs(r, c, visited, prev_height):
        if (r, c) in visited:
            return
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return
        # Why `< prev_height`? We're going UPHILL (reverse flow).
        # Water flows from high to low, so tracing backward means
        # we can only move to cells >= current height.
        if heights[r][c] < prev_height:
            return

        visited.add((r, c))

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            dfs(r + dr, c + dc, visited, heights[r][c])

    # Start from Pacific (top and left edges)
    for c in range(cols):
        dfs(0, c, pacific, 0)
    for r in range(rows):
        dfs(r, 0, pacific, 0)

    # Start from Atlantic (bottom and right edges)
    for c in range(cols):
        dfs(rows - 1, c, atlantic, 0)
    for r in range(rows):
        dfs(r, cols - 1, atlantic, 0)

    # Intersection: cells reachable from BOTH oceans = cells where
    # water can flow to both Pacific AND Atlantic.
    return list(pacific & atlantic)
```

---

#### Pattern 7C: Surrounded Regions

**Problem:** LeetCode 130 - Capture surrounded regions

```python
def solve(board: list[list[str]]) -> None:
    if not board:
        return

    rows, cols = len(board), len(board[0])

    # KEY INSIGHT: Instead of "find surrounded O's" (hard),
    # flip the question: "find UN-surrounded O's" (easy — they touch an edge).
    # Mark edge-connected O's as safe, then capture the rest.
    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return
        # Why `!= 'O'`? Stop if it's 'X' (wall) or 'E' (already marked safe).
        if board[r][c] != 'O':
            return

        board[r][c] = 'E'  # 'E' = Escaped (safe, connected to edge)

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            dfs(r + dr, c + dc)

    # Start DFS from ALL edge cells — any O connected to an edge escapes.
    for r in range(rows):
        dfs(r, 0)         # Left edge
        dfs(r, cols - 1)  # Right edge
    for c in range(cols):
        dfs(0, c)         # Top edge
        dfs(rows - 1, c)  # Bottom edge

    # Final pass — three-way conversion:
    # O → X (not connected to edge = captured/surrounded)
    # E → O (was connected to edge = restore to original)
    for r in range(rows):
        for c in range(cols):
            if board[r][c] == 'O':
                board[r][c] = 'X'
            elif board[r][c] == 'E':
                board[r][c] = 'O'
```

---

<a name="post-processing"></a>
## 5. Post-Processing Reference

| Problem Type | Return Value | Notes |
|--------------|--------------|-------|
| **Shortest path** | Distance or -1 | Check if target reached |
| **Path exists** | Boolean | Simple visited check |
| **All paths** | List of paths | Backtracking |
| **Connected components** | Count | Each DFS/BFS from unvisited |
| **Topological sort** | Order or empty | Empty if cycle |
| **Bipartite** | Boolean | Check for conflicts |

---

<a name="pitfalls"></a>
## 6. Common Pitfalls & Solutions

### Pitfall 1: Not Marking as Visited Before Adding to Queue

```python
# WRONG: Same node added multiple times
for neighbor in graph[node]:
    if neighbor not in visited:
        queue.append(neighbor)
        # visited.add(neighbor) should be HERE!
```

**Solution:** Mark visited when adding to queue, not when processing
```python
for neighbor in graph[node]:
    if neighbor not in visited:
        visited.add(neighbor)  # Mark immediately
        queue.append(neighbor)
```

---

### Pitfall 2: Using List as Visited for Large Graphs

```python
# SLOW: O(n) lookup
visited = []
if node not in visited:  # O(n)
```

**Solution:** Use set
```python
visited = set()
if node not in visited:  # O(1)
```

---

### Pitfall 3: Forgetting Disconnected Components

```python
# WRONG: Only explores from node 0
dfs(0)
```

**Solution:** Iterate over all nodes
```python
for i in range(n):
    if i not in visited:
        dfs(i)
```

---

### Pitfall 4: Stack Overflow in DFS

**Problem:** Very deep graphs cause recursion limit exceeded

**Solution:** Use iterative DFS or increase limit
```python
import sys
sys.setrecursionlimit(10000)
# Or use iterative version
```

---

### Pitfall 5: Modifying Grid During BFS Without Care

```python
# Can cause issues if checking before marking
if grid[nr][nc] == 1:
    queue.append((nr, nc))
    # Should mark here, not later!
```

---

### Pitfall 6: Wrong Direction Array

```python
# WRONG: Diagonal directions when only cardinal needed
directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1)]
```

**Solution:** Match problem requirements
```python
# Cardinal only
directions = [(0,1), (1,0), (0,-1), (-1,0)]
# With diagonals
directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
```

---

<a name="recognition"></a>
## 7. Problem Recognition Framework

### Step 1: Is it a Graph Problem?

**Graph indicators:**
- Explicit edges/connections
- Grid (implicit graph)
- Networks, relationships
- State transitions
- Dependencies

### Step 2: What Type of Graph?

| Type | Characteristics |
|------|----------------|
| Directed | One-way edges |
| Undirected | Two-way edges |
| Weighted | Edges have costs |
| Unweighted | All edges equal |
| DAG | Directed, no cycles |
| Grid | 2D matrix traversal |

### Step 3: What's Being Asked?

| Question | Algorithm |
|----------|-----------|
| Shortest path (unweighted) | BFS |
| Shortest path (weighted) | Dijkstra |
| Any path exists? | DFS or BFS |
| All paths | DFS backtracking |
| Connected components | DFS/BFS from each unvisited |
| Cycle detection | DFS with colors |
| Ordering with dependencies | Topological sort |
| Two-colorable? | BFS/DFS bipartite check |

### Decision Tree

```
               Graph Problem
                    ↓
         ┌─────────┴─────────┐
     Shortest Path?        Other
         ↓                    ↓
    ┌────┴────┐        ┌──────┼──────┐
Weighted  Unweighted   Cycle  Order  Connect
    ↓         ↓          ↓      ↓      ↓
Dijkstra    BFS        DFS   TopoSort DFS/BFS
```

---

<a name="checklist"></a>
## 8. Interview Preparation Checklist

### Before the Interview

**Master the fundamentals:**
- [ ] Can write BFS template from memory
- [ ] Can write DFS template (both recursive and iterative)
- [ ] Know when to use BFS vs DFS
- [ ] Understand Dijkstra's algorithm
- [ ] Can do topological sort (both BFS and DFS versions)

**Practice pattern recognition:**
- [ ] Can identify graph problems quickly
- [ ] Know which algorithm fits which problem
- [ ] Understand cycle detection approaches

**Know the patterns:**
- [ ] Connected components
- [ ] Shortest path (unweighted and weighted)
- [ ] Cycle detection
- [ ] Topological sort
- [ ] Bipartite checking
- [ ] Grid traversal

**Common problems solved:**
- [ ] LC 200: Number of Islands
- [ ] LC 207/210: Course Schedule
- [ ] LC 127: Word Ladder
- [ ] LC 133: Clone Graph
- [ ] LC 743: Network Delay Time
- [ ] LC 785: Is Graph Bipartite

### During the Interview

**1. Clarify (30 seconds)**
- Directed or undirected?
- Weighted or unweighted?
- Can have cycles?
- What to return?

**2. Identify pattern (30 seconds)**
- Shortest path → BFS (unweighted) or Dijkstra
- Connectivity → DFS
- Dependencies → Topological sort

**3. Code (3-4 minutes)**
- Build adjacency list
- Initialize visited set
- Choose BFS queue or DFS stack
- Process and track result

**4. Test (1-2 minutes)**
- Empty graph
- Single node
- Disconnected graph
- Cycle (if applicable)

**5. Analyze (30 seconds)**
- Time: O(V + E) for traversal
- Space: O(V) for visited

---

## 9. Quick Reference Cards

### BFS Template
```python
def bfs(start):
    visited = {start}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
```

### DFS Template
```python
def dfs(node, visited):
    visited.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(neighbor, visited)
```

### Dijkstra Template
```python
def dijkstra(start):
    dist = {start: 0}
    heap = [(0, start)]
    while heap:
        d, node = heappop(heap)
        if d > dist.get(node, inf): continue
        for neighbor, weight in graph[node]:
            new_d = d + weight
            if new_d < dist.get(neighbor, inf):
                dist[neighbor] = new_d
                heappush(heap, (new_d, neighbor))
    return dist
```

---

## 10. Complexity Reference

| Algorithm | Time | Space |
|-----------|------|-------|
| BFS/DFS | O(V + E) | O(V) |
| Dijkstra | O((V + E) log V) | O(V) |
| Topological Sort | O(V + E) | O(V) |
| Grid Traversal | O(R × C) | O(R × C) |

Where V = vertices, E = edges, R = rows, C = columns

---

## Final Thoughts

**Remember:**
1. BFS = shortest path in unweighted graph
2. DFS = explore all possibilities, detect cycles
3. Mark visited **before** adding to queue (BFS)
4. Use colors (unvisited/visiting/visited) for cycle detection in directed graphs
5. Topological sort only works on DAGs

**When stuck:**
1. Draw the graph and trace through
2. Ask: "Do I need shortest path or just any path?"
3. Consider: "Is this really a graph problem?"
4. Check: "Am I handling all connected components?"

---

## Appendix: Practice Problem Set

### Easy
- 733. Flood Fill
- 997. Find the Town Judge
- 1971. Find if Path Exists in Graph

### Medium
- 200. Number of Islands
- 207. Course Schedule
- 210. Course Schedule II
- 133. Clone Graph
- 417. Pacific Atlantic Water Flow
- 542. 01 Matrix
- 695. Max Area of Island
- 743. Network Delay Time
- 785. Is Graph Bipartite
- 994. Rotting Oranges
- 1091. Shortest Path in Binary Matrix

### Hard
- 127. Word Ladder
- 269. Alien Dictionary
- 329. Longest Increasing Path in Matrix
- 787. Cheapest Flights Within K Stops
- 1192. Critical Connections in a Network

**Recommended Practice Order:**
1. Start with LC 200 (basic DFS/BFS)
2. Practice LC 207, 210 (topological sort)
3. Do LC 994, 542 (multi-source BFS)
4. Master LC 743 (Dijkstra)
5. Try LC 127 (advanced BFS)

---

## Appendix: Conditional Quick Reference

### BFS Conditionals

| Condition | Where Used | Why |
|-----------|-----------|-----|
| `neighbor not in visited` (before enqueue) | BFS Template A | Early marking: each node queued exactly once, saves memory |
| `level_size = len(queue)` | BFS with Levels | Snapshot separates current level from next level's nodes |
| `grid[nr][nc] == 0` | Binary Matrix BFS | Checks both "open cell" and "unvisited" (visited cells marked as 1) |
| `grid[nr][nc] == 1` | Rotting Oranges | Only fresh oranges can be rotted; rotten/empty are skipped |
| `fresh == 0` | Rotting Oranges (end) | If any fresh orange is unreachable, return -1 |
| `distances[nr][nc] > distances[r][c] + 1` | Multi-Source BFS / 01 Matrix | Doubles as visited check: only process if found shorter path |

### DFS Conditionals

| Condition | Where Used | Why |
|-----------|-----------|-----|
| `node in visited` (after pop) | Iterative DFS | Late marking: stack may contain duplicates; skip already-processed |
| `neighbor not in path` | All Paths DFS | Allow revisiting on different paths (unlike permanent visited set) |
| `neighbor != parent` | Undirected Cycle | Don't count the edge we came from as a cycle |
| `color[neighbor] == GRAY` | DFS Topo Sort | Back edge to ancestor in current path = cycle detected |
| `color[neighbor] == color[node]` | Bipartite Check | Same-team conflict = not bipartite |

### Shortest Path Conditionals

| Condition | Where Used | Why |
|-----------|-----------|-----|
| `dist > distances[node]` | Dijkstra | Skip stale heap entries; first pop is always shortest |
| `new_dist < distances[neighbor]` | Dijkstra Relaxation | Only push if found a strictly shorter path |
| `stops > k` | Cheapest Flights | Prune paths exceeding the stop limit |
| `best[node] <= stops` | Cheapest Flights | Skip if already reached with fewer stops |

### Special Graph Conditionals

| Condition | Where Used | Why |
|-----------|-----------|-----|
| `original in cloned` | Clone Graph | Prevents infinite loops on cycles; reuses existing clones |
| `heights[r][c] < prev_height` | Pacific Atlantic | Reverse flow: only go uphill (water flows downhill) |
| `board[r][c] != 'O'` | Surrounded Regions | Stop at walls (X) or already-safe cells (E) |
| `in_degree[neighbor] == 0` | Kahn's Topo Sort | All prerequisites done → neighbor is ready |
| `len(result) != num_nodes` | Kahn's Cycle Check | Nodes stuck in cycles never reach in-degree 0 |

Good luck with your interview preparation!
