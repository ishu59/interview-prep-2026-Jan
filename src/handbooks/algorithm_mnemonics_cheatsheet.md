# Algorithm Mnemonics & Memory Tricks for Coding Interviews

A comprehensive collection of mnemonics, memory aids, trigger words, and quick-recall rules
for all 17 interview patterns -- organized from foundational to advanced.

---

## Table of Contents

**Array & String Fundamentals**
1. [Two Pointers](#two-pointers)
2. [Sliding Window](#sliding-window)
3. [Binary Search](#binary-search)
4. [Monotonic Stack / Queue](#monotonic-stack)

**Sorting & Selection**
5. [Intervals](#intervals)
6. [Heaps / Priority Queues](#heaps)
7. [Greedy Algorithms](#greedy)

**Tree & Graph**
8. [Trees](#trees)
9. [Tries](#tries)
10. [Graphs (BFS / DFS)](#graphs)
11. [Matrix / Grid](#matrix)
12. [Union Find](#union-find)

**Search & Optimization**
13. [Backtracking](#backtracking)
14. [Divide and Conquer](#divide-and-conquer)
15. [Dynamic Programming](#dp)
16. [Advanced DP](#advanced-dp)
17. [Bit Manipulation](#bit-manipulation)

**Quick Reference**
- [Cross-Pattern Decision Table](#decision-table)
- [Constraint-Based Complexity Guide](#constraints)
- [Spaced Repetition Schedule](#spaced-repetition)

---

<a name="two-pointers"></a>
## 1. TWO POINTERS

### The "Pinch, Slide, or Parallel" Mental Model
**Mnemonic: "Pinch from ends, Slide together, or walk Parallel arrays"**

- **PINCH (Opposite Ends)**: Pointers start at opposite ends, move inward like a pincer. Use for: pair sum in sorted array, palindrome check, container with most water.
- **SLIDE (Same Direction)**: Both pointers start from one end, move the same way at different speeds. Use for: remove duplicates, fast/slow cycle detection, linked list middle.
- **PARALLEL (Two Arrays)**: One pointer per array, advance based on comparison. Use for: merge sorted arrays, intersection of sorted arrays, shortest word distance.

### Trigger Words
| Keyword | Think Two Pointers |
|---|---|
| "Sorted array" + "find pair" | Opposite-end pointers |
| "Palindrome" | Compare from both ends |
| "Remove duplicates in-place" | Slow/fast same direction |
| "Container with most water" | Shrink from the shorter side |
| "3Sum / 4Sum" | Fix one, two-pointer the rest |

### The Golden Rule
**"Two pointers reduces O(n²) to O(n) when data has a sorted or structural constraint."**

If you're about to write a nested loop on sorted data (or data with another constraint like a linked list), stop and think two pointers.

> Sources: [GeeksforGeeks](https://www.geeksforgeeks.org/dsa/two-pointers-technique/), [Hello Interview](https://www.hellointerview.com/learn/code/two-pointers/overview), [AlgoMap](https://algomap.io/lessons/2-pointers)

---

<a name="sliding-window"></a>
## 2. SLIDING WINDOW

### The Critical Shrink Rule
```
LONGEST valid window  → shrink when INVALID
SHORTEST valid window → shrink when VALID
```

This is the #1 thing to memorize. Get this wrong and the entire solution breaks.

### Fixed vs Variable Window Decision
- **Fixed**: Problem says "window of size k" or "average of k elements" → Template A
- **Variable**: Problem says "longest/shortest satisfying condition" → Template B

### The "Exactly K" Trick
**Mnemonic: "exactly(K) = atMost(K) - atMost(K-1)"**

Whenever a problem asks for "exactly K", convert it to two "at most" calls.

### Trigger Words
| Keyword | It's Sliding Window |
|---|---|
| "Contiguous subarray" | Always |
| "Substring" | Always |
| "Window of size k" | Fixed window |
| "Longest/shortest with condition" | Variable window |
| "Maximum sum of size k" | Fixed window |
| "At most K distinct" | Variable window |

### When It's NOT Sliding Window
**"Negative numbers + sum target = use Prefix Sum instead."**

Sliding window only works when adding elements can only make the window "more invalid" and removing elements can only make it "more valid." Negative numbers break this monotonicity.

> Sources: [Built In](https://builtin.com/data-science/sliding-window-algorithm), [LeetCopilot](https://leetcopilot.dev/leetcode-pattern/sliding-window/guide), [Labuladong](https://labuladong.online/en/algo/essential-technique/sliding-window-framework/)

---

<a name="binary-search"></a>
## 3. BINARY SEARCH

### The Golden Rule
**"Sorted + need O(log n) = Binary Search"**

If the interviewer pushes back on your O(n) solution for a sorted input, they're hinting binary search.

### The Overflow-Safe Mid Calculation
**Mnemonic: "Never add, always subtract"**
```python
mid = lo + (hi - lo) // 2   # SAFE
mid = (lo + hi) // 2         # OVERFLOW RISK
```

### Template Decision
| Loop Condition | When to Use |
|---|---|
| `while lo < hi` | **Preferred.** Boundary finding: converge to a single point. Also works for exact match. |
| `while lo <= hi` | Standard search when you need to check every element and return -1 on miss |

**Handbook recommends `while lo < hi` as the default** -- it avoids off-by-one errors and naturally converges.

### The Two Mid Formulas
```python
mid = lo + (hi - lo) // 2       # LEFT-BIASED (rounds down) -- use with lo = mid + 1
mid = lo + (hi - lo + 1) // 2   # RIGHT-BIASED (rounds up) -- use with hi = mid - 1
```

### Infinite Loop Pitfall
**"If `lo = mid` with left-biased mid, you get an infinite loop."**

When your update is `lo = mid` (not `lo = mid + 1`), you MUST use right-biased mid. Otherwise `lo` never advances when `hi - lo == 1`.

### Beyond Arrays
**"Binary search works on any monotonic function, not just arrays."**

If a function is monotonically increasing/decreasing over a range, you can binary search on the answer space. Examples: "minimum capacity to ship in D days", "koko eating bananas."

> Sources: [Interview Cake](https://www.interviewcake.com/concept/java/binary-search), [Hello Interview](https://www.hellointerview.com/learn/code/binary-search/overview), [Interviewing.io](https://interviewing.io/binary-search-interview-questions)

---

<a name="monotonic-stack"></a>
## 4. MONOTONIC STACK / QUEUE

### The Golden Inverse Rule
**"Greater needs Decreasing, Smaller needs Increasing"**

The stack type is the OPPOSITE of what you're looking for:
- **Decreasing stack** → finds **greater** elements
- **Increasing stack** → finds **smaller** elements

### The Operator Template Trick
**"The operator in the while-loop decides the stack type"**
```
while stack and stack[-1] [OPERATOR] current:
    pop
push current
```
- `>` (or `>=`) → Monotonic Increasing (pops bigger ones)
- `<` (or `<=`) → Monotonic Decreasing (pops smaller ones)

**Note:** Use strict `>` / `<` to allow equal elements on the stack (needed for problems with duplicates). Use `>=` / `<=` to keep only strictly monotonic elements. The handbook uses strict operators by default.

### Trigger Words
| Keyword | Stack Type |
|---|---|
| "Next Greater Element" | Decreasing stack, scan L→R |
| "Next Smaller Element" | Increasing stack, scan L→R |
| "Previous Greater/Smaller" | Same stack type, scan L→R, record answer at push time |
| "Largest rectangle in histogram" | Increasing stack + sentinels |
| "Trapping rain water" | Stack or two pointers |
| "Daily temperatures" | Decreasing stack |
| "Stock span" | Decreasing stack |
| "Remove K digits" | Increasing stack (greedy build) |

### Stack vs Deque
**"Need a sliding WINDOW? Use a DEQUE. Otherwise, STACK."**

### The Complexity Guarantee
**"Each element enters once, leaves once -- always O(n)"**

> Sources: [GeeksforGeeks](https://www.geeksforgeeks.org/dsa/introduction-to-monotonic-stack-2/), [Hello Interview](https://www.hellointerview.com/learn/code/stack/monotonic-stack), [LeetCode Discuss](https://leetcode.com/discuss/post/2347639/a-comprehensive-guide-and-template-for-m-irii/)

---

<a name="intervals"></a>
## 5. INTERVALS

### The Sorting Decision Rule
```
MERGE problems  → sort by START time
SCHEDULE problems → sort by END time
```

**Why end time for scheduling?** Picking the interval that ends earliest leaves maximum room for future intervals.

### Overlap Detection Formula
**"Two intervals overlap if each one starts before the other ends."**
```python
overlap = a.start < b.end and b.start < a.end   # strict: for scheduling (no touching)
overlap = a.start <= b.end and b.start <= a.end  # non-strict: for merge (touching counts)
```
**For merge problems** (LC 56), use `<=` because `[1,3]` and `[3,5]` should merge. **For scheduling** (LC 252), use `<` because touching intervals don't conflict.

### Trigger Words
- "Overlapping intervals", "merge intervals"
- "Meeting rooms", "schedule", "calendar"
- "Non-overlapping", "minimum arrows"
- "Insert interval"

### The Template
1. **Sort** (by start or end)
2. **Iterate** linearly
3. **Compare current with last processed**

> Sources: [Hello Interview](https://www.hellointerview.com/learn/code/intervals/overview), [GeeksforGeeks](https://www.geeksforgeeks.org/dsa/merging-intervals/)

---

<a name="heaps"></a>
## 6. HEAPS / PRIORITY QUEUES

### The Top K Rule (Inverted!)
```
Top K LARGEST  → use MIN heap of size K
Top K SMALLEST → use MAX heap of size K
```

**Why inverted?** The heap's root is the "worst" of the K best. A min-heap root is the smallest among the K largest -- that's exactly the Kth largest element.

### Python heapq Trick
**"Python heapq = MIN heap. NEGATE for MAX heap."**
```python
heapq.heappush(heap, -val)    # insert as max heap
result = -heapq.heappop(heap)  # extract as max heap
```

### Trigger Words
| Keyword | Heap Pattern |
|---|---|
| "Kth largest/smallest" | Min/max heap of size K |
| "K closest points" | Max heap of size K |
| "Top K frequent" | Min heap of size K |
| "Merge K sorted lists" | Min heap of K elements |
| "Find median from stream" | Two heaps (max + min) |
| "Task scheduler" | Max heap + cooldown |

### Heap vs Sorting
**"Need repeated min/max access? Heap. Sort once and done? Sort."**

> Sources: [Tech Interview Handbook](https://www.techinterviewhandbook.org/algorithms/heap/), [AlgoMaster](https://algomaster.io/learn/dsa/top-k-elements-introduction)

---

<a name="greedy"></a>
## 7. GREEDY ALGORITHMS

### The "No Regret" Test
**"If you'll NEVER REGRET the choice later, go Greedy. If you MIGHT regret but subproblems overlap, go DP. If you need ALL solutions, go Backtracking."**

### CLIP Checklist
**"CLIP = Check Local, Independent, Progressive"**
- **C**hoices are **L**ocal: only needs local information
- **I**ndependent: one choice doesn't eliminate better future options (doesn't require considering all possibilities)
- **P**rogressive: problem strictly shrinks after each choice

All three hold → Greedy. Any fails → Consider DP or Backtracking.

### The Meta-Pattern
**"When in doubt, SORT it out"**

Most greedy problems follow: **Sort → Iterate → Decide → Collect**

| Problem Type | Sort By |
|---|---|
| Activity/interval selection | End time (earliest) |
| Merge intervals | Start time (earliest) |
| Job scheduling with deadlines | Deadline (earliest) |
| Fractional knapsack | Value/weight ratio (highest) |

### Two Proof Techniques
- **GSA (Greedy Stays Ahead)**: Show greedy is always winning at every step
- **EA (Exchange Argument)**: Show swapping any choice for greedy's choice never makes it worse

### GC + OS = Greedy is Boss
- **GC** = Greedy Choice Property
- **OS** = Optimal Substructure
- Both hold → Greedy works. Only OS → need DP. Neither → Backtracking.

> Sources: [Stanford CS161](https://web.stanford.edu/class/archive/cs/cs161/cs161.1138/handouts/120%20Guide%20to%20Greedy%20Algorithms.pdf), [AlgoMaster](https://algomaster.io/learn/dsa/greedy-introduction), [Interview Cake](https://www.interviewcake.com/concept/java/greedy)

---

<a name="trees"></a>
## 8. TREES

### Traversal Order: "The Name Says When"
- **PRE**-order = Root **PRE**-viously (before children): **Root → Left → Right**
- **IN**-order = Root **IN** the middle: **Left → Root → Right**
- **POST**-order = Root **POST**-poned (after children): **Left → Right → Root**

### The Snake Crawl Method
Place 3 dots around each node (left, bottom, right). Crawl around the tree:
- **Left dot** → Preorder
- **Bottom dot** → Inorder
- **Right dot** → Postorder

### BFS vs DFS on Trees
**"BFS = Queue = Level order. DFS = Stack/Recursion = In/Pre/Post order."**

| Need | Use |
|---|---|
| Level-order, minimum depth, nearest leaf | BFS (Queue) |
| Path sum, validate BST, diameter, max depth | DFS (Recursion) |
| Serialize/deserialize | Either (DFS usually simpler) |

### Key Insight
**"Inorder traversal of a BST gives sorted output."**

> Sources: [GeeksforGeeks](https://www.geeksforgeeks.org/dsa/tree-traversals-inorder-preorder-and-postorder/), [Quora](https://www.quora.com/How-do-I-remember-preorder-postorder-and-inorder-traversal), [Medium](https://medium.com/analytics-vidhya/an-easy-trick-to-derive-tree-traversal-results-in-a-single-look-8506e83974e4)

---

<a name="tries"></a>
## 9. TRIES

### The One Rule
**"If you need PREFIX lookups, XOR-based search, or wildcard matching across many words, it's a Trie."**

### Trie vs HashMap
- **HashMap**: Fast exact-match lookup only, less memory overhead for small sets
- **Trie**: Returns ALL words matching a prefix, shares prefixes to save space when words overlap heavily, preserves lexicographic order. Note: can use MORE memory than a HashMap for sparse word sets due to child pointers.

### Trigger Words
- "Prefix search", "autocomplete", "dictionary"
- "Spell check", "word search II" (multiple words in grid)
- "Lexicographic order"
- "Maximum XOR" (binary trie)
- "Wildcard matching" (`.` matches any character)

### Mental Model
**"A Trie is a tree of characters. Each root-to-node path (where `is_end=True`) spells a word."**

Words can end at internal nodes, not just leaves. For example, "car" and "card" share the same path up to 'r', but "car" has `is_end=True` at 'r' (an internal node).

> Sources: [Interview Cake](https://www.interviewcake.com/concept/java/trie), [GeeksforGeeks](https://www.geeksforgeeks.org/dsa/trie-insert-and-search/), [HeyCoach](https://heycoach.in/blog/trie-vs-hashmap/)

---

<a name="graphs"></a>
## 10. GRAPHS (BFS / DFS)

### The Core Decision
**"Closest/shortest → BFS. Complete/all paths → DFS."**

| Need | Use | Data Structure |
|---|---|---|
| Shortest path (unweighted) | BFS | Queue (FIFO) |
| Level-by-level exploration | BFS | Queue (FIFO) |
| Explore all paths | DFS | Stack / Recursion |
| Detect cycles | DFS (3-color) or Kahn's BFS | Visited + In-path set, or in-degree array |
| Topological sort | DFS or Kahn's BFS | Stack or Queue |
| Connected components | Either | Visited set |

### Topological Sort Mnemonic
**"Cook before eat = dependencies must be resolved first"**

Process nodes with 0 in-degree first (Kahn's), or post-order DFS (reversed).

### Cycle Detection Quick Rule
- **Kahn's BFS**: If `len(order) < num_nodes` → cycle exists
- **DFS**: If you visit a GRAY (in-progress) node → cycle (back edge)

### Graph Representation
**"Adjacency list for sparse, matrix for dense. Interviews usually use adjacency list."**

> Sources: [GeeksforGeeks](https://www.geeksforgeeks.org/dsa/difference-between-bfs-and-dfs/), [LeetCopilot](https://leetcopilot.dev/blog/when-to-use-bfs-vs-dfs-in-binary-tree-interviews)

---

<a name="matrix"></a>
## 11. MATRIX / GRID

### Direction Array: "URDL = You aRe Done, Leave!"
```python
directions = [(-1,0), (0,1), (1,0), (0,-1)]  # Up, Right, Down, Left
```

### The Boundary Check
**"Before you step, check the fence: 0 <= r < rows AND 0 <= c < cols"**
```python
if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]:
```

### BFS vs DFS on Grids
**"Shortest path = BFS. Connected regions = Either. All paths = DFS."**

### Island Template
**"For every unvisited land cell, launch a DFS/BFS -- that's one island."**

### Spiral Traversal
**"Peel the Onion: Right → Down → Left → Up, then shrink boundaries."**

After traversing each wall, move that wall inward.

### Matrix Rotation
**"Rotate 90° clockwise = Transpose + Reverse each row"**

### Sorted Matrix Search
- **Fully sorted (LC 74)**: Treat m×n matrix as 1D sorted array: `row = mid // cols, col = mid % cols`
- **Row + column sorted (LC 240)**: Start at top-right corner, staircase walk: go left if too big, go down if too small. O(m + n).

> Sources: [Tech Interview Handbook](https://www.techinterviewhandbook.org/algorithms/matrix/), [GeeksforGeeks](https://www.geeksforgeeks.org/dsa/print-a-given-matrix-in-spiral-form/), [Labuladong](https://labuladong.online/algo/en/frequency-interview/island-dfs-summary/)

---

<a name="union-find"></a>
## 12. UNION FIND

### The One-Line Rule
**"Dynamic connectivity = Union Find"**

### Two Operations, Two Optimizations
- **Find**: Which group does element belong to?
- **Union**: Merge two groups

Optimizations:
- **Path compression**: Flatten the tree during Find
- **Union by rank**: Attach smaller tree under larger

**"Path compression + Union by rank = O(α(n)) ≈ O(1) amortized"**

(α is inverse Ackermann -- practically constant for any realistic n)

### Trigger Words
- "Connected components", "are nodes connected?"
- "Redundant connection", "detect cycle in undirected graph"
- "Accounts merge", "number of provinces"
- "MST (Kruskal's algorithm)"

### Union Find vs BFS/DFS
**"Edges added dynamically → Union Find. Static graph with one-time query → BFS/DFS."**

Note: Union Find is also useful on static graphs when you need **repeated** "are X and Y connected?" queries -- O(α(n)) per query vs O(V+E) per BFS/DFS.

### Key Limitation
**"Union Find can union but can't un-union."** Deletions are not supported. If edges are removed, consider rebuilding or using a different approach.

> Sources: [CP-Algorithms](https://cp-algorithms.com/data_structures/disjoint_set_union.html), [GeeksforGeeks](https://www.geeksforgeeks.org/dsa/introduction-to-disjoint-set-data-structure-or-union-find-algorithm/)

---

<a name="backtracking"></a>
## 13. BACKTRACKING

### The Template: "Choose, Explore, Unchoose"
```python
def backtrack(path):
    if is_solution(path):
        result.append(path.copy())  # IMPORTANT: .copy() since path is mutable
        return
    for candidate in get_candidates(path):
        path.append(candidate)    # CHOOSE
        backtrack(path)           # EXPLORE
        path.pop()                # UNCHOOSE
```
**Note:** Use `.append()`/`.pop()` for lists (ordered, allows duplicates). Use `.add()`/`.remove()` only for set-based `used` tracking.

### Permutations vs Combinations vs Subsets
| Type | Constraint | Template Tweak |
|---|---|---|
| Permutations | Order matters, use all elements | Track `used[]` set |
| Combinations | Order doesn't matter, choose K | Pass `start` index, loop from `start` |
| Subsets | All possible subsets | Include/exclude each element |

### Trigger Words
- "All permutations", "all combinations", "all subsets"
- "Generate all", "N-Queens", "Sudoku solver"
- "Word search", "partition into palindromes"

### Pruning Mnemonic
**"Cut branches early -- check validity BEFORE recursing, not after."**

Pruning reduces search space from O(n^n) to O(n!).

> Sources: [Medium](https://medium.com/@albertoarrigoni/the-choose-explore-unchoose-pattern-for-backtracking-c0a519a3c2e8), [AlgoMonster](https://algo.monster/problems/backtracking_pruning), [Labuladong](https://labuladong.gitbook.io/algo-en/iii.-algorithmic-thinking/subset_permutation_combination)

---

<a name="divide-and-conquer"></a>
## 14. DIVIDE AND CONQUER

### The Three Steps: "DCC"
1. **D**ivide: Split into smaller subproblems
2. **C**onquer: Solve each recursively
3. **C**ombine: Merge sub-solutions

### Merge Sort vs Quick Sort
**"Merge Sort = Easy Split, Hard Merge. Quick Sort = Hard Split, Easy Merge."**

| Aspect | Merge Sort | Quick Sort |
|---|---|---|
| Work happens during | COMBINING (merge) | DIVIDING (partition) |
| Stability | Stable | Not stable |
| Worst case | Always O(n log n) | O(n²) with bad pivots |
| Space | O(n) extra | O(log n) in-place |

### Quick Select
**"QuickSelect = QuickSort but only recurse ONE side"**

"Why sort the whole closet when you only need to find one shirt?"

### Master Theorem Cheat
T(n) = aT(n/b) + O(n^d) -- compare log_b(a) to d:
- log_b(a) > d → **O(n^(log_b(a)))** (recursion dominates)
- log_b(a) = d → **O(n^d · log n)** (balanced)
- log_b(a) < d → **O(n^d)** (work-per-level dominates)

### When to Use Which
**"Need ORDER → Merge Sort. Need the Kth → Quick Select. Need to SEARCH → Binary Search."**

> Sources: [Khan Academy](https://www.khanacademy.org/computing/computer-science/algorithms/merge-sort/a/divide-and-conquer-algorithms), [UCI](https://ics.uci.edu/~eppstein/161/960118.html)

---

<a name="dp"></a>
## 15. DYNAMIC PROGRAMMING

### The Recognition Rule
**"Overlapping subproblems + Optimal substructure = DP"**

### The DP Recipe (STBAO+)
1. **S**tate: What variables define a subproblem? (What dimensions does `dp` have?)
2. **T**ransition: How does a bigger problem relate to smaller ones? (The recurrence relation)
3. **B**ase case: What are the trivially known values? (Initialize the DP table)
4. **A**nswer: Which cell has the final answer? (`dp[n]`, `dp[n][W]`, `max(dp)`, etc.)
5. **O**rder: What order to fill the table? (Ensure dependencies are computed first)
6. **Optimize**: Can you reduce space? (Rolling array, 1D instead of 2D)

The handbook also teaches the **FAST Method**: **F**ind recursive solution → **A**nalyze for overlapping subproblems → **S**ave results (memoize) → **T**urn bottom-up. Both frameworks arrive at the same solution.

### Trigger Words
| Keyword | DP Pattern |
|---|---|
| "Number of ways" | Counting DP |
| "Minimum cost / maximum profit" | Optimization DP |
| "Is it possible" | Feasibility DP |
| "Longest / shortest" | Optimization DP |
| "Can you partition into..." | Subset sum / knapsack |

### Top-Down vs Bottom-Up
- **Top-Down (Memoization)**: Recursion + cache. Natural to write.
- **Bottom-Up (Tabulation)**: Iterative, base cases first. Better space optimization.

### The "Caching" Insight
**"If you're solving the same subproblem multiple times in recursion, cache it -- that's DP."**

### DP vs Greedy Quick Test
**"Can a locally optimal choice ever be wrong later? YES → DP. NO → Greedy."**

> Sources: [FreeCodeCamp](https://www.freecodecamp.org/news/follow-these-steps-to-solve-any-dynamic-programming-interview-problem-cc98e508cd0e/), [AlgoMaster](https://blog.algomaster.io/p/20-patterns-to-master-dynamic-programming), [EnjoyAlgorithms](https://www.enjoyalgorithms.com/blog/top-down-memoization-vs-bottom-up-tabulation/)

---

<a name="advanced-dp"></a>
## 16. ADVANCED DP

### Bitmask DP
**Trigger: "n ≤ 20 + assign/visit all items = Bitmask DP"**

Use integer bits as boolean flags. Bit i = 1 means element i is used.
- Traveling Salesman, assignment problems, Hamiltonian path
- Complexity: O(2^n · n)

### State Machine DP
**Trigger: "Multiple states with transitions = State Machine DP"**

Draw the states and transitions. Classic: buy/sell stock with cooldown.
- States: **hold** (holding stock), **sold** (just sold, in cooldown), **rest** (not holding, free to buy)
- Transitions: rest→buy→hold, hold→sell→sold, sold→rest, rest→rest, hold→hold
- Each day, compute all states from previous day's values

### Interval DP
**Trigger: "Merge/split ranges optimally = Interval DP"**

Try every split point k between i and j, solve both halves, combine.
- Key: "Think about the LAST operation in the interval"
- Classic: Burst Balloons, Matrix Chain Multiplication

> Sources: [USACO Guide](https://usaco.guide/gold/dp-bitmasks), [Codeforces](https://codeforces.com/blog/entry/81516), [AlgoCademy](https://algocademy.com/blog/mastering-bitmask-dynamic-programming-a-comprehensive-guide/)

---

<a name="bit-manipulation"></a>
## 17. BIT MANIPULATION

### XOR Identity Tricks
```
x ^ x = 0     (anything XOR itself is 0)
x ^ 0 = x     (anything XOR 0 is itself)
x ^ y ^ y = x (XOR is self-inverse)
```
**Key properties:** XOR is **commutative** (`a ^ b == b ^ a`) and **associative** (`(a ^ b) ^ c == a ^ (b ^ c)`). This means order doesn't matter -- you can XOR elements in any order and get the same result.

**Application**: Find single number where all others appear twice → XOR everything.

### The "CSCT" Operations
| Operation | Formula | Mnemonic |
|---|---|---|
| **C**heck bit i | `(n >> i) & 1` | AND with mask |
| **S**et bit i | `n \| (1 << i)` | OR with mask |
| **C**lear bit i | `n & ~(1 << i)` | AND with inverted mask |
| **T**oggle bit i | `n ^ (1 << i)` | XOR with mask |

### Power Tricks
**Warning: Operator precedence!** `&` has lower precedence than `==` in Python. Always use parentheses:
- **Is power of 2?** `(n & (n-1)) == 0 and n > 0`
- **Is odd?** `(n & 1) == 1`
- **Clear rightmost 1-bit**: `n & (n-1)`
- **Isolate rightmost 1-bit**: `n & (-n)`
- **Count set bits**: Repeatedly `n = n & (n-1)`, count iterations

### Trigger Words
- "Single number", "missing number"
- "Power of two", "counting bits / Hamming weight"
- "Bit difference", "swap without temp"

> Sources: [Interview Cake](https://www.interviewcake.com/article/java/bit-manipulation), [GeeksforGeeks](https://www.geeksforgeeks.org/dsa/bits-manipulation-important-tactics/)

---

<a name="decision-table"></a>
## CROSS-PATTERN DECISION TABLE

### "If You See X, Think Y"

| Problem Signal | Pattern |
|---|---|
| Sorted array + find pair | Two Pointers |
| Contiguous subarray/substring + optimize | Sliding Window |
| Subarray sum = K with negatives | Prefix Sum + HashMap |
| Sorted + need O(log n) | Binary Search |
| Next/previous greater/smaller | Monotonic Stack |
| Sliding window max/min | Monotonic Deque |
| Overlapping intervals / merge | Intervals (sort by start) |
| Maximum non-overlapping / schedule | Intervals (sort by end) |
| Kth largest/smallest / top K | Heap |
| Merge K sorted | Heap |
| Find median from stream | Two Heaps |
| Prefix search / autocomplete | Trie |
| Level-order / shortest path (unweighted) | BFS |
| All paths / cycle detection | DFS |
| Connected components (static) | BFS/DFS |
| Connected components (dynamic) | Union Find |
| Number of islands / grid flood fill | Grid DFS/BFS |
| Shortest path in grid | Grid BFS |
| All permutations / combinations | Backtracking |
| Generate all / constraint satisfaction | Backtracking |
| Sort + merge / Kth element | Divide and Conquer |
| Count inversions | Modified Merge Sort |
| Min/max cost / number of ways | Dynamic Programming |
| Longest subsequence | DP |
| n ≤ 20 + visit/assign all | Bitmask DP |
| Multiple states with transitions | State Machine DP |
| Single number / power of 2 | Bit Manipulation |

### Distinguishing Similar Patterns

**Two Pointers vs Sliding Window:**
"Need the stuff IN the middle → Sliding Window. Only need THE ENDS → Two Pointers."

**Sliding Window vs Prefix Sum:**
"All positive elements → Sliding Window. Negatives present → Prefix Sum."

**DP vs Greedy vs Backtracking:**
1. Can you make a greedy choice that's never wrong? → **Greedy** (fastest, O(n log n) typically)
2. Need optimal solution with overlapping subproblems? → **DP** (polynomial, avoids recomputation)
3. Need ALL solutions or constraint satisfaction? → **Backtracking** (exponential, explores all paths)

**BFS vs DFS:**
"Closest/nearest = BFS. Complete/all paths = DFS."

**Union Find vs BFS/DFS for connectivity:**
"Edges added dynamically = Union Find. Static one-time query = BFS/DFS. Repeated queries on static graph = either works, UF has better per-query cost."

### Subarray vs Subsequence vs Subset
| Type | Contiguous? | Order? | Count | Technique |
|---|---|---|---|---|
| Subarray | Yes | Preserved | n(n+1)/2 | Sliding Window, Prefix Sum |
| Subsequence | No | Preserved | 2^n - 1 | DP (LCS, LIS) |
| Subset | No | Doesn't matter | 2^n | Backtracking, Bitmask |

---

<a name="constraints"></a>
## CONSTRAINT-BASED COMPLEXITY GUIDE

**"Read the constraints, know your algorithm."**

| Input Size (n) | Max Complexity | Algorithms |
|---|---|---|
| n ≤ 12 | O(n!) | Permutations, brute force |
| n ≤ 20 | O(2^n) | Backtracking, bitmask DP |
| n ≤ 100 | O(n^4) | 4 nested loops (rare) |
| n ≤ 500 | O(n^3) | Floyd-Warshall, cubic DP |
| n ≤ 10^4 | O(n^2) | Simple DP, nested loops |
| n ≤ 10^5-10^6 | O(n log n) | Sorting, divide & conquer, heap |
| n ≤ 10^8 | O(n) | Linear scan, two pointers, sliding window |
| n > 10^8 | O(log n) or O(1) | Binary search, math |

---

<a name="spaced-repetition"></a>
## SPACED REPETITION SCHEDULE

To actually retain these patterns long-term:

| Day | Action |
|---|---|
| **Day 1** | Learn pattern + solve 2-3 problems |
| **Day 3** | Re-solve from memory (no notes) |
| **Day 7** | Re-solve again, explain out loud |
| **Day 14** | Mixed review (interleave with other patterns) |
| **Day 30** | Final re-solve -- if correct without notes, it's in long-term memory |

### Key Study Techniques
1. **Pattern-first batching**: Solve 5-7 problems of the SAME pattern before moving on
2. **Interleaved review**: Mix patterns in review sessions (not during initial learning)
3. **Active recall**: Always attempt from scratch before checking solution
4. **Teach it**: Explain your solution out loud as if teaching someone

> Sources: [LeetCopilot](https://leetcopilot.dev/blog/how-to-practice-leetcode-without-memorizing-deliberate-practice), [Red Green Code](https://www.redgreencode.com/leetcode-tip-10-planning-a-spaced-repetition-schedule/)
