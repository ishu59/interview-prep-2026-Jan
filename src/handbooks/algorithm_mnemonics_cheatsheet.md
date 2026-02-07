# Algorithm Mnemonics & Memory Tricks for Coding Interviews

A comprehensive collection of mnemonics, memory aids, acronyms, and catchy phrases
organized by topic to help memorize algorithm patterns.

---

## 1. MONOTONIC STACK / QUEUE

### The Golden Inverse Rule
**Mnemonic: "Greater needs Decreasing, Smaller needs Increasing"**

This is the single most important rule to memorize. It is an inverse relationship:
- **Decreasing stack** --> finds **greater** elements (next greater, previous greater)
- **Increasing stack** --> finds **smaller** elements (next smaller, previous smaller)

Think of it as: "The stack type is the OPPOSITE of what you're looking for."

> Sources:
> - [GeeksforGeeks - Introduction to Monotonic Stack](https://www.geeksforgeeks.org/dsa/introduction-to-monotonic-stack-2/)
> - [Hello Interview - Monotonic Stack](https://www.hellointerview.com/learn/code/stack/monotonic-stack)
> - [LeetCode Discuss - Comprehensive Guide](https://leetcode.com/discuss/post/2347639/a-comprehensive-guide-and-template-for-m-irii/)

---

### The "Taller Person in Line" Analogy
**Mnemonic: "Walk down a line of people -- short people become irrelevant once a tall person appears"**

Imagine people standing in a line. You want to find each person's "next taller person":
- Walk through the line; keep a stack of "candidates."
- If the next person is TALLER than the one on top of the stack, POP the shorter person.
- The shorter person is now irrelevant -- the taller person "blocks" them.
- The popped person's answer is the current taller person.

This captures the core invariant: **elements get popped when they meet their answer.**

> Source: [PurpleTutor - Monotonic Stack](https://purpletutor.com/monotonic-stack/)

---

### The "Buildings Blocking the View" Analogy
**Mnemonic: "A taller building blocks all shorter buildings behind it"**

Imagine looking at a city skyline from the right:
- Any building shorter than the current one is "blocked" and irrelevant.
- Only keep buildings that are progressively taller (decreasing stack from top).
- When a new tall building appears, all shorter ones in the stack get their answer.

This helps remember why elements are discarded: they can never be the "next greater" for any future element once something taller stands in front.

> Source: [Medium - Monotonic Stack Playbook](https://medium.com/@megha_bh/monotonic-stack-playbook-a-visual-guide-with-mind-map-e07f24f14e17)

---

### Trigger Words -- When to Reach for a Monotonic Stack
**Mnemonic: "NGE, NSE, PGE, PSE -- if you see these letters, it's a mono-stack plea"**

Watch for these phrases in problem statements:
| Trigger Phrase | Stack Type | Scan Direction |
|---|---|---|
| "Next Greater Element" (NGE) | Decreasing | Left to Right |
| "Next Smaller Element" (NSE) | Increasing | Left to Right |
| "Previous Greater Element" (PGE) | Decreasing | Right to Left |
| "Previous Smaller Element" (PSE) | Increasing | Right to Left |

Also watch for: "largest rectangle", "maximum area", "stock span", "daily temperatures",
"trapping rain water", "remove K digits".

> Sources:
> - [Medium - Monotonic Stack for Efficient Problem Solving](https://medium.com/@manuchaitanya/monotonic-stack-for-efficient-problem-solving-next-greater-next-smaller-previous-greater-and-6c63d0572644)
> - [LeetCode Discuss - Monotonic Stack Guide + List of Problems](https://leetcode.com/discuss/post/5148505/Monotonic-Stack-Guide-+-List-of-Problems/)

---

### The Operator Template Trick
**Mnemonic: "The OPERATOR in the while-loop decides the stack type"**

```
while stack is not empty AND stack.top() [OPERATOR] current:
    pop from stack
push current
```

- `>` or `>=` in the while condition --> **Monotonic Increasing** stack (pops bigger ones)
- `<` or `<=` in the while condition --> **Monotonic Decreasing** stack (pops smaller ones)

Just remember: **the operator you write in the while-loop defines everything.**

> Source: [LeetCode Discuss - Comprehensive Guide and Template](https://leetcode.com/discuss/post/2347639/a-comprehensive-guide-and-template-for-m-irii/)

---

### Stack vs Queue Decision
**Mnemonic: "Need a sliding WINDOW? Use a QUEUE (deque). Otherwise, STACK."**

- Monotonic **Stack** --> next/previous greater/smaller for each element
- Monotonic **Queue (Deque)** --> sliding window maximum/minimum

> Source: [LeetCopilot - Monotonic Stack & Queue Guide](https://leetcopilot.dev/leetcode-pattern/monotonic-stack-queue/guide)

---

### The Complexity Guarantee
**Mnemonic: "Each element enters once, leaves once -- always O(n)"**

Even though there is a while loop inside a for loop, each element can only be pushed
once and popped once across the entire traversal. Total operations = 2n = O(n).

> Source: [Algo Monster - Monotonic Stack Intro](https://algo.monster/problems/mono_stack_intro)

---

## 2. GREEDY ALGORITHMS

### The Two Properties Test
**Mnemonic: "GC + OS = Greedy is Boss"**
- **GC** = Greedy Choice Property (local optimal leads to global optimal)
- **OS** = Optimal Substructure (optimal solution contains optimal sub-solutions)

If both properties hold, greedy works. If only OS holds (but not GC), you likely need DP.

> Sources:
> - [GeeksforGeeks - Greedy vs DP](https://www.geeksforgeeks.org/dsa/greedy-approach-vs-dynamic-programming/)
> - [Wikipedia - Greedy Algorithm](https://en.wikipedia.org/wiki/Greedy_algorithm)

---

### The "No Regret" Decision Rule
**Mnemonic: "If you'll NEVER REGRET the choice later, go Greedy. If you MIGHT regret, go DP."**

Ask yourself: "Can a locally optimal choice ever turn out to be wrong later?"
- **NO** (no future regret) --> Greedy works
- **YES** (future choices depend on past) --> Need DP

Concrete test:
- Fractional Knapsack: You can take fractions, so taking the highest ratio first is
  never regretted --> GREEDY
- 0/1 Knapsack: Taking one item might prevent a better combination later --> DP

> Sources:
> - [Interview Cake - Greedy Algorithms](https://www.interviewcake.com/concept/java/greedy)
> - [Baeldung - Greedy vs DP](https://www.baeldung.com/cs/greedy-approach-vs-dynamic-programming)

---

### The "Can You Undo?" Test
**Mnemonic: "Greedy never looks back. If the problem needs you to look back, it's DP."**

Greedy algorithms NEVER reconsider their choices. If the problem structure requires
reconsidering or undoing a previous decision, greedy will fail.

Rule of thumb:
- Choices are INDEPENDENT of future --> Greedy
- Choices CONSTRAIN future options --> DP

> Source: [Medium - Greedy vs DP: When to Choose What](https://medium.com/@alok.gathe20/greedy-algorithms-vs-dynamic-programming-when-to-choose-what-9c87d2d46650)

---

### The Greedy "Sort First" Pattern
**Mnemonic: "When in doubt, SORT it out"**

Most greedy interview problems follow this meta-pattern:
1. **Sort** the input by some metric
2. **Iterate** once, making the greedy choice at each step
3. **Collect** the result

Common "sort by" strategies:

| Problem Type | Sort By | Why |
|---|---|---|
| Activity/Interval Selection | End time (earliest first) | Frees up maximum remaining time |
| Job Scheduling with Deadlines | Deadline (earliest first) | Handles urgency first |
| Fractional Knapsack | Value/Weight ratio (highest first) | Maximizes value per unit |
| Minimum Platforms | Both arrival and departure times | Track overlap |
| Assign Cookies | Both greed factors and cookie sizes | Match smallest satisfiable first |

> Sources:
> - [GeeksforGeeks - Activity Selection](https://www.geeksforgeeks.org/dsa/activity-selection-problem-greedy-algo-1/)
> - [GeeksforGeeks - Scheduling in Greedy](https://www.geeksforgeeks.org/dsa/scheduling-in-greedy-algorithms/)

---

### Interval Scheduling -- The #1 Greedy Pattern
**Mnemonic: "Sort by END time, not start time. Earliest finish = most room left."**

This is the most commonly confused greedy pattern:
- WRONG: Sorting by start time (common trap!)
- RIGHT: Sorting by end time (finish time)

Why? By picking the interval that ends earliest, you leave maximum room for
subsequent intervals. This is the canonical "exchange argument" proof.

> Source: [Princeton - Greedy Algorithms Chapter 4](https://www.cs.princeton.edu/~wayne/kleinberg-tardos/pearson/04GreedyAlgorithms-2x2.pdf)

---

### Two Proof Techniques
**Mnemonic: "GSA or EA -- pick your proof"**

**1. Greedy Stays Ahead (GSA)**
- Show that at every step, greedy's partial solution is at least as good as
  any other solution's partial solution.
- "Greedy is always winning or tied at every checkpoint."

**2. Exchange Argument (EA)**
- Take any optimal solution that differs from greedy's solution.
- Show you can SWAP pieces to make it look like greedy's solution without
  making it worse.
- "Any optimal solution can be transformed into the greedy solution."

> Sources:
> - [Stanford CS161 - Guide to Greedy Algorithms](https://web.stanford.edu/class/archive/cs/cs161/cs161.1138/handouts/120%20Guide%20to%20Greedy%20Algorithms.pdf)
> - [Cornell CS482 - Greedy Stays Ahead](https://www.cs.cornell.edu/courses/cs482/2004su/handouts/greedy_ahead.pdf)
> - [UMass - Exchange Arguments](https://people.cs.umass.edu/~marius/class/cs311-fa18/lec6-nup.pdf)

---

### Greedy vs DP Quick-Fire Checklist
**Mnemonic: "CLIP" -- Check Local, Independent, Progressive**

- **C**hoices are **L**ocal: Each decision only needs local information
- **I**ndependent: One choice does not affect the validity of future choices
- **P**rogressive: Once a choice is made, the problem strictly shrinks

If all three hold --> Greedy. If any fails --> Consider DP.

> Source: [AlgoMaster - Greedy Introduction](https://algomaster.io/learn/dsa/greedy-introduction)

---

## 3. DIVIDE AND CONQUER

### The Three Steps
**Mnemonic: "DCC" -- Divide, Conquer, Combine**

1. **D**ivide: Break the problem into smaller subproblems
2. **C**onquer: Solve each subproblem recursively (base case = small enough to solve directly)
3. **C**ombine: Merge the sub-solutions into the solution for the original problem

Some sources use "Divide, Conquer, Merge" -- same idea.

> Source: [Khan Academy - Divide and Conquer](https://www.khanacademy.org/computing/computer-science/algorithms/merge-sort/a/divide-and-conquer-algorithms)

---

### Merge Sort vs Quick Sort -- The Duality
**Mnemonic: "Merge Sort = Easy Split, Hard Merge. Quick Sort = Hard Split, Easy Merge."**

This is the single most important thing to remember about these two algorithms:

| Aspect | Merge Sort | Quick Sort |
|---|---|---|
| Divide step | EASY (split in half blindly) | HARD (partition around pivot) |
| Combine step | HARD (merge two sorted halves) | EASY (already sorted, just concatenate) |
| Where work happens | During COMBINING (merge) | During DIVIDING (partition) |
| Stability | Stable | Not stable |
| Space | O(n) extra | O(log n) extra (in-place) |
| Worst case | Always O(n log n) | O(n^2) with bad pivots |

Memory trick: "**M**erge sort **M**erges (combine is hard). **Q**uick sort partitions (**Q**uickly divides)."

> Sources:
> - [UCI - Sorting by Divide and Conquer](https://ics.uci.edu/~eppstein/161/960118.html)
> - [InterviewKickstart - Quicksort vs Merge Sort](https://interviewkickstart.com/blogs/learn/quicksort-vs-merge-sort)
> - [University of Toronto - Quicksort](https://www.cs.toronto.edu/~david/course-notes/csc110-111/18-sorting/06-quicksort.html)

---

### Quick Select -- The "Lazy Quicksort"
**Mnemonic: "QuickSelect = Quicksort but only recurse ONE side"**

When you need the Kth smallest/largest element:
- Don't sort everything (O(n log n))
- Use QuickSelect: partition, then only recurse into the side that contains K
- Average O(n), worst case O(n^2)

Think of it as: "Why sort the whole closet when you only need to find one shirt?"

> Sources:
> - [Wikipedia - Quickselect](https://en.wikipedia.org/wiki/Quickselect)
> - [GeeksforGeeks - Quickselect Algorithm](https://www.geeksforgeeks.org/dsa/quickselect-algorithm/)

---

### When to Use Which D&C Algorithm
**Mnemonic: "Need ORDER? Merge Sort. Need the Kth? Quick Select. Need to SEARCH? Binary Search."**

| Goal | Algorithm | Time |
|---|---|---|
| Sort a linked list | Merge Sort | O(n log n) |
| Sort an array (general) | Quick Sort | O(n log n) avg |
| Sort with stability guarantee | Merge Sort | O(n log n) |
| Find Kth smallest/largest | Quick Select | O(n) avg |
| Find element in sorted data | Binary Search | O(log n) |
| Count inversions | Modified Merge Sort | O(n log n) |
| Closest pair of points | Divide and Conquer | O(n log n) |

> Sources:
> - [GeeksforGeeks - Introduction to Divide and Conquer](https://www.geeksforgeeks.org/dsa/introduction-to-divide-and-conquer-algorithm/)
> - [Codedamn - Divide and Conquer in Merge Sort and Quick Sort](https://codedamn.com/news/algorithms/divide-conquer-merge-sort-quick-sort)

---

### The Master Theorem "Cheat"
**Mnemonic: "T(n) = aT(n/b) + O(n^d) -- compare log_b(a) to d"**

For recurrence T(n) = aT(n/b) + O(n^d):
- If log_b(a) > d --> O(n^(log_b(a)))  ... "recursion dominates"
- If log_b(a) = d --> O(n^d * log n)   ... "balanced"
- If log_b(a) < d --> O(n^d)           ... "work-per-level dominates"

Quick applications:
- Merge Sort: T(n) = 2T(n/2) + O(n) --> a=2, b=2, d=1 --> log_2(2)=1=d --> O(n log n)
- Binary Search: T(n) = T(n/2) + O(1) --> a=1, b=2, d=0 --> log_2(1)=0=d --> O(log n)

> Source: [UTulsa - Divide-Conquer-Glue Algorithms](https://tylermoore.utulsa.edu/courses/cse3353/slides/l11-handout.pdf)

---

## 4. MATRIX / GRID PROBLEMS

### Direction Array -- The Core Building Block
**Mnemonic: "URDL" (Up, Right, Down, Left) -- think "You Are Done, Leave!"**

4-directional movement (the most common):
```
dr = [-1, 0, 1,  0]   // row deltas
dc = [ 0, 1, 0, -1]   // col deltas
```

Mapping:
| Index | Direction | dr | dc | Memory Aid |
|---|---|---|---|---|
| 0 | Up | -1 | 0 | Row decreases |
| 1 | Right | 0 | +1 | Col increases |
| 2 | Down | +1 | 0 | Row increases |
| 3 | Left | 0 | -1 | Col decreases |

Alternative compact form (also common):
```
directions = [(-1,0), (0,1), (1,0), (0,-1)]
```

8-directional (including diagonals) -- add the four corners:
```
dr = [-1, -1, -1, 0, 0, 1, 1, 1]
dc = [-1,  0,  1,-1, 1,-1, 0, 1]
```

> Sources:
> - [GeeksforGeeks - BFS on 2D Array](https://www.geeksforgeeks.org/dsa/breadth-first-traversal-bfs-on-a-2d-array/)
> - [Tech Interview Handbook - Matrix Cheatsheet](https://www.techinterviewhandbook.org/algorithms/matrix/)
> - [Codeforces - Efficient Grid Navigation](https://codeforces.com/blog/entry/78827)

---

### BFS vs DFS on Grids -- When to Use Which
**Mnemonic: "Shortest = BFS. Connected = Either. All paths = DFS."**

| Problem Type | Use | Why |
|---|---|---|
| Shortest path (unweighted) | BFS | Explores level by level = distance order |
| Number of islands | DFS or BFS | Just need connected components |
| Flood fill | DFS or BFS | Just need to mark connected region |
| All paths from A to B | DFS + backtracking | Need to explore every possibility |
| Nearest exit / minimum steps | BFS | Shortest path problem |
| Word search (path exists?) | DFS | Backtracking with path constraint |
| Rotting oranges (multi-source) | BFS | Multi-source shortest distance |

**Key rule**: If the problem says "minimum", "shortest", "fewest steps" --> **BFS**.
If it says "all paths", "exists a path", "connected" --> **DFS** (usually simpler).

> Sources:
> - [GeeksforGeeks - When to Use DFS or BFS](https://www.geeksforgeeks.org/dsa/when-to-use-dfs-or-bfs-to-solve-a-graph-problem/)
> - [Interviewing.io - BFS Interview Questions](https://interviewing.io/breadth-first-search-interview-questions)

---

### Spiral Matrix Traversal
**Mnemonic: "Peel the Onion: Right, Down, Left, Up -- then shrink"**

The spiral order is always: **Right -> Down -> Left -> Up**, then repeat on the
inner layer. Think of peeling an onion layer by layer from outside in.

Use four boundary variables:
```
top = 0, bottom = rows-1, left = 0, right = cols-1
```

After each direction:
- After going RIGHT across top row --> top++ (shrink top boundary down)
- After going DOWN along right col --> right-- (shrink right boundary left)
- After going LEFT across bottom row --> bottom-- (shrink bottom boundary up)
- After going UP along left col --> left++ (shrink left boundary right)

**Memory aid**: "After you traverse a wall, the wall moves INWARD."

> Sources:
> - [GeeksforGeeks - Print Matrix in Spiral Form](https://www.geeksforgeeks.org/dsa/print-a-given-matrix-in-spiral-form/)
> - [AlgoMap - Spiral Matrix Solution](https://algomap.io/problems/spiral-matrix)
> - [EnjoyAlgorithms - Spiral Traversal](https://www.enjoyalgorithms.com/blog/print-matrix-in-spiral-order/)

---

### The Boundary Check Pattern
**Mnemonic: "Before you step, check the fence: 0 <= r < rows AND 0 <= c < cols"**

The #1 source of bugs in grid problems is going out of bounds. ALWAYS validate:
```python
def is_valid(r, c, rows, cols):
    return 0 <= r < rows and 0 <= c < cols
```

Also check: not visited, not a wall/obstacle.

Full neighbor generation pattern:
```python
for dr, dc in [(-1,0), (0,1), (1,0), (0,-1)]:
    nr, nc = r + dr, c + dc
    if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]:
        # process neighbor
```

> Source: [Tech Interview Handbook - Matrix Cheatsheet](https://www.techinterviewhandbook.org/algorithms/matrix/)

---

### Grid Traversal -- The "Island" Template
**Mnemonic: "For every unvisited land cell, launch a DFS/BFS -- that's one island"**

The Number of Islands pattern is the foundation for many grid problems:
```
count = 0
for each cell (r, c):
    if cell is land AND not visited:
        DFS/BFS from (r, c) marking all connected land as visited
        count++
return count
```

This same template applies to:
- Number of islands (LC 200)
- Max area of island (LC 695)
- Number of provinces (LC 547)
- Surrounded regions (LC 130)
- Pacific Atlantic water flow (LC 417)

> Source: [Labuladong - Solve All Island Problems with DFS](https://labuladong.online/algo/en/frequency-interview/island-dfs-summary/)

---

### Matrix Rotation / Transpose Trick
**Mnemonic: "Rotate 90 clockwise = Transpose + Reverse each row"**

- Rotate 90 degrees clockwise: Transpose the matrix, then reverse each row
- Rotate 90 degrees counter-clockwise: Transpose the matrix, then reverse each column
- Rotate 180 degrees: Reverse each row, then reverse each column (or rotate 90 twice)

Transpose means swapping matrix[i][j] with matrix[j][i].

> Source: [Tech Interview Handbook - Matrix Cheatsheet](https://www.techinterviewhandbook.org/algorithms/matrix/)

---

## CROSS-TOPIC CHEAT SHEET: QUICK PATTERN RECOGNITION

| If you see... | Think... | Pattern |
|---|---|---|
| "Next greater/smaller element" | Monotonic Stack | Stack (decreasing for greater, increasing for smaller) |
| "Sliding window maximum/minimum" | Monotonic Queue (Deque) | Deque maintaining monotonic order |
| "Minimum number of intervals" | Greedy (sort by end time) | Sort + greedy select |
| "Shortest path in grid" | BFS | Queue-based level traversal |
| "Number of islands / connected components" | DFS/BFS on grid | Visited array + traversal |
| "Kth largest/smallest" | Quick Select | Partition + recurse one side |
| "Sort linked list" | Merge Sort | Recursive split + merge |
| "Largest rectangle in histogram" | Monotonic Stack | Decreasing stack for boundaries |
| "Trapping rain water" | Monotonic Stack or Two Pointers | Stack-based or pointer-based |
| "Spiral order" | Boundary shrinking | Right-Down-Left-Up + shrink |

---

## SPACED REPETITION SCHEDULE

To actually memorize these patterns, use the following intervals:
- **Day 1**: Learn and solve a problem using the pattern
- **Day 4**: Re-solve from memory (no notes)
- **Day 11**: Re-solve again
- **Day 30**: Final re-solve -- if correct without notes, it's in long-term memory

> Source: [LeetCopilot - How to Practice Without Memorizing](https://leetcopilot.dev/blog/how-to-practice-leetcode-without-memorizing-deliberate-practice)
