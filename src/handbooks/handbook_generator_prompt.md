# Master Prompt: Interview Handbook Generator

## How to Use This Prompt

1. Copy the prompt below (between the `---` markers)
2. Replace `[TOPIC]` with your desired topic
3. Replace `[PHILOSOPHY]` with a one-sentence philosophy for the topic
4. Replace `[FIRST_PRINCIPLES]` with 2-3 real-world analogies
5. Replace `[PATTERNS]` with 5-7 specific patterns to cover
6. Replace `[PROBLEMS]` with 15-20 specific LeetCode problems (with numbers)
7. Paste into Claude and get your comprehensive handbook

All 17 core handbooks are already created (see list at the bottom). Use this prompt to create additional handbooks or regenerate existing ones with improvements.

---

## THE PROMPT (Copy everything below this line)

---

I am preparing for FAANG-level coding interviews and need a comprehensive, detailed handbook for **[TOPIC]**. I want this to be similar in quality and structure to a professional interview preparation guide.

The handbook must follow this EXACT structure. Every section is required.

---

### OPENING FORMAT

Start with:

```markdown
# The Complete [TOPIC] Handbook
> A template-based approach for mastering [TOPIC] in coding interviews

**Philosophy:** [PHILOSOPHY]
```

### SECTION STRUCTURE (10 numbered sections + Final Thoughts + Appendix)

#### Section 1: Understanding the Core Philosophy (15-20% of content)

This section has a specific format with three required opening subsections, then deeper technical content:

**Required opening subsections:**
- **First Principles** -- 2-3 physical-world analogies that make the concept intuitive to a non-programmer. Each analogy should be a named concept (e.g., "The Coin Cashier", "The Tournament Bracket") followed by 2-3 sentences explaining how it maps to the algorithm.
- **No-Jargon Translation** -- A bullet list defining every key term in plain English (no CS jargon).
- **Mental Model** -- A single-sentence metaphor in a blockquote (`> "A greedy algorithm is like..."`) that captures the essence of the pattern.

**Then deeper technical content:**
- When does this pattern work / fail? (with concrete examples showing both)
- Key distinctions (e.g., Greedy vs DP, BFS vs DFS)
- Visual Understanding -- ASCII art showing the core concept
- Any proof techniques or correctness arguments relevant to the pattern

#### Section 2: Master Templates (20-25% of content)

Provide 3-6 Python templates covering the main variations. Include:
- **Decision matrix** (table): "If you see X → use Template Y"
- Each template should have:
  - A descriptive name (e.g., "Template A: Sort-then-Iterate")
  - When to use it (one-liner)
  - Complete Python code with comments explaining WHY each line exists
  - Time/space complexity
  - Example problem it applies to

#### Section 3: Pattern Classification Guide (10% of content)

Categorize all problems into 5-7 distinct patterns with:
- Pattern name
- One-sentence description
- Key signal / trigger words
- Example problems (LeetCode numbers)
- Table format preferred

#### Section 4: Complete Pattern Library (35-40% of content -- the bulk)

This is the largest section. Organize into 5-7 PATTERNS, each with 2-3 fully solved problems.

**Each problem MUST follow this exact format:**
```
### PATTERN N: [Pattern Name]

#### Pattern NA: [Problem Name]

**Problem:** LeetCode [NUMBER] - [Problem statement in 1-2 sentences]

**Example:**
[Input/Output/Explanation in a code block]

**Key Insight:** [2-3 sentences explaining WHY this approach works and what makes it non-obvious. This is the most important part -- explain the "aha moment".]

**Visual Trace:**
[ASCII art showing step-by-step execution on the example input. Show state changes, pointer movements, stack contents, etc.]

**Solution:**
[Complete Python solution with comments explaining key decisions]

**Complexity:** Time O(...), Space O(...) with brief justification.
```

**Target: 15-20 fully solved LeetCode problems per handbook.** Every solution must be complete, runnable Python code.

#### Section 5: Post-Processing Reference (2-3% of content)

A single table mapping problem types to return values and post-processing notes:

```markdown
| Problem Type | Return Value | Post-Processing Notes |
|---|---|---|
| [type] | [what to return] | [edge cases, conversions] |
```

#### Section 6: Common Pitfalls & Solutions (8-10% of content)

5-7 pitfalls, each with:
- A descriptive name (e.g., "Pitfall 1: Sorting by the Wrong Criterion")
- **Wrong code** in a Python code block with a comment explaining what goes wrong
- **Why it fails** (concrete failing test case)
- **Correct code** in a Python code block with the fix

#### Section 7: Problem Recognition Framework (5% of content)

A decision tree or flowchart in a code block showing:
```
"keyword in problem statement"  ->  Pattern name
```

Map 10-15 common problem phrasings to the specific pattern within this topic.

#### Section 8: Interview Preparation Checklist (5% of content)

Two sub-sections:

**Before the Interview:**
- Checkbox list of fundamentals to master
- Checkbox list of templates to memorize
- Checkbox list of specific LC problems to have solved

**During the Interview:**
- Numbered timeline: Clarify (30s) → Identify pattern (30s) → Explain approach (1 min) → Code (3-4 min) → Test (1-2 min) → Analyze complexity (30s)
- Specific questions to ask and things to state for THIS topic

#### Section 9: Quick Reference Cards (3-5% of content)

Condensed reference cards with:
- All templates in minimal form (just the skeleton, no comments)
- Key formulas and relationships
- "If you see X, do Y" quick-reference table

#### Section 10: Complexity Reference (2-3% of content)

Table of all patterns with their time/space complexity:

```markdown
| Pattern | Time | Space | Notes |
|---|---|---|---|
```

#### Final Thoughts (1%)

2-3 sentences of encouragement and a "key takeaway" quote in a blockquote.

#### Appendix: Practice Problem Set (2%)

Problems organized by difficulty tier (Starter → Intermediate → Advanced) with recommended order and brief notes on what each tests.

---

### STYLE REQUIREMENTS

**Code:**
- Python only
- Clean, readable code with meaningful variable names
- Comments explaining WHY, not WHAT
- Show both wrong and right approaches in Pitfalls section

**Visuals:**
- ASCII diagrams for Visual Trace in every problem
- Show state changes step by step
- Use arrows, annotations, and clear labeling

**Formatting:**
- Use `<a name="anchor"></a>` for section anchors
- Table of Contents at the top with anchor links
- `---` horizontal rules between major sections
- Code blocks with ```python for all solutions
- Tables for all comparison/reference data

**Tone:**
- Direct and practical -- no filler
- Every sentence should teach something
- Use bold for key terms on first introduction
- Use blockquotes for mental models and key insights

---

### DEPTH & LENGTH REQUIREMENTS

- Target: 1,500-2,200 lines of markdown
- 15-20 fully solved LeetCode problems with complete Python solutions
- Each major pattern: 150-250 lines including 2-3 problems
- Section 1 (Philosophy): 80-120 lines
- Section 2 (Templates): 100-180 lines

---

### SPECIFIC CONTENT TO COVER

**First Principles Analogies:**
[FIRST_PRINCIPLES]

**Patterns to cover:**
[PATTERNS]

**Problems to include (with LeetCode numbers):**
[PROBLEMS]

---

### MY BACKGROUND

- Software engineer preparing for senior-level FAANG interviews
- Python for all code examples
- I learn best with: "WHY" explanations, visual traces, and physical-world analogies
- I want to MASTER this topic, not just memorize patterns

### SUCCESS CRITERIA

After studying this handbook, I should be able to:
1. Recognize the pattern within 30 seconds of reading a problem
2. Choose the right template confidently
3. Code the solution from memory
4. Explain my approach clearly to an interviewer
5. Handle variations and follow-up questions
6. Avoid common pitfalls specific to this topic

**Please begin creating "The Complete [TOPIC] Handbook" now.**

---

## END OF PROMPT

---

# Quick Topic Customization Reference

## All 17 Core Handbooks (Complete)

All handbooks below have been created and follow the structure above.

| # | Handbook File | Topic | Key Patterns |
|---|---|---|---|
| 1 | `sliding_window_handbook.md` | Sliding Window | Fixed window, variable window, counting subarrays, exactly-K trick |
| 2 | `two_pointers_handbook.md` | Two Pointers | Opposite direction, same direction, fast/slow, partition |
| 3 | `binary_search_handbook.md` | Binary Search | Standard search, boundary finding, search on answer space |
| 4 | `dynamic_programming_handbook.md` | Dynamic Programming | 1D, 2D, knapsack, LCS, LIS, state machine |
| 5 | `advanced_dp_handbook.md` | Advanced DP | Bitmask DP, interval DP, state machine DP, tree DP |
| 6 | `trees_handbook.md` | Trees | Traversals, BST, path problems, construction, LCA |
| 7 | `graphs_bfs_dfs_handbook.md` | Graphs (BFS/DFS) | Shortest path, cycle detection, topological sort, bipartite |
| 8 | `backtracking_handbook.md` | Backtracking | Permutations, combinations, subsets, constraint satisfaction |
| 9 | `heaps_handbook.md` | Heaps / Priority Queues | Top-K, merge K sorted, two heaps, task scheduling |
| 10 | `intervals_handbook.md` | Intervals | Merge, insert, scheduling, overlap counting |
| 11 | `tries_handbook.md` | Tries | Prefix trees, word search, autocomplete |
| 12 | `union_find_handbook.md` | Union Find | Connected components, cycle detection, MST |
| 13 | `bit_manipulation_handbook.md` | Bit Manipulation | XOR tricks, bit counting, masks, subset generation |
| 14 | `monotonic_stack_queue_handbook.md` | Monotonic Stack/Queue | NGE, histogram, trapping rain water, sliding window max |
| 15 | `greedy_algorithms_handbook.md` | Greedy Algorithms | Interval scheduling, jump game, digit manipulation, two-pass |
| 16 | `divide_and_conquer_handbook.md` | Divide and Conquer | Merge sort variants, quick select, tree construction |
| 17 | `matrix_grid_handbook.md` | Matrix / Grid | Islands, grid BFS/DFS, matrix DP, spiral, rotation |

## Example Customizations for New Topics

### For Segment Trees / Fenwick Trees:
```
[TOPIC] = Segment Trees and Fenwick Trees (Binary Indexed Trees)
[PHILOSOPHY] = Segment trees and Fenwick trees are not about storing data -- they're about precomputing range queries so that both updates and queries happen in O(log n) instead of O(n).
[FIRST_PRINCIPLES] =
- "The Corporate Org Chart": To find the total sales of a division, you don't ask every employee -- you ask the VP who already has their team's total. Segment trees store pre-aggregated results at each level.
- "The Running Scoreboard": Instead of recalculating the total score from scratch after each play, the scoreboard adds the delta. Fenwick trees store cumulative deltas.
[PATTERNS] = Range sum query, Range min/max query, Lazy propagation, Point update + range query, Range update + point query, 2D range queries
[PROBLEMS] = LC 307 (Range Sum Query - Mutable), LC 315 (Count of Smaller Numbers After Self), LC 493 (Reverse Pairs), LC 218 (The Skyline Problem), LC 699 (Falling Squares), LC 1157 (Online Majority Element in Subarray)
```

### For String Algorithms:
```
[TOPIC] = String Algorithms (KMP, Rabin-Karp, Z-Algorithm)
[PHILOSOPHY] = String matching is not about comparing character by character -- it's about using preprocessing to skip impossible positions, turning O(n*m) into O(n+m).
[FIRST_PRINCIPLES] =
- "The License Plate Scanner": When looking for "ABC" in a stream of characters, if you see "ABD", you don't restart from scratch -- you already know "AB" matched, so you only need to check if the next character starts a new match from a known position.
- "The Fingerprint Check": Instead of comparing strings character by character, compute a numeric "fingerprint" (hash) and compare numbers. Only do full comparison when fingerprints match.
[PATTERNS] = KMP pattern matching, Rabin-Karp rolling hash, Z-algorithm, Longest palindromic substring (Manacher's), Suffix arrays, String hashing
[PROBLEMS] = LC 28 (Find the Index of the First Occurrence), LC 214 (Shortest Palindrome), LC 459 (Repeated Substring Pattern), LC 5 (Longest Palindromic Substring), LC 1392 (Longest Happy Prefix), LC 686 (Repeated String Match)
```

### For Graph Advanced (Shortest Path):
```
[TOPIC] = Advanced Graph Algorithms (Shortest Path, MST, Network Flow)
[PHILOSOPHY] = Advanced graph algorithms are not separate algorithms to memorize -- they're variations of BFS/DFS with different priority rules for which node to visit next.
[FIRST_PRINCIPLES] =
- "The GPS Navigator": Dijkstra's is like a GPS that always expands the closest unexplored intersection first. It never revisits a settled node because any other path to it would be longer.
- "The Cheapest Flight Finder": Bellman-Ford is like checking all flights K times -- after K rounds, you've found the cheapest path using at most K flights. Negative prices (rebates) are fine.
[PATTERNS] = Dijkstra's (single-source, non-negative), Bellman-Ford (negative edges), Floyd-Warshall (all-pairs), Kruskal's/Prim's MST, 0-1 BFS, A* search
[PROBLEMS] = LC 743 (Network Delay Time), LC 787 (Cheapest Flights Within K Stops), LC 1514 (Path with Maximum Probability), LC 1584 (Min Cost to Connect All Points), LC 1631 (Path With Minimum Effort), LC 778 (Swim in Rising Water)
```

---

# Pro Tips for Using This Prompt

1. **Be specific about problems**: Always include LeetCode numbers. The more specific, the better the output.
2. **First Principles matter most**: Good analogies in the philosophy section make the entire handbook more memorable.
3. **Iterate after generation**: Ask for expansions:
   - "Can you add 3 more problems to Pattern 2?"
   - "Can you add a visual trace for the solution to LC 743?"
   - "Can you expand the pitfalls section with 2 more common mistakes?"
4. **Cross-reference**: After creating a handbook, update `algorithm_mnemonics_cheatsheet.md` with the key mnemonics and trigger words.
5. **Quality check**: Verify all Python solutions are correct by testing them. Fix any issues before studying.

---

# Expected Output Quality

A well-generated handbook should have:
- 1,500-2,200 lines of markdown
- 3-6 master templates with decision matrix
- 5-7 distinct patterns
- 15-20 fully solved LeetCode problems with visual traces
- 5-7 pitfalls with wrong/right code comparisons
- Complete complexity reference table
- Interview-ready checklist

This structure has been validated across all 17 existing handbooks.
