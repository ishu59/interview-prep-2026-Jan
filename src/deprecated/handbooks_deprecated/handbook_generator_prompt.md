# Master Prompt: Interview Handbook Generator

## How to Use This Prompt

1. Copy the prompt below
2. Replace `[TOPIC]` with your desired topic (e.g., "Dynamic Programming", "BFS/DFS", "Trees and Tries", etc.)
3. Replace `[TOPIC_EXAMPLES]` with 2-3 specific examples of that topic
4. Paste into Claude and get your comprehensive handbook!

---

## THE PROMPT (Copy everything below this line)

---

I am preparing for FAANG-level coding interviews and need a comprehensive, detailed handbook for **[TOPIC]**. I want this to be similar in quality and structure to a professional interview preparation guide.

## Requirements:

### 1. FOCUS ON EXPLAINABILITY & READABILITY
This is THE MOST IMPORTANT requirement. Every concept, pattern, and code snippet must answer "WHY this way?":
- Why does this approach work?
- Why this code structure over alternatives?
- Why these specific steps?
- What's the intuition behind this pattern?

**Never just show code - always explain the reasoning.**

### 2. COMPREHENSIVE STRUCTURE

Please organize the handbook with these sections:

#### A. Core Philosophy & Fundamentals (15-20% of content)
- Mental models for understanding [TOPIC]
- Common misconceptions and how to think correctly
- Fundamental principles that apply across all problems
- Key terminology and concepts

#### B. Master Templates (20-25% of content)
- Provide 2-4 core templates that cover most use cases
- Each template should include:
  - When to use it
  - Why it's structured this way
  - Step-by-step breakdown
  - Common variations
  - Time/space complexity
- Make templates memorization-friendly
- Use consistent naming and structure

#### C. Pattern Classification (10-15% of content)
- Categorize problems into 5-8 distinct patterns
- For each pattern:
  - Clear identification criteria
  - Mental model / visualization
  - When it applies
  - Example problem types
- Include a decision tree or flowchart for pattern recognition

#### D. Complete Pattern Library (35-40% of content)
- For EACH pattern, provide:
  - **Pattern Name & Use Case**
  - **Recognition:** How to identify this pattern in a problem
  - **Approach:** Step-by-step strategy
  - **Template Code:** Clean, well-commented implementation
  - **Visual Example:** Diagram or trace showing execution
  - **Complexity Analysis:** Time and space with explanation
  - **Common Variations:** How the pattern adapts
  - **Related Problems:** 2-3 LeetCode examples
  
Cover these patterns at minimum:
[List 6-8 specific patterns for your topic here]

Examples:
- For DP: 1D DP, 2D DP, Knapsack, LIS, LCS, State Machine, etc.
- For BFS/DFS: Level-order, Path finding, Cycle detection, Topological sort, etc.
- For Trees: Traversals, LCA, Path sums, Serialization, BST operations, etc.

#### E. Problem Recognition Framework (5-8% of content)
- Decision tree: How to identify which pattern to use
- Keywords/phrases in problem statements that signal patterns
- The "IFTTT" test or similar heuristic for this topic
- Step-by-step approach to analyze a new problem

#### F. Common Pitfalls & Solutions (5-8% of content)
- Top 10 mistakes people make
- For each mistake:
  - What goes wrong (with example)
  - Why it happens
  - How to fix it
  - How to avoid it

#### G. Post-Processing & Edge Cases (3-5% of content)
- Common edge cases for this topic
- Validation and boundary checking
- Off-by-one errors specific to this topic
- Input validation patterns

#### H. Interview Execution Guide (3-5% of content)
- Pre-interview checklist
- During-interview strategy
- Time management (how long to spend on each phase)
- Communication tips specific to this topic
- How to test and verify solutions

#### I. Quick Reference Section (5% of content)
- Cheat sheet with all templates
- Complexity reference table
- Pattern recognition quick guide
- Common code snippets

#### J. Practice Problem Set (5% of content)
- Organized by difficulty: Easy, Medium, Hard
- Organized by pattern
- Recommended practice order
- Time targets for each difficulty

### 3. STYLE REQUIREMENTS

**Clarity:**
- Use simple, direct language
- Break complex concepts into digestible chunks
- Use analogies and real-world examples
- Progressive disclosure (simple first, then complexity)

**Visuals:**
- Use ASCII diagrams, trees, arrays, tables
- Show before/after states
- Trace execution with examples
- Use arrows and annotations

**Code Quality:**
- Clean, readable code with meaningful variable names
- Comprehensive comments explaining WHY, not just WHAT
- Consistent style across all examples
- Show both correct and incorrect approaches

**Formatting:**
- Clear headers and sub-headers
- Use tables for comparisons
- Use code blocks with syntax highlighting
- Use callout boxes for important notes (⚠️, ✅, ❌, 💡)
- Bullet points and numbered lists

### 4. DEPTH REQUIREMENTS

**Go Deep on Core Concepts:**
- Don't just list patterns - explain the intuition
- Show why certain approaches fail
- Explain complexity analysis step-by-step
- Provide multiple examples for complex patterns

**Cover Edge Cases:**
- Empty inputs
- Single element
- Duplicates (if applicable)
- Maximum/minimum constraints
- Special conditions

**Show Variations:**
- How patterns combine
- How to modify for different constraints
- How to optimize further

### 5. INTERVIEW FOCUS

Remember, this is for CODING INTERVIEWS, so:
- Prioritize patterns that appear frequently in interviews
- Include FAANG-specific examples (mention which companies ask what)
- Show how to communicate solution approach
- Include complexity analysis (mandatory in interviews)
- Show how to optimize brute force → optimal solution
- Include "follow-up question" sections

### 6. SPECIFIC EXAMPLES TO INCLUDE

Please ensure you cover these specific problems/scenarios:
[TOPIC_EXAMPLES]

For example:
- For DP: Coin change, House robber, Longest increasing subsequence, Edit distance
- For BFS/DFS: Number of islands, Course schedule, Word ladder, Clone graph
- For Trees: Validate BST, Lowest common ancestor, Serialize/deserialize, Path sum

### 7. QUALITY STANDARDS

The handbook should be:
- ✅ Comprehensive: 95%+ pattern coverage for this topic
- ✅ Self-contained: Can be studied standalone
- ✅ Practical: Immediately usable in interviews
- ✅ Clear: Readable in one sitting (but detailed enough for deep study)
- ✅ Structured: Easy to find information quickly
- ✅ Memorable: Templates stick in your mind

### 8. LENGTH & DETAIL

- Aim for 8,000-15,000 words (comprehensive but focused)
- Each major pattern should get 500-800 words
- Don't skimp on explanations - detail is valued
- Use examples liberally
- Repeat key concepts in different contexts

### 9. OUTPUT FORMAT

Please provide the handbook in Markdown format with:
- Clear table of contents with anchor links
- Consistent heading hierarchy
- Code blocks with language tags
- Tables where appropriate
- Visual separators between major sections

---

## ADDITIONAL CONTEXT

**My Background:**
- I am a software engineer preparing for senior-level FAANG interviews
- I prefer Python for interviews (use Python for code examples)
- I learn best with: explanations of WHY, visual aids, and multiple examples
- I want to MASTER this topic, not just memorize patterns

**Success Criteria:**
After studying this handbook, I should be able to:
1. Recognize the pattern within 30 seconds of reading a problem
2. Choose the right template/approach confidently
3. Code the solution from memory with correct complexity
4. Explain my approach clearly to an interviewer
5. Handle variations and follow-up questions
6. Avoid common pitfalls specific to this topic

**Please begin creating "The Complete [TOPIC] Handbook" now.**

Make it comprehensive, clear, and interview-focused. Take your time to ensure quality - this is a resource I'll use intensively for the next few weeks.

---

## END OF PROMPT

---

# Quick Topic Customization Guide

## For Dynamic Programming:
Replace [TOPIC] with: **Dynamic Programming**
Replace [TOPIC_EXAMPLES] with:
- Coin Change (LC 322)
- House Robber (LC 198)
- Longest Increasing Subsequence (LC 300)
- Edit Distance (LC 72)
- Knapsack variants
- State machine DP (Best Time to Buy/Sell Stock)

Add to pattern list:
- 1D DP (Linear problems)
- 2D DP (Grid/string problems)
- Knapsack (0/1, unbounded, bounded)
- Longest Common Subsequence/Substring
- State Machine DP
- DP on Trees
- Bitmask DP
- Interval DP

## For BFS/DFS:
Replace [TOPIC] with: **BFS and DFS (Graph Traversal)**
Replace [TOPIC_EXAMPLES] with:
- Number of Islands (LC 200)
- Course Schedule (LC 207)
- Word Ladder (LC 127)
- Clone Graph (LC 133)
- All paths source to target
- Shortest path in binary matrix

Add to pattern list:
- Level-order BFS
- Shortest path BFS
- Multi-source BFS
- DFS with backtracking
- Cycle detection
- Topological sort
- Connected components
- Bidirectional BFS

## For Trees and Tries:
Replace [TOPIC] with: **Trees and Tries**
Replace [TOPIC_EXAMPLES] with:
- Validate BST (LC 98)
- Lowest Common Ancestor (LC 236)
- Serialize and Deserialize Tree (LC 297)
- Path Sum variants
- Implement Trie (LC 208)
- Word Search II (LC 212)

Add to pattern list:
- Tree traversals (inorder, preorder, postorder)
- Level-order traversal
- Path problems (path sum, path to node)
- Tree construction (from traversals)
- BST operations
- Lowest common ancestor
- Tree serialization
- Trie construction
- Trie search variants

## For Backtracking:
Replace [TOPIC] with: **Backtracking**
Replace [TOPIC_EXAMPLES] with:
- Subsets (LC 78)
- Permutations (LC 46)
- Combination Sum (LC 39)
- N-Queens (LC 51)
- Palindrome Partitioning (LC 131)
- Word Search (LC 79)

Add to pattern list:
- Subsets/Combinations
- Permutations
- Constraint satisfaction
- Path finding
- Board problems
- Partition problems
- Generate parentheses

## For Sliding Window:
Replace [TOPIC] with: **Sliding Window**
Replace [TOPIC_EXAMPLES] with:
- Longest Substring Without Repeating Characters (LC 3)
- Minimum Window Substring (LC 76)
- Longest Repeating Character Replacement (LC 424)
- Max Consecutive Ones III (LC 1004)

Add to pattern list:
- Fixed window size
- Variable window size
- String problems
- Two pointers + window
- Window with constraints

## For Two Pointers:
Replace [TOPIC] with: **Two Pointers**
Replace [TOPIC_EXAMPLES] with:
- Two Sum II (LC 167)
- 3Sum (LC 15)
- Container With Most Water (LC 11)
- Trapping Rain Water (LC 42)
- Remove Duplicates (LC 26)

Add to pattern list:
- Opposite direction pointers
- Same direction pointers (fast/slow)
- Partition problems
- Sum problems
- Merge operations

## For Heaps (Priority Queues):
Replace [TOPIC] with: **Heaps and Priority Queues**
Replace [TOPIC_EXAMPLES] with:
- Kth Largest Element (LC 215)
- Merge K Sorted Lists (LC 23)
- Top K Frequent Elements (LC 347)
- Find Median from Data Stream (LC 295)

Add to pattern list:
- K-way merge
- Top K elements
- Median finding (two heaps)
- Meeting rooms / intervals
- Task scheduling

## For Intervals:
Replace [TOPIC] with: **Interval Problems**
Replace [TOPIC_EXAMPLES] with:
- Merge Intervals (LC 56)
- Insert Interval (LC 57)
- Meeting Rooms II (LC 253)
- Non-overlapping Intervals (LC 435)

Add to pattern list:
- Interval merging
- Interval insertion
- Interval intersection
- Overlapping intervals
- Interval scheduling

## For Graphs (Advanced):
Replace [TOPIC] with: **Advanced Graph Algorithms**
Replace [TOPIC_EXAMPLES] with:
- Dijkstra's Algorithm
- Union-Find / Disjoint Set
- Minimum Spanning Tree (Kruskal/Prim)
- Floyd-Warshall
- Bellman-Ford

Add to pattern list:
- Shortest path (single source)
- Shortest path (all pairs)
- Union-Find operations
- Minimum spanning tree
- Strongly connected components

## For Bit Manipulation:
Replace [TOPIC] with: **Bit Manipulation**
Replace [TOPIC_EXAMPLES] with:
- Single Number (LC 136)
- Number of 1 Bits (LC 191)
- Counting Bits (LC 338)
- Sum of Two Integers (LC 371)

Add to pattern list:
- Basic operations (set, clear, toggle bits)
- XOR properties
- Bit masks
- Power of two
- Subset generation with bits

---

# Pro Tips for Using This Prompt

1. **Start with fundamentals:** Do Binary Search, Two Pointers, Sliding Window first
2. **Then move to core patterns:** BFS/DFS, Trees, DP
3. **Finally advanced topics:** Graphs, Union-Find, Segment Trees

4. **Customize the prompt:** Add your specific pain points or companies you're targeting

5. **Iterate:** If first output isn't perfect, ask for:
   - "Can you expand section X with more examples?"
   - "Can you add more visual diagrams for pattern Y?"
   - "Can you explain Z in simpler terms?"

6. **Build a library:** Create handbooks for 8-10 core topics and you'll have comprehensive coverage

7. **Use follow-up prompts:**
   - "Create a practice schedule using these handbooks"
   - "Generate a mock interview problem set from these patterns"
   - "Create flashcards for quick review from this handbook"

---

# Expected Output Quality

When you use this prompt, you should get:
- 📖 8,000-15,000 word comprehensive guide
- 🎯 4-8 master templates ready to memorize
- 📊 10-15 distinct patterns fully explained
- 💡 Clear "why" reasoning throughout
- ✅ Complete code implementations
- 📈 Complexity analysis for each approach
- 🎓 Interview-ready format
- 📝 50+ practice problems organized by pattern

This is the same quality standard we achieved for the Binary Search handbook!
