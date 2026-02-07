# EXAMPLE: Ready-to-Use Prompt for Dynamic Programming Handbook

Copy everything below this line and paste into Claude:

---

I am preparing for FAANG-level coding interviews and need a comprehensive, detailed handbook for **Dynamic Programming**. I want this to be similar in quality and structure to a professional interview preparation guide.

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
- Mental models for understanding Dynamic Programming
- Common misconceptions and how to think correctly
- Fundamental principles that apply across all problems
- Key terminology and concepts (memoization vs tabulation, overlapping subproblems, optimal substructure)

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
1. **1D DP (Linear problems)** - Climbing stairs, house robber, decode ways
2. **2D DP (Grid problems)** - Unique paths, minimum path sum, dungeon game
3. **Knapsack variants** - 0/1 knapsack, unbounded knapsack, target sum
4. **Longest Common Subsequence/Substring** - LCS, edit distance, shortest common supersequence
5. **Longest Increasing Subsequence** - LIS and variations, Russian doll envelopes
6. **State Machine DP** - Best time to buy/sell stock with various constraints
7. **Partition DP** - Palindrome partitioning, burst balloons
8. **DP on Trees** - House robber III, binary tree cameras
9. **Bitmask DP** - Traveling salesman, assignments with constraints
10. **Interval DP** - Matrix chain multiplication, burst balloons

#### E. Problem Recognition Framework (5-8% of content)
- Decision tree: How to identify which pattern to use
- Keywords/phrases in problem statements that signal DP patterns
- The "optimal substructure" and "overlapping subproblems" test
- Step-by-step approach to analyze a new DP problem
- How to identify state and transitions

#### F. Common Pitfalls & Solutions (5-8% of content)
- Top 10 mistakes people make
- For each mistake:
  - What goes wrong (with example)
  - Why it happens
  - How to fix it
  - How to avoid it

Examples:
- Incorrect base cases
- Wrong state definition
- Missing states
- Wrong transition formula
- Not considering all subproblems
- Integer overflow in DP arrays
- Space optimization bugs

#### G. Post-Processing & Edge Cases (3-5% of content)
- Common edge cases for DP problems
- Validation and boundary checking
- Off-by-one errors specific to DP
- Input validation patterns
- When to use 0-indexed vs 1-indexed

#### H. Interview Execution Guide (3-5% of content)
- Pre-interview checklist
- During-interview strategy:
  1. Identify optimal substructure
  2. Define state and transitions
  3. Write recurrence relation
  4. Decide memoization vs tabulation
  5. Implement and test
  6. Optimize space if needed
- Time management (how long to spend on each phase)
- Communication tips: How to explain DP thinking to interviewer
- How to test and verify DP solutions

#### I. Quick Reference Section (5% of content)
- Cheat sheet with all templates
- Complexity reference table
- Pattern recognition quick guide
- Common recurrence relations
- Space optimization techniques

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
- Use ASCII diagrams for DP tables
- Show state transitions with arrays
- Trace execution with examples
- Use arrows and annotations for state relationships

**Code Quality:**
- Clean, readable Python code with meaningful variable names
- Comprehensive comments explaining WHY, not just WHAT
- Consistent style across all examples
- Show both brute force → memoization → tabulation progression

**Formatting:**
- Clear headers and sub-headers
- Use tables for state definitions and transitions
- Use code blocks with Python syntax highlighting
- Use callout boxes for important notes (⚠️, ✅, ❌, 💡)
- Bullet points and numbered lists

### 4. DEPTH REQUIREMENTS

**Go Deep on Core Concepts:**
- Don't just list patterns - explain how to derive the recurrence
- Show why greedy fails and DP is needed
- Explain complexity analysis step-by-step
- Provide multiple examples for complex patterns

**Cover Edge Cases:**
- Empty inputs
- Single element
- All same elements
- Maximum constraints (array size, values)
- Negative numbers (when applicable)

**Show Variations:**
- How patterns combine
- How to modify for different constraints
- Space optimization (2D → 1D array)
- Print the actual solution, not just the value

### 5. INTERVIEW FOCUS

Remember, this is for CODING INTERVIEWS, so:
- Prioritize patterns that appear frequently in interviews
- Include FAANG-specific examples
- Show how to communicate the DP approach ("I notice this has optimal substructure because...")
- Include complexity analysis (mandatory in interviews)
- Show progression: Brute force → Memoization → Tabulation → Space optimized
- Include "follow-up question" sections (e.g., "What if we need to print the path?")

### 6. SPECIFIC EXAMPLES TO INCLUDE

Please ensure you cover these specific problems:
- **Coin Change (LC 322)** - Classic unbounded knapsack
- **House Robber (LC 198)** - 1D DP with constraints
- **House Robber II (LC 213)** - Circular constraint variant
- **Longest Increasing Subsequence (LC 300)** - With O(n log n) optimization
- **Edit Distance (LC 72)** - 2D DP, string matching
- **Unique Paths (LC 62)** - Grid DP
- **Longest Common Subsequence (LC 1143)** - String DP
- **Word Break (LC 139)** - DP with hash set
- **Best Time to Buy and Sell Stock III (LC 123)** - State machine DP
- **Partition Equal Subset Sum (LC 416)** - 0/1 Knapsack variant
- **Palindrome Partitioning II (LC 132)** - Partition DP
- **Burst Balloons (LC 312)** - Interval DP
- **Target Sum (LC 494)** - Knapsack with count
- **Decode Ways (LC 91)** - 1D DP with conditions

### 7. QUALITY STANDARDS

The handbook should be:
- ✅ Comprehensive: 95%+ pattern coverage for Dynamic Programming
- ✅ Self-contained: Can be studied standalone
- ✅ Practical: Immediately usable in interviews
- ✅ Clear: Readable in one sitting (but detailed enough for deep study)
- ✅ Structured: Easy to find information quickly
- ✅ Memorable: Templates stick in your mind

### 8. LENGTH & DETAIL

- Aim for 10,000-15,000 words (comprehensive but focused)
- Each major pattern should get 800-1000 words
- Don't skimp on explanations - detail is valued
- Use examples liberally
- Repeat key concepts in different contexts (recurrence definition, state transitions)

### 9. OUTPUT FORMAT

Please provide the handbook in Markdown format with:
- Clear table of contents with anchor links
- Consistent heading hierarchy
- Python code blocks with syntax highlighting
- Tables for state definitions and transitions
- Visual separators between major sections

---

## ADDITIONAL CONTEXT

**My Background:**
- I am a software engineer preparing for senior-level FAANG interviews
- I prefer Python for interviews (use Python for all code examples)
- I learn best with: explanations of WHY, visual aids, and multiple examples
- I want to MASTER Dynamic Programming, not just memorize patterns

**Success Criteria:**
After studying this handbook, I should be able to:
1. Recognize a DP problem within 30 seconds
2. Define the state and transition relation confidently
3. Choose between memoization and tabulation appropriately
4. Code the solution from scratch with correct complexity
5. Explain my approach clearly to an interviewer
6. Handle variations and follow-up questions
7. Optimize space when asked

**Please begin creating "The Complete Dynamic Programming Handbook" now.**

Make it comprehensive, clear, and interview-focused. Take your time to ensure quality - this is a resource I'll use intensively for the next few weeks.

Focus especially on:
- Clear state definition process
- How to derive recurrence relations
- When to use top-down vs bottom-up
- Space optimization techniques
- How to handle "print the solution" follow-ups
