# Interview Preparation Roadmap: Handbook Study Order

## Recommended Study Sequence

### Phase 1: Fundamentals (Weeks 1-2)
**Goal:** Master the building blocks that appear in 60%+ of problems

1. **Binary Search** ✅ (Already completed!)
   - Time: 2-3 days
   - Why first: Clean algorithm, teaches template thinking
   - Practice: 15-20 problems

2. **Two Pointers**
   - Time: 2-3 days
   - Why: Simpler than most topics, frequent in easy/medium problems
   - Practice: 15-20 problems

3. **Sliding Window**
   - Time: 2-3 days
   - Why: Builds on two pointers, very common pattern
   - Practice: 15-20 problems

### Phase 2: Core Data Structures (Weeks 3-4)
**Goal:** Master the most tested data structures

4. **Trees and Tree Traversals**
   - Time: 4-5 days
   - Why: Appears in 30% of interviews, many variations
   - Focus areas: DFS traversals, BFS level-order, path problems
   - Practice: 25-30 problems

5. **Heaps and Priority Queues**
   - Time: 2-3 days
   - Why: Powerful for optimization problems
   - Practice: 12-15 problems

### Phase 3: Graph Algorithms (Week 5)
**Goal:** Handle complex traversal and connectivity problems

6. **BFS and DFS (Graphs)**
   - Time: 4-5 days
   - Why: Critical for many medium/hard problems
   - Focus areas: Shortest path, cycle detection, topological sort
   - Practice: 25-30 problems

7. **Union-Find (Disjoint Set)**
   - Time: 1-2 days
   - Why: Elegant solution for connectivity problems
   - Practice: 8-10 problems

### Phase 4: Advanced Patterns (Weeks 6-7)
**Goal:** Handle the trickiest interview patterns

8. **Dynamic Programming**
   - Time: 6-7 days (most important, needs time!)
   - Why: Appears in 25% of interviews, tests problem-solving depth
   - Focus areas: 1D, 2D, knapsack, LCS, state machine
   - Practice: 30-40 problems

9. **Backtracking**
   - Time: 3-4 days
   - Why: Common in combination/permutation problems
   - Practice: 15-20 problems

### Phase 5: Specialized Topics (Week 8+)
**Goal:** Round out knowledge for hard problems

10. **Tries**
    - Time: 2 days
    - Why: String problems, prefix matching
    - Practice: 8-10 problems

11. **Intervals**
    - Time: 2 days
    - Why: Scheduling problems common in real scenarios
    - Practice: 10-12 problems

12. **Bit Manipulation**
    - Time: 1-2 days
    - Why: Occasional but elegant solutions
    - Practice: 8-10 problems

13. **Advanced DP** (If time permits)
    - Time: 3-4 days
    - Focus: Bitmask DP, tree DP, digit DP
    - Practice: 10-15 problems

---

## Quick Priority Matrix

**Must Master (Do these first):**
- Binary Search ✅
- Two Pointers
- Trees (DFS/BFS)
- Dynamic Programming
- BFS/DFS Graphs

**Should Know (High ROI):**
- Sliding Window
- Heaps
- Backtracking
- Tries
- Intervals

**Good to Know (Nice to have):**
- Union-Find
- Bit Manipulation
- Advanced graph algorithms
- Segment Trees

---

## Weekly Study Plan (8-Week Intensive)

### Week 1: Foundations
- Mon-Wed: Binary Search (handbook + 20 problems)
- Thu-Fri: Two Pointers (handbook + 15 problems)
- Weekend: Review + mock interview

### Week 2: Arrays & Strings
- Mon-Wed: Sliding Window (handbook + 15 problems)
- Thu-Fri: String problems using learned techniques
- Weekend: Review + mock interview

### Week 3: Trees Part 1
- Mon-Tue: Tree fundamentals + DFS (handbook)
- Wed-Thu: Practice DFS problems (15 problems)
- Fri: BFS level-order (handbook section)
- Weekend: BFS problems (10 problems)

### Week 4: Trees Part 2
- Mon-Tue: Tree construction, BST, path problems
- Wed-Thu: Practice (15 problems)
- Fri: Heaps (handbook)
- Weekend: Heap problems (12 problems) + review

### Week 5: Graphs
- Mon-Tue: BFS/DFS graphs (handbook)
- Wed: Shortest path, cycle detection
- Thu: Topological sort, bipartite
- Fri: Union-Find (handbook)
- Weekend: Graph problems (20 problems)

### Week 6: Dynamic Programming Part 1
- Mon: DP philosophy, 1D patterns (handbook)
- Tue: 2D patterns (handbook)
- Wed: Knapsack patterns (handbook)
- Thu-Fri: Practice 1D, 2D, Knapsack (15 problems)
- Weekend: More practice (10 problems)

### Week 7: Dynamic Programming Part 2
- Mon: LCS/LIS patterns (handbook)
- Tue: State machine DP (handbook)
- Wed: Practice LCS/LIS/State (10 problems)
- Thu-Fri: Backtracking (handbook + 15 problems)
- Weekend: Review all DP + mock

### Week 8: Specialization
- Mon: Tries (handbook + 8 problems)
- Tue: Intervals (handbook + 10 problems)
- Wed: Bit manipulation (handbook + 8 problems)
- Thu-Fri: Weak areas + hard problems
- Weekend: Final review + full mock interviews

---

## Daily Study Schedule (Recommended)

### Weekdays (3-4 hours/day)
- **6:00-7:00 AM:** Read handbook section (60 min)
- **12:00-12:30 PM:** Review morning content (30 min)
- **7:00-9:00 PM:** Solve 2-3 problems (120 min)
- **9:00-9:30 PM:** Review solutions, update notes (30 min)

### Weekends (6-8 hours/day)
- **Morning:** Complete handbook if not finished
- **Afternoon:** Problem solving sprint (10-15 problems)
- **Evening:** Mock interview or review session

---

## Handbook Creation Order

Based on priority and dependencies:

1. **Binary Search** ✅ Done!
2. **Two Pointers** (use handbook generator prompt)
3. **Sliding Window** (use handbook generator prompt)
4. **Trees and Tries** (use handbook generator prompt)
5. **BFS/DFS** (use handbook generator prompt)
6. **Dynamic Programming** (example prompt provided!)
7. **Backtracking** (use handbook generator prompt)
8. **Heaps** (use handbook generator prompt)
9. **Intervals** (use handbook generator prompt)
10. **Union-Find** (use handbook generator prompt)

**Pro Tip:** Create handbooks 1-2 topics ahead of your study schedule. This gives you time to absorb and lets you start studying immediately.

---

## Topic Dependency Map

```
Binary Search (standalone)
    ↓
Two Pointers (standalone)
    ↓
Sliding Window (builds on two pointers)

Trees (standalone)
    ↓
BFS/DFS Graphs (uses tree concepts)
    ↓
Union-Find (graph connectivity)
    ↓
Topological Sort (uses BFS/DFS)

Dynamic Programming (standalone, but harder)
    ↓
Advanced DP (builds on basic DP)

Backtracking (standalone)
    ↓
Tries (uses recursion/backtracking concepts)

Heaps (standalone)
Intervals (standalone, uses sorting)
Bit Manipulation (standalone)
```

**Key Insight:** Do Trees before Graphs. Do basic DP before advanced DP. Most others are independent.

---

## Metrics to Track

**Per Topic:**
- [ ] Handbook read and understood
- [ ] All templates memorized
- [ ] Easy problems: 80%+ solve rate
- [ ] Medium problems: 60%+ solve rate
- [ ] Hard problems: 30%+ solve rate (optional)
- [ ] Can solve medium in <20 minutes

**Overall Progress:**
- Total problems solved: _____ / 250 (target)
- Mock interviews completed: _____ / 8 (weekly)
- Topics mastered: _____ / 10

---

## When You're Ready

**Week 8+ Onwards:**
- Focus on company-specific problem lists
- Do full-length mock interviews (2-3 per week)
- Review all handbooks (1 per day)
- Practice explaining solutions out loud
- Time yourself strictly
- Focus on communication, not just coding

**Red Flags (Need more prep):**
- Can't identify pattern in 1 minute
- Can't code medium in 25 minutes
- Struggling with >50% of medium problems
- Can't explain your approach clearly

**Green Flags (Ready to interview):**
- Pattern recognition: <30 seconds
- Medium problems: <20 minutes
- Can explain approach before coding
- Comfortable with follow-ups
- 250+ problems solved
- 8+ mock interviews done

---

## Additional Resources

**For Each Topic:**
1. **Read the handbook** (1-2 hours)
2. **Do LeetCode explore card** (if available)
3. **Watch NeetCode video** (for difficult patterns)
4. **Practice on LeetCode** (sort by acceptance rate)
5. **Review handbook** before sleep (retention)

**Weekly:**
- 2 mock interviews (Pramp, Interviewing.io)
- 1 system design review (if applicable)
- 1 behavioral prep session

**Monthly:**
- Review all handbooks
- Redo problems you struggled with
- Update weak areas list

---

## Customization Tips

**If you have less time (4 weeks):**
- Focus on: Binary Search, Two Pointers, Trees, DP, BFS/DFS
- Do only medium problems
- 1 mock interview per week

**If you have more time (12+ weeks):**
- Add advanced topics: Segment trees, Fenwick trees
- Do more hard problems
- Deeper dive into DP variants
- Company-specific prep earlier

**If you're rusty:**
- Add Week 0: Refresh arrays, strings, hash maps
- Spend 2 weeks on each major topic
- Do more easy problems first

**If you're strong:**
- Skip easy problems
- Focus on hard problems
- Add advanced topics early
- More mock interviews (3+ per week)

---

Good luck with your preparation! 🚀

Remember: **Consistency > Intensity**
Better to study 3 hours daily for 8 weeks than 8 hours daily for 3 weeks.
