# The Complete Trees and Tree Traversals Handbook
> A template-based approach for mastering tree problems in coding interviews

**Philosophy:** Tree problems are not about memorizing traversal orders. It's about **choosing the right traversal for the problem structure** and understanding that most tree problems reduce to a few fundamental patterns: traversal, path finding, tree construction, and structural comparison.

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

- **The Family Tree**: Every person (node) has at most two children. To answer a question about the whole family, you ask each child to answer it for their branch, then combine. Grandma doesn't need to visit every descendant herself -- she just asks her two kids, who ask theirs, and so on.
- **The Russian Nesting Doll**: Every subtree is itself a complete tree. The trick that makes tree problems tractable is that solving the big tree = solving smaller trees + combining results at the current node.

### No-Jargon Translation

- **Node**: one item in the tree
- **Root**: the topmost node -- the starting point
- **Leaf**: a node with no children -- the bottom
- **Subtree**: any node plus everything below it -- a mini-tree
- **DFS**: going deep before going wide -- like exploring one hallway fully before trying the next
- **BFS**: going wide before going deep -- checking every room on this floor before going downstairs
- **Preorder/inorder/postorder**: when you process the current node -- before visiting children, between them, or after

### Mental Model

> "A tree is a Russian nesting doll: to solve the whole thing, open it up, solve each smaller doll inside, and combine the answers on the way back out."

---

### The Tree Node Structure

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

### Why Trees Are Fundamental

Trees appear everywhere:
- File systems
- DOM (web pages)
- Organization hierarchies
- Decision trees
- Expression parsing

### The Key Insight: Recursive Structure

Every tree problem exploits the fact that:
> **A tree is a root node connected to smaller subtrees**

This means most solutions follow the pattern:
1. Handle base case (null node or leaf)
2. Process current node
3. Recursively solve for subtrees
4. Combine results

### The Four Traversal Orders

```
        1
       / \
      2   3
     / \
    4   5

Preorder  (Root-Left-Right): 1, 2, 4, 5, 3
Inorder   (Left-Root-Right): 4, 2, 5, 1, 3
Postorder (Left-Right-Root): 4, 5, 2, 3, 1
Level Order (BFS):          1, 2, 3, 4, 5
```

### When to Use Each Traversal

| Traversal | When to Use | Example Problems |
|-----------|-------------|------------------|
| **Preorder** | Process node before children, copying trees | Serialize tree, clone tree |
| **Inorder** | BST problems (gives sorted order) | Validate BST, kth smallest |
| **Postorder** | Need children's results first | Height, diameter, subtree sum |
| **Level Order** | Level-by-level processing | Right side view, zigzag |

**Memory Aid:**
- **Pre**order: Process **pre**viously (before children)
- **In**order: Process **in** the middle (between children)
- **Post**order: Process **post**eriorly (after children)

---

<a name="master-templates"></a>
## 2. The Master Templates

### Template A: Basic Recursive DFS

```python
def dfs(node: TreeNode) -> ResultType:
    """
    Basic template for tree DFS.
    Most tree problems follow this structure.
    """
    # Base case: why check for None?
    # Recursion will always reach past the leaves. A leaf's children are None.
    # Without this check, accessing node.left or node.val would crash.
    # This is the "bottom of the nesting doll" -- nothing left to open.
    if not node:
        return base_value

    # Process current node (preorder position)

    # Recurse on children
    left_result = dfs(node.left)
    right_result = dfs(node.right)

    # Process current node (postorder position)

    # Combine and return result
    return combine(node.val, left_result, right_result)
```

**Preorder vs Postorder Processing:**
```python
def preorder_style(node):
    if not node:
        return
    # Process HERE for preorder (before children)
    print(node.val)
    preorder_style(node.left)
    preorder_style(node.right)

def postorder_style(node):
    if not node:
        return 0
    left = postorder_style(node.left)
    right = postorder_style(node.right)
    # Process HERE for postorder (after children)
    return 1 + max(left, right)  # e.g., height
```

---

### Template B: DFS with Path/State Tracking

```python
def dfs_with_state(node: TreeNode, state: State, result: list):
    """
    Template for problems needing path or accumulated state.
    State is passed DOWN, results collected along the way.
    """
    if not node:
        return

    # Update state with current node
    new_state = update_state(state, node)

    # Check if we found a valid result
    if is_valid(node, new_state):
        result.append(capture_result(new_state))

    # Recurse with updated state
    dfs_with_state(node.left, new_state, result)
    dfs_with_state(node.right, new_state, result)

    # Backtrack if state is mutable (like a list)
    # backtrack(state, node)
```

**Example: Root-to-leaf paths**
```python
def all_paths(root: TreeNode) -> list[list[int]]:
    result = []

    def dfs(node, path):
        if not node:
            return

        path.append(node.val)

        # Why check BOTH left AND right are None?
        # A leaf is a node with NO children at all. If only one child
        # is None, the node is NOT a leaf -- it still has a subtree.
        # We only record a complete path when we've truly reached the end.
        if not node.left and not node.right:  # Leaf
            result.append(path.copy())
        else:
            dfs(node.left, path)
            dfs(node.right, path)

        path.pop()  # Backtrack

    dfs(root, [])
    return result
```

---

### Template C: Iterative DFS (Using Stack)

```python
def iterative_dfs(root: TreeNode) -> list:
    """
    Iterative preorder using stack.
    Useful when recursion depth is a concern.
    """
    if not root:
        return []

    result = []
    stack = [root]

    # Why does `while stack` terminate?
    # Each node is pushed at most once and popped exactly once.
    # We never re-add a node, so the stack shrinks to empty.
    while stack:
        node = stack.pop()
        result.append(node.val)

        # Push right first so left is processed first (stack is LIFO).
        # Why check `if node.right`? Pushing None would cause a crash
        # when we later try to access node.val on a None object.
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)

    return result
```

**Iterative Inorder (More complex):**
```python
def iterative_inorder(root: TreeNode) -> list:
    result = []
    stack = []
    current = root

    # Why `current or stack`? Two conditions because we alternate between
    # two phases: (1) drilling left (`current` is not None), and
    # (2) backtracking via the stack (`stack` is not empty).
    # The loop ends only when BOTH are exhausted -- no node left to
    # drill into AND no node saved to backtrack to.
    while current or stack:
        # Go all the way left -- push each node so we can return to it
        while current:
            stack.append(current)
            current = current.left

        # Process node
        current = stack.pop()
        result.append(current.val)

        # Move to right subtree (may be None, which just means
        # the inner `while current` won't execute next iteration)
        current = current.right

    return result
```

---

### Template D: Level Order (BFS)

```python
from collections import deque

def level_order(root: TreeNode) -> list[list[int]]:
    """
    Template for BFS / level-order traversal.
    Processes tree level by level.
    """
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        # Why snapshot the length here?
        # The queue currently holds ALL nodes of the current level.
        # As we process them, we add their children (next level) to the
        # SAME queue. Snapshotting `level_size` before the loop ensures
        # we only process THIS level's nodes, not the new children.
        level_size = len(queue)
        level = []

        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)

            # Only enqueue non-None children; None means no child exists
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(level)

    return result
```

**Key Insight:** The `for` loop with `level_size` ensures we process exactly one level at a time. Without this snapshot, newly added children would be processed in the same iteration, mixing levels together.

---

### Template E: DFS with Global State

```python
class Solution:
    """
    Template for problems needing global state across recursive calls.
    Use instance variable or closure to track global result.
    """
    def solve(self, root: TreeNode) -> int:
        self.result = initial_value

        def dfs(node):
            if not node:
                return base_value

            left = dfs(node.left)
            right = dfs(node.right)

            # Update global result
            self.result = update_global(self.result, node, left, right)

            # Return local result for parent
            return local_result(node, left, right)

        dfs(root)
        return self.result
```

**Example: Tree Diameter**
```python
def diameterOfBinaryTree(root: TreeNode) -> int:
    diameter = 0

    def height(node):
        nonlocal diameter
        if not node:
            return 0

        left_height = height(node.left)
        right_height = height(node.right)

        # Diameter through this node
        diameter = max(diameter, left_height + right_height)

        return 1 + max(left_height, right_height)

    height(root)
    return diameter
```

---

### Quick Decision Matrix

| Problem Type | Template | Key Characteristic |
|--------------|----------|-------------------|
| Simple traversal | A | Just visit all nodes |
| Height/depth | A (postorder) | Need children's results |
| Path problems | B | Track state down the path |
| Level-by-level | D (BFS) | Need level information |
| BST operations | A + inorder | Exploit sorted property |
| Max path sum | E | Global tracking |
| Tree construction | A | Build from traversals |

---

<a name="pattern-guide"></a>
## 3. Pattern Classification Guide

### Category 1: Basic Traversals
- Visit all nodes in specific order
- **Template A or C**
- Examples: Preorder, inorder, postorder traversal

### Category 2: Tree Properties
- Calculate height, depth, diameter, width
- **Template A (postorder) or E**
- Examples: Maximum depth, diameter, balanced check

### Category 3: Path Problems
- Root-to-leaf paths, path sum
- **Template B**
- Examples: Path sum, all paths, longest path

### Category 4: Level-Based Problems
- Process level by level
- **Template D (BFS)**
- Examples: Level order, right side view, zigzag

### Category 5: BST Problems
- Exploit sorted property of BST
- **Template A with inorder**
- Examples: Validate BST, kth smallest, LCA

### Category 6: Tree Construction
- Build tree from traversals
- **Template A**
- Examples: From preorder+inorder, from array

### Category 7: Tree Comparison
- Compare structure or values
- **Template A (parallel recursion)**
- Examples: Same tree, symmetric tree, subtree

### Category 8: Lowest Common Ancestor
- Find shared ancestor
- **Template A (postorder)**
- Examples: LCA in binary tree, LCA in BST

---

<a name="patterns"></a>
## 4. Complete Pattern Library

### PATTERN 1: Basic Traversals

---

#### Pattern 1A: Preorder Traversal

**Problem:** LeetCode 144 - Binary Tree Preorder Traversal

**Recursive:**
```python
def preorderTraversal(root: TreeNode) -> list[int]:
    result = []

    def dfs(node):
        # Why `if not node: return`?
        # Recursion always goes past the leaves -- a leaf's children are None.
        # Without this guard, `node.val` on None would crash immediately.
        if not node:
            return
        result.append(node.val)  # Process root BEFORE children (pre-order)
        dfs(node.left)           # Then left
        dfs(node.right)          # Then right

    dfs(root)
    return result
```

**Iterative:**
```python
def preorderTraversal_iterative(root: TreeNode) -> list[int]:
    # Why guard `if not root` before building the stack?
    # Avoids initializing and entering the while loop for an empty tree;
    # pushing None onto the stack would crash on `node.val` immediately.
    if not root:
        return []

    result = []
    stack = [root]

    while stack:
        node = stack.pop()
        result.append(node.val)
        # Why push right BEFORE left?
        # The stack is LIFO, so whatever is pushed last is processed first.
        # Pushing right first ensures left is popped next, preserving preorder.
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)

    return result
```

**Complexity:** Time O(n), Space O(h) where h = height

---

#### Pattern 1B: Inorder Traversal

**Problem:** LeetCode 94 - Binary Tree Inorder Traversal

**Recursive:**
```python
def inorderTraversal(root: TreeNode) -> list[int]:
    result = []

    def dfs(node):
        # Why `if not node: return`?
        # Every recursive call will eventually receive None (past a leaf).
        # Without this, accessing node.left or node.val crashes.
        if not node:
            return
        dfs(node.left)           # First go all the way LEFT
        result.append(node.val)  # THEN process current node (inorder = sorted for BST)
        dfs(node.right)          # THEN right subtree

    dfs(root)
    return result
```

**Iterative:**
```python
def inorderTraversal_iterative(root: TreeNode) -> list[int]:
    result = []
    stack = []
    current = root

    # Why `current or stack` (two conditions)?
    # We alternate between two phases: drilling left (current != None) and
    # backtracking (stack not empty). Both must be exhausted before we stop.
    while current or stack:
        # Go left as far as possible
        while current:
            stack.append(current)
            current = current.left

        # Process node
        current = stack.pop()
        result.append(current.val)

        # Move to right subtree
        current = current.right

    return result
```

**Key for BST:** Inorder of BST gives sorted order!

---

#### Pattern 1C: Postorder Traversal

**Problem:** LeetCode 145 - Binary Tree Postorder Traversal

**Recursive:**
```python
def postorderTraversal(root: TreeNode) -> list[int]:
    result = []

    def dfs(node):
        # Why `if not node: return`?
        # Base case for all recursive tree functions -- leaf children are None.
        # Without this guard we crash when trying to access node.left.
        if not node:
            return
        dfs(node.left)           # First left subtree
        dfs(node.right)          # Then right subtree
        result.append(node.val)  # Process root LAST (postorder = children first)

    dfs(root)
    return result
```

**Iterative (using two stacks):**
```python
def postorderTraversal_iterative(root: TreeNode) -> list[int]:
    if not root:
        return []

    result = []
    stack = [root]

    while stack:
        node = stack.pop()
        result.append(node.val)
        # Why push left BEFORE right here (opposite of preorder)?
        # We reverse the result at the end. Reversing a modified preorder
        # (root-right-left) gives postorder (left-right-root).
        if node.left:
            stack.append(node.left)
        if node.right:
            stack.append(node.right)

    return result[::-1]  # Reverse to get postorder
```

---

#### Pattern 1D: Level Order Traversal

**Problem:** LeetCode 102 - Binary Tree Level Order Traversal

```python
from collections import deque

def levelOrder(root: TreeNode) -> list[list[int]]:
    # Why return [] for None root (not crash)?
    # An empty tree has no levels; this prevents accessing .left/.right on None.
    if not root:
        return []

    result = []
    queue = deque([root])

    # Why `while queue`?
    # The queue is non-empty as long as there are unprocessed nodes.
    # When all nodes have been dequeued and no children added, it empties.
    while queue:
        # Why snapshot `len(queue)` before the inner loop?
        # At this moment, the queue holds exactly the nodes of the current level.
        # The inner loop will enqueue next-level children into the same queue.
        # Snapshotting prevents accidentally processing those new children now.
        level_size = len(queue)
        level = []

        # Why `for _ in range(level_size)` and not `while queue`?
        # We want to process EXACTLY this level's nodes -- no more, no less.
        # Using `while queue` here would consume next-level nodes too, mixing levels.
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)

            # Why guard with `if node.left`?
            # Appending None would cause a crash when we call node.val later.
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(level)

    return result
```

**Complexity:** Time O(n), Space O(w) where w = max width

---

### PATTERN 2: Tree Properties

---

#### Pattern 2A: Maximum Depth

**Problem:** LeetCode 104 - Maximum Depth of Binary Tree

```python
def maxDepth(root: TreeNode) -> int:
    # Why return 0 for None (not -1 or crash)?
    # A None node contributes 0 edges -- it represents the absence of a subtree.
    # Returning 0 means: "this path has no length", which combines cleanly with +1.
    if not root:
        return 0

    left_depth = maxDepth(root.left)
    right_depth = maxDepth(root.right)

    # Why `max(left_depth, right_depth) + 1`?
    # We take the LONGER of the two child paths -- the farthest leaf defines depth.
    # The +1 counts the current node itself as one level.
    return 1 + max(left_depth, right_depth)
```

**Iterative (BFS):**
```python
def maxDepth_bfs(root: TreeNode) -> int:
    if not root:
        return 0

    depth = 0
    queue = deque([root])

    # Why increment `depth` once per level (outside the inner loop)?
    # Each pass through the outer while loop processes exactly one full level.
    # Incrementing here ties depth to "number of levels fully processed".
    while queue:
        depth += 1
        for _ in range(len(queue)):
            node = queue.popleft()
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

    return depth
```

---

#### Pattern 2B: Balanced Binary Tree

**Problem:** LeetCode 110 - Check if tree is height-balanced

**Key Insight:** Check height while also verifying balance

```python
def isBalanced(root: TreeNode) -> bool:
    def check_height(node):
        """Returns height if balanced, -1 if unbalanced."""
        # Why return 0 for None?
        # An absent node has height 0 -- it contributes nothing to depth.
        # This is the same base case used in maxDepth.
        if not node:
            return 0

        left_height = check_height(node.left)
        # Why check `== -1`? This is early-termination propagation.
        # If ANY subtree below is unbalanced, -1 bubbles up all the way
        # to the root. No point checking further -- the whole tree fails.
        if left_height == -1:
            return -1

        right_height = check_height(node.right)
        if right_height == -1:
            return -1

        # Why `> 1` and not `>= 1`?
        # A balanced tree allows heights to differ by AT MOST 1.
        # diff=0 (perfectly balanced) and diff=1 (slightly uneven) are OK.
        # Only diff=2 or more is unbalanced, hence `> 1`.
        if abs(left_height - right_height) > 1:
            return -1

        # Why `max` (not `min`) for height here?
        # Height = longest path to a leaf. We must account for the deeper side
        # because that's what the parent will compare against its other child.
        return 1 + max(left_height, right_height)

    return check_height(root) != -1
```

**Why return -1 for unbalanced?**
- Early termination: once we find imbalance, propagate up
- Avoids redundant computation

---

#### Pattern 2C: Diameter of Binary Tree

**Problem:** LeetCode 543 - Find longest path between any two nodes

**Key Insight:** At each node, diameter could be `left_height + right_height`

```python
def diameterOfBinaryTree(root: TreeNode) -> int:
    diameter = 0

    def height(node):
        nonlocal diameter
        # Why return 0 for None?
        # A missing subtree has zero depth; adding 0 to the other side's depth
        # correctly gives the one-sided path length for a node with one child.
        if not node:
            return 0

        left_h = height(node.left)
        right_h = height(node.right)

        # Why `left_h + right_h` (ADD, not max) for diameter?
        # The longest path THROUGH this node uses BOTH arms: it goes down the left
        # subtree AND down the right subtree. Max would only use one arm.
        # Example: node with left_h=2, right_h=3 → diameter candidate = 5 edges.
        diameter = max(diameter, left_h + right_h)

        # Why return `max(left_h, right_h) + 1` (not sum) to the parent?
        # The parent needs to know the longest single path extending DOWNWARD.
        # A path can't go down both arms and still be a valid simple path upward.
        return 1 + max(left_h, right_h)

    height(root)
    return diameter
```

**Visual:**
```
        1
       / \
      2   3
     / \
    4   5

At node 2: left_h=1 (node 4), right_h=1 (node 5)
           diameter through 2 = 1 + 1 = 2 (path: 4-2-5)

At node 1: left_h=2 (through 2), right_h=1 (node 3)
           diameter through 1 = 2 + 1 = 3 (path: 4-2-1-3 or 5-2-1-3)
```

---

#### Pattern 2D: Minimum Depth

**Problem:** LeetCode 111 - Minimum depth to nearest leaf

**Caution:** Must reach a LEAF node!

```python
def minDepth(root: TreeNode) -> int:
    # Why return 0 for None (same as maxDepth)?
    # None means no subtree at all; depth is 0. The caller adds +1 for itself.
    if not root:
        return 0

    # Why not just `min(left, right)` like maxDepth?
    # If one child is None, that side returns 0, but a None child is
    # NOT a leaf -- we haven't reached a real endpoint. Taking min
    # would incorrectly say depth=1 for a node with one child.
    # We MUST go through the existing child to find a real leaf.
    if not root.left:
        return 1 + minDepth(root.right)
    if not root.right:
        return 1 + minDepth(root.left)

    # Why `min` here (vs `max` in maxDepth)?
    # We want the SHORTEST root-to-leaf path, so we take the shallower child.
    return 1 + min(minDepth(root.left), minDepth(root.right))
```

**BFS approach (often faster for this problem):**
```python
def minDepth_bfs(root: TreeNode) -> int:
    if not root:
        return 0

    queue = deque([(root, 1)])

    while queue:
        node, depth = queue.popleft()

        # Why return immediately at the first leaf?
        # BFS explores level by level, so the FIRST leaf we encounter
        # is guaranteed to be at the shallowest depth. No need to continue.
        if not node.left and not node.right:
            return depth

        if node.left:
            queue.append((node.left, depth + 1))
        if node.right:
            queue.append((node.right, depth + 1))

    return 0
```

---

### PATTERN 3: Path Problems

---

#### Pattern 3A: Path Sum

**Problem:** LeetCode 112 - Does root-to-leaf path with given sum exist?

```python
def hasPathSum(root: TreeNode, targetSum: int) -> bool:
    # Why return False for None and not True?
    # None is not a leaf -- it's the absence of a node.
    # A path must end at an actual leaf, so hitting None means
    # this branch has no valid endpoint.
    if not root:
        return False

    # Why check the sum only at a LEAF, not at every node?
    # The problem requires a ROOT-TO-LEAF path. If we check at internal
    # nodes, we might return True for a partial path that doesn't reach
    # a leaf. Example: tree [1,2], target=1 -- node 1 matches, but
    # the path 1->2 doesn't sum to 1 at a leaf.
    # Why check BOTH `not node.left AND not node.right`?
    # A leaf has NO children at all. If only one is None, the other subtree
    # still exists and we must continue down it.
    if not root.left and not root.right:
        return root.val == targetSum

    # Why subtract `root.val` before recursing (not after)?
    # We're tracking how much target remains BELOW this node.
    # Subtracting here lets the recursive call compare against 0 at the leaf.
    remaining = targetSum - root.val
    return (hasPathSum(root.left, remaining) or
            hasPathSum(root.right, remaining))
```

---

#### Pattern 3B: Path Sum II (Find All Paths)

**Problem:** LeetCode 113 - Find all root-to-leaf paths with given sum

```python
def pathSum(root: TreeNode, targetSum: int) -> list[list[int]]:
    result = []

    def dfs(node, remaining, path):
        # Why return early for None?
        # Same as hasPathSum -- None is not a leaf; no valid path ends here.
        if not node:
            return

        path.append(node.val)

        # Why check `remaining == node.val` at a leaf (not `remaining == 0`)?
        # We haven't subtracted node.val yet at this point -- we check if the
        # current node's value EXACTLY accounts for the remaining budget.
        # Checking `== 0` would require subtracting first, but we do it inline here.
        if not node.left and not node.right:
            if remaining == node.val:
                result.append(path.copy())
        else:
            dfs(node.left, remaining - node.val, path)
            dfs(node.right, remaining - node.val, path)

        path.pop()  # Backtrack

    dfs(root, targetSum, [])
    return result
```

**Why `path.copy()`?**
- `path` is mutable and changes as we backtrack
- We need to save a snapshot when we find a valid path

---

#### Pattern 3C: Binary Tree Maximum Path Sum

**Problem:** LeetCode 124 - Maximum sum path (can start/end anywhere)

```python
def maxPathSum(root: TreeNode) -> int:
    max_sum = float('-inf')

    def max_gain(node):
        nonlocal max_sum
        # Why return 0 for None (not -inf)?
        # A missing subtree contributes nothing. Returning 0 means "don't use
        # this arm", which is correct because we only extend a path if it adds value.
        if not node:
            return 0

        # Why `max(..., 0)`? We clamp negative gains to 0.
        # If a subtree has negative total, taking it hurts the path sum.
        # Choosing 0 means "ignore that arm" -- don't extend through it.
        left_gain = max(max_gain(node.left), 0)
        right_gain = max(max_gain(node.right), 0)

        # Why ADD both gains (left + right) for path_sum?
        # The best path THROUGH this node can extend in BOTH directions simultaneously.
        # This is the "diameter" trick applied to weighted paths.
        path_sum = node.val + left_gain + right_gain
        max_sum = max(max_sum, path_sum)

        # Why return only `max(left_gain, right_gain)` to the parent (not both)?
        # A path continuing upward is a single chain -- it can only come FROM one side.
        # Returning both would create a "Y" shape which isn't a valid simple path.
        return node.val + max(left_gain, right_gain)

    max_gain(root)
    return max_sum
```

**Key Insight:**
- When computing path through node: can use BOTH children
- When returning to parent: can only use ONE child (path can't split)

---

#### Pattern 3D: Sum Root to Leaf Numbers

**Problem:** LeetCode 129 - Sum of all root-to-leaf numbers

```python
def sumNumbers(root: TreeNode) -> int:
    def dfs(node, current_num):
        # Why return 0 for None?
        # A None branch contributes no number to the sum.
        # Returning 0 is the additive identity -- it leaves the total unchanged.
        if not node:
            return 0

        current_num = current_num * 10 + node.val

        # Why return `current_num` at a leaf instead of continuing to recurse?
        # At a leaf we've formed a complete number. Recursing further would
        # call dfs(None, ...) and return 0, which is correct but wasteful.
        # Returning here is the natural exit: the number is fully built.
        if not node.left and not node.right:
            return current_num

        return dfs(node.left, current_num) + dfs(node.right, current_num)

    return dfs(root, 0)
```

---

### PATTERN 4: Level-Based Problems

---

#### Pattern 4A: Binary Tree Right Side View

**Problem:** LeetCode 199 - Values visible from right side

```python
def rightSideView(root: TreeNode) -> list[int]:
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)

        for i in range(level_size):
            node = queue.popleft()

            # Why check `i == level_size - 1`?
            # BFS processes nodes left-to-right within a level. The LAST node
            # processed at each level is the rightmost one -- the one visible
            # from the right side. We only record that final node.
            if i == level_size - 1:
                result.append(node.val)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

    return result
```

**DFS approach:**
```python
def rightSideView_dfs(root: TreeNode) -> list[int]:
    result = []

    def dfs(node, depth):
        # Why return for None?
        # Prevents accessing node.val and ensures we don't record phantom nodes.
        if not node:
            return

        # Why check `depth == len(result)`?
        # We traverse right-first, so the first node seen at each depth IS the
        # rightmost one. `len(result)` equals the number of depths already recorded,
        # so equality means we're seeing a new depth for the first time.
        if depth == len(result):
            result.append(node.val)

        dfs(node.right, depth + 1)  # Right first!
        dfs(node.left, depth + 1)

    dfs(root, 0)
    return result
```

---

#### Pattern 4B: Binary Tree Zigzag Level Order

**Problem:** LeetCode 103 - Alternate left-to-right and right-to-left

```python
def zigzagLevelOrder(root: TreeNode) -> list[list[int]]:
    if not root:
        return []

    result = []
    queue = deque([root])
    left_to_right = True

    while queue:
        level_size = len(queue)
        level = deque()

        for _ in range(level_size):
            node = queue.popleft()

            if left_to_right:
                level.append(node.val)
            else:
                level.appendleft(node.val)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(list(level))
        left_to_right = not left_to_right

    return result
```

---

#### Pattern 4C: Average of Levels

**Problem:** LeetCode 637 - Average value at each level

```python
def averageOfLevels(root: TreeNode) -> list[float]:
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        level_sum = 0

        for _ in range(level_size):
            node = queue.popleft()
            level_sum += node.val

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(level_sum / level_size)

    return result
```

---

### PATTERN 5: BST Problems

---

#### Pattern 5A: Validate BST

**Problem:** LeetCode 98 - Check if valid BST

**Key Insight:** Pass valid range down the tree

```python
def isValidBST(root: TreeNode) -> bool:
    def validate(node, min_val, max_val):
        # Why return True for None?
        # An absent node vacuously satisfies all BST constraints.
        # It's the base case: nothing left to validate means "all good".
        if not node:
            return True

        # Why use a RANGE `(min_val, max_val)` instead of just comparing to parent?
        # BST requires every LEFT descendant (not just direct child) to be < root,
        # and every RIGHT descendant > root. The range enforces the full ancestor chain.
        # Example: in [5, 4, 6, null, null, 3, 7], node 3 is right child of 4 but
        # still violates the constraint `3 > 5` (root). Without passing down min_val,
        # this would be missed.
        if node.val <= min_val or node.val >= max_val:
            return False

        # Why pass `node.val` as the new max for left, and new min for right?
        # BST invariant: left subtree values must be < current node (upper bound),
        # right subtree values must be > current node (lower bound).
        return (validate(node.left, min_val, node.val) and
                validate(node.right, node.val, max_val))

    return validate(root, float('-inf'), float('inf'))
```

**Alternative: Inorder should be sorted**
```python
def isValidBST_inorder(root: TreeNode) -> bool:
    prev = float('-inf')

    def inorder(node):
        nonlocal prev
        if not node:
            return True

        if not inorder(node.left):
            return False

        # Why `node.val <= prev` and not `< prev`?
        # BST requires STRICT ordering -- duplicate values are not allowed.
        # If node.val == prev, the tree violates BST property (must be strictly greater).
        if node.val <= prev:
            return False
        prev = node.val

        return inorder(node.right)

    return inorder(root)
```

---

#### Pattern 5B: Kth Smallest in BST

**Problem:** LeetCode 230 - Find kth smallest element

```python
def kthSmallest(root: TreeNode, k: int) -> int:
    count = 0
    result = None

    def inorder(node):
        nonlocal count, result
        # Why `result is not None` as a second early-exit condition?
        # Once we've found the kth element, we don't need to visit any more nodes.
        # This short-circuits the entire remaining traversal for efficiency.
        if not node or result is not None:
            return

        # Why inorder (left-root-right)?
        # Inorder traversal of a BST yields values in SORTED ASCENDING order.
        # The kth node visited in inorder is the kth smallest element.
        inorder(node.left)

        count += 1
        if count == k:
            result = node.val
            return

        inorder(node.right)

    inorder(root)
    return result
```

**Iterative:**
```python
def kthSmallest_iterative(root: TreeNode, k: int) -> int:
    stack = []
    current = root

    while current or stack:
        while current:
            stack.append(current)
            current = current.left

        current = stack.pop()
        k -= 1
        if k == 0:
            return current.val

        current = current.right

    return -1
```

---

#### Pattern 5C: Lowest Common Ancestor in BST

**Problem:** LeetCode 235 - LCA in BST

**Key Insight:** Use BST property to guide search

```python
def lowestCommonAncestor(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    while root:
        # Why check BOTH p and q against root.val (not just one)?
        # We need both targets to be on the same side to keep descending.
        # If they're on opposite sides -- or one equals root -- we've found the LCA.
        if p.val < root.val and q.val < root.val:
            # Both in left subtree
            root = root.left
        elif p.val > root.val and q.val > root.val:
            # Both in right subtree
            root = root.right
        else:
            # Split point: p and q are on different sides (or one equals root).
            # This node is the lowest point where both are in the same subtree.
            return root

    return None
```

**Recursive:**
```python
def lowestCommonAncestor_recursive(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    # Why go left when BOTH values are less than root?
    # BST property: if both targets are smaller, the LCA must be in the left subtree.
    if p.val < root.val and q.val < root.val:
        return lowestCommonAncestor_recursive(root.left, p, q)
    # Why go right when BOTH values are greater?
    # Same logic -- if both are larger, LCA is in the right subtree.
    if p.val > root.val and q.val > root.val:
        return lowestCommonAncestor_recursive(root.right, p, q)
    # Why return `root` otherwise?
    # The split point: one target is <= root, the other >= root.
    # No deeper node can be ancestor of both -- root is the LCA.
    return root
```

---

#### Pattern 5D: Search in BST

**Problem:** LeetCode 700 - Find node with given value

```python
def searchBST(root: TreeNode, val: int) -> TreeNode:
    while root:
        if val == root.val:
            return root
        # Why go LEFT when `val < root.val`?
        # BST invariant: all values in the left subtree are strictly less than root.
        # If target is smaller, it cannot be in the right subtree -- prune that half.
        elif val < root.val:
            root = root.left
        # Why go RIGHT when `val > root.val`?
        # All right subtree values are strictly greater -- target must be there.
        else:
            root = root.right
    # Why return None when root becomes None?
    # We've exhausted all BST paths without finding val -- it doesn't exist.
    return None
```

---

### PATTERN 6: Tree Construction

---

#### Pattern 6A: Build Tree from Preorder and Inorder

**Problem:** LeetCode 105 - Construct tree from preorder and inorder traversal

**Key Insight:**
- Preorder first element is root
- Find root in inorder to determine left/right subtree sizes

```python
def buildTree(preorder: list[int], inorder: list[int]) -> TreeNode:
    # Why return None when `preorder` is empty?
    # No preorder elements means no nodes to construct -- the subtree is absent.
    # This is the base case that terminates the recursion.
    if not preorder:
        return None

    # Why use `preorder[0]` as the root?
    # Preorder visits root FIRST, so the first element is always the current root.
    root_val = preorder[0]
    root = TreeNode(root_val)

    # Why find root_val in inorder?
    # In inorder traversal, everything LEFT of the root index belongs to the left
    # subtree, and everything RIGHT belongs to the right subtree.
    # This split tells us the SIZE of each subtree, letting us slice preorder too.
    root_idx = inorder.index(root_val)

    # Split arrays
    left_inorder = inorder[:root_idx]
    right_inorder = inorder[root_idx + 1:]

    left_preorder = preorder[1:1 + len(left_inorder)]
    right_preorder = preorder[1 + len(left_inorder):]

    # Recursively build subtrees
    root.left = buildTree(left_preorder, left_inorder)
    root.right = buildTree(right_preorder, right_inorder)

    return root
```

**Optimized with hashmap:**
```python
def buildTree_optimized(preorder: list[int], inorder: list[int]) -> TreeNode:
    inorder_idx = {val: idx for idx, val in enumerate(inorder)}

    def build(pre_left, pre_right, in_left, in_right):
        if pre_left > pre_right:
            return None

        root_val = preorder[pre_left]
        root = TreeNode(root_val)

        root_idx = inorder_idx[root_val]
        left_size = root_idx - in_left

        root.left = build(pre_left + 1, pre_left + left_size,
                         in_left, root_idx - 1)
        root.right = build(pre_left + left_size + 1, pre_right,
                          root_idx + 1, in_right)

        return root

    return build(0, len(preorder) - 1, 0, len(inorder) - 1)
```

---

#### Pattern 6B: Build Tree from Inorder and Postorder

**Problem:** LeetCode 106 - Construct tree from inorder and postorder

```python
def buildTree(inorder: list[int], postorder: list[int]) -> TreeNode:
    if not postorder:
        return None

    # Last element of postorder is root
    root_val = postorder[-1]
    root = TreeNode(root_val)

    root_idx = inorder.index(root_val)

    # Build right first (postorder is left-right-root)
    root.left = buildTree(inorder[:root_idx], postorder[:root_idx])
    root.right = buildTree(inorder[root_idx + 1:], postorder[root_idx:-1])

    return root
```

---

#### Pattern 6C: Sorted Array to BST

**Problem:** LeetCode 108 - Convert sorted array to height-balanced BST

```python
def sortedArrayToBST(nums: list[int]) -> TreeNode:
    def build(left, right):
        # Why `left > right` as base case (not `left == right`)?
        # When left > right, the subarray is empty -- no node to create.
        # `left == right` still has one valid element to build, so we must NOT stop there.
        if left > right:
            return None

        # Why choose the MIDDLE element as root?
        # Picking the middle element splits the array evenly into two halves,
        # producing a height-balanced BST. Picking any other index would skew the tree.
        mid = (left + right) // 2
        root = TreeNode(nums[mid])
        root.left = build(left, mid - 1)
        root.right = build(mid + 1, right)

        return root

    return build(0, len(nums) - 1)
```

---

### PATTERN 7: Tree Comparison

---

#### Pattern 7A: Same Tree

**Problem:** LeetCode 100 - Check if two trees are identical

```python
def isSameTree(p: TreeNode, q: TreeNode) -> bool:
    # Why check `not p and not q` FIRST?
    # Both being None means both subtrees ended at the same depth -- they match here.
    # This must come before the single-None check to avoid returning False incorrectly.
    if not p and not q:
        return True
    # Why `not p or not q` (asymmetry) returns False?
    # At this point we know they're not BOTH None. If exactly ONE is None,
    # the trees have different shapes -- they can't be identical.
    if not p or not q:
        return False
    # Why check `p.val != q.val` before recursing?
    # Early exit: if current node values differ, no need to recurse into children.
    if p.val != q.val:
        return False

    return isSameTree(p.left, q.left) and isSameTree(p.right, q.right)
```

---

#### Pattern 7B: Symmetric Tree

**Problem:** LeetCode 101 - Check if tree is mirror of itself

```python
def isSymmetric(root: TreeNode) -> bool:
    def is_mirror(left, right):
        # Why `not left and not right` → True?
        # Both arms ended at the same depth simultaneously -- symmetric so far.
        if not left and not right:
            return True
        # Why `not left or not right` → False?
        # One arm ended before the other -- the tree is asymmetric here.
        if not left or not right:
            return False
        if left.val != right.val:
            return False

        # Why compare `left.left` with `right.right` and `left.right` with `right.left`?
        # Mirror symmetry means outer children mirror each other, and inner children too.
        # left.left <-> right.right is the outer pair; left.right <-> right.left is inner.
        return (is_mirror(left.left, right.right) and
                is_mirror(left.right, right.left))

    return is_mirror(root, root)
```

**Iterative:**
```python
def isSymmetric_iterative(root: TreeNode) -> bool:
    queue = deque([(root, root)])

    while queue:
        left, right = queue.popleft()

        if not left and not right:
            continue
        if not left or not right:
            return False
        if left.val != right.val:
            return False

        queue.append((left.left, right.right))
        queue.append((left.right, right.left))

    return True
```

---

#### Pattern 7C: Subtree of Another Tree

**Problem:** LeetCode 572 - Check if s is subtree of t

```python
def isSubtree(root: TreeNode, subRoot: TreeNode) -> bool:
    def is_same(p, q):
        if not p and not q:
            return True
        if not p or not q:
            return False
        return (p.val == q.val and
                is_same(p.left, q.left) and
                is_same(p.right, q.right))

    def dfs(node):
        # Why return False for None (not True)?
        # We've gone past a leaf without finding a matching subtree root.
        # Returning False signals "subRoot not found in this branch".
        if not node:
            return False
        # Why check `is_same` before recursing into children?
        # If the current node matches subRoot exactly, we've found it -- no need
        # to look deeper. Checking children first would be wasteful (they can't match).
        if is_same(node, subRoot):
            return True
        # Why `or` (short-circuit) between left and right?
        # We just need ONE branch to contain the subtree; finding it in either is enough.
        return dfs(node.left) or dfs(node.right)

    return dfs(root)
```

---

### PATTERN 8: Lowest Common Ancestor

---

#### Pattern 8A: LCA in Binary Tree

**Problem:** LeetCode 236 - Find LCA of two nodes in binary tree

```python
def lowestCommonAncestor(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    # Why return None for None root?
    # We've gone past a leaf without finding p or q -- signal "not found" upward.
    if not root:
        return None

    # Why return root immediately when `root == p or root == q`?
    # If the current node IS one of the targets, it must be an ancestor of itself.
    # We don't need to search deeper: even if the other target is a descendant,
    # this node is still the LCA (it's higher up). Return it and let the caller decide.
    if root == p or root == q:
        return root

    # Search in subtrees
    left = lowestCommonAncestor(root.left, p, q)
    right = lowestCommonAncestor(root.right, p, q)

    # Why `if left and right: return root`?
    # Both subtrees returned a non-None result, meaning p was found in one side
    # and q in the other. The CURRENT node is the split point -- the LCA.
    if left and right:
        return root

    # Why `return left if left else right`?
    # Only one subtree found a target (or neither). Propagate whichever is non-None
    # upward. If both are None, returns None (target not in this subtree at all).
    return left if left else right
```

**Visual:**
```
        3
       / \
      5   1
     / \ / \
    6  2 0  8

LCA(5, 1) = 3 (found in different subtrees)
LCA(5, 6) = 5 (5 is ancestor of 6)
```

---

<a name="post-processing"></a>
## 5. Post-Processing Reference

| Problem Type | Return Value | Base Case |
|--------------|--------------|-----------|
| **Traversal** | List of values | Empty list if null |
| **Property (height)** | Integer | 0 if null |
| **Boolean check** | True/False | True for null (usually) |
| **Find node** | Node or None | None if not found |
| **Path problems** | List or sum | 0 or empty for null |
| **Construction** | TreeNode | None if empty input |

---

<a name="pitfalls"></a>
## 6. Common Pitfalls & Solutions

### Pitfall 1: Forgetting Null Checks

```python
# WRONG: Will crash on null
def height(node):
    return 1 + max(height(node.left), height(node.right))
```

**Solution:**
```python
def height(node):
    if not node:
        return 0
    return 1 + max(height(node.left), height(node.right))
```

---

### Pitfall 2: Leaf vs Null Confusion

**Problem:** Treating null nodes as leaves

```python
# WRONG: Returns True for null nodes
def hasPathSum(root, target):
    if not root:
        return target == 0  # Null is NOT a leaf!
```

**Solution:** Check for leaf explicitly:
```python
def hasPathSum(root, target):
    if not root:
        return False
    if not root.left and not root.right:  # IS a leaf
        return root.val == target
```

---

### Pitfall 3: Not Copying Path in Backtracking

```python
# WRONG: All paths point to same list
if is_valid:
    result.append(path)  # path will be modified later!
```

**Solution:**
```python
if is_valid:
    result.append(path.copy())  # or list(path) or path[:]
```

---

### Pitfall 4: BST Validation with Wrong Bounds

```python
# WRONG: Only checks immediate parent
def isValidBST(root):
    if not root:
        return True
    if root.left and root.left.val >= root.val:
        return False
    # Misses: left subtree nodes could be larger than root
```

**Solution:** Pass valid range down:
```python
def validate(node, min_val, max_val):
    if not node:
        return True
    if node.val <= min_val or node.val >= max_val:
        return False
    return validate(node.left, min_val, node.val) and \
           validate(node.right, node.val, max_val)
```

---

### Pitfall 5: Stack Overflow on Deep Trees

**Problem:** Very deep trees cause recursion limit exceeded

**Solution:** Use iterative approach or increase recursion limit:
```python
import sys
sys.setrecursionlimit(10000)  # Use with caution

# Or better: use iterative approach
```

---

### Pitfall 6: Modifying Tree During Traversal

```python
# DANGEROUS: Modifying while traversing
def prune(node):
    if some_condition:
        node.left = None  # May cause issues
    prune(node.left)  # node.left is now None!
```

**Solution:** Be careful about order of operations, or use postorder for modifications.

---

<a name="recognition"></a>
## 7. Problem Recognition Framework

### Step 1: Identify Problem Category

| Clue | Category |
|------|----------|
| "Traverse", "visit all nodes" | Traversal |
| "Height", "depth", "diameter" | Property |
| "Path", "root to leaf" | Path |
| "Level", "layer", "row" | Level-based |
| "BST", "sorted", "search" | BST |
| "Build", "construct" | Construction |
| "Same", "equal", "mirror" | Comparison |
| "Ancestor", "parent" | LCA |

### Step 2: Choose Traversal Order

| Need | Traversal |
|------|-----------|
| Process before children | Preorder |
| BST in sorted order | Inorder |
| Need children's results | Postorder |
| Level by level | BFS |

### Step 3: Determine Return Type

| Problem | Returns |
|---------|---------|
| Search for node | Node or None |
| Calculate property | Integer |
| Validate condition | Boolean |
| Collect results | List |

### Decision Tree

```
                    Tree Problem
                         ↓
            ┌────────────┼────────────┐
            │            │            │
       Property      Path/Search    Structural
            │            │            │
     ┌──────┴──────┐     │     ┌──────┴──────┐
   Height    Balanced   │   Compare    Construct
   Diameter    Width    │   Same/Mirror  From Arrays
      ↓          ↓      │      ↓            ↓
  Postorder   BFS    ┌──┴──┐  Parallel    Divide &
                     │     │  Recursion   Conquer
                   Path   BST
                   Sum    Ops
                     ↓      ↓
                DFS+State Inorder
```

---

<a name="checklist"></a>
## 8. Interview Preparation Checklist

### Before the Interview

**Master the fundamentals:**
- [ ] Can write all traversals (recursive + iterative)
- [ ] Understand when to use each traversal
- [ ] Know preorder vs postorder processing
- [ ] Can do BFS level-order from memory

**Practice pattern recognition:**
- [ ] Can identify problem category quickly
- [ ] Know which template fits which problem
- [ ] Understand BST properties and applications

**Know the patterns:**
- [ ] Basic traversals
- [ ] Height/depth/diameter
- [ ] Path sum problems
- [ ] Level order variations
- [ ] BST operations
- [ ] Tree construction
- [ ] Same/symmetric/subtree
- [ ] LCA

**Common problems solved:**
- [ ] LC 104: Maximum Depth
- [ ] LC 110: Balanced Binary Tree
- [ ] LC 543: Diameter
- [ ] LC 112/113: Path Sum
- [ ] LC 102: Level Order
- [ ] LC 98: Validate BST
- [ ] LC 230: Kth Smallest in BST
- [ ] LC 105: Build from Preorder + Inorder
- [ ] LC 236: LCA

### During the Interview

**1. Clarify (30 seconds)**
- Binary tree or BST?
- What to return? (value, node, boolean)
- Can tree be null? Empty?
- Unique values?

**2. Identify pattern (30 seconds)**
- Which category?
- Which traversal order?
- DFS or BFS?

**3. Code (3-4 minutes)**
- Write base case first
- Recursive structure
- Process current node
- Combine results

**4. Test (1-2 minutes)**
- Null tree
- Single node
- Unbalanced (all left or all right)
- Example from problem

**5. Analyze (30 seconds)**
- Time: Usually O(n)
- Space: O(h) for recursion, O(w) for BFS

---

## 9. Quick Reference Cards

### DFS Template (Recursive)
```python
def dfs(node):
    if not node:
        return base_value
    # Preorder: process here
    left = dfs(node.left)
    right = dfs(node.right)
    # Postorder: process here
    return combine(node.val, left, right)
```

### BFS Template
```python
def bfs(root):
    if not root:
        return []
    queue = deque([root])
    while queue:
        for _ in range(len(queue)):
            node = queue.popleft()
            process(node)
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
```

### Path Template
```python
def paths(node, state, result):
    if not node:
        return
    state.append(node.val)
    if is_leaf(node):
        result.append(state.copy())
    paths(node.left, state, result)
    paths(node.right, state, result)
    state.pop()  # backtrack
```

---

## 10. Complexity Reference

| Operation | Time | Space |
|-----------|------|-------|
| Any traversal | O(n) | O(h) recursive, O(w) BFS |
| BST search | O(h) | O(h) or O(1) iterative |
| Build from traversals | O(n) or O(n²) | O(n) |
| LCA | O(n) | O(h) |

Where: n = nodes, h = height, w = max width

**Height bounds:**
- Balanced tree: h = O(log n)
- Skewed tree: h = O(n)

---

## Final Thoughts

**Remember:**
1. Most tree problems reduce to choosing the right traversal
2. Postorder when you need children's results first
3. Inorder for BST gives sorted order
4. BFS for level-by-level processing
5. Always handle null case first

**When stuck:**
1. Draw a small tree and trace through
2. Ask: "Do I need to process before or after children?"
3. Consider both DFS and BFS approaches
4. For BST, think about how sorted property helps

---

## Appendix: Practice Problem Set

### Easy
- 94/144/145. Binary Tree Traversals
- 100. Same Tree
- 101. Symmetric Tree
- 104. Maximum Depth
- 108. Sorted Array to BST
- 110. Balanced Binary Tree
- 112. Path Sum
- 226. Invert Binary Tree
- 543. Diameter of Binary Tree
- 700. Search in BST

### Medium
- 98. Validate Binary Search Tree
- 102. Binary Tree Level Order Traversal
- 103. Binary Tree Zigzag Level Order
- 105. Construct from Preorder and Inorder
- 113. Path Sum II
- 199. Binary Tree Right Side View
- 230. Kth Smallest Element in BST
- 236. Lowest Common Ancestor
- 437. Path Sum III
- 450. Delete Node in BST

### Hard
- 124. Binary Tree Maximum Path Sum
- 297. Serialize and Deserialize Binary Tree
- 968. Binary Tree Cameras

**Recommended Practice Order:**
1. Master basic traversals (94, 144, 145, 102)
2. Practice property calculations (104, 110, 543)
3. Do path problems (112, 113)
4. Master BST operations (98, 230, 235)
5. Try construction problems (105, 108)
6. Attempt hard problems (124, 297)

Good luck with your interview preparation!

---

## Appendix: Conditional Quick Reference

This table lists every key condition used in this handbook, its plain-English meaning, and the intuition behind it.

### A. Base Cases & Null Checks

| Condition | Plain English | Why it works |
|-----------|---------------|--------------|
| `if not node: return 0` | Node is absent; return zero | Recursion always reaches past leaves. Returning 0 (the additive identity for depth/height) lets the parent's `+1` combine cleanly without crashing. |
| `if not node: return None` | Node is absent; nothing found | Signals "target not in this subtree" for search and LCA problems. The caller can then check if both sides returned non-None. |
| `if not node: return False` | Node is absent; path invalid | Used in path-sum and subtree problems where None is NOT a valid endpoint -- a path must terminate at a real leaf. |
| `if not node: return True` | Node is absent; constraint satisfied | Used in BST validation: an empty subtree vacuously satisfies all BST constraints. |
| `if not preorder: return None` | No elements left; no node to build | Base case for tree construction -- an empty array means no subtree to create. |
| `if left > right: return None` | Subarray is empty; stop recursion | Used in sorted-array-to-BST: `left == right` still has one element, but `left > right` means the slice is exhausted. |
| `if not root: return []` | Empty tree; return empty result | Guards BFS/level-order so we never push None into the queue or enter the while loop. |

### B. BST Property Conditions

| Condition | Plain English | Why it works |
|-----------|---------------|--------------|
| `if val < root.val: go left` | Target is smaller; must be in left subtree | BST invariant: ALL left-subtree values are strictly less than the root. Smaller target cannot be on the right. |
| `if val > root.val: go right` | Target is larger; must be in right subtree | BST invariant: ALL right-subtree values are strictly greater than the root. |
| `if node.val <= min_val or node.val >= max_val: return False` | Node violates its valid range | Propagating a range (not just the direct parent) catches violations from ancestors further up the tree. |
| `if node.val <= prev: return False` (inorder) | Current value not strictly greater than previous | Inorder BST traversal must produce strictly ascending values. Equal values violate the BST strict-ordering requirement. |
| `if p.val < root.val and q.val < root.val: go left` | Both targets are smaller; LCA is in left subtree | BST LCA: if both targets are on the same side, the LCA cannot be at the current root or above it -- descend to narrow down. |
| `else: return root` (BST LCA) | Targets split across current node | The current node is the lowest point where both targets are "reachable" from different sides -- it is the LCA by definition. |

### C. DFS Path & Depth Conditions

| Condition | Plain English | Why it works |
|-----------|---------------|--------------|
| `if not node.left and not node.right` | Current node is a leaf | A leaf has NO children at all. Checking only one child would incorrectly treat a one-child node as a leaf endpoint. |
| `return 1 + max(left_h, right_h)` | Height = deeper child's height + 1 | We must report the LONGEST path downward (for diameter and balance checks). Min would undercount the reachable depth. |
| `diameter = max(diameter, left_h + right_h)` | Diameter candidate = both arms added | The longest path THROUGH a node goes down BOTH subtrees simultaneously. Using max (only one arm) would miss this two-directional path. |
| `left_gain = max(max_gain(node.left), 0)` | Clamp negative subtree gain to zero | If a subtree's maximum path sum is negative, taking it shrinks our total. Clamping to 0 means "don't extend through that arm". |
| `if root == p or root == q: return root` (LCA) | Found one target; it must be the LCA or an ancestor of the other | If one target is an ancestor of the other, we return immediately -- no deeper search needed because nothing below can be a higher ancestor. |
| `return left if left else right` (LCA) | Propagate the found target upward | If only one subtree found a target (the other returned None), we bubble the found node up so the caller can check if it also finds the other side. |
| `if abs(left_height - right_height) > 1: return -1` | Height difference exceeds 1; tree is unbalanced | AVL-style balance: difference of 0 or 1 is allowed. Returning -1 as a sentinel short-circuits all further computation once imbalance is detected. |

### D. BFS Level-Order Conditions

| Condition | Plain English | Why it works |
|-----------|---------------|--------------|
| `while queue` | Keep processing as long as nodes remain | The queue is non-empty exactly when unprocessed nodes exist. When all nodes have been dequeued and no new children added, the loop ends naturally. |
| `level_size = len(queue)` (snapshot before inner loop) | Freeze the count of current-level nodes | As we process nodes we enqueue their children into the same queue. Snapshotting first ensures we process ONLY this level's nodes, not the newly added next level. |
| `for _ in range(level_size)` | Process exactly this level's nodes | Using `while queue` for the inner loop would consume next-level nodes too, mixing levels and breaking per-level grouping. |
| `if i == level_size - 1: record` (right side view) | Last node in this level is the rightmost visible one | BFS processes left-to-right within each level, so the final node dequeued per level is the rightmost one -- exactly what the right side view requires. |
| `if node.left: queue.append(node.left)` | Only enqueue non-None children | Enqueueing None would crash later when we try to access `node.val` on a None object dequeued from the queue. |
