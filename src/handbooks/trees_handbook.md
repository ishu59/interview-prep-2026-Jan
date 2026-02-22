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
        if not node:
            return
        result.append(node.val)  # Process root
        dfs(node.left)           # Then left
        dfs(node.right)          # Then right

    dfs(root)
    return result
```

**Iterative:**
```python
def preorderTraversal_iterative(root: TreeNode) -> list[int]:
    if not root:
        return []

    result = []
    stack = [root]

    while stack:
        node = stack.pop()
        result.append(node.val)
        # Push right first (LIFO: left processed first)
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
        if not node:
            return
        dfs(node.left)           # First left
        result.append(node.val)  # Process root
        dfs(node.right)          # Then right

    dfs(root)
    return result
```

**Iterative:**
```python
def inorderTraversal_iterative(root: TreeNode) -> list[int]:
    result = []
    stack = []
    current = root

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
        if not node:
            return
        dfs(node.left)           # First left
        dfs(node.right)          # Then right
        result.append(node.val)  # Process root last

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
        # Push left first (we'll reverse at end)
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
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        level = []

        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)

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
    if not root:
        return 0

    left_depth = maxDepth(root.left)
    right_depth = maxDepth(root.right)

    return 1 + max(left_depth, right_depth)
```

**Iterative (BFS):**
```python
def maxDepth_bfs(root: TreeNode) -> int:
    if not root:
        return 0

    depth = 0
    queue = deque([root])

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
        if not node:
            return 0

        left_h = height(node.left)
        right_h = height(node.right)

        # Update diameter (path through this node)
        diameter = max(diameter, left_h + right_h)

        # Return height for parent's calculation
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
    if not root.left and not root.right:
        return root.val == targetSum

    # Recurse with reduced target
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
        if not node:
            return

        path.append(node.val)

        # Check leaf
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
        if not node:
            return 0

        # Max gain from left/right (ignore negative paths)
        left_gain = max(max_gain(node.left), 0)
        right_gain = max(max_gain(node.right), 0)

        # Path through current node
        path_sum = node.val + left_gain + right_gain
        max_sum = max(max_sum, path_sum)

        # Return max gain if we continue path upward
        # Can only pick ONE child to continue
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
        if not node:
            return 0

        current_num = current_num * 10 + node.val

        # Leaf node
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

            # Last node of this level is visible from right
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
        if not node:
            return

        # First node we see at this depth (going right first)
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
        if not node:
            return True

        if node.val <= min_val or node.val >= max_val:
            return False

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
        if not node or result is not None:
            return

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
        if p.val < root.val and q.val < root.val:
            # Both in left subtree
            root = root.left
        elif p.val > root.val and q.val > root.val:
            # Both in right subtree
            root = root.right
        else:
            # Split point - this is LCA
            return root

    return None
```

**Recursive:**
```python
def lowestCommonAncestor_recursive(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    if p.val < root.val and q.val < root.val:
        return lowestCommonAncestor_recursive(root.left, p, q)
    if p.val > root.val and q.val > root.val:
        return lowestCommonAncestor_recursive(root.right, p, q)
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
        elif val < root.val:
            root = root.left
        else:
            root = root.right
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
    if not preorder:
        return None

    # First element of preorder is root
    root_val = preorder[0]
    root = TreeNode(root_val)

    # Find root in inorder
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
        if left > right:
            return None

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
    if not p and not q:
        return True
    if not p or not q:
        return False
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
        if not left and not right:
            return True
        if not left or not right:
            return False
        if left.val != right.val:
            return False

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
        if not node:
            return False
        if is_same(node, subRoot):
            return True
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
    if not root:
        return None

    # If current node is p or q, it's an ancestor
    if root == p or root == q:
        return root

    # Search in subtrees
    left = lowestCommonAncestor(root.left, p, q)
    right = lowestCommonAncestor(root.right, p, q)

    # If both subtrees return non-null, current node is LCA
    if left and right:
        return root

    # Otherwise, return the non-null result
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
