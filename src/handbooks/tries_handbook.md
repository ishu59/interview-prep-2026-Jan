# The Complete Tries Handbook
> A template-based approach for mastering Tries in coding interviews

**Philosophy:** A Trie is not just a tree for storing strings. It's about **sharing common prefixes** to enable O(L) operations where L is the word length, regardless of how many words are stored.

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

- **The Filing Cabinet with Tabbed Folders**: Imagine a filing cabinet where the top drawer has 26 tabs (A-Z). Under each tab, there's another set of 26 tabs for the second letter. Under each of those, another 26 for the third letter. To find "CAT", you open C, then A, then T. Words that share a prefix (CAR, CAT, CAN) share the same drawers up to where they diverge.
- **The Shared Prefix**: The whole magic of a trie is that common beginnings are stored once. If you have 1000 words starting with "pre-", the "p-r-e" path exists only once, not 1000 times.

### No-Jargon Translation

- **Trie**: a tree where each level represents one character position, and paths from root to leaves spell out words -- pronounced "try"
- **Node**: one letter position
- **Prefix**: the beginning portion of a word -- like "un" in "undo", "undo", "under"
- **Terminal/end marker**: a flag on a node saying "a complete word ends here" -- because "car" ends at 'r' even though "card" continues further
- **Child**: the next possible letter from this position

### Mental Model

> "A trie is a filing cabinet where each drawer is one letter deep, and words that start the same way share the same drawers -- so looking up any word takes exactly as many drawer-opens as the word has letters."

---

### What is a Trie?

A Trie (pronounced "try") is a tree-like data structure for storing strings where:
- Each node represents a character
- Path from root to node represents a prefix
- Words sharing prefixes share the same path

```
Words: ["app", "apple", "application", "apt"]

            (root)
              |
              a
              |
              p
             / \
            p   t*
           /
          l*
          |
          e*
          |
          ...
          |
          n*

* marks end of a word
```

### Why Tries?

| Operation | Array/Set | Trie |
|-----------|----------|------|
| Insert word | O(1) amortized | O(L) |
| Search word | O(L) average | O(L) |
| Search prefix | O(n × L) | O(L) |
| Autocomplete | O(n × L) | O(L + results) |

**Key Insight:** Tries excel at **prefix-based operations** because all words with a common prefix share the same path.

### Time & Space Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Insert | O(L) | O(L) per word |
| Search | O(L) | - |
| StartsWith | O(L) | - |
| Total Space | - | O(N × L × A) |

Where L = word length, N = number of words, A = alphabet size

### The Trade-off

**Pros:**
- O(L) search regardless of dictionary size
- Efficient prefix matching
- Autocomplete support
- Lexicographic ordering

**Cons:**
- More space than hash set (especially for sparse words)
- Cache-unfriendly (pointer chasing)

---

<a name="master-templates"></a>
## 2. The Master Templates

### Template A: Basic Trie (Array-based)

```python
class TrieNode:
    def __init__(self):
        self.children = [None] * 26  # For lowercase a-z
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def _char_to_index(self, char: str) -> int:
        return ord(char) - ord('a')

    def insert(self, word: str) -> None:
        node = self.root
        for char in word:
            index = self._char_to_index(char)
            # If no child exists for this character, create one.
            # This is insertion: we build the path that does not yet exist.
            if not node.children[index]:
                node.children[index] = TrieNode()
            node = node.children[index]
        # Why set is_end here? Because "app" is a prefix of "apple",
        # but we need to know "app" is also a complete word on its own.
        # Without this flag, search("app") could not distinguish
        # "app was inserted" from "app is just a prefix of apple".
        node.is_end = True

    def search(self, word: str) -> bool:
        node = self._find_node(word)
        # Two conditions must both be true:
        # - `node is not None`: the full path for the word exists in the trie
        # - `node.is_end`: someone actually inserted this exact word,
        #   not just a longer word that shares this prefix
        return node is not None and node.is_end

    def startsWith(self, prefix: str) -> bool:
        # Unlike search, we do NOT check is_end here.
        # A prefix just needs the path to exist -- it does not need to
        # be a complete word. So `is not None` alone is sufficient.
        return self._find_node(prefix) is not None

    def _find_node(self, prefix: str) -> TrieNode:
        node = self.root
        for char in prefix:
            index = self._char_to_index(char)
            # If the child slot is empty (None), the trie has no path
            # for this character at this position -- the prefix does
            # not exist, so return None immediately.
            if not node.children[index]:
                return None
            node = node.children[index]
        return node
```

---

### Template B: Trie with Dictionary (Flexible)

```python
class TrieNode:
    def __init__(self):
        self.children = {}  # char -> TrieNode
        self.is_end = False
        self.word = None    # Optional: store the word itself

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for char in word:
            # If this character has no child node yet, create one.
            # Unlike an array-based trie, the dict starts empty,
            # so we only allocate nodes for characters that actually appear.
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        # Mark this node as a complete word (see Template A for why).
        node.is_end = True
        # Storing the full word here lets us retrieve it directly
        # during DFS without rebuilding it character by character.
        node.word = word  # Optional

    def search(self, word: str) -> bool:
        node = self._find_node(word)
        # Must check both: path exists AND it marks a complete word.
        return node is not None and node.is_end

    def startsWith(self, prefix: str) -> bool:
        # Only need the path to exist -- no is_end check needed.
        return self._find_node(prefix) is not None

    def _find_node(self, prefix: str) -> TrieNode:
        node = self.root
        for char in prefix:
            # If the character is missing from children, the trie
            # has no continuation for this prefix -- return None.
            if char not in node.children:
                return None
            node = node.children[char]
        return node
```

---

### Template C: Trie with Count

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.count = 0      # Words ending at this node
        self.prefix_count = 0  # Words with this prefix

class TrieWithCount:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            # Increment prefix_count at every node along the path,
            # because every node on this path is a prefix of the word
            # being inserted (e.g., inserting "apple" increments counts
            # at nodes for "a", "ap", "app", "appl", "apple").
            node.prefix_count += 1
        # Only increment count at the final node -- this tracks how
        # many times this exact word (not just prefix) was inserted.
        node.count += 1

    def countWordsEqualTo(self, word: str) -> int:
        node = self._find_node(word)
        # If node is None, the word was never inserted; return 0.
        # Otherwise return count (exact word matches, not prefix matches).
        return node.count if node else 0

    def countWordsStartingWith(self, prefix: str) -> int:
        node = self._find_node(prefix)
        # If node is None, no word with this prefix exists; return 0.
        # Otherwise prefix_count tells us how many inserted words
        # pass through this node (i.e., have this prefix).
        return node.prefix_count if node else 0

    def erase(self, word: str) -> None:
        node = self.root
        for char in word:
            # If the path does not exist, the word was never inserted,
            # so there is nothing to erase -- return early.
            if char not in node.children:
                return
            node = node.children[char]
            # Decrement prefix_count at each node along the path,
            # reversing what insert did.
            node.prefix_count -= 1
        node.count -= 1

    def _find_node(self, prefix: str) -> TrieNode:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node
```

---

### Template D: Trie with Wildcard Search

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class WildcardTrie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    def search(self, word: str) -> bool:
        """Search with '.' as wildcard matching any character."""
        def dfs(node, index):
            # Base case: we have matched every character in the word.
            # Now check is_end: did a complete word end at this node?
            # Just reaching the node is not enough (e.g., path "app"
            # exists because of "apple", but "app" is only a word if
            # is_end is True).
            if index == len(word):
                return node.is_end

            char = word[index]
            # Why check `char == '.'`?
            # '.' is a wildcard that can match ANY single character.
            # We cannot just follow one path -- we must try every
            # child because any character could be the match. If ANY
            # child path leads to a successful match, we return True.
            if char == '.':
                for child in node.children.values():
                    if dfs(child, index + 1):
                        return True
                # None of the children led to a match.
                return False
            else:
                # For a regular character, the path must exist.
                # If it does not, the word cannot be in the trie.
                if char not in node.children:
                    return False
                return dfs(node.children[char], index + 1)

        return dfs(self.root, 0)
```

---

### Template E: Trie for Numbers (XOR Problems)

```python
class BitTrie:
    """Trie for storing binary representations of numbers."""
    def __init__(self, max_bits=32):
        self.root = {}
        self.max_bits = max_bits

    def insert(self, num: int) -> None:
        node = self.root
        # Why iterate from MSB (most significant bit) to LSB?
        # XOR maximization is greedy: a 1 in a higher bit position
        # contributes more than all lower bits combined. So we must
        # store and compare from the top bit downward.
        for i in range(self.max_bits - 1, -1, -1):
            bit = (num >> i) & 1
            if bit not in node:
                node[bit] = {}
            node = node[bit]

    def max_xor(self, num: int) -> int:
        """Find number in trie that gives maximum XOR with num."""
        node = self.root
        result = 0
        for i in range(self.max_bits - 1, -1, -1):
            bit = (num >> i) & 1
            # To maximize XOR, we want the opposite bit at each position.
            # XOR gives 1 when bits differ, so picking the opposite bit
            # sets this bit position to 1 in the result.
            want = 1 - bit
            if want in node:
                # The opposite bit exists -- this bit position contributes
                # to the XOR result, so set it in the output.
                result |= (1 << i)
                node = node[want]
            elif bit in node:
                # The opposite bit does not exist; we must follow the same
                # bit (XOR is 0 at this position -- no contribution).
                node = node[bit]
            else:
                # No path exists at all (empty trie branch) -- stop early.
                break
        return result
```

---

### Quick Decision Matrix

| Problem Type | Template | Key Feature |
|--------------|----------|-------------|
| Basic insert/search | A or B | Standard operations |
| Count words/prefixes | C | Count tracking |
| Wildcard matching | D | DFS with backtracking |
| XOR maximum | E | Bit-based trie |
| Autocomplete | B + DFS | Collect all words from node |
| Word search in grid | B + DFS | Combined with backtracking |

---

<a name="pattern-guide"></a>
## 3. Pattern Classification Guide

### Category 1: Basic Trie Operations
- Insert, search, prefix search
- **Template A or B**

### Category 2: Counting
- Count words equal to / starting with prefix
- **Template C**

### Category 3: Pattern Matching
- Wildcard characters
- Regex-like patterns
- **Template D**

### Category 4: Autocomplete / Suggestions
- Find all words with prefix
- Top k words with prefix
- **Template B + DFS**

### Category 5: XOR Problems
- Maximum XOR pair
- XOR queries
- **Template E**

### Category 6: Word Games
- Word search in grid
- Boggle
- **Template B + backtracking**

---

<a name="patterns"></a>
## 4. Complete Pattern Library

### PATTERN 1: Basic Trie Operations

---

#### Pattern 1A: Implement Trie

**Problem:** LeetCode 208 - Implement Trie (Prefix Tree)

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for char in word:
            # Create the child node only if it does not exist yet.
            # If it already exists (from a previously inserted word
            # sharing the same prefix), just follow the existing path.
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        # Mark the end of a complete word. Without this, we could not
        # tell "app" (inserted) from "app" (just a prefix of "apple").
        node.is_end = True

    def search(self, word: str) -> bool:
        node = self.root
        for char in word:
            # If the character is missing, the word was never inserted.
            # This is the key difference from insert: insert creates
            # the missing node, search returns False.
            if char not in node.children:
                return False
            node = node.children[char]
        # Why `return node.is_end` and not `return True`?
        # The full path might exist only because a longer word was inserted.
        # Example: if "apple" was inserted, the path for "app" exists, but
        # "app" was not inserted -- is_end=False guards against false positives.
        return node.is_end

    def startsWith(self, prefix: str) -> bool:
        node = self.root
        for char in prefix:
            # Why `char not in node.children`?
            # The prefix diverges from all stored words at this character.
            # No inserted word begins with the characters typed so far.
            if char not in node.children:
                return False
            node = node.children[char]
        # Why `return True` (not `return node.is_end`)?
        # startsWith only asks "does any stored word begin with this prefix?"
        # The prefix does NOT need to be a complete word itself -- even a
        # pure prefix like "app" (from "apple") should return True.
        return True
```

---

#### Pattern 1B: Design Add and Search Words Data Structure

**Problem:** LeetCode 211 - Support '.' as wildcard

```python
class WordDictionary:
    def __init__(self):
        self.root = {}

    def addWord(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        # '$' is the end-of-word sentinel. Using a special key avoids
        # needing a separate TrieNode class -- just nested dicts.
        node['$'] = True

    def search(self, word: str) -> bool:
        def dfs(node, i):
            # Base case: all characters matched. Check if a complete
            # word ends here by looking for the '$' sentinel.
            if i == len(word):
                return '$' in node

            char = word[i]
            # Why `char == '.'`?
            # '.' is a wildcard that must match exactly one character.
            # We cannot follow a single path -- we must branch into every
            # real child because any letter could satisfy the wildcard.
            if char == '.':
                # '.' matches any character, so we must try EVERY child.
                # We skip '$' because it is not a real character -- it is
                # the end-of-word marker and cannot match '.'.
                for key in node:
                    # Why `key != '$'`?
                    # '$' is an internal sentinel for end-of-word, not an
                    # actual character in any string. Recursing into it
                    # would treat the sentinel as a real letter, causing
                    # incorrect matches.
                    if key != '$' and dfs(node[key], i + 1):
                        return True
                return False
            else:
                # Exact character match: the path must exist.
                # Why `char not in node`?
                # If this exact character has no child, no stored word
                # follows this path -- return False immediately rather
                # than crashing on a missing key.
                if char not in node:
                    return False
                return dfs(node[char], i + 1)

        return dfs(self.root, 0)
```

---

### PATTERN 2: Word Search Problems

---

#### Pattern 2A: Word Search II

**Problem:** LeetCode 212 - Find all words from dictionary in grid

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        # Why store the whole word instead of just is_end?
        # When DFS finds a match deep in the grid, we need to know
        # WHICH word was found. Storing the word here avoids having
        # to reconstruct it by tracking the path of characters.
        self.word = None

def findWords(board: list[list[str]], words: list[str]) -> list[str]:
    # Build Trie
    root = TrieNode()
    for word in words:
        node = root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.word = word

    rows, cols = len(board), len(board[0])
    result = []

    def dfs(r, c, node):
        char = board[r][c]
        # If the current cell's character has no matching child in the
        # trie, no dictionary word can be formed along this path.
        # Pruning here avoids exploring dead-end paths entirely.
        if char not in node.children:
            return

        next_node = node.children[char]

        # Why check `next_node.word` (not just `next_node.is_end`)?
        # We stored the full word string here, so a non-None value
        # means a complete dictionary word ends at this trie node.
        if next_node.word:
            result.append(next_node.word)
            # Set to None so the same word is not collected again if
            # the same path is reached from a different starting cell.
            next_node.word = None

        # Mark visited by replacing with '#'. This prevents the DFS
        # from revisiting this cell in the current path (no cycles).
        board[r][c] = '#'

        # Explore all 4 neighbors
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            # Why `0 <= nr < rows and 0 <= nc < cols`?
            # Grid boundaries: stepping outside means we fell off the board.
            # Why `board[nr][nc] != '#'`?
            # '#' marks cells already on the current DFS path. Revisiting
            # one would create a cycle, using the same grid cell twice in
            # one word -- which is not allowed.
            if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] != '#':
                dfs(nr, nc, next_node)

        # Restore the original character (backtracking) so other
        # starting cells can use this cell in their paths.
        board[r][c] = char

        # Why `not next_node.children`?
        # If the node has no children left, it is a dead leaf -- no more
        # words can be discovered through it. Deleting it prunes the trie
        # so future DFS calls skip this branch immediately, a key
        # optimization for large boards with many starting cells.
        if not next_node.children:
            del node.children[char]

    # Start DFS from every cell -- any cell could be the first letter
    # of a dictionary word.
    for i in range(rows):
        for j in range(cols):
            dfs(i, j, root)

    return result
```

---

### PATTERN 3: Prefix-Based Problems

---

#### Pattern 3A: Longest Common Prefix

**Problem:** Using Trie to find longest common prefix of all strings

```python
def longestCommonPrefix(strs: list[str]) -> str:
    # Edge case: no strings means no common prefix.
    if not strs:
        return ""

    # Build Trie
    root = {}
    for word in strs:
        # If any word is empty, the common prefix is empty --
        # every word must share the prefix, and "" has nothing.
        if not word:
            return ""
        node = root
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        node['$'] = True

    # Find LCP: follow the trie path while it does not branch.
    prefix = []
    node = root
    while True:
        # Why `'$' in node`?
        # If an end-of-word sentinel exists here, one of the input strings
        # terminates at this depth. Extending the prefix further would
        # exclude that shorter string, so we must stop.
        # Why `len(node) != 1`?
        # More than one child means the strings diverge here (e.g., "flower"
        # goes to 'l' while "flow" ends at '$'). A single child means all
        # strings still agree on the next character.
        if '$' in node or len(node) != 1:
            break
        char = next(iter(node.keys()))
        prefix.append(char)
        node = node[char]

    return ''.join(prefix)
```

**Alternative (character-by-character):**
```python
def longestCommonPrefix_simple(strs: list[str]) -> str:
    if not strs:
        return ""

    prefix = strs[0]
    for s in strs[1:]:
        # Shrink the prefix until it matches the start of s.
        # Why `while` and not `if`? Because we may need to chop off
        # more than one character -- e.g., prefix="flower" and
        # s="flight" requires shrinking to "fl".
        # Why `while not s.startswith(prefix)` (not `if`)?
        # A single chop may not be enough -- "flower" vs "flight" requires
        # removing 'e', 'r', 'w', 'o' one at a time until "fl" remains.
        # `if` would only remove one character per string, giving wrong results.
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            # Why `if not prefix`?
            # Once the prefix shrinks to empty, no common prefix exists.
            # Returning immediately avoids an infinite loop (an empty string
            # is always a prefix, so `while` would never terminate).
            if not prefix:
                return ""

    return prefix
```

---

#### Pattern 3B: Search Suggestions System

**Problem:** LeetCode 1268 - Return suggestions for each prefix

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.suggestions = []  # Store up to 3 words

def suggestedProducts(products: list[str], searchWord: str) -> list[list[str]]:
    products.sort()  # Sort for lexicographic order

    # Build Trie
    root = TrieNode()
    for product in products:
        node = root
        for char in product:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            # Why `len(node.suggestions) < 3`?
            # The problem asks for at most 3 suggestions per prefix.
            # Because products are pre-sorted lexicographically, the first
            # 3 that pass through this node are already the lexicographically
            # smallest -- so we cap at 3 and ignore later products.
            if len(node.suggestions) < 3:
                node.suggestions.append(product)

    # Search
    result = []
    node = root
    for char in searchWord:
        # Why `node` (truthiness check first)?
        # Once a previous character fell off the trie, `node` was set to
        # None. Checking it first avoids a crash on `char in node.children`.
        # Why `char in node.children`?
        # The typed character may not match any stored product at this depth,
        # meaning no product shares this prefix -- no suggestions possible.
        if node and char in node.children:
            node = node.children[char]
            result.append(node.suggestions)
        else:
            # Once we leave the trie, every subsequent character also
            # has no suggestions, so set node = None to short-circuit.
            node = None
            result.append([])

    return result
```

---

#### Pattern 3C: Replace Words

**Problem:** LeetCode 648 - Replace words with their shortest root

```python
def replaceWords(dictionary: list[str], sentence: str) -> str:
    # Build Trie
    root = {}
    for word in dictionary:
        node = root
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        node['$'] = word  # Store the root word

    def find_root(word):
        node = root
        for char in word:
            # Why `char not in node`?
            # If the trie has no path for this character, no dictionary
            # root starts with this prefix -- the word has no replacement,
            # so return it unchanged.
            if char not in node:
                return word
            node = node[char]
            # Why check '$' at every step (not just at the end)?
            # We want the SHORTEST root. The first time we hit a '$'
            # marker, we have found the shortest dictionary root that
            # is a prefix of this word. Returning immediately ensures
            # "cat" is preferred over "cattle" as a root for "cattle".
            if '$' in node:
                return node['$']
        return word

    words = sentence.split()
    return ' '.join(find_root(word) for word in words)
```

---

### PATTERN 4: Autocomplete

---

#### Pattern 4A: Top K Frequent Words with Prefix

```python
import heapq
from collections import defaultdict

class AutocompleteSystem:
    def __init__(self, sentences: list[str], times: list[int]):
        self.root = {}
        self.current_node = self.root
        self.current_input = []
        self.counts = defaultdict(int)

        for sentence, count in zip(sentences, times):
            self.counts[sentence] = count
            self._insert(sentence)

    def _insert(self, sentence: str) -> None:
        node = self.root
        for char in sentence:
            if char not in node:
                node[char] = {}
            node = node[char]
        node['$'] = sentence

    def _get_all_sentences(self, node) -> list[str]:
        """Get all sentences under this node."""
        sentences = []

        def dfs(n):
            # Why `'$' in n`?
            # '$' signals that a complete sentence ends at this trie node.
            # Without this check we would only collect sentences at leaf
            # nodes, missing any sentence that is a prefix of another.
            if '$' in n:
                sentences.append(n['$'])
            for char in n:
                # Why `char != '$'`?
                # '$' is the end-of-word sentinel, not a real character.
                # Recursing into it would crash because n['$'] is True
                # (a bool, not a dict), so we must skip it.
                if char != '$':
                    dfs(n[char])

        dfs(node)
        return sentences

    def input(self, c: str) -> list[str]:
        # '#' signals the user finished typing a sentence.
        # We save it, insert it into the trie, and reset state.
        if c == '#':
            sentence = ''.join(self.current_input)
            self.counts[sentence] += 1
            self._insert(sentence)
            self.current_input = []
            self.current_node = self.root
            return []

        self.current_input.append(c)

        # Why `self.current_node is None`?
        # Once any character in the current input fell off the trie,
        # the running prefix cannot match any sentence. We use None as
        # a sentinel to avoid re-checking the trie on every subsequent
        # character -- early exit for all remaining characters.
        if self.current_node is None:
            return []

        # Why `c not in self.current_node`?
        # The newly typed character has no branch in the trie at this
        # depth, so no stored sentence shares the current prefix.
        # Set to None so future characters also short-circuit immediately.
        if c not in self.current_node:
            self.current_node = None
            return []

        self.current_node = self.current_node[c]

        # Get all sentences with current prefix
        sentences = self._get_all_sentences(self.current_node)

        # Sort by frequency (desc) then lexicographically.
        # Negative count gives descending frequency; 'x' gives
        # ascending alphabetical order as tiebreaker.
        sentences.sort(key=lambda x: (-self.counts[x], x))

        return sentences[:3]
```

---

### PATTERN 5: XOR Problems

---

#### Pattern 5A: Maximum XOR of Two Numbers

**Problem:** LeetCode 421 - Find maximum XOR of any two numbers

```python
def findMaximumXOR(nums: list[int]) -> int:
    # Find max number of bits needed
    max_num = max(nums)
    max_bits = max_num.bit_length()

    # Build Trie
    root = {}
    for num in nums:
        node = root
        for i in range(max_bits - 1, -1, -1):
            bit = (num >> i) & 1
            if bit not in node:
                node[bit] = {}
            node = node[bit]

    # Find max XOR for each number
    max_xor = 0
    for num in nums:
        node = root
        current_xor = 0
        for i in range(max_bits - 1, -1, -1):
            bit = (num >> i) & 1
            # Want opposite bit for max XOR.
            # XOR of two equal bits is 0; XOR of opposite bits is 1.
            # Choosing the opposite at every position greedily maximizes
            # the result because higher bits contribute more value.
            want = 1 - bit
            # Why `want in node`?
            # If the preferred (opposite) bit exists in the trie, take it:
            # this bit position becomes 1 in the XOR result.
            if want in node:
                current_xor |= (1 << i)
                node = node[want]
            else:
                # The opposite bit does not exist; we must follow the same
                # bit. XOR at this position is 0 -- no contribution to result.
                node = node[bit]
        max_xor = max(max_xor, current_xor)

    return max_xor
```

---

#### Pattern 5B: Maximum XOR With Element From Array

**Problem:** LeetCode 1707 - Answer queries for max XOR with bound

```python
def maximizeXor(nums: list[int], queries: list[list[int]]) -> list[int]:
    nums.sort()

    # Sort queries by limit, keep original index
    indexed_queries = sorted(enumerate(queries), key=lambda x: x[1][1])

    root = {}
    max_bits = 32
    result = [-1] * len(queries)
    j = 0

    def insert(num):
        node = root
        for i in range(max_bits - 1, -1, -1):
            bit = (num >> i) & 1
            if bit not in node:
                node[bit] = {}
            node = node[bit]

    def query(num):
        # Why `not root`?
        # If no numbers have been inserted yet (trie is empty), there is
        # no valid pair to XOR with. Return -1 per problem specification.
        if not root:
            return -1
        node = root
        xor_val = 0
        for i in range(max_bits - 1, -1, -1):
            bit = (num >> i) & 1
            want = 1 - bit
            # Why `want in node`?
            # Prefer the opposite bit to maximize XOR. If it exists in
            # the trie (some number has this bit value), take the greedy
            # choice and record a 1 at this bit position.
            if want in node:
                xor_val |= (1 << i)
                node = node[want]
            # Why `elif bit in node`?
            # Opposite bit unavailable; settle for the same bit (XOR = 0
            # here). We still need to continue traversing to lower bits.
            elif bit in node:
                node = node[bit]
            else:
                # Neither bit exists -- no valid number in trie matches.
                # This should not happen if the trie is correctly built,
                # but we guard against it to avoid a KeyError crash.
                return -1
        return xor_val

    for idx, (x, m) in indexed_queries:
        # Why `nums[j] <= m`?
        # Each query has a bound m: only numbers <= m may be used in the XOR.
        # Because both nums and queries are sorted by bound, a two-pointer
        # approach works: insert nums one by one until we exceed the current
        # query's limit, then answer the query with only valid numbers in the trie.
        # Why `j < len(nums)`?
        # Guard against reading past the end of the nums array.
        while j < len(nums) and nums[j] <= m:
            insert(nums[j])
            j += 1
        result[idx] = query(x)

    return result
```

---

### PATTERN 6: Counting Problems

---

#### Pattern 6A: Count Pairs With XOR in a Range

**Problem:** Count pairs where l <= (a XOR b) <= r

```python
def countPairs(nums: list[int], low: int, high: int) -> int:
    def count_less_than(limit):
        """Count pairs with XOR < limit."""
        root = {}
        count = 0

        for num in nums:
            # Query: how many previous numbers XOR with num < limit
            node = root
            for i in range(14, -1, -1):
                # Why `node is None`?
                # The trie ran out of branches -- no more numbers satisfy
                # the XOR constraint at this depth. Stop early.
                if node is None:
                    break

                bit = (num >> i) & 1
                limit_bit = (limit >> i) & 1

                # Why `limit_bit == 1`?
                # When the limit has a 1 at this bit position, any XOR
                # result with a 0 here is automatically less than the
                # limit (the higher bit is 0 < 1). We can count ALL
                # numbers in the "same bit" subtree immediately and then
                # continue exploring the "opposite bit" branch to see if
                # XOR can equal 1 here but still be under limit in lower bits.
                if limit_bit == 1:
                    # If we go same as bit, XOR is 0 for this position
                    # All numbers in that subtree contribute
                    if bit in node:
                        count += node[bit].get('count', 0)
                    # Continue with opposite (XOR = 1)
                    node = node.get(1 - bit)
                else:
                    # Why follow the same bit when limit_bit == 0?
                    # We need XOR to be 0 at this position (since limit
                    # is 0 here). Going opposite would make XOR = 1 at
                    # this position, which already exceeds the limit bit.
                    # Must go same to keep XOR < limit
                    node = node.get(bit)

            # Insert current number
            node = root
            for i in range(14, -1, -1):
                bit = (num >> i) & 1
                if bit not in node:
                    node[bit] = {'count': 0}
                node = node[bit]
                node['count'] = node.get('count', 0) + 1

        return count

    return count_less_than(high + 1) - count_less_than(low)
```

---

### PATTERN 7: Stream Processing

---

#### Pattern 7A: Stream of Characters

**Problem:** LeetCode 1032 - Check if suffix of stream matches any word

```python
class StreamChecker:
    def __init__(self, words: list[str]):
        # Build Trie with reversed words
        self.root = {}
        for word in words:
            node = self.root
            for char in reversed(word):
                if char not in node:
                    node[char] = {}
                node = node[char]
            node['$'] = True

        self.stream = []

    def query(self, letter: str) -> bool:
        self.stream.append(letter)

        # Check if any word is suffix of current stream.
        # We walk backwards through the stream because the trie stores
        # reversed words -- the last character is the trie's first level.
        node = self.root
        for i in range(len(self.stream) - 1, -1, -1):
            char = self.stream[i]
            # Why `char not in node`?
            # The current suffix character has no matching branch in the
            # reversed-word trie, so no dictionary word ends with this
            # suffix. Return False immediately (no point going further back).
            if char not in node:
                return False
            node = node[char]
            # Why `'$' in node` inside the loop (not just at the end)?
            # Words can have different lengths. Checking at each step lets
            # us detect a match as soon as the shortest matching suffix is
            # confirmed, rather than requiring a full loop iteration.
            if '$' in node:
                return True

        return False
```

---

<a name="post-processing"></a>
## 5. Post-Processing Reference

| Operation | Result | Notes |
|-----------|--------|-------|
| **Search word** | Boolean | Check is_end |
| **Search prefix** | Boolean | Node exists |
| **Get all words** | List | DFS from node |
| **Get count** | Integer | Track in node |
| **Max XOR** | Integer | Greedy opposite bits |

---

<a name="pitfalls"></a>
## 6. Common Pitfalls & Solutions

### Pitfall 1: Forgetting End Marker

```python
# WRONG: No way to distinguish prefix from complete word
def search(self, word):
    node = self._find_node(word)
    return node is not None  # "app" matches even if only "apple" inserted
```

**Solution:** Use `is_end` flag

---

### Pitfall 2: Not Handling Empty String

```python
# Edge case: empty string
trie.insert("")  # Should this be valid?
trie.search("")  # Should return True if inserted
```

---

### Pitfall 3: Memory Explosion

```python
# Can use lots of memory with sparse data
# Each node can have up to 26 children
```

**Solution:** Use dictionary for sparse tries, or implement compression

---

### Pitfall 4: Modifying Trie During Search

```python
# DANGEROUS: Modifying while iterating
for char in node.children:
    del node.children[char]  # Don't do this!
```

---

<a name="recognition"></a>
## 7. Problem Recognition Framework

### Step 1: Is Trie Appropriate?

**Good indicators:**
1. Multiple strings with shared prefixes
2. Prefix-based queries (startsWith)
3. Autocomplete or suggestions
4. Multiple pattern matching
5. XOR maximization

**NOT Trie if:**
- Single string operations
- No prefix relationships
- Exact match only (hash set simpler)

### Step 2: Choose Trie Variant

| Feature Needed | Variant |
|----------------|---------|
| Basic insert/search | Template A or B |
| Count words | Template C |
| Wildcards | Template D |
| XOR problems | Template E |

### Decision Tree

```
        Need prefix operations?
               ↓
         ┌─────┴─────┐
        Yes          No
         ↓            ↓
    Multiple       Use Hash
    strings?
         ↓
    ┌────┴────┐
   Yes        No
    ↓          ↓
  Trie      Simpler approach

    Trie Type:
    - XOR → BitTrie
    - Wildcards → DFS search
    - Count → CountTrie
    - Basic → Standard Trie
```

---

<a name="checklist"></a>
## 8. Interview Preparation Checklist

### Before the Interview

**Master the fundamentals:**
- [ ] Can implement basic Trie from memory
- [ ] Understand insert, search, startsWith
- [ ] Know dictionary vs array children trade-off
- [ ] Can implement wildcard search

**Know the patterns:**
- [ ] Basic Trie operations
- [ ] Word search in grid
- [ ] Autocomplete
- [ ] XOR maximum

**Common problems solved:**
- [ ] LC 208: Implement Trie
- [ ] LC 211: Design Add and Search Words
- [ ] LC 212: Word Search II
- [ ] LC 421: Maximum XOR of Two Numbers
- [ ] LC 648: Replace Words

### During the Interview

**1. Clarify (30 seconds)**
- Character set? (a-z, alphanumeric, any?)
- Case sensitive?
- Prefix queries needed?

**2. Identify need for Trie (30 seconds)**
- Multiple strings?
- Prefix-based operations?
- Multiple lookups?

**3. Code (3-4 minutes)**
- TrieNode class
- Insert method
- Search/query method

**4. Test (1-2 minutes)**
- Empty string
- Single character
- Common prefixes
- No match case

---

## 9. Quick Reference Cards

### Basic Trie
```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

def insert(root, word):
    node = root
    for char in word:
        if char not in node.children:
            node.children[char] = TrieNode()
        node = node.children[char]
    node.is_end = True
```

### XOR Trie Insert
```python
def insert(root, num, bits=32):
    node = root
    for i in range(bits - 1, -1, -1):
        bit = (num >> i) & 1
        if bit not in node:
            node[bit] = {}
        node = node[bit]
```

---

## 10. Complexity Reference

| Operation | Time | Space |
|-----------|------|-------|
| Insert | O(L) | O(L) |
| Search | O(L) | O(1) |
| StartsWith | O(L) | O(1) |
| All words with prefix | O(L + R) | O(R) |
| Total Trie space | O(N × L) | - |

Where L = word length, N = number of words, R = results

---

## Final Thoughts

**Remember:**
1. Trie = tree where path represents prefix
2. Mark end of words with `is_end` flag
3. Dictionary children are more flexible than array
4. XOR Trie stores bits from MSB to LSB
5. Prune branches when possible for optimization

**When stuck:**
1. Draw the Trie structure
2. Trace through insert and search
3. Consider if simpler structure (hash set) works
4. For XOR, think about maximizing bit by bit

---

## Appendix: Practice Problem Set

### Medium
- 208. Implement Trie
- 211. Design Add and Search Words Data Structure
- 421. Maximum XOR of Two Numbers
- 648. Replace Words
- 677. Map Sum Pairs
- 720. Longest Word in Dictionary
- 1268. Search Suggestions System

### Hard
- 212. Word Search II
- 336. Palindrome Pairs
- 472. Concatenated Words
- 1032. Stream of Characters
- 1707. Maximum XOR With an Element From Array

**Recommended Practice Order:**
1. Basic: 208 → 211
2. Applications: 648 → 720 → 1268
3. Word Search: 212
4. XOR: 421 → 1707

Good luck with your interview preparation!

---

## Appendix: Conditional Quick Reference

This table lists every key condition used in this handbook, its plain-English meaning, and the intuition behind it.

### A. Insert / Search Core

| Condition | Plain English | Why it works |
|-----------|---------------|--------------|
| `char not in node.children` | This character has no child node yet | Prevents a KeyError crash; on insert we create the node, on search/startsWith we return False/None immediately |
| `not node.children[index]` | The array slot for this character is still None | Array-based trie equivalent of the dict check above -- None means no child was ever created for this character |
| `node.is_end` | A complete word was explicitly inserted ending at this node | Distinguishes an inserted word from a word that is merely a prefix of a longer stored word (e.g., "app" vs "apple") |
| `node is not None and node.is_end` | The full path exists AND marks a complete word | Both halves required: path alone could be a prefix; is_end alone is unreachable if path is missing |
| `return node.is_end` (in search) | Return whether a real word ends here, not just True | Returning True would falsely match prefixes; is_end is False for nodes that only exist as interior stops |
| `return True` (in startsWith) | Any existing path is a valid prefix | startsWith does not require a complete word -- the path existing is sufficient proof |
| `node is not None` (in startsWith) | The entire prefix path was found without breaking | If any character was missing, _find_node returned None, meaning the prefix does not exist |

### B. Prefix / Wildcard Matching

| Condition | Plain English | Why it works |
|-----------|---------------|--------------|
| `char == '.'` | Current pattern character is a wildcard | '.' must match every possible letter, so we branch into all children instead of following one path |
| `key != '$'` (when iterating wildcard children) | Skip the end-of-word sentinel when expanding '.' | '$' is an internal marker (bool True), not a real character; recursing into it would crash or give wrong results |
| `'$' in node` (search end-of-word check) | A complete word/sentence ends at this node | Sentinel-based end marker -- checking at every step catches any-length matches, not just the deepest node |
| `'$' in node or len(node) != 1` (LCP loop) | A word ends here OR the trie branches here | Either condition means the common prefix cannot grow: a branch means strings diverge, a word-end means a shorter string would be excluded |
| `node and char in node.children` (suggestions search) | Node still valid AND this character has a branch | Guards against both a None node (previously fell off trie) and a missing character at the current depth |

### C. Word Break / DP on Trie

| Condition | Plain English | Why it works |
|-----------|---------------|--------------|
| `char not in node` (find_root loop) | Trie has no continuation for this character | The sentence word has no dictionary root as a prefix -- return the word unchanged rather than crashing |
| `if '$' in node` (find_root, inside loop) | The shortest dictionary root ends here | Checking at every step (not just the end) ensures we return the SHORTEST matching root, not a longer one deeper in the trie |
| `while not s.startswith(prefix)` | Current prefix does not match this string's start | Uses while (not if) because multiple characters may need removing -- "flower" vs "flight" shrinks from 6 to 2 characters |
| `if not prefix` (inside LCP while loop) | The prefix shrunk to empty -- no common prefix exists | Prevents an infinite loop (empty string always satisfies startswith) and returns the correct early-exit answer |
| `nums[j] <= m` (XOR with bound, while loop) | This number is within the query's allowed limit | Offline sorted queries: only insert numbers that satisfy the bound before answering each query, ensuring correctness |

### D. Delete / Cleanup Conditions

| Condition | Plain English | Why it works |
|-----------|---------------|--------------|
| `if char not in node.children` (erase early exit) | Word path does not exist -- nothing to erase | Gracefully handles erasing a word that was never inserted; prevents decrementing counts below zero |
| `if not next_node.children` (Word Search II prune) | Trie node is now a dead leaf with no remaining words | Once all words through this node are found, the branch is useless; deleting it prunes future DFS calls and speeds up the algorithm |
| `if self.current_node is None` (autocomplete guard) | A previous character already fell off the trie | Sentinel-based short-circuit: once we leave the trie, no subsequent character can match -- avoids re-checking the trie on every keystroke |
| `if not root` (XOR query guard) | Trie is completely empty -- no numbers inserted yet | Returning -1 immediately matches the problem spec and avoids a crash when traversing an empty dict |
| `if node is None: break` (countPairs loop) | Trie branch exhausted during XOR counting traversal | No valid paths remain at this bit depth; breaking prevents a NoneType attribute error and correctly terminates the count |
