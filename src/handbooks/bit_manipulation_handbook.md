# The Complete Bit Manipulation Handbook
> A template-based approach for mastering bit manipulation in coding interviews

**Philosophy:** Bit manipulation is not about memorizing tricks. It's about understanding that **integers are just sequences of bits**, and many operations become elegant single-line solutions when viewed through this lens.

---

## Table of Contents
1. [Understanding the Core Philosophy](#core-philosophy)
2. [The Master Templates](#master-templates)
3. [Pattern Classification Guide](#pattern-guide)
4. [Complete Pattern Library](#patterns)
5. [Common Pitfalls & Solutions](#pitfalls)
6. [Problem Recognition Framework](#recognition)
7. [Interview Preparation Checklist](#checklist)

---

<a name="core-philosophy"></a>
## 1. Understanding the Core Philosophy

### First Principles

- **The Row of Light Switches**: Every integer is a row of light switches (bits), each either ON (1) or OFF (0). The number 13 is just switches `1101` -- the 8-switch, the 4-switch, and the 1-switch are on. All bit manipulation is just flipping, checking, or combining switches.
- **The XOR Trick (The Toggle)**: XOR is a special switch combiner: if two switches are the same, the result is OFF; if different, ON. The magical property: if you XOR something with itself, it cancels out (turns OFF). This is why XOR finds "the one that's different" -- everything paired cancels, leaving the loner.

### No-Jargon Translation

- **Bit**: one switch: 0 or 1
- **AND**: both switches must be ON for the result to be ON -- like two keys both needed to open a lock
- **OR**: either switch ON makes the result ON
- **XOR**: switches must be different for ON -- like a toggle
- **NOT**: flip every switch
- **Left shift**: slide all switches left, padding with 0s on the right -- multiplies by 2
- **Right shift**: slide right -- divides by 2
- **Mask**: a specific switch pattern you use to test or set other switches -- like a stencil
- **Two's complement**: how computers represent negative numbers -- flip all bits and add 1

### Mental Model

> "Bit manipulation treats every number as a row of light switches, and every operation -- AND, OR, XOR, shift -- is a way to flip, check, or combine those switches without touching the others."

---

### Binary Representation

Every integer is stored as binary (base 2):
```
Decimal 13 = Binary 1101

Position:  3  2  1  0  (right to left, 0-indexed)
Bit:       1  1  0  1
Value:     8  4  0  1  = 13
```

### Essential Bit Operations

| Operator | Name | Description | Example |
|----------|------|-------------|---------|
| `&` | AND | 1 if both bits are 1 | `5 & 3 = 101 & 011 = 001 = 1` |
| `\|` | OR | 1 if either bit is 1 | `5 \| 3 = 101 \| 011 = 111 = 7` |
| `^` | XOR | 1 if bits are different | `5 ^ 3 = 101 ^ 011 = 110 = 6` |
| `~` | NOT | Flip all bits | `~5 = ...11111010` |
| `<<` | Left shift | Multiply by 2^n | `5 << 1 = 1010 = 10` |
| `>>` | Right shift | Divide by 2^n | `5 >> 1 = 10 = 2` |

### XOR Properties (The Most Important!)

```python
a ^ 0 = a           # XOR with 0 gives same number
a ^ a = 0           # XOR with itself gives 0
a ^ b ^ a = b       # XOR is self-inverse
a ^ b = b ^ a       # Commutative
(a ^ b) ^ c = a ^ (b ^ c)  # Associative
```

**Key Insight:** XOR of all numbers where one appears once and others appear twice → the unique number remains.

### Common Bit Tricks

```python
# Check if n is even/odd
# Why? The last bit IS the 1s place. Even numbers end in 0, odd in 1.
(n & 1) == 0  # even (MUST use parentheses — & has lower precedence than ==)
(n & 1) == 1  # odd

# Check if n is power of 2
# Why does n & (n-1) work? Powers of 2 have exactly ONE set bit: 1000.
# n-1 flips that bit and sets all lower bits: 0111.
# AND them: 1000 & 0111 = 0000. Only powers of 2 give 0.
# Non-power: 6=110, 5=101, 110&101=100 ≠ 0.
n > 0 and (n & (n - 1)) == 0

# Get i-th bit (0-indexed from right)
(n >> i) & 1

# Set i-th bit to 1
n | (1 << i)

# Clear i-th bit to 0
n & ~(1 << i)

# Toggle i-th bit
n ^ (1 << i)

# Clear all bits from LSB to i (inclusive)
n & ~((1 << (i + 1)) - 1)

# Clear all bits from MSB to i (inclusive)
n & ((1 << i) - 1)

# Get lowest set bit (rightmost 1)
# Why does n & (-n) work? In two's complement, -n = ~n + 1.
# Example: n=12=1100, -n=0100. AND: 1100 & 0100 = 0100 (the lowest 1).
# Flipping all bits and adding 1 turns off everything above the lowest 1.
n & (-n)  # or n & ~(n - 1)

# Clear lowest set bit
# Why? n-1 flips the lowest 1 and all bits below it.
# Example: n=12=1100, n-1=1011. AND: 1100 & 1011 = 1000 (lowest 1 gone).
n & (n - 1)

# Count set bits (Brian Kernighan's Algorithm)
# Each iteration clears exactly one set bit, so we loop
# only k times where k = number of 1s (not 32 times).
count = 0
while n:
    n &= (n - 1)  # Clear lowest set bit
    count += 1
```

### Two's Complement (Negative Numbers)

```
In 8-bit:
 5 = 00000101
-5 = 11111011  (flip bits, add 1)

-n = ~n + 1
~n = -n - 1
```

---

<a name="master-templates"></a>
## 2. The Master Templates

### Template A: Find Single Number (XOR)

```python
def findSingle(nums: list[int]) -> int:
    """
    Find the number that appears once when others appear twice.
    Uses XOR property: a ^ a = 0
    """
    result = 0
    for num in nums:
        result ^= num
    return result
```

---

### Template B: Count Set Bits

```python
def countBits(n: int) -> int:
    """
    Count number of 1s in binary representation.
    Brian Kernighan's algorithm.
    """
    count = 0
    while n:
        n &= (n - 1)  # Clear lowest set bit
        count += 1
    return count
```

---

### Template C: Check/Set/Clear/Toggle Bit

```python
class BitOperations:
    @staticmethod
    def get_bit(n: int, i: int) -> int:
        """Get i-th bit (0 or 1)."""
        return (n >> i) & 1

    @staticmethod
    def set_bit(n: int, i: int) -> int:
        """Set i-th bit to 1."""
        return n | (1 << i)

    @staticmethod
    def clear_bit(n: int, i: int) -> int:
        """Clear i-th bit to 0."""
        return n & ~(1 << i)

    @staticmethod
    def toggle_bit(n: int, i: int) -> int:
        """Toggle i-th bit."""
        return n ^ (1 << i)

    @staticmethod
    def update_bit(n: int, i: int, value: int) -> int:
        """Set i-th bit to value (0 or 1)."""
        mask = ~(1 << i)
        return (n & mask) | (value << i)
```

---

### Template D: Iterate Through Set Bits

```python
def iterate_set_bits(n: int):
    """
    Process each set bit in n.
    """
    # Why this approach instead of checking all 32 bits?
    # We only loop k times (k = number of set bits), skipping zeros entirely.
    while n:
        lowest_bit = n & (-n)  # Isolate lowest set bit (e.g., 12=1100 → 0100)
        bit_position = (lowest_bit - 1).bit_length()  # Convert to position index
        # Process bit at bit_position
        yield bit_position
        n &= (n - 1)  # Clear that bit and move to the next one
```

---

### Template E: Bitmask Subset Enumeration

```python
def enumerate_subsets(mask: int):
    """
    Enumerate all subsets of a bitmask.
    """
    # Why `(submask - 1) & mask`? Subtracting 1 "steps down" to the next
    # smaller number, and ANDing with mask keeps only bits within the
    # original set. This generates all subsets in decreasing order.
    # Example: mask=0b110 → submasks: 110, 100, 010, then 0 (empty).
    submask = mask
    while submask:
        yield submask
        submask = (submask - 1) & mask
    yield 0  # Empty subset
```

---

### Template F: Bit DP State

```python
def bitmask_dp(n: int, elements: list):
    """
    DP where state is a bitmask representing chosen elements.
    """
    # dp[mask] = answer for subset represented by mask
    dp = [0] * (1 << n)
    dp[0] = base_value

    for mask in range(1, 1 << n):
        for i in range(n):
            if mask & (1 << i):  # i-th element is in subset
                prev_mask = mask ^ (1 << i)  # Subset without i
                dp[mask] = combine(dp[prev_mask], elements[i])

    return dp[(1 << n) - 1]  # All elements
```

---

### Quick Decision Matrix

| Problem Type | Key Operation | Template |
|--------------|---------------|----------|
| Find unique | XOR | A |
| Count bits | n & (n-1) | B |
| Manipulate bit | Shift + mask | C |
| Power of 2 | n & (n-1) | - |
| Subset enumeration | Decrement + AND | E |
| State compression | Bitmask DP | F |

---

<a name="pattern-guide"></a>
## 3. Pattern Classification Guide

### Category 1: Single Number Problems
- Find element appearing different times
- **XOR template**

### Category 2: Bit Counting
- Count set bits, bit differences
- **Brian Kernighan's algorithm**

### Category 3: Power of 2
- Check power, round to power
- **n & (n-1) trick**

### Category 4: Bit Manipulation
- Get/set/clear/toggle specific bits
- **Mask operations**

### Category 5: State Compression
- DP with subsets as states
- **Bitmask DP**

### Category 6: Mathematical
- Add without +, divide without /
- **Bit-level arithmetic**

---

<a name="patterns"></a>
## 4. Complete Pattern Library

### PATTERN 1: Single Number Problems

---

#### Pattern 1A: Single Number (XOR)

**Problem:** LeetCode 136 - Find element appearing once, others twice

```python
def singleNumber(nums: list[int]) -> int:
    result = 0
    for num in nums:
        result ^= num
    return result
```

**Why it works:**
```
[4, 1, 2, 1, 2]
4 ^ 1 ^ 2 ^ 1 ^ 2
= 4 ^ (1 ^ 1) ^ (2 ^ 2)
= 4 ^ 0 ^ 0
= 4
```

---

#### Pattern 1B: Single Number II (Appears Once, Others Three Times)

**Problem:** LeetCode 137

```python
def singleNumber(nums: list[int]) -> int:
    # Strategy: For each bit position, count how many numbers have a 1 there.
    # Numbers appearing 3× contribute 3 (or 0 mod 3) to the count.
    # The single number contributes 1. So bit_sum % 3 reveals the single number's bit.
    result = 0
    for i in range(32):
        # Why `(num >> i) & 1`? Shift bit i to position 0, then mask
        # with 1 to extract just that bit (0 or 1).
        bit_sum = sum((num >> i) & 1 for num in nums)
        # Why `% 3`? Triples cancel out. If remainder is non-zero,
        # the single number has a 1 at this position.
        if bit_sum % 3:
            result |= (1 << i)

    # Why this check? Python integers are unbounded, but we need 32-bit
    # signed behavior. If bit 31 is set, it's negative in 32-bit.
    if result >= (1 << 31):
        result -= (1 << 32)
    return result
```

**Alternative (state machine):**
```python
def singleNumber_state(nums: list[int]) -> int:
    # State machine: tracks bit counts mod 3 across two variables.
    # ones = bits seen 1 time (mod 3), twos = bits seen 2 times (mod 3).
    # After 3 occurrences, both reset to 0. The single number stays in `ones`.
    ones = twos = 0
    for num in nums:
        # `& ~twos`: if a bit is already in twos, don't let it into ones
        ones = (ones ^ num) & ~twos
        # `& ~ones`: if a bit just moved to ones, don't let it into twos
        twos = (twos ^ num) & ~ones
    return ones  # Bits that appeared exactly once
```

---

#### Pattern 1C: Single Number III (Two Numbers Appear Once)

**Problem:** LeetCode 260

```python
def singleNumber(nums: list[int]) -> list[int]:
    # Step 1: XOR all numbers. Pairs cancel, leaving a ^ b.
    xor = 0
    for num in nums:
        xor ^= num

    # Step 2: Find ANY bit where a and b differ (use lowest for simplicity).
    # Why `xor & (-xor)`? This isolates the lowest set bit in a^b.
    # Since a^b has a 1 wherever a and b differ, this gives us a
    # "separator" bit to split them into different groups.
    diff_bit = xor & (-xor)

    # Step 3: Partition ALL numbers into two groups based on this bit.
    # a and b land in different groups (they differ at this bit).
    # All paired numbers land in the SAME group (same value = same bit).
    # XOR within each group cancels pairs, leaving just a or just b.
    a = b = 0
    for num in nums:
        if num & diff_bit:
            a ^= num
        else:
            b ^= num

    return [a, b]
```

---

### PATTERN 2: Counting Bits

---

#### Pattern 2A: Number of 1 Bits

**Problem:** LeetCode 191 - Count set bits (Hamming weight)

```python
def hammingWeight(n: int) -> int:
    count = 0
    while n:
        n &= (n - 1)  # Clear lowest set bit
        count += 1
    return count
```

**Alternative:**
```python
def hammingWeight_builtin(n: int) -> int:
    return bin(n).count('1')
```

---

#### Pattern 2B: Counting Bits (Array)

**Problem:** LeetCode 338 - Count bits for 0 to n

```python
def countBits(n: int) -> list[int]:
    result = [0] * (n + 1)
    for i in range(1, n + 1):
        # Why `result[i >> 1] + (i & 1)`?
        # i >> 1 = i with last bit removed (e.g., 13=1101 → 6=110).
        # i & 1 = the last bit itself (0 or 1).
        # So: bits(1101) = bits(110) + 1. We reuse the already-computed answer
        # for the number without its last bit, then add back that last bit.
        result[i] = result[i >> 1] + (i & 1)
    return result
```

**Alternative using n & (n-1):**
```python
def countBits_alt(n: int) -> list[int]:
    result = [0] * (n + 1)
    for i in range(1, n + 1):
        # Why `i & (i-1)`? Clears the lowest set bit.
        # So bits(i) = bits(i with lowest 1 removed) + 1.
        # Example: i=12=1100, i&(i-1)=1000. bits(1100) = bits(1000) + 1.
        result[i] = result[i & (i - 1)] + 1
    return result
```

---

#### Pattern 2C: Hamming Distance

**Problem:** LeetCode 461 - Bit positions where x and y differ

```python
def hammingDistance(x: int, y: int) -> int:
    xor = x ^ y
    count = 0
    while xor:
        xor &= (xor - 1)
        count += 1
    return count
```

---

### PATTERN 3: Power of 2

---

#### Pattern 3A: Power of Two

**Problem:** LeetCode 231

```python
def isPowerOfTwo(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0
```

**Why it works:**
```
Powers of 2 have exactly one set bit:
1 = 0001
2 = 0010
4 = 0100
8 = 1000

n & (n-1) clears the lowest set bit
If result is 0, there was only one bit
```

---

#### Pattern 3B: Power of Four

**Problem:** LeetCode 342

```python
def isPowerOfFour(n: int) -> bool:
    # Three checks:
    # 1. n > 0: rule out 0 and negatives
    # 2. n & (n-1) == 0: must be a power of 2 (single set bit)
    # 3. n & 0x55555555 != 0: the set bit must be at an EVEN position
    #    0x55555555 = 01010101... in binary (bits at positions 0, 2, 4, 6...)
    #    Powers of 4: 1(pos 0), 4(pos 2), 16(pos 4) — all at even positions.
    #    Powers of 2 but not 4: 2(pos 1), 8(pos 3) — at odd positions.
    return n > 0 and (n & (n - 1)) == 0 and (n & 0x55555555) != 0
```

---

### PATTERN 4: Bit Reversal

---

#### Pattern 4A: Reverse Bits

**Problem:** LeetCode 190

```python
def reverseBits(n: int) -> int:
    result = 0
    for _ in range(32):
        # Why `(result << 1) | (n & 1)`?
        # 1. Shift result left to make room for next bit
        # 2. Extract n's last bit with n & 1
        # 3. Append it to result with OR
        # Like reading digits right-to-left and writing left-to-right.
        result = (result << 1) | (n & 1)
        n >>= 1  # Move to n's next bit
    return result
```

**Optimized with divide and conquer:**
```python
def reverseBits_fast(n: int) -> int:
    n = ((n & 0xffff0000) >> 16) | ((n & 0x0000ffff) << 16)
    n = ((n & 0xff00ff00) >> 8) | ((n & 0x00ff00ff) << 8)
    n = ((n & 0xf0f0f0f0) >> 4) | ((n & 0x0f0f0f0f) << 4)
    n = ((n & 0xcccccccc) >> 2) | ((n & 0x33333333) << 2)
    n = ((n & 0xaaaaaaaa) >> 1) | ((n & 0x55555555) << 1)
    return n
```

---

### PATTERN 5: Missing/Duplicate Numbers

---

#### Pattern 5A: Missing Number

**Problem:** LeetCode 268 - Find missing in [0, n]

```python
def missingNumber(nums: list[int]) -> int:
    # XOR all indices (0..n-1) with all values AND n itself.
    # If no number were missing, XOR of indices 0..n would cancel with
    # XOR of values 0..n. The missing number doesn't cancel → it remains.
    # Example: nums=[0,1,3], n=3. XOR: 3^(0^0)^(1^1)^(2^3) = 3^0^0^2^3 = 2.
    n = len(nums)
    result = n  # Start with n (indices only go 0..n-1, so n isn't covered)
    for i, num in enumerate(nums):
        result ^= i ^ num
    return result
```

**Alternative (math):**
```python
def missingNumber_math(nums: list[int]) -> int:
    n = len(nums)
    expected_sum = n * (n + 1) // 2
    return expected_sum - sum(nums)
```

---

#### Pattern 5B: Find the Duplicate Number

**Problem:** LeetCode 287 - Using bit manipulation

```python
def findDuplicate(nums: list[int]) -> int:
    n = len(nums) - 1
    result = 0

    # Strategy: For each bit position, count how many numbers in nums
    # have that bit set vs. how many in [1..n] have it set.
    # The duplicate adds an extra 1 at its bit positions, so
    # count_nums > count_expected reveals the duplicate's bits.
    for bit in range(32):
        mask = 1 << bit
        count_nums = sum(1 for num in nums if num & mask)
        count_expected = sum(1 for i in range(1, n + 1) if i & mask)

        # Why `>`? The duplicate number has a 1 at this bit,
        # contributing one extra count. Reconstruct it bit by bit.
        if count_nums > count_expected:
            result |= mask

    return result
```

---

### PATTERN 6: Arithmetic with Bits

---

#### Pattern 6A: Sum of Two Integers (Without +)

**Problem:** LeetCode 371

```python
def getSum(a: int, b: int) -> int:
    # 32-bit mask to simulate fixed-width integer overflow in Python
    MASK = 0xFFFFFFFF
    MAX_INT = 0x7FFFFFFF

    # Why loop until b == 0? Each iteration handles one "round" of carries.
    # Like grade-school addition: add digits, carry the 1, repeat until no carries.
    while b != 0:
        # a & b: bits where BOTH are 1 → these positions generate a carry.
        # << 1: carry goes to the NEXT higher position.
        carry = (a & b) << 1
        # a ^ b: sum WITHOUT carries (0+0=0, 1+0=1, 0+1=1, 1+1=0-with-carry).
        a = (a ^ b) & MASK
        b = carry & MASK  # Process remaining carries in next iteration

    # Why this conversion? Python integers are unbounded, but we need
    # 32-bit signed behavior. If bit 31 is set, convert to negative.
    return a if a <= MAX_INT else ~(a ^ MASK)
```

**Why it works:**
```
a XOR b: sum without carry
a AND b << 1: carry positions
Repeat until no carry
```

---

#### Pattern 6B: Divide Two Integers (Without /)

**Problem:** LeetCode 29

```python
def divide(dividend: int, divisor: int) -> int:
    MAX_INT = 2**31 - 1
    MIN_INT = -2**31

    # Why this special case? -2^31 / -1 = 2^31, which overflows 32-bit signed int.
    if dividend == MIN_INT and divisor == -1:
        return MAX_INT

    # Why `!=`? XOR-like logic: different signs → negative result.
    negative = (dividend < 0) != (divisor < 0)

    # Work with positive values
    a, b = abs(dividend), abs(divisor)
    result = 0

    # Strategy: repeatedly subtract the largest possible multiple of b.
    # Like long division: find how many times b fits, but using bit shifts
    # (doubling) to jump exponentially instead of subtracting one-by-one.
    while a >= b:
        temp, multiple = b, 1
        # Double temp until it would exceed a. This finds the largest
        # power-of-2 multiple of b that fits in a.
        # Example: a=17, b=3 → temp goes 3→6→12 (24 would exceed), multiple=4.
        while a >= (temp << 1):
            temp <<= 1    # Double the subtraction amount
            multiple <<= 1  # Double the quotient contribution
        a -= temp       # Subtract that chunk
        result += multiple  # Add its contribution to quotient

    return -result if negative else result
```

---

### PATTERN 7: Advanced Applications

---

#### Pattern 7A: Maximum XOR of Two Numbers

**Problem:** LeetCode 421

```python
def findMaximumXOR(nums: list[int]) -> int:
    max_xor = 0
    mask = 0

    # Build the answer bit by bit, from MSB to LSB (greedy).
    # At each position, ask: "Can I set this bit to 1 in the result?"
    for i in range(31, -1, -1):
        # Why accumulate mask? We only care about bits from position i upward.
        # Masking strips lower bits we haven't decided yet.
        mask |= (1 << i)
        prefixes = {num & mask for num in nums}

        # Greedily try to set this bit: "what if the answer has a 1 here?"
        candidate = max_xor | (1 << i)

        for prefix in prefixes:
            # Why `prefix ^ candidate in prefixes`?
            # If a^b = candidate, then a^candidate = b.
            # So if both `prefix` and `prefix ^ candidate` exist in our set,
            # there exist two numbers whose XOR achieves the candidate value.
            if prefix ^ candidate in prefixes:
                max_xor = candidate
                break

    return max_xor
```

---

#### Pattern 7B: Subsets (Bitmask)

**Problem:** LeetCode 78 - Generate all subsets

```python
def subsets(nums: list[int]) -> list[list[int]]:
    n = len(nums)
    result = []

    # Why `1 << n`? There are 2^n subsets. Each mask from 0 to 2^n-1
    # represents one subset: bit i = 1 means "include nums[i]."
    # Example: n=3, mask=5=101 → include nums[0] and nums[2].
    for mask in range(1 << n):
        subset = []
        for i in range(n):
            if mask & (1 << i):  # Is bit i set? → include nums[i]
                subset.append(nums[i])
        result.append(subset)

    return result
```

---

#### Pattern 7C: Minimum XOR Sum of Two Arrays

**Problem:** LeetCode 1879 - Bitmask DP

```python
def minimumXORSum(nums1: list[int], nums2: list[int]) -> int:
    n = len(nums1)
    dp = [float('inf')] * (1 << n)
    dp[0] = 0

    for mask in range(1 << n):
        k = bin(mask).count('1')  # Position in nums1
        if k >= n:
            continue

        for j in range(n):
            if not (mask & (1 << j)):  # j not used
                new_mask = mask | (1 << j)
                dp[new_mask] = min(dp[new_mask],
                                   dp[mask] + (nums1[k] ^ nums2[j]))

    return dp[(1 << n) - 1]
```

---

<a name="pitfalls"></a>
## 5. Common Pitfalls & Solutions

### Pitfall 1: Negative Numbers in Python

```python
# Python integers have arbitrary precision
# -1 >> 1 is still -1 (not a large positive)

# Use masking for 32-bit behavior
MASK = 0xFFFFFFFF
result = (result & MASK)
```

### Pitfall 2: Operator Precedence

```python
# WRONG: & has lower precedence than ==
if n & 1 == 0:  # Parsed as n & (1 == 0)

# CORRECT: Use parentheses
if (n & 1) == 0:
```

### Pitfall 3: Shift Overflow

```python
# Shifting by 32+ bits on 32-bit integer
1 << 32  # Python handles this, but other languages don't
```

### Pitfall 4: XOR Associativity Assumption

```python
# XOR is associative and commutative
# But order matters when building result bit by bit
```

---

<a name="recognition"></a>
## 6. Problem Recognition Framework

### Step 1: Is it a Bit Problem?

**Indicators:**
- "XOR", "AND", "OR" mentioned
- "Binary representation"
- "Single/unique element"
- Power of 2
- State compression

### Step 2: Choose Strategy

| Problem | Strategy |
|---------|----------|
| Find unique | XOR |
| Count bits | Brian Kernighan |
| Power of 2 | n & (n-1) |
| Subset DP | Bitmask |

---

<a name="checklist"></a>
## 7. Interview Preparation Checklist

### Before the Interview

**Master the fundamentals:**
- [ ] Know all bit operators and their behavior
- [ ] Understand XOR properties
- [ ] Can implement Brian Kernighan's algorithm
- [ ] Know n & (n-1) clears lowest bit
- [ ] Can get/set/clear specific bits

**Know the patterns:**
- [ ] Single number (XOR)
- [ ] Count set bits
- [ ] Power of 2 checks
- [ ] Bit reversal
- [ ] Bitmask subsets

**Common problems solved:**
- [ ] LC 136: Single Number
- [ ] LC 191: Number of 1 Bits
- [ ] LC 231: Power of Two
- [ ] LC 268: Missing Number
- [ ] LC 338: Counting Bits

### During the Interview

**1. Clarify (30 seconds)**
- 32-bit or 64-bit?
- Handle negative numbers?
- Expected range?

**2. Identify pattern (30 seconds)**
- XOR for finding unique
- n & (n-1) for counting/power of 2
- Bit manipulation for specific positions

**3. Code (3-4 minutes)**
- Use appropriate operators
- Handle edge cases
- Consider negative numbers

**4. Test (1-2 minutes)**
- 0, 1, -1
- Powers of 2
- Maximum values

---

## 8. Quick Reference Cards

### Essential Formulas
```python
# Get bit i
(n >> i) & 1

# Set bit i
n | (1 << i)

# Clear bit i
n & ~(1 << i)

# Toggle bit i
n ^ (1 << i)

# Lowest set bit
n & (-n)

# Clear lowest set bit
n & (n - 1)

# Check power of 2
n > 0 and n & (n - 1) == 0
```

### XOR Properties
```python
a ^ 0 = a
a ^ a = 0
a ^ b ^ a = b
```

---

## 9. Complexity Reference

| Operation | Time |
|-----------|------|
| Single XOR scan | O(n) |
| Count bits | O(k) where k = set bits |
| Bit manipulation | O(1) |
| Bitmask DP | O(2^n × n) |

---

## Final Thoughts

**Remember:**
1. XOR is the most useful: `a ^ a = 0`, `a ^ 0 = a`
2. `n & (n-1)` clears lowest set bit
3. `n & (-n)` isolates lowest set bit
4. Python integers are arbitrary precision
5. Bit manipulation gives O(1) for many operations

---

## Appendix: Practice Problem Set

### Easy
- 136. Single Number
- 191. Number of 1 Bits
- 231. Power of Two
- 268. Missing Number
- 338. Counting Bits
- 461. Hamming Distance

### Medium
- 137. Single Number II
- 190. Reverse Bits
- 260. Single Number III
- 371. Sum of Two Integers
- 421. Maximum XOR of Two Numbers

### Hard
- 29. Divide Two Integers

**Recommended Practice Order:**
1. Basic: 136 → 191 → 231 → 338
2. Advanced: 137 → 260
3. Applications: 268 → 371 → 421

---

## Appendix: Conditional Quick Reference

### Core Bit Operations (Why They Work)

| Operation | Code | Binary Example | Why It Works |
|-----------|------|----------------|-------------|
| Check bit i | `(n >> i) & 1` | n=13=1101, i=2 → 11 & 1 = 1 | Shift bit to position 0, mask with 1 |
| Set bit i | `n \| (1 << i)` | n=9=1001, i=1 → 1001 \| 0010 = 1011 | OR with a 1 at position i forces it on |
| Clear bit i | `n & ~(1 << i)` | n=13=1101, i=2 → 1101 & 1011 = 1001 | AND with 0 at position i forces it off |
| Toggle bit i | `n ^ (1 << i)` | n=13=1101, i=2 → 1101 ^ 0100 = 1001 | XOR flips: 1→0, 0→1 |
| Lowest set bit | `n & (-n)` | n=12=1100 → 1100 & 0100 = 0100 | Two's complement flips above lowest 1 |
| Clear lowest bit | `n & (n-1)` | n=12=1100 → 1100 & 1011 = 1000 | n-1 flips lowest 1 and bits below it |
| Power of 2? | `n & (n-1) == 0` | n=8=1000 → 1000 & 0111 = 0 ✓ | Powers of 2 have exactly one set bit |

### XOR Pattern Conditionals

| Condition | Where Used | Why |
|-----------|-----------|-----|
| `result ^= num` (all nums) | Single Number I | Pairs cancel (a^a=0), leaving the loner |
| `bit_sum % 3` | Single Number II | Triples cancel mod 3, leaving single's bit |
| `(ones ^ num) & ~twos` | Single Number II (state machine) | State transitions: count mod 3 across two vars |
| `xor & (-xor)` | Single Number III | Find a differing bit between the two unique numbers |
| `num & diff_bit` | Single Number III | Partition into two groups to isolate each unique number |

### Arithmetic Conditionals

| Condition | Where Used | Why |
|-----------|-----------|-----|
| `b != 0` | Sum without + | Keep looping while carries remain |
| `(a & b) << 1` | Sum without + | AND finds carry positions; shift left moves carry to next column |
| `a ^ b` | Sum without + | XOR = addition without carries |
| `a >= (temp << 1)` | Divide without / | Double divisor until it would exceed dividend |
| `a >= b` | Divide without / | Keep subtracting while dividend has room |

### DP & Enumeration Conditionals

| Condition | Where Used | Why |
|-----------|-----------|-----|
| `mask & (1 << i)` | Bitmask DP/Subsets | Check if item i is in the current subset |
| `mask \| (1 << i)` | Bitmask DP | Add item i to subset |
| `(1 << n) - 1` | Full mask | All n bits set = all items selected |
| `(submask - 1) & mask` | Subset enumeration | Step to next smaller subset within mask |
| `result >= (1 << 31)` | 32-bit signed | Bit 31 set = negative in signed 32-bit |

Good luck with your interview preparation!
