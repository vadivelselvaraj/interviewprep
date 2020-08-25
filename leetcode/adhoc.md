## 1. Reverse an integer

Given a 32-bit signed integer, reverse digits of an integer.

**Note:** Assume we are dealing with an environment which could only store integers within the 32-bit signed integer range: [−231,  231 − 1]. For the purpose of this problem, assume that your function returns 0 when the reversed integer overflows.

**Hint:**
* pop and push digits logic like `reversedNumber = reversedNumber * 10 + pop` can cause an overflow. So, we need to catch it before the overflow happens.
* If `reversedNumber = reversedNumber * 10 + pop` can cause an overflow, then `reversedNumber >= INTMAX/10`. So, the below are true.
  * If `reversedNumber > INTMAX/10`, then `reversedNumber = reversedNumber * 10 + pop` will cause an overflow.
  * If `reversedNumber == INTMAX/10`, then `reversedNumber = reversedNumber * 10 + pop` will cause an overflow iff `pop > 7`(rightmost digit of 2^31).

[LeetCode link](https://leetcode.com/problems/reverse-integer/)

<details>
<summary>Click here to see code</summary>

```python
def reverse(self, x: int) -> int:
        if x >=-9 and x <= 9:
            return x
        
        negativeSign = True if x < 0 else False
        
        signedX = -1 * x if x < 0 else x
        result, t = 0, signedX
        divider = 10
        
        leftLimit, rightLimit = (2**31), (2**31)-1
        
        while t:
            digitToAdd = t%10
            if (
                not negativeSign
                and ( ( result == (rightLimit // 10) and digitToAdd > 7 ) or result > (rightLimit // 10) )
            ) or (
                negativeSign
                and ( ( result == (leftLimit // 10) and digitToAdd > 8 ) or (result > leftLimit // 10) )
            ):
                return 0

            result = (result * divider) + digitToAdd
            t //= 10
        
        return -1 * result if negativeSign else result
```

</details>

## 2.) Pow(x, n)

Implement pow(x, n), which calculates x raised to the power n

**Hint:**
* If n is -ve, set `x = 1/x and n = -1 * n`.
* If n is odd, `result = x * (x^n/2)^2`. Otherwise `result = (x^n/2)^2`

[LeetCode link](https://leetcode.com/problems/)

<details>
<summary>Click here to see code</summary>

```python
def myPow(self, x: float, n: int) -> float:
    if n == 0:
        return 1
    if x == 0:
        return 0
    
    result, power, prod = 1.0, n, x
    if n < 0:
        prod = 1/prod
        power = -power
    
    while power:
        if power & 1:
            result *= prod
        prod *= prod
        power = power >> 1

    return result
```

</details>

## 3.) Sqrt(x)
Implement `int sqrt(int x)`.

Compute and return the square root of x, where x is guaranteed to be a non-negative integer.

Since the return type is an integer, the decimal digits are truncated and only the integer part of the result is returned.

## Example 1:
```
Input: 4
Output: 2
Example 2:
```
```
Input: 8
Output: 2
Explanation: The square root of 8 is 2.82842..., and since 
             the decimal part is truncated, 2 is returned.
```
**Hint:**
* If n < 2, return n.
* Otherwise, do a binary search between 2 <= x < x/2

[LeetCode link](https://leetcode.com/problems/sqrtx/)

<details>
<summary>Click here to see code</summary>

```python
def mySqrt(self, x: int) -> int:
    if x < 2:
        return x
    
    # do a binary search between 2 <= x < x/2
    left, right = 2, x//2
    
    while left <= right:
        mid = (left + right) >> 1
        t = mid * mid

        if t == x:
            return mid
        elif t < x:
            left = mid + 1
        elif t > x:
            right = mid - 1
        
    return right
```

</details>

## 4.) Nested List Weight Sum
Given a nested list of integers, return the sum of all integers in the list weighted by their depth.

Each element is either an integer, or a list -- whose elements may also be integers or other lists.

## Example 1:
```
Input: [[1,1],2,[1,1]]
Output: 10 

Explanation: Four 1's at depth 2, one 2 at depth 1.
```
## Example 2:
```
Input: [1,[4,[6]]]
Output: 27 
Explanation: One 1 at depth 1, one 4 at depth 2, and one 6 at depth 3; 1 + 4*2 + 6*3 = 27.
```
**Hint:**
* Do a simple DFS on the nested list

[LeetCode link](https://leetcode.com/problems/)

<details>
<summary>Click here to see code</summary>

```python
class Solution:
    def depthSum(self, nestedList: List[NestedInteger]) -> int:
        if not nestedList:
            return 0
        result = 0
        
        def dfs(nums, depth):
            nonlocal result
            for num in nums:
                if num.isInteger():
                    result += (num.getInteger() * depth)
                else:
                    dfs(num.getList(), depth + 1)
                    
        dfs(nestedList, 1)
        return result
```

</details>

## 5.) Monotonic Array

An array is monotonic if it is either monotone increasing or monotone decreasing.

An array A is monotone increasing if for all `i <= j, A[i] <= A[j]`.  An array A is monotone decreasing if for all `i <= j, A[i] >= A[j]`.

Return true if and only if the given array A is monotonic.


### Example 1:
```
Input: [1,2,2,3]
Output: true
```
### Example 2:
```
Input: [6,5,4,4]
Output: true
```
### Example 3:
```
Input: [1,3,2]
Output: false
```
### Example 4:
```
Input: [1,2,4,5]
Output: true
```
### Example 5:
```
Input: [1,1,1]
Output: true
```

Note:

* `11 <= A.length <= 50000`
* `-100000 <= A[i] <= 100000`

**Hint:**
* Set increasing, decreasing flags to true.
* Iterate through the array and set these flags to false whenever their respective conditions aren't meet.
* return true if any of the flags are true

[LeetCode link](https://leetcode.com/problems/monotonic-array/)

<details>
<summary>Click here to see code</summary>

```python
def isMonotonic(self, A: List[int]) -> bool:
    if not A or len(A) <= 1:
        return True
    
    increase = decrease = True
    for i in range(1, len(A)):
        if A[i-1] > A[i]:
            increase = False
        if A[i-1] < A[i]:
            decrease = False
    
    return increase or decrease
```

</details>

## 6.) Verifying an Alien Dictionary

In an alien language, surprisingly they also use english lowercase letters, but possibly in a different order. The order of the alphabet is some permutation of lowercase letters.

Given a sequence of words written in the alien language, and the order of the alphabet, return true if and only if the given words are sorted lexicographicaly in this alien language.

### Example 1:
```
Input: words = ["hello","leetcode"], order = "hlabcdefgijkmnopqrstuvwxyz"
Output: true
Explanation: As 'h' comes before 'l' in this language, then the sequence is sorted.
```
### Example 2:
```
Input: words = ["word","world","row"], order = "worldabcefghijkmnpqstuvxyz"
Output: false
Explanation: As 'd' comes after 'l' in this language, then words[0] > words[1], hence the sequence is unsorted.
```
### Example 3:
```
Input: words = ["apple","app"], order = "abcdefghijklmnopqrstuvwxyz"
Output: false
```
*Explanation:* The first three characters "app" match, and the second string is shorter (in size.) According to lexicographical rules "apple" > "app", because 'l' > '∅', where '∅' is defined as the blank character which is less than any other character ([More info](https://en.wikipedia.org/wiki/Lexicographical_order)).
### Constraints:

- 1 <= words.length <= 100
- 1 <= words[i].length <= 20
- order.length == 26
- All characters in words[i] and order are English lowercase letters.

**Hint:**
* Iterate through the array comparing adjacent elements and making sure that the first character difference is lexicographically sorted. Also, ensure that when there are no character differences, the smallest word appears first.

[LeetCode link](https://leetcode.com/problems/verifying-an-alien-dictionary/)

<details>
<summary>Click here to see code</summary>

```python
    def isAlienSorted(self, words: List[str], order: str) -> bool:
        if not words:
            return True
        
        orderMap = { char: order for order, char in enumerate(order) }
        for i in range(1, len(words)):
            w1 = words[i-1]
            w2 = words[i]
            
            for j in range(min(len(w1), len(w2))):
                if w1[j] != w2[j]:
                    if orderMap[w1[j]] > orderMap[w2[j]]:
                        return False
                    break
            else:
                if len(w1) > len(w2):
                    return False
        
        return True
```

</details>

## 7.) Add Strings

Given two non-negative integers num1 and num2 represented as string, return the sum of num1 and num2.

### Note:

- The length of both num1 and num2 is < 5100.
- Both num1 and num2 contains only digits 0-9.
- Both num1 and num2 does not contain any leading zero.
- You must not use any built-in BigInteger library or convert the inputs to integer directly.

**Hint:**
* Use text book addition. Be careful to insert the carry, value at the first index of the resultant array and to populate the array with string so that `''.join(arr)` would work.

[LeetCode link](https://leetcode.com/problems/add-strings/)

<details>
<summary>Click here to see code</summary>

```python
def addStrings(self, num1: str, num2: str) -> str:
    if not num1:
        return num2
    elif not num2:
        return num1

    result = []
    i, j, carry = len(num1) - 1, len(num2) - 1, 0
    
    while i >= 0 or j >= 0:
        n1, n2 = 0, 0
        if i >= 0:
            n1 = ord(num1[i]) - ord('0')
            i -= 1
        if j >= 0:
            n2 = ord(num2[j]) - ord('0')
            j -= 1

        currentSum = n1 + n2 + carry
        carry, digit = divmod(currentSum, 10)
        result.insert(0, str(digit))
    
    if carry:
        result.insert(0, str(carry))
    
    return ''.join(result)
```

</details>

## 8.) Add Binary
Given two binary strings, return their sum (also a binary string).

The input strings are both non-empty and contains only characters 1 or 0.

## Example 1:
```
Input: a = "11", b = "1"
Output: "100"
```
## Example 2:
```
Input: a = "1010", b = "1011"
Output: "10101"
 ```

### Constraints:

- Each string consists only of '0' or '1' characters.
- 1 <= a.length, b.length <= 10^4
- Each string is either "0" or doesn't contain any leading zero.

**Hint:**
* `1 + 1 => 10`, `1 + 1 + 1 => 11`
* Use text book addition for adding two binary strings. Iterate through the indices and fix the 

[LeetCode link](https://leetcode.com/problems/add-binary/)

<details>
<summary>Click here to see code</summary>

```python
def addBinary(self, a: str, b: str) -> str:
    if not a:
        return b
    elif not b:
        return a

    result = []
    i, j, carry = len(a) - 1, len(b) - 1, 0

    while i >= 0 or j >= 0:
        n1, n2 = 0, 0
        if i >= 0:
            n1 = ord(a[i]) - ord('0')
            i -= 1
        if j >= 0:
            n2 = ord(b[j]) - ord('0')
            j -= 1

        carry += n1 + n2
        if carry & 1:
            result.insert(0, '1')
        else:
            result.insert(0, '0')
        carry = carry >> 1

    if carry == 1:
        result.insert(0, '1')

    return ''.join(result)
```

</details>

## 10.) Plus One

Given a non-empty array of digits representing a non-negative integer, increment one to the integer.

The digits are stored such that the most significant digit is at the head of the list, and each element in the array contains a single digit.

You may assume the integer does not contain any leading zero, except the number 0 itself.

 

## Example 1:
```
Input: digits = [1,2,3]
Output: [1,2,4]
Explanation: The array represents the integer 123.
```
## Example 2:
```
Input: digits = [4,3,2,1]
Output: [4,3,2,2]
Explanation: The array represents the integer 4321.
```
## Example 3:
```
Input: digits = [0]
Output: [1]
``` 

### Constraints:

- 1 <= digits.length <= 100
- 0 <= digits[i] <= 9
**Hint:**
* 

[LeetCode link](https://leetcode.com/problems/plus-one/)

<details>
<summary>Click here to see code</summary>

```python
    def plusOne(self, digits: List[int]) -> List[int]:
        if not digits:
            return []
        
        carry = 1
        result = []
        for num in digits[::-1]:
            t = num + carry
            carry, value = divmod(t, 10)
            result.insert(0, value)
        
        if carry:
            result.insert(0, carry)
            
        return result
```

</details>

## 11.) Merge Sorted Array

Given two sorted integer arrays nums1 and nums2, merge nums2 into nums1 as one sorted array.

### Note:

- The number of elements initialized in nums1 and nums2 are m and n respectively.
- You may assume that nums1 has enough space (size that is equal to m + n) to hold additional elements from nums2.
## Example:
```
Input:
nums1 = [1,2,3,0,0,0], m = 3
nums2 = [2,5,6],       n = 3

Output: [1,2,2,3,5,6]
```

### Constraints:

- -10^9 <= nums1[i], nums2[i] <= 10^9
- nums1.length == m + n
- nums2.length == n
**Hint:**
* Use a `three pointer` approach. Two pointers should point to the end of each arrays while the third one should point to the `m+n-1` index of the first array. 

[LeetCode link](https://leetcode.com/problems/merge-sorted-array/)

<details>
<summary>Click here to see code</summary>

```python
def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
    """
    Do not return anything, modify nums1 in-place instead.
    """
    if not nums1 and not nums2:
        return
    elif not nums2:
        return
    
    i, j, k = m-1, n-1, m+n-1
    while i >= 0 and j >= 0:
        if nums1[i] > nums2[j]:
            nums1[k] = nums1[i]
            i -=1
        else:
            nums1[k] = nums2[j]
            j -= 1
        k -= 1
    
    if j >= 0:
        nums1[:k+1] = nums2[:j+1]
```

</details>


## 12.) Valid Palindrome II

Given a non-empty string s, you may delete at most one character. Judge whether you can make it a palindrome.

## Example 1:
```
Input: "aba"
Output: True
```
## Example 2:
```
Input: "abca"
Output: True
Explanation: You could delete the character 'c'.
```
### Note:
The string will only contain lowercase characters a-z. The maximum length of the string is 50000.

**Hint:**
* Use the usual logic for palindromic check. When the low and high pointer characters doesn't match, check if eliminating the chars pointed by them yields a palindrome. Otherwise, return false.

[LeetCode link](https://leetcode.com/problems/valid-palindrome-ii/)

<details>
<summary>Click here to see code</summary>

```python
def validPalindrome(self, s: str) -> bool:
    if not s:
        return True

    def _isPalindrome(low, high):
        while low < high:
            if s[low] != s[high]:
                return False
            low += 1
            high -= 1
        return True
    
    low, high = 0, len(s) - 1
    while low < high:
        if s[low] == s[high]:
            low += 1
            high -= 1
        else:
            # if removing low or high ensures the string is a palindrome,
            # returns true.
            if _isPalindrome(low+1, high):
                return True
            if _isPalindrome(low, high-1):
                return True
            return False
    
    return True
```

</details>

# Template:

Use this template to create new problems

## .) 


**Hint:**
* 

[LeetCode link]()

<details>
<summary>Click here to see code</summary>

```python
```

</details>