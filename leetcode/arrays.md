# Arrays

## 1.) Subarray Sum Equals K

Given an array of integers and an integer k, you need to find the total number of continuous subarrays whose sum equals to k.

**Hint:**
* [Prefix Sum](https://en.wikipedia.org/wiki/Prefix_sum) technique
* Note: `prefixSumMap[currentSum] += 1` should be done only at the end.

[LeetCode link](https://leetcode.com/problems/subarray-sum-equals-k/)

<details>
<summary>Click here to see code</summary>

```python
    def subarraySum(self, nums: List[int], k: int) -> int:
        if not nums:
            return 0
        
        prefixSumMap = defaultdict(int)
        count, currentSum = 0, 0
        
        for num in nums:
            currentSum += num
            
            # Case 1: Contiguous sum from the beginning
            if currentSum == k:
                count += 1
            
            # Case 2: Contiguous sum from the rest
            if currentSum - k in prefixSumMap:
                count += prefixSumMap[currentSum - k]

            # Add the current sum
            prefixSumMap[currentSum] += 1
            
        return count
```

</details>

## 2.) Merge Intervals

Given a collection of intervals, merge all overlapping intervals.

### Example 1:

__Input:__ intervals = `[[1,3],[2,6],[8,10],[15,18]]`

__Output:__ `[[1,6],[8,10],[15,18]]`

__Explanation:__ Since intervals `[1,3]` and `[2,6]` overlaps, merge them into [1,6].

### Example 2:

__Input:__ intervals = `[[1,4],[4,5]]`

__Output:__ `[[1,5]]`

__Explanation:__ Intervals `[1,4]` and `[4,5]` are considered overlapping.

### Follow up

How do you add intervals and merge them for a large stream of intervals? (Facebook Follow-up Question)

**Hint:**
* __Approach 1:__ Sort the array. Then do a pass to merge the intervals as we move forward as per the invariant `If a[i] can't merge with a[i-1], then a[i+1] cannot merge with a[i-1] due to the sort.` To merge `a[i]` with `a[i-1]`, use `a[i-1][end] = max( a[i-1][end], a[i][end] )` eg: `[1,4], [2,3]` & `[1,3], [2,4]`. Time Complexity: _O(nlogn)_
* __Approach 2:__ Interval Trees.

[LeetCode link](https://leetcode.com/problems/merge-intervals/)

<details>
<summary>Click here to see code</summary>

```python
def merge(self, intervals: List[List[int]]) -> List[List[int]]:
    intervals.sort(key=lambda x: x[0])
    
    result = []
    for interval in intervals:
        # If it's the first item in the result or if the current interval doesn't
        # overlap with the previous one, we append it directly to the result.
        if not result or result[-1][1] < interval[0]:
            result.append(interval)
        else:
            # Merge the current interval into the last one.
            result[-1][1] = max(result[-1][1], interval[1])

    return result
```

</details>


## 2.) Move Zeroes

Given an array `nums`, write a function to move all 0's to the end of it while maintaining the relative order of the non-zero elements.

### Example 1:

__Input:__  `[0,1,0,3,12]`

__Output:__ `[1,3,12,0,0]`

### Note:

- You must do this in-place without making a copy of the array.
- Minimize the total number of operations.

**Hint:**
* __Approach 1:__ Move the non-zero elements to their correct positions while counting the number of non-zero elements. Finally, write zeroes to positions after the last non-zero element.
* __Approach 2:__ Use a two pointer approach and swap elements in place only if the first pointer doesn't point to zero and the second pointer points to zero. Increment the second pointer only when it doesn't point to a zero.

[LeetCode link](https://leetcode.com/problems/move-zeroes/)

<details>
<summary>Click here to see code</summary>

## Approach 1:

```python
def moveZeroes(self, nums: List[int]) -> None:
    count = 0 # Count of non-zero elements 
      
    # Traverse the array. If element  
    # encountered is non-zero, then 
    # replace the element at index 
    # 'count' with this element 
    for i in range(n): 
        if nums[i] != 0: 
              
            # here count is incremented 
            nums[count] = nums[i] 
            count+=1
      
    # Now all non-zero elements have been 
    # shifted to front and 'count' is set 
    # as index of first 0. Make all  
    # elements 0 from count to end. 
    while count < n: 
        nums[count] = 0
        count += 1
```


## Approach 2:

```python
def moveZeroes(self, nums: List[int]) -> None:
    """
    Do not return anything, modify nums in-place instead.
    """
    if not nums:
        return
    
    j = 0

    for i in range(len(nums)):
        if nums[i] != 0 and nums[j] == 0:
            nums[j], nums[i] = nums[i], nums[j]
        if nums[j] != 0:
            j += 1
```

</details>

## 3.) Reorder data log files

You have an array of logs.  Each log is a space delimited string of words.

For each log, the first word in each log is an alphanumeric identifier.  Then, either:

- Each word after the identifier will consist only of lowercase letters, or;
- Each word after the identifier will consist only of digits.
We will call these two varieties of logs letter-logs and digit-logs.  It is guaranteed that each log has at least one word after its identifier.

Reorder the logs so that all of the letter-logs come before any digit-log.  The letter-logs are ordered lexicographically ignoring identifier, with the identifier used in case of ties.  The digit-logs should be put in their original order.

Return the final order of the logs.

### Example 1:

**Input:** `logs = ["dig1 8 1 5 1","let1 art can","dig2 3 6","let2 own kit dig","let3 art zero"]`

**Output:** `["let1 art can","let3 art zero","let2 own kit dig","dig1 8 1 5 1","dig2 3 6"]`


### Constraints:

- 0 <= logs.length <= 100
- 3 <= logs[i].length <= 100
- logs[i] is guaranteed to have an identifier, and a word after the identifier.

**Hint:**
* Use python sort function with `key` set to a custom method that handles all the cases cited in the question.

[LeetCode link](https://leetcode.com/problems/reorder-data-in-log-files/)

<details>
<summary>Click here to see code</summary>

```python
def reorderLogFiles(self, logs: List[str]) -> List[str]:
    if not logs:
        return []

    def compareFunc(log):
        key, rest = log.split(' ', 1)
        return (0, rest, key) if rest[0].isalpha() else (1,)

    logs.sort(key=compareFunc)
    
    return logs
```
**Note:** `(1,)` above denotes that it's a tuple rather than an invalid function call. It can also be return as `tuple([1])`.

</details>

## 4.) Intersection of Two Arrays
Given two arrays, write a function to compute their intersection.

### Example 1:

**Input:** nums1 = [1,2,2,1], nums2 = [2,2]

**Output:** [2]

### Example 2:

**Input:** nums1 = [4,9,5], nums2 = [9,4,9,8,4]

**Output:** [9,4]

### Note:

- Each element in the result must be unique.
- The result can be in any order.

**Hint:**
* __Approach 1:__ Convert lists to sets and then do an intersection. i.e. `set1 & set2`
    * _Time Complexity:_ `O(m + n)`
    * _Space Complexity:_ `O(m + n)`
* __Approach 2:__ Use a hashmap to store the frequency of items in the list that has min no. of elements and then do a linear scan of the other list to identify the common elements.
    * _Time Complexity:_ `O(m + n)`
    * _Space Complexity:_ `O( min(m, n) )`

[LeetCode link](https://leetcode.com/problems/intersection-of-two-arrays/)

<details>
<summary>Click here to see code</summary>

```python
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        if not nums1 or not nums2:
            return []
        
        numMap = defaultdict(int)

        len1, len2 = len(nums1), len(nums2)
        minLengthList, maxLengthList = nums1, nums2
 
        if min(len1, len2) == len2:
            minLengthList = nums2
            maxLengthList = nums1
        
        for num in minLengthList:
            numMap[num] += 1
        
        result = []
        for num in maxLengthList:
            if num in numMap and numMap[num] > 0:
                result.append(num)
                numMap[num] = 0
        
        return result
```

</details>

## 5.) Intersection of Two Arrays II
Given two arrays, write a function to compute their intersection.

### Example 1:

**Input:** `nums1 = [1,2,2,1], nums2 = [2,2]`

**Output:** `[2,2]`

### Example 2:

**Input:** `nums1 = [4,9,5], nums2 = [9,4,9,8,4]`

**Output:** `[4,9]`

### Note:

- Each element in the result should appear as many times as it shows in both arrays.
- The result can be in any order.

### Follow up:

- What if the given array is already sorted? How would you optimize your algorithm?
- What if nums1's size is small compared to nums2's size? Which algorithm is better?
- What if elements of nums2 are stored on disk, and the memory is limited such that you cannot load all elements into the memory at once?

**Hint:**
* __Approach 1:__ Convert lists to sets and then do an intersection. i.e. `set1 & set2`
    * _Time Complexity:_ `O(m + n)`
    * _Space Complexity:_ `O(m + n)`
* __Approach 2:__ Use a hashmap to store the frequency of items in the list that has min no. of elements and then do a linear scan of the other list to identify the common elements.
    * _Time Complexity:_ `O(m + n)`
    * _Space Complexity:_ `O( min(m, n) )`
* __Approach 3:__ Sort both the lists. Then, Use a hashmap to store the frequency of items in the list that has min no. of elements and then do a linear scan of the other list to identify the common elements.
    * _Time Complexity:_ `O(m + n)`
    * _Space Complexity:_ `O( min(m, n) )`


- What if the given array is already sorted? How would you optimize your algorithm?
    - Using an `O(n+m)` linear scan algorithm, it can be done in constant time.
- What if nums1's size is small compared to nums2's size? Which algorithm is better?
    - Using the hashmap approach by hashing the smaller size list should do.
- What if elements of nums2 are stored on disk, and the memory is limited such that you cannot load all elements into the memory at once?
    - If anyone of the list or the range of the chars in the list fits in memory, hashmap approach should work out.
    - If neither of the array fit, we can apply partial processing strategies.
        * Split the numeric range into subranges that fits into memory. Modify hashmap approach to collect counts only within a given subrange and call the method multiple times for each subrange.
        * Use an external sort for both arrays. Load them and process arrays sequentially.

[LeetCode link](https://leetcode.com/problems/intersection-of-two-arrays-ii/)

<details>
<summary>Click here to see code</summary>

```python
from collections import defaultdict
class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        if not nums1 or not nums2:
            return []
        
        numMap = defaultdict(int)

        len1, len2 = len(nums1), len(nums2)
        minLengthList, maxLengthList = nums1, nums2
 
        if min(len1, len2) == len2:
            minLengthList = nums2
            maxLengthList = nums1
        
        for num in minLengthList:
            numMap[num] += 1
        
        result = []
        for num in maxLengthList:
            if num in numMap and numMap[num] > 0:
                result.append(num)
                numMap[num] -= 1
        
        return result
```

</details>

## 6.) Longest Substring with At Most K Distinct Characters
Given a string, find the length of the longest substring T that contains at most k distinct characters.

### Example 1:

**Input:** `s = "eceba", k = 2`

**Output:** `3`

**Explanation:** `T is "ece" which its length is 3.`


### Example 2:

**Input:** `s = "aa", k = 1`

**Output:** `2`

**Explanation:** `T is "aa" which its length is 2.`


**Hint:**
* Use a sliding window approach and a hashmap. Keep track of the start and end of the window. Move the start pointer only after the window has more than k unique characters(length of the hashmap at that instant).
    * Note that if the range of characters is 256, then the time complexity is independent of the hashmap space. So, it'd an `O(1)` space complexity.
* Use a sliding window and ordered dict(linked hash map). Store the last index for each character in the ordered dict so that it can be used when the sliding window moves out of a character. Ordered dict helps in adding a key, getting a key from the first/last pos, deleting a key in `O(1)` time.

[LeetCode link](https://leetcode.com/problems/longest-substring-with-at-most-k-distinct-characters/)

<details>
<summary>Click here to see code</summary>

## Approach 1: Using hashmap

```python
class Solution:
    def lengthOfLongestSubstringKDistinct(self, s: str, k: int) -> int:
        if k <= 0:
            return 0
        
        frequencyMap = defaultdict(int)
        
        start, end = 0, 0
        maxLength = 0
        for end in range(len(s)):
            frequencyMap[ s[end] ] += 1
            
            while len(frequencyMap) > k:
                substrStartChar = s[start]
                frequencyMap[substrStartChar] -= 1
                if frequencyMap[substrStartChar] == 0:
                    del frequencyMap[substrStartChar]
                start += 1
            
            maxLength = max(maxLength, end - start + 1)
        
        return maxLength
```
*Note:*
- The line `while len(frequencyMap) > k:` can't be replaced with `if len(frequencyMap) > k:` because for cases like `aeeeiou` where when the sliding window enters `o`, it should drop all the `e's` inorder to count the accurate distinct characters at `o`. So, the time complexity is `O(nxm)` where m is the avg. no. of times each character is repeated in the array.

## Approach 2: Using ordered dict

```python
from collections import OrderedDict
class Solution:
    def lengthOfLongestSubstringKDistinct(self, s: 'str', k: 'int') -> 'int':
        n = len(s) 
        if k == 0 or n == 0:
            return 0
        
        # sliding window left and right pointers
        left, right = 0, 0
        # hashmap character -> its rightmost position 
        # in the sliding window
        hashmap = OrderedDict()

        max_len = 1
        
        while right < n:
            character = s[right]
            # if character is already in the hashmap -
            # delete it, so that after insert it becomes
            # the rightmost element in the hashmap
            if character in hashmap:
                del hashmap[character]
            hashmap[character] = right
            right += 1

            # slidewindow contains k + 1 characters
            if len(hashmap) == k + 1:
                # delete the leftmost character
                _, del_idx = hashmap.popitem(last = False)
                # move left pointer of the slidewindow
                left = del_idx + 1

            max_len = max(max_len, right - left)

        return max_len
```        
</details>

## 7.) Intersection of Three Sorted Arrays

Given three integer arrays arr1, arr2 and arr3 sorted in strictly increasing order, return a sorted array of only the integers that appeared in all three arrays.

### Example 1:

**Input:** arr1 = [1,2,3,4,5], arr2 = [1,2,5,7,9], arr3 = [1,3,4,5,8]

**Output:** [1,5]

**Explanation:** Only 1 and 5 appeared in the three arrays.

### Constraints:

- `1 <= arr1.length, arr2.length, arr3.length <= 1000`
- `1 <= arr1[i], arr2[i], arr3[i] <= 2000`

**Hint:**
* Write a method that computes intersection for two sorted arrays and use that to find the intersection for 3 arrays.
* The intersection of two sorted arrays can be done in `min(len(list1), len(list2))` time.

[LeetCode link](https://leetcode.com/problems/intersection-of-three-sorted-arrays/)

<details>
<summary>Click here to see code</summary>

```python
class Solution:
    def arraysIntersection(self, arr1: List[int], arr2: List[int], arr3: List[int]) -> List[int]:
        
        def intersectionTwoArrays(list1: List[int], list2: List[int]) -> List[int]:
            if not list1 or not list2:
                return []
            
            result = []
            
            i, j = 0, 0
            while i < len(list1) and j < len(list2):
                if list1[i] < list2[j]:
                    i += 1
                elif list2[j] < list1[i]:
                    j += 1
                else:
                    result.append(list1[i])
                    i += 1
                    j += 1

            return result
            
        return intersectionTwoArrays(intersectionTwoArrays(arr1, arr2), arr3)
```

</details>


## 8.) Next Permutation

Implement *next* permutation, which rearranges numbers into the lexicographically next greater permutation of numbers.

If such arrangement is not possible, it must rearrange it as the lowest possible order (ie, sorted in ascending order).

The replacement must be in-place and use only constant extra memory.

Here are some examples. Inputs are in the left-hand column and its corresponding outputs are in the right-hand column.

`1,2,3 → 1,3,2`
`3,2,1 → 1,2,3`
`1,1,5 → 1,5,1`

**Hint:**
* Find the first number from the right that is less than it's successive number i.e `a[i] < a[i+1]`. Swap `a[i]` with the number that is first higher than it starting from the right as well.
* Now reverse the array starting from `a[i+1]` till the end of the array. This would yield the next highest permutation as well as handle the case of `3 2 1` and transform it to `1 2 3`.

<img src="https://leetcode.com/media/original_images/31_Next_Permutation.gif">

[LeetCode link](https://leetcode.com/problems/next-permutation/)

<details>
<summary>Click here to see code</summary>

```python
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        if not nums:
            return
        
        i = len(nums) - 2
        while i >= 0 and nums[i] >= nums[i + 1]:
            i -= 1
        
        if i >= 0:
            j = len(nums) - 1
            while j >= 0 and nums[j] <= nums[i]:
                j -= 1

            nums[i], nums[j] = nums[j], nums[i]

        def reverse(arr, start):
            i, j = start, len(arr) - 1

            while i < j:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
                j -= 1
        
        reverse(nums, i + 1)
```

</details>

## 9.) 3Sum

Given an array _nums_ of n integers, are there elements a, b, c in nums such that _a + b + c = 0_? Find all unique triplets in the array which gives the sum of zero.

### Note:

The solution set must not contain duplicate triplets.

### Example:

Given array `nums = [-1, 0, 1, 2, -1, -4]`,

A solution set is:
```
[
  [-1, 0, 1],
  [-1, -1, 2]
]
```
**Hint:**
* Sort the array. Loop through index 0, say i. For each pass, use two pointers; one starting from i + 1 and the other from the end of the array incrementing/decrementing them as necessary.

[LeetCode link](https://leetcode.com/problems/3sum/)

<details>
<summary>Click here to see code</summary>

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        if not nums or len(nums) < 3:
            return []

        result = []
        
        nums.sort()
        for i in range(len(nums) - 2):
            if nums[i] > 0:
                break

            if i != 0 and nums[i-1] == nums[i]:
                continue
            
            low, high = i + 1, len(nums) - 1
            
            while low < high:
                sumTriplet = nums[i] + nums[low] + nums[high]
                if sumTriplet == 0:
                    result.append([nums[i], nums[low], nums[high]])
                    low += 1
                    high -= 1
                    while low < high and nums[low] == nums[low - 1]:
                        low += 1
                elif sumTriplet < 0:
                    low += 1
                elif sumTriplet > 0:
                    high -= 1

        return result
```

</details>

## 10.) Search a 2D Matrix

Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties:

Integers in each row are sorted from left to right.
The first integer of each row is greater than the last integer of the previous row.

### Example 1:
```
Input:
matrix = [
  [1,   3,  5,  7],
  [10, 11, 16, 20],
  [23, 30, 34, 50]
]
target = 3
Output: true
```
### Example 2:
```
Input:
matrix = [
  [1,   3,  5,  7],
  [10, 11, 16, 20],
  [23, 30, 34, 50]
]
target = 13
Output: false
```

**Hint:**
* Consider the entire matrix as a sorted array starting at 0 and ending at m*n-1. Row, Column can be determined using the math `[index // colSize][index % colSize]`. Now, do a binary search for this. Time Complexity: `O(log(mn))`.

[LeetCode link](https://leetcode.com/problems/search-a-2d-matrix/)

<details>
<summary>Click here to see code</summary>

### Approach 1: Binary search on entire matrix
```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        m = len(matrix)
        if m == 0:
            return False
        n = len(matrix[0])
        
        # binary search
        left, right = 0, m * n - 1
        while left <= right:
                pivot_idx = (left + right) // 2
                pivot_element = matrix[pivot_idx // n][pivot_idx % n]
                if target == pivot_element:
                    return True
                else:
                    if target < pivot_element:
                        right = pivot_idx - 1
                    else:
                        left = pivot_idx + 1
        return False
```

### Approach 2: O(m+n) search
```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        if not matrix:
            return False
        
        rowSize, colSize = len(matrix), len(matrix[0])
        i, j = rowSize - 1, 0
        
        while i >= 0 and j < colSize:
            if matrix[i][j] == target:
                return True
            elif matrix[i][j] < target:
                j += 1
            else:                
                i -= 1

        return False
```

</details>

# Template:

Use this template to create new problems

## .) 


**Hint:**
* 

[LeetCode link](https://leetcode.com/problems/)

<details>
<summary>Click here to see code</summary>

```python
```

</details>