# Strings
## 1.) Remove vowels from a string
<details>
<summary>Click here to see code</summary>

```python
def removeVowels(input):
	result = ''
	for character in input:
		if character not in "aeiou":
			result = result + character
	return result
```

</details> 

## 2.) Reverse words in a string

Given an input string, reverse the string word by word.

[LeetCode link](https://leetcode.com/problems/reverse-words-in-a-string/)
<details>
<summary>Click here to see code</summary>

## Approach 1: Built-in functions

```python
class Solution:
    def reverseWords(self, s: str) -> str:
        return ' '.join(reversed(list(filter(str.strip, s.strip().split(' ')))))
```

## Approach 2: Loop through each word

```python
class Solution:
    def trim_spaces(self, s: str) -> list:
        left, right = 0, len(s) - 1
        # remove leading spaces
        while left <= right and s[left] == ' ':
            left += 1
        
        # remove trailing spaces
        while left <= right and s[right] == ' ':
            right -= 1
        
        # reduce multiple spaces to single one
        output = []
        while left <= right:
            if s[left] != ' ':
                output.append(s[left])
            elif output[-1] != ' ':
                output.append(s[left])
            left += 1
        
        return output
            
    def reverse(self, l: list, left: int, right: int) -> None:
        while left < right:
            l[left], l[right] = l[right], l[left]
            left, right = left + 1, right - 1
            
    def reverse_each_word(self, l: list) -> None:
        n = len(l)
        start = end = 0
        
        while start < n:
            # go to the end of the word
            while end < n and l[end] != ' ':
                end += 1
            # reverse the word
            self.reverse(l, start, end - 1)
            # move to the next word
            start = end + 1
            end += 1
                
    def reverseWords(self, s: str) -> str:
        # convert string to char array 
        # and trim spaces at the same time
        l = self.trim_spaces(s)
        
        # reverse the whole string
        self.reverse(l, 0, len(l) - 1)
        
        # reverse each word
        self.reverse_each_word(l)
        
        return ''.join(l)
```

</details>

## 3.) Positions of Large Groups

In a string S of lowercase letters, these letters form consecutive groups of the same character.

For example, a string like S = "abbxxxxzyy" has the groups "a", "bb", "xxxx", "z" and "yy".

Call a group large if it has 3 or more characters.  We would like the starting and ending positions of every large group.

The final answer should be in lexicographic order.

### Example 1:
```
Input: "abbxxxxzzy"
Output: [[3,6]]
Explanation: "xxxx" is the single large group with starting  3 and ending positions 6.
```
### Example 2:
```
Input: "abc"
Output: []
Explanation: We have "a","b" and "c" but no large group.
```
### Example 3:
```
Input: "abcdddeeeeaabbbcd"
Output: [[3,5],[6,9],[12,14]]
```

**Questions to ask**
- Characters only between 'A' and 'Z'?

**Test Cases to consider**
- `"aaa": [[0,2]]`

**Hint:**
* Two pointer approach: Start with (start, end) and make sure it denotes the start and end of the large group. Move on to the next pair when the there is a mismatch.

[LeetCode link](https://leetcode.com/problems/positions-of-large-groups/solution/)

<details>
<summary>Click here to see code</summary>

```python
def largeGroupPositions(self, S: str) -> List[List[int]]:
    if not S:
        return []
    
    result, index = [], 0
    
    while index < len(S) - 1:
        count, start, end = 1, index, index
        while index < len(S) - 1 and S[index] == S[index + 1]:
            count += 1
            index += 1
            end = index
        else:
            index += 1
        
        if count >= 3:
            result.append([start, end])

    return result
```

```python
def largeGroupPositions(self, S: str) -> List[List[int]]:
    if not S:
        return []
    
    ans = []
    i = 0 # The start of each group
    for j in range(len(S)):
        if j == len(S) - 1 or S[j] != S[j+1]:
            # Here, [i, j] represents a group.
            if j-i+1 >= 3:
                ans.append([i, j])
            i = j + 1
    return ans
```

</details>

## 4.) Minimum window substring

Given two strings s and t, return the minimum window in s which will contain all the characters in t. If there is no such window in s that covers all characters in t, return the empty string "".

Note that If there is such a window, it is guaranteed that there will always be only one unique minimum window in s.

**Questions to ask**

**Test Cases to consider**

**Hint:**
* 

[LeetCode link](https://leetcode.com/problems/minimum-window-substring/)

<details>
<summary>Click here to see code</summary>
The idea is we use a variable-length sliding window which is gradually applied across the string. We use two pointers: start and end to mark the sliding window. We start by fixing the start pointer and moving the end pointer to the right. The way we determine the current window is a valid one is by checking if all the target letters have been found in the current window. If we are in a valid sliding window, we first make note of the sliding window of the most minimum length we have seen so far. Next we try to contract the sliding window by moving the start pointer. If the sliding window continues to be valid, we note the new minimum sliding window. If it becomes invalid (all letters of the target have been bypassed), we break out of the inner loop and go back to moving the end pointer to the right.

```python
def found_target(target_len):
    return target_len == 0

class Solution(object):
    def minWindow(self, search_string, target):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        target_letter_counts = collections.Counter(target)
        start = 0
        end = 0
        min_window = ""
        target_len = len(target)        
        
        for end in range(len(search_string)):
			# If we see a target letter, decrease the total target letter count
			if target_letter_counts[search_string[end]] > 0:
                target_len -= 1

            # Decrease the letter count for the current letter
			# If the letter is not a target letter, the count just becomes -ve
			target_letter_counts[search_string[end]] -= 1
            
			# If all letters in the target are found:
            while found_target(target_len):
                window_len = end - start + 1
                if not min_window or window_len < len(min_window):
					# Note the new minimum window
                    min_window = search_string[start : end + 1]
                    
				# Increase the letter count of the current letter
                target_letter_counts[search_string[start]] += 1
                
				# If all target letters have been seen and now, a target letter is seen with count > 0
				# Increase the target length to be found. This will break out of the loop
                if target_letter_counts[search_string[start]] > 0:
                    target_len += 1
                    
                start+=1
                
        return min_window
```

</details>

# Template for adding a new question

## 1.) Question label

Question text

**Questions to ask**

**Test Cases to consider**

**Hint:**
* 

[LeetCode link]()

<details>
<summary>Click here to see code</summary>

```python
```

</details>