# Arrays

## 1.)  Subarray Sum Equals K

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