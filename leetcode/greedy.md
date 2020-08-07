## 1.) Task Scheduler

You are given a char array representing tasks CPU need to do. It contains capital letters A to Z where each letter represents a different task. Tasks could be done without the original order of the array. Each task is done in one unit of time. For each unit of time, the CPU could complete either one task or just be idle.

However, there is a non-negative integer n that represents the cooldown period between two same tasks (the same letter in the array), that is that there must be at least n units of time between any two same tasks.

You need to return the least number of units of times that the CPU will take to finish all the given tasks.

**Questions to ask**

None

**Test Cases to consider**
- n=0

**Hint:**
* Take a greedy approach. Compute frequencies of all tasks and sort it.
* Answer: _len(tasks) + minIdleTime_
* To compute minIdleTime, do the below.
  * _maxIdleTime = (frequency_of_task_with_max_freq - 1) * n_.
  * Subtract _maxIdleTime_ with the frequency of other less frequency items at each step until the idleTime doesn't drop below zero or doesn't exceed _frequency_of_task_with_max_freq - 1_ as a less frequency item cannot exceed this slots.
* Remember to ensure that _idleTime_ doesn't drop below zero.

[LeetCode link](https://leetcode.com/problems/task-scheduler/)

<details>
<summary>Click here to see code</summary>

```python
class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        frequency = [0] * 26
        
        # Compute frequency for all character
        for task in tasks:
            frequency[ ord(task) - ord('A') ] += 1
        
        # Sort Frequency in ascending order
        frequency.sort()
        
        # Fetch the max frequency and compute max idle time
        maxFrequency = frequency.pop() 
        idleTime = (maxFrequency - 1) * n
        
        while frequency and idleTime >= 0:
            currentFrequency = frequency.pop()
            idleTime -= min(maxFrequency - 1, currentFrequency)
            
        return len(tasks) + max(0, idleTime)
```

</details>
