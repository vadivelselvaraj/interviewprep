# Graphs

## 1.) Sum of distances in tree

An undirected, connected tree with N nodes labelled `0...N-1` and `N-1` edges are given.

The ith edge connects nodes `edges[i][0]` and `edges[i][1]` together.

Return a list `ans`, where `ans[i]` is the sum of the distances between node i and all other nodes.

### Example 1:
```
Input: N = 6, edges = [[0,1],[0,2],[2,3],[2,4],[2,5]]
Output: [8,12,6,10,10,10]
Explanation: 
Here is a diagram of the given tree:
  0
 / \
1   2
   /|\
  3 4 5
We can see that dist(0,1) + dist(0,2) + dist(0,3) + dist(0,4) + dist(0,5)
equals 1 + 1 + 2 + 2 + 2 = 8.  Hence, answer[0] = 8, and so on.
```

**Note:** `1 <= N <= 10000`

**Hint:**
* Use a postorder DFS and memoize the `(node, parent)` calls in order to compute the below.
  * `distanceSum[i] = sum( distanceSum[Ci] + nodeCount[Ci] )`, where `Ci` is the children of node `i`
  * A straight forward implementation of the above would be `O(n^2)` as we've to dfs for each children. To fix that, we need to memoize it.

[LeetCode link](https://leetcode.com/problems/sum-of-distances-in-tree/)

<details>
<summary>Click here to see code</summary>

## Approach 1: Dynamic Programming
Time complexity: `O(2|E|)`, where `E` is the number of edges i.e. `O(E)`
Space Complexity: `O(E)`, since we store twice the number of edges in the memoization map.
```python
from collections import defaultdict, namedtuple

Result = namedtuple('Result', ('distanceSum', 'nodeCount'))
class Solution:
    def sumOfDistancesInTree(self, N: int, edges: List[List[int]]) -> List[int]:
        if N == 0:
            return []
        
        # Construct graph with adjacency list
        graph = defaultdict(set)
        for u, v in edges:
            graph[u].add(v)
            graph[v].add(u)
        
        # Result holder
        distanceSum = [0] * N
        
        mem = {}
        def dfs(node, parent):
            if (node, parent) in mem:
                return mem[(node, parent)]
            
            distanceSum = 0
            nodeCount = 1
            
            for child in graph[node]:
                if child != parent:
                    childResult = dfs(child, node)
                    nodeCount += childResult.nodeCount
                    distanceSum += childResult.distanceSum + childResult.nodeCount
            
            mem[(node, parent)] = Result(distanceSum, nodeCount)
            return mem[(node, parent)]
        
        
        for i in range(N):
            distanceSum[i] = dfs(i, -1).distanceSum
        
        return distanceSum
```

## Approach 2: Subtree Sum and Count
Two DFS
```python
class Solution(object):
    def sumOfDistancesInTree(self, N, edges):
        graph = collections.defaultdict(set)
        for u, v in edges:
            graph[u].add(v)
            graph[v].add(u)

        count = [1] * N
        ans = [0] * N
        def dfs(node = 0, parent = None):
            for child in graph[node]:
                if child != parent:
                    dfs(child, node)
                    count[node] += count[child]
                    ans[node] += ans[child] + count[child]

        def dfs2(node = 0, parent = None):
            for child in graph[node]:
                if child != parent:
                    ans[child] = ans[node] - count[child] + N - count[child]
                    dfs2(child, node)

        dfs()
        dfs2()
        return ans
```

</details>



# Template:

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