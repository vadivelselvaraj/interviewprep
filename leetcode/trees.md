# Problems

## 1.) Post Order traversal of a binary tree
*Hint:* Traverse the root node only after traversing the children using a single stack.
<details>
<summary>Click here to see code</summary>

```python
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        result = []

        # Base Case
        if not root:
            return result
        
        stack = [(root, False)]
        while len(stack) > 0:
            current, childrenTraversed = stack.pop()
            if current:
                if childrenTraversed:
                    result.append(current.val)
                else:
                    stack.append((current, True))
                    if current.right:
                        stack.append((current.right, False))
                    if current.left:
                        stack.append((current.left, False))
        
        return result
```

</details>  

## 2.) Post Order traversal of a nary tree
*Hint:* Traverse the root node only after traversing the children using a single stack.
<details>
<summary>Click here to see code</summary>

```python
def postorder(self, root: 'Node') -> List[int]:
	result = []

	# Base Case
	if not root:
		return result
	
	stack = [(root, False)]
	while len(stack) > 0:
		current, childrenTraversed = stack.pop()
		if current:
			if childrenTraversed:
				result.append(current.val)
			else:
				stack.append((current, True))
				for child in current.children[::-1]:
					stack.append((child, False))
	
	return result
```

</details> 


## 3.) Pre Order traversal of a n-ary tree
*Hint:* Traverse the root node only after traversing the children using a single stack.
<details>
<summary>Click here to see code</summary>

```python
def preorder(self, root: 'Node') -> List[int]:
	result = []

	# Base Case
	if not root:
		return result

	stack = [(root, False)]
	while len(stack) > 0:
		current, childrenTraversed = stack.pop()
		if current:
			if childrenTraversed:
				result.append(current.val)
			else:
				for child in current.children[::-1]:
					stack.append((child, False))
				stack.append((current, True))

	return result
```

</details> 

## 4.) Pre Order traversal of a n-ary tree
*Hint:* Traverse the root node only after traversing the children using a single stack.
<details>
<summary>Click here to see code</summary>

```python
def preorder(self, root: 'Node') -> List[int]:
	result = []

	# Base Case
	if not root:
		return result

	stack = [(root, False)]
	while len(stack) > 0:
		current, childrenTraversed = stack.pop()
		if current:
			if childrenTraversed:
				result.append(current.val)
			else:
				for child in current.children[::-1]:
					stack.append((child, False))
				stack.append((current, True))

	return result
```

</details> 


## 5.) Inorder traversal of a binary tree
*Hint:* Traverse the root node only after traversing the left child using a single stack.
<details>
<summary>Click here to see code</summary>

## Recursive Traversal
```python
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        return self.inorderTraversal(root.left) +  [ root.val ] + self.inorderTraversal(root.right) if root else 
```

## Iterative traversal

```python
def inorderTraversal(self, root: TreeNode) -> List[int]:
	result = []
	stack = [(root, False)]

	while len(stack) > 0:
		current, leftTreeTraversed = stack.pop()
		if current:
			if leftTreeTraversed:
				result.append(current.val)
			else:
				if current.right:
					stack.append((current.right, False))
				stack.append((current, True))
				if current.left:
					stack.append((current.left, False))

	return result
```

## Morris Algorithm to do Inorder traversal in constant space

```python
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        temp, pred, result = root, None, []

        while temp:
            if not temp.left:
                result.append(temp.val)
                temp = temp.right
            else:
                # find inOrder predecessor of the current node
                pred = temp.left
                while pred.right != None and pred.right != temp:
                    pred = pred.right
                
                # If pred isn't set already, let's set it.
                if not pred.right:
                    pred.right = temp
                    pred = temp
                    temp = temp.left
                else:
                    # Let's remove the predecessor link
                    pred.right = None
                    result.append(temp.val)
                    temp = temp.right
                    
        return result
```
</details> 

## 6.) Level traversal of a n-ary tree
*Hint:* Enqueue the children nodes of the current level nodes at each level and add them to the output for each iteration. Use a single queue.
<details>
<summary>Click here to see code</summary>

```python
def levelOrder(self, root: 'Node') -> List[List[int]]:
	result = []
	if not root:
		return result
	
	currentLevelNodes = [root]
	while currentLevelNodes:
		result.append([node.val for node in currentLevelNodes])
		nextLevelNodes = []
		for node in currentLevelNodes:
			for child in node.children:
				nextLevelNodes.append(child)
		currentLevelNodes = nextLevelNodes
		
	return result
```

</details> 


## 7.) Max Depth of a binary tree
Maximum depth of a node is the length of the longest path between the root node and a leaf node.

*Hint:* MaxDepth = 1 + max(leftMaxDepth, rightMaxDepth)
<details>
<summary>Click here to see code</summary>

```python
def maxDepth(self, root: TreeNode) -> int:
	if not root:
		return 0

	maxDepth = 0
	stack = [(root, 1)]

	while stack:
		node, currentDepth = stack.pop()
		if node:
			maxDepth = max(maxDepth, currentDepth)
			stack.append((node.left, currentDepth + 1))
			stack.append((node.right, currentDepth + 1))
	
	return maxDepth
```

</details> 

## 8.) Min Depth of a binary tree
Minimum depth is the number of nodes along the shortest path from the root node down to the nearest leaf node.

*Hint:* Update minDepth only when the node is a leaf node. Depth = 1 + min(leftMinDepth, rightMinDepth)
<details>
<summary>Click here to see code</summary>

```python
def minDepth(self, root: TreeNode) -> int:
	if not root:
		return 0

	minDepth = math.inf
	stack = [(root, 1)]

	while stack:
		node, currentDepth = stack.pop()
		if node:
			# Update minDepth iff it's leaf node.
			if not node.left and not node.right:
				minDepth = min(minDepth, currentDepth)
			stack.append((node.left, currentDepth + 1))
			stack.append((node.right, currentDepth + 1))
	
	return minDepth
```

</details>


## 9.) Max Depth of a n-ary tree
The maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

*Hint:* MaxDepth = 1 + max(leftMaxDepth, rightMaxDepth)
<details>
<summary>Click here to see code</summary>

```python
def maxDepth(self, root: 'Node') -> int:
	if not root:
		return 0

	maxDepth = 0
	stack = [(root, 1)]

	while stack:
		node, currentDepth = stack.pop()
		if node:
			maxDepth = max(maxDepth, currentDepth)
			for child in node.children:
				stack.append((child, currentDepth + 1))

	return maxDepth
```

</details> 


## 10.) Check if a binary tree is height-balanced
A binary tree in which the left and right subtrees of every node differ in height by no more than 1.

*Hint:*
* Use a bottom up approach to save the cost of computing heights for each node in the tree.
* Check if the child subtrees are balanced. If they are, use their heights to determine if the current subtree is balanced as well as to calculate the current subtree's height.

[LeetCode link](https://leetcode.com/problems/balanced-binary-tree/)

<details>
<summary>Click here to see code</summary>

```python
def _isBalanced(self, root: TreeNode) -> (bool, int):
	if not root:
		return True, -1
	
	isLeftBalanced, leftHeight = self._isBalanced(root.left)
	if not isLeftBalanced:
		return False, leftHeight
	
	isRightBalanced, rightHeight = self._isBalanced(root.right)
	if not isRightBalanced:
		return False, rightHeight
	
	return ( abs(leftHeight - rightHeight) < 2, 1 + max(leftHeight, rightHeight) )
	
	
def isBalanced(self, root: TreeNode) -> bool:
	return self._isBalanced(root)[0]
```

</details> 

## 11.) Same tree
Given two binary trees, write a function to check if they are the same or not.

Two binary trees are considered the same if they are structurally identical and the nodes have the same value.

*Hint:*
* Are not equal if anyone of them is null while the other isn't.
* Are not equal if their values differ.
* Recursively check if left and right children are the same.
[LeetCode link](https://leetcode.com/problems/same-tree/)

<details>
<summary>Click here to see code</summary>

## Recursion

```python
def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
	if not p and not q:
		return True

	if not p or not q:
		return False
	
	if p.val != q.val:
		return False
	
	return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
```

## Iteration

```python
def isSameTree(self, p, q):
	"""
	:type p: TreeNode
	:type q: TreeNode
	:rtype: bool
	"""    
	def check(p, q):
		# if both are None
		if not p and not q:
			return True
		# one of p and q is None
		if not q or not p:
			return False
		if p.val != q.val:
			return False
		return True

	stack = [(p, q)]
	while stack:
		p, q = stack.pop()
		if not check(p, q):
			return False

		if p:
			stack.append((p.left, q.left))
			stack.append((p.right, q.right))

	return True
```
</details> 

## 12.) Leaf-similar trees
Two binary trees are considered leaf-similar if their leaf value sequence is the same.

*Hint:*
* Find the leaf value sequence for both given trees. Afterwards, we can compare them to see if they are equal or not.
* To find the leaf value sequence of a tree, we use a depth first search. Our dfs function writes the node's value if it is a leaf, and then recursively explores each child. This is guaranteed to visit each leaf in left-to-right order, as left-children are fully explored before right-children.

[LeetCode link](https://leetcode.com/problems/leaf-similar-trees/)

<details>
<summary>Click here to see code</summary>

```python
class Solution:
    def dfs(self, root: TreeNode, nodes: List[int]) -> None:
        if not root:
            return nodes

        if not root.left and not root.right:
            nodes.append(root.val)

        self.dfs(root.left, nodes)
        self.dfs(root.right, nodes)
    
    def leafSimilar(self, root1: TreeNode, root2: TreeNode) -> bool:
        root1Nodes, root2Nodes = [], []

        self.dfs(root1, root1Nodes)
        self.dfs(root2, root2Nodes)
        
        return root1Nodes == root2Nodes
```

</details>

## 13.) Single Element in a Sorted Array
You are given a sorted array consisting of only integers where every element appears exactly twice, except for one element which appears exactly once. Find this single element that appears only once.

*Hint:*
* Observe that the array will be odd-lengthed.
* Invariant: The subarray with the culprit will be odd-lengthed.
* Construct 4 cases and code that.
	* array[mid] == array[mid + 1], culprit is on the left side
	* array[mid] == array[mid + 1], culprit is on the right side
	* array[mid] == array[mid - 1], culprit is on the left side
	* array[mid] == array[mid - 1], culprit is on the right side

[LeetCode link](https://leetcode.com/problems/single-element-in-a-sorted-array/)

<details>
<summary>Click here to see code</summary>

```python
def singleNonDuplicate(self, nums: List[int]) -> int:
	low, high = 0, len(nums) - 1
	while low < high:
		mid = low + (high - low) // 2
		# print(f"low: {low}, mid: {mid}, high: {high}")
		if nums[mid] == nums[mid-1]:
			rightHalfLength = (high - mid)
				# Left side contains culprit
			if rightHalfLength % 2 == 0:
				high = mid - 2
			else:
				# Right side contains culprit
				low = mid + 1
		elif nums[mid] == nums[mid+1]:
			leftHalfLength = (mid - low)
				# Right side contains culprit
			if leftHalfLength % 2 == 0:
				low = mid + 2
			else:
				# Left side contains culprit
				high = mid - 1
		else:
			return nums[mid]
	return nums[low]
```

</details>

## 14.) Minimum Distance Between BST Nodes
Given a Binary Search Tree (BST) with the root node root, return the minimum difference between the values of any two different nodes in the tree.

*Hint:*
* Invariant: In an sorted array, the difference between any two consecutive elements would yield the minimum value.
* Do an in order traversal, find the inorder predecessor and update the min as you go.

[LeetCode link](https://leetcode.com/problems/minimum-distance-between-bst-nodes/)

<details>
<summary>Click here to see code</summary>

```python
class Solution:
    def minDiffInBST(self, root: TreeNode) -> int:
        self.minDiff = math.inf
        self.predecessor = None
        
        def inorderTraversal(node: TreeNode):
            if not node:
                return
            inorderTraversal(node.left)

            # update min iff there are two nodes to compare
            if self.predecessor:
                self.minDiff = min(self.minDiff, node.val - self.predecessor.val)

            self.predecessor = node
            inorderTraversal(node.right)
        
        inorderTraversal(root)
        return self.minDiff
```

</details>


## 15.) Diameter of a Binary tree
Given a binary tree, you need to compute the length of the diameter of the tree. The diameter of a binary tree is the length of the longest path between any two nodes in a tree. This path may or may not pass through the root.

*Hint:*
* Compute height for each node as _1 + max(leftHeight, rightHeight)_.
* While doing so, also compute diameter assuming that it passes through the current node i.e. _diameter = (leftHeight + rightHeight)_

[LeetCode link](https://leetcode.com/problems/diameter-of-binary-tree/)

<details>
<summary>Click here to see code</summary>

```python
class Solution:
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        diameter = 0

        def height(root: TreeNode):
            if not root:
                return 0
            
            nonlocal diameter

            leftHeight = height(root.left)
            rightHeight = height(root.right)

            # Computer the diameter assuming the path passes through the current node
            diameter = max(diameter, leftHeight + rightHeight)
            
            # return the height of the current node
            return 1 + max(leftHeight, rightHeight)
        
        depth(root)
        return diameter
```

</details>

## 16.) Diameter of a n-ary tree
Given a root of an N-ary tree, you need to compute the length of the diameter of the tree.

The diameter of an N-ary tree is the length of the longest path between any two nodes in the tree. This path may or may not pass through the root.

*Hint:*
* Compute height for each node as _1 + max(height of all children)_.
* While doing so, also compute diameter assuming that it passes through the current node i.e. _diameter = (sum of two children with max height)_

[LeetCode link](https://leetcode.com/problems/diameter-of-n-ary-tree/)

<details>
<summary>Click here to see code</summary>

```python
class Solution:
    def diameter(self, root: 'Node') -> int:
        """
        :type root: 'Node'
        :rtype: int
        """
        diameter = 0
        
        def getTwoMaxElements(inputList: List) -> List[int]:
            if not inputList or len(inputList) <= 1:
                return inputList

            max1, max2 = inputList[0], inputList[1]
            for index in range(1, len(inputList)):
                if max1 < inputList[index]:
                    max2 = max1
                    max1 = inputList[index]
                elif max2 < inputList[index]:
                    max2 = inputList[index]
            
            return [max1, max2]

            
        def height(root: Node):
            if not root:
                return 0
            
            heightOfChildren = []
            for child in root.children:
                heightOfChildren.append( height(child) )
            
            nonlocal diameter
            oldDiameter = diameter
            
            diameter = max(diameter, sum(getTwoMaxElements(heightOfChildren)))

            return 1 if len(heightOfChildren) == 0 else 1 + max(heightOfChildren)

        height(root)
        return diameter
```

</details>

## 17.) Time Needed to Inform All Employees
A company has n employees with a unique ID for each employee from 0 to n - 1. The head of the company has is the one with headID.

Each employee has one direct manager given in the manager array where manager[i] is the direct manager of the i-th employee, manager[headID] = -1. Also it's guaranteed that the subordination relationships have a tree structure.

The head of the company wants to inform all the employees of the company of an urgent piece of news. He will inform his direct subordinates and they will inform their subordinates and so on until all employees know about the urgent news.

The i-th employee needs informTime[i] minutes to inform all of his direct subordinates (i.e After informTime[i] minutes, all his direct subordinates can start spreading the news).

Return the number of minutes needed to inform all the employees about the urgent news.

*Hint:*
* Required: The max time that it takes for an employee at the bottom to get the news.
* Do a BFS or DFS and return the max time from any of the leaf nodes.

[LeetCode link](https://leetcode.com/problems/time-needed-to-inform-all-employees/)

<details>
<summary>Click here to see code</summary>

## Construct an actual tree 

```python
from collections import deque

class Node(object):
    def __init__(self, id, informTime=None):
        self.id = id
        self.informTime = informTime
        self.children = []
        self.informedAt = 0

    def __str__(self):
        return f"id: {self.id}, children: { [str(child) for child in self.children] }"

class Solution:
    def numOfMinutes(self, n: int, headID: int, manager: List[int], informTime: List[int]) -> int:
        if n == 1:
            return informTime[0]

        def createTree():
            nodeMap = {}
            for i in range(n):
                nodeMap[i] = Node(i, informTime[i])

            # root node is the head of the company
            root = nodeMap[headID]

            # Create tree by looping through the manager array
            for index, item in enumerate(manager):
                if item == -1:
                    continue

                managerNode = nodeMap[item]
                childNode = nodeMap[index]

                managerNode.children.append(childNode)

            return root
        
        # Perform BFS on the graph and update the informedAt time for all of the nodes
        def getMaxTime(root):
            queue = deque()
            queue.append(root)
            maxTime = 0

            while len(queue) > 0:
                currentNode = queue.popleft()
                informTimeForChildren = currentNode.informedAt
                if currentNode.informTime:
                    informTimeForChildren = informTimeForChildren + currentNode.informTime

                maxTime = max(maxTime, informTimeForChildren)

                # Update informedAt for all children
                for child in currentNode.children:
                    child.informedAt = informTimeForChildren
                    queue.append(child)    
            return maxTime

        root = createTree()
        return getMaxTime(root)
```

## Without an actual tree
```python
from collections import deque
class Solution:
    def numOfMinutes(self, n: int, headID: int, manager: List[int], informTime: List[int]) -> int:
        if n == 1:
            return informTime[0]
        
        childrenMap = defaultdict(list)
        
        # Setup parent to children map
        for index, item in enumerate(manager):
            childrenMap[item].append(index)
        
        # perform bfs
        maxTime = 0
        
        queue = deque()
        queue.append((headID, 0))
        
        while len(queue) > 0:
            node, timeToInform = queue.popleft()
            timeForChildren = timeToInform + informTime[node]
            maxTime = max(maxTime, timeForChildren)
            
            # If node has children, let's update the informTime
            if node in childrenMap:
                for child in childrenMap[node]:
                    queue.append((child, timeForChildren))
            
        return maxTime
```
</details>


## 18.) Binary Tree Vertical Order Traversal
Given a binary tree, return the vertical order traversal of its nodes' values. (ie, from top to bottom, column by column).

If two nodes are in the same row and column, the order should be from left to right.

*Hint:*
* Do BFS so that nodes can be traversed from top to bottom and from left to right.
* Ensure that left child is first traversed before right.
* Note: Don't sort the map that stores the nodes by vertical distance as in a skewed tree scenario, the time complexity might increase to NlogN. To skip the sorting, maintain minColumn, maxColumn during BFS and use that to populate the result.

[LeetCode link](https://leetcode.com/problems/binary-tree-vertical-order-traversal/)

<details>
<summary>Click here to see code</summary>

```python
class Solution:
    def verticalOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        
        queue = deque()
        queue.append((root, 0))
        verticalDistanceToNodeMap = defaultdict(list)
        minColumn, maxColumn = math.inf, -math.inf

        while len(queue) > 0:
            node, verticalDistance = queue.popleft()
            if node:
                verticalDistanceToNodeMap[verticalDistance].append(node.val)
                minColumn = min(minColumn, verticalDistance)
                maxColumn = max(maxColumn, verticalDistance)

                if node.left:
                    queue.append((node.left, verticalDistance - 1))
                if node.right:
                    queue.append((node.right, verticalDistance + 1))

        result = [ verticalDistanceToNodeMap[col] for col in range(minColumn, maxColumn + 1) ]
            
        return result 
```

</details>

## 19.) Binary Tree Vertical Order Traversal
Given a binary tree, return the vertical order traversal of its nodes values.

For each node at position (X, Y), its left and right children respectively will be at positions (X-1, Y-1) and (X+1, Y-1).

Running a vertical line from X = -infinity to X = +infinity, whenever the vertical line touches some nodes, we report the values of the nodes in order from top to bottom (decreasing Y coordinates).

If two nodes have the same position, then the value of the node that is reported first is the value that is smaller.

Return an list of non-empty reports in order of X coordinate.  Every report will have a list of values of nodes

*Hint:*
* Do BFS so that nodes can be traversed from top to bottom and from left to right.
* Capture elements in a column wise dict as a tuple (row, node.val).
* To sort efficiently, maintain minColumn, maxColumn during BFS and sort the tuples that are within each column.

[LeetCode link](https://leetcode.com/problems/vertical-order-traversal-of-a-binary-tree/)

<details>
<summary>Click here to see code</summary>

```python
class Solution:
    def verticalTraversal(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        
        queue = deque()
        queue.append((root, 0, 0))
        colToNodeMap = defaultdict(list)
        minCol, maxCol = math.inf, -math.inf

        while queue:
            node, row, col = queue.popleft()
            if node:
                colToNodeMap[col].append((row, node.val))
                minCol = min(minCol, col)
                maxCol = max(maxCol, col)

                if node.left:
                    queue.append((node.left, row + 1, col - 1))
                if node.right:
                    queue.append((node.right, row + 1, col + 1))

        result = []
        for key in range(minCol, maxCol + 1):
            result.append( [ val for row, val in sorted(colToNodeMap[key]) ] )
            
        return result 
```

</details>

## 20.)  Maximum Width of Binary Tree
Given a binary tree, write a function to get the maximum width of the given tree. The maximum width of a tree is the maximum width among all levels.

The width of one level is defined as the length between the end-nodes (the leftmost and right most non-null nodes in the level, where the null nodes between the end-nodes are also counted into the length calculation.

*Hint:*
* Do BFS so that nodes can be traversed from top to bottom and from left to right.
* Enqueue nodes as (node, index) where index is as below.
  * root = 1
  * leftChild = 2*parentIndex
  * rightChild = 2*parentIndex + 1
* At the end of each level, update maxWidth using _max(maxWidth, leftMostNodeIndex - leftMostNodeIndex + 1)_

[LeetCode link](https://leetcode.com/problems/maximum-width-of-binary-tree/)

<details>
<summary>Click here to see code</summary>

```python
from collections import deque
class Solution:
    def widthOfBinaryTree(self, root: TreeNode) -> int:
        if not root:
            return 0
        
        queue = deque()
        queue.append((root, 1))
        maxWidth = 0
        
        while queue:
            _, leftNodeIndex = queue[0]
            currentLevelLength = len(queue)
            rightNodeIndex = None
            
            for i in range(currentLevelLength):
                node, index = queue.popleft()
                if i == currentLevelLength - 1:
                    rightNodeIndex = index
                if node.left:
                    queue.append((node.left, 2*index))
                if node.right:
                    queue.append((node.right, 2*index + 1))
            
            maxWidth = max(maxWidth, rightNodeIndex - leftNodeIndex + 1)
        return maxWidth
```

</details>

20.) Find Largest Value in Each Tree Row
find the largest value in each row of a binary tree.

*Hint:*
* Do level order traversal and for each row find the max element.

[LeetCode link](https://leetcode.com/problems/find-largest-value-in-each-tree-row/)

<details>
<summary>Click here to see code</summary>

```python
from collections import deque
class Solution:
    def largestValues(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        
        queue = deque()
        queue.append(root)
        
        result = []
        while queue:
            currentLevelLength = len(queue)
            
            currentLevelMax = -math.inf
            for _ in range(currentLevelLength):
                node = queue.popleft()
                currentLevelMax = max(currentLevelMax, node.val)

                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            result.append(currentLevelMax)
        
        return result
```

</details>

21.) Check Completeness of a Binary Tree
Given a binary tree, determine if it is a complete binary tree.

*Hint:*
* Do BFS and verify the property _index(leftChild) = 2*(parentIndex)_ and _index(rightChild) = 2*(parentIndex) + 1_.
* The property can be verified by by making sure that the nodes visited in the BFS so far equals the index for the node currently processed.

[LeetCode link](https://leetcode.com/problems/check-completeness-of-a-binary-tree/)

<details>
<summary>Click here to see code</summary>

```python
    def isCompleteTree(self, root: TreeNode) -> bool:
        if not root:
            return True
        
        # Perform BFS
        queue = deque()
        queue.append((root, 1))
        totalNodesSoFar = 0
        
        while queue:
            node, index = queue.popleft()
            totalNodesSoFar += 1
            
            if totalNodesSoFar != index:
                return False

            if node.left:
                queue.append((node.left, index*2))
            if node.right:
                queue.append((node.right, index*2+1))

        return True
```

</details>

## 21.) Cousins in Binary Tree

In a binary tree, the root node is at depth 0, and children of each depth k node are at depth k+1.

Two nodes of a binary tree are cousins if they have the same depth, but have different parents.

We are given the root of a binary tree with unique values, and the values x and y of two different nodes in the tree.

Return true if and only if the nodes corresponding to the values x and y are cousins.

*Hint:*
* Do level order traversal and verify the below.
  * Both nodes are found at the same level.
  * Both nodes have different parents.
  * Otherwise, return false.

[LeetCode link](https://leetcode.com/problems/cousins-in-binary-tree/)

<details>
<summary>Click here to see code</summary>

```python
from collections import deque
class Solution:
    def isCousins(self, root: TreeNode, x: int, y: int) -> bool:
        if not root:
            return False
        
        queue = deque()
        queue.append((root, None))

        xFound, yFound = False, False
        xParent, yParent = None, None

        while queue:
            currentLevelLength = len(queue)
            
            for _ in range(currentLevelLength):
                node, parent = queue.popleft()
                if node.val == x:
                    xFound = True
                    xParent = parent
                elif node.val == y:
                    yFound = True
                    yParent = parent

                if node.left:
                    queue.append((node.left, node.val))
                if node.right:
                    queue.append((node.right, node.val))

            # If we've found both the nodes, see if their parents are different
            if xFound and yFound:
                return xParent != yParent
            # If only one of the nodes are found, other node exists in a different depth.
            elif xFound or yFound:
                return False
                
        return False

```

</details>

## 22.) All Nodes Distance K in Binary Tree

We are given a binary tree (with root node root), a target node, and an integer value K.

Return a list of the values of all nodes that have a distance K from the target node.  The answer can be returned in any order.

*Hint:*
* Mark parents of all nodes and then do a BFS assuming neighbors as __(node.left, node.right, node.parent)__

[LeetCode link](https://leetcode.com/problems/all-nodes-distance-k-in-binary-tree/)

<details>
<summary>Click here to see code</summary>

```python
    def distanceK(self, root: TreeNode, target: TreeNode, K: int) -> List[int]:
        if not root or not target:
            []
        
        # Do DFS to get the target node and to annotate parents of all nodes
        def dfs(root, parent):
            if root:
                root.parent = parent
                if root.left:
                    dfs(root.left, root)
                if root.right:
                    dfs(root.right, root)

        dfs(root, None)
        
        # Do BFS to get the nodes at depth K
        queue = deque([(target, 0)])
        visitedSet = set()
        
        while queue:
            if queue[0][1] == K:
                return [ node.val for node, depth in queue ]

            node, depth = queue.popleft()
            visitedSet.add(node)
            
            for neighbor in (node.left, node.right, node.parent):
                if neighbor and neighbor not in visitedSet:
                    visitedSet.add(neighbor)
                    queue.append((neighbor, depth + 1))
        
        return []
```

</details>

## 23.) Binary Search Tree Iterator

Implement an iterator over a binary search tree (BST). Your iterator will be initialized with the root node of a BST.

Calling next() will return the next smallest number in the BST.

*Note:*
- next() and hasNext() should run in average O(1) time and uses O(h) memory, where h is the height of the tree.
- You may assume that next() call will always be valid, that is, there will be at least a next smallest number in the BST when next() is called.

**Hint:**
* Do controlled DFS. i.e. load up all the left nodes in stack in advance and during each call of the next function.

[LeetCode link](https://leetcode.com/problems/binary-search-tree-iterator/)

<details>
<summary>Click here to see code</summary>

```python
class BSTIterator:

    def __init__(self, root: TreeNode):
        self.root = root
        self.stack = []
        self.traverseLeftNodes(root)
        
    def traverseLeftNodes(self, node: TreeNode):
        while node:
            self.stack.append(node)
            node = node.left

    def next(self) -> int:
        """
        @return the next smallest number
        """
        smallestNode = self.stack.pop()
        if smallestNode.right:
            self.traverseLeftNodes(smallestNode.right)

        return smallestNode.val
        

    def hasNext(self) -> bool:
        """
        @return whether we have a next smallest number
        """
        return len(self.stack) > 0


# Your BSTIterator object will be instantiated and called as such:
# obj = BSTIterator(root)
# param_1 = obj.next()
# param_2 = obj.hasNext()
```

</details>

Note: Time Complexity of the solution is O(N) at the worst case i.e a skewed tree. But since the requirement is for an average(or amortized) time, this solution would suffice.


## 24.) Binary Tree Paths

Given a binary tree, return all root-to-leaf paths.

Note: A leaf is a node with no children.

**Hint:**
* Do DFS, append each node visited to a path string and finally append the entire path when a leaf node is reached.

[LeetCode link](https://leetcode.com/problems/binary-tree-paths/)

<details>
<summary>Click here to see code</summary>

## Approach 1: Iteration

```python
    def binaryTreePaths(self, root: TreeNode) -> List[str]:
        if not root:
            return []

        result = []
        
        queue = deque([(root, str(root.val))])
        
        while queue:
            node, path = queue.popleft()
            if node:
                if not node.left and not node.right:
                    result.append(path)
                else:
                    if node.left:
                        queue.append((node.left, f"{path}->{node.left.val}"))
                    if node.right:
                        queue.append((node.right, f"{path}->{node.right.val}"))

        return result
```

## Approach 2: Recursion

```python
    def binaryTreePaths(self, root: TreeNode) -> List[str]:
        if not root:
            return []

        result = []
        def dfs(node: TreeNode, path: str):
            if node:
                if not node.left and not node.right:
                    nonlocal result
                    result.append(path)
                elif node.left:
                    dfs(node.left, f"{path}->{node.left.val}")
                if node.right:
                    dfs(node.right, f"{path}->{node.right.val}")

        dfs(root, str(root.val))

        return result
```

</details>



## 25.) Binary Tree Maximum Path Sum

Given a non-empty binary tree, find the maximum path sum.

For this problem, a path is defined as any sequence of nodes from some starting node to any node in the tree along the parent-child connections. The path must contain at least one node and does not need to go through the root.

**Hint:**
* Do a DFS and at each node, return _max( node.val, node.val+max( leftsum, rightSum ) )_.
* While doing so, also keep track of the maxPathSum so far including the

[LeetCode link](https://leetcode.com/problems/binary-tree-maximum-path-sum/)

<details>
<summary>Click here to see code</summary>

```python
class Solution:
    def maxPathSum(self, root: TreeNode) -> int:
        if not root:
            return 0
        
        maxPathSum = -math.inf
        def dfs(node: TreeNode):
            if not node:
                return 0

            leftSum = dfs(node.left)
            rightSum = dfs(node.right)
            
            nonlocal maxPathSum
            currentPathSum = max(
                node.val,
                node.val + max(leftSum, rightSum)
            )
            maxPathSum = max(maxPathSum, currentPathSum, node.val + leftSum + rightSum)
            
            return currentPathSum
            
                
        dfs(root)
        print()

        return maxPathSum
```

</details>

## 26.) Longest Univalue path

Given a binary tree, find the length of the longest path where each node in the path has the same value. This path may or may not pass through the root.

The length of path between two nodes is represented by the number of edges between them.

**Hint:**
* Do DFS and increase the path length iff _node.val == node.left.val_ or the converse and _reset it to zero_ otherwise.

[LeetCode link](https://leetcode.com/problems/longest-univalue-path/)

<details>
<summary>Click here to see code</summary>

```python
    def longestUnivaluePath(self, root: TreeNode) -> int:
        if not root:
            return 0

        longestPath = 0
        def dfs(node: TreeNode):
            if not node:
                return 0

            leftPathLength, rightPathLength = 0, 0
            if node.left:
                leftPathLength = dfs(node.left)
            if node.right:
                rightPathLength = dfs(node.right)

            nonlocal longestPath
            # If the nodes are equal, let's increase the path length.
            # Otherwise, set it to zero.
            if node.left and node.val == node.left.val:
                leftPathLength += 1
            else:
                leftPathLength = 0

            if node.right and node.val == node.right.val:
                rightPathLength += 1
            else:
                rightPathLength = 0
            
            longestPath = max(longestPath, leftPathLength + rightPathLength)
            
            return  max(leftPathLength, rightPathLength)
  
        dfs(root)
        return longestPath
```

</details>

## 27.) Path Sum I

Given a binary tree and a sum, determine if the tree has a root-to-leaf path such that adding up all the values along the path equals the given sum.

Note: A leaf is a node with no children.

**Hint:**
* Do a DFS; a top down approach, passing down the sum as we recur down the tree and once the leaf node is reached, check if the give sum is reached. If so, return true.

[LeetCode link](https://leetcode.com/problems/path-sum/)

<details>
<summary>Click here to see code</summary>

```python
class Solution:
    def hasPathSum(self, root: TreeNode, sum: int) -> bool:
                if not root:
                    return False
                
                def _hasPathSum(node: TreeNode, pathSum: int):
                    if not node:
                        return False

                    nonlocal sum
                    if not node.left and not node.right:
                        if sum == node.val + pathSum:
                            return True

                    return _hasPathSum(node.left, node.val + pathSum) or _hasPathSum(node.right, node.val + pathSum)

                return _hasPathSum(root, 0)
```

</details>


## 28.) Path Sum II

Given a binary tree and a sum, find all root-to-leaf paths where each path's sum equals the given sum.

Note: A leaf is a node with no children.

**Hint:**
* Do top down traversal via a DFS, recuring down the sum along the way until the intended sum is reached.
* Note: We should pop a node once all the subtree under the node are traversed.
* Remember to use _list(pathList)_ to do a deep copy of the _pathList_ so that the `result` i.e. _List[List[int]]_ doesn't carry a pointer to the modified _pathList_.

[LeetCode link](https://leetcode.com/problems/path-sum-ii/)

<details>
<summary>Click here to see code</summary>

```python
from copy import deepcopy
class Solution:
    def pathSum(self, root: TreeNode, sum: int) -> List[List[int]]:
        if not root:
            return []
        
        result = []
        def _pathSum(node: TreeNode, pathSum: int, pathList: List[int]):
            if not node:
                return

            currentPathSum = node.val + pathSum
            pathList.append(node.val)
            
            nonlocal sum, result
            if not node.left and not node.right:
                if sum == currentPathSum:
                    result.append(deepcopy(pathList))
            if node.left:
                _pathSum(node.left, currentPathSum, pathList)
            if node.right:
                _pathSum(node.right, currentPathSum, pathList)

            pathList.pop()

        _pathSum(root, 0, [])
        return result
```

</details>


## 29.) Path Sum III

You are given a binary tree in which each node contains an integer value.

Find the number of paths that sum to a given value.

The path does not need to start or end at the root or a leaf, but it must go downwards (traveling only from parent nodes to child nodes).

**Hint:**
* Do top down traversal via a DFS, recuring down the sum along the way until the intended sum is reached.
* Note: We should pop a node once all the subtree under the node are traversed.
* Remember to use _list(pathList)_ to do a deep copy of the _pathList_ so that the `result` i.e. _List[List[int]]_ doesn't carry a pointer to the modified _pathList_.

[LeetCode link](https://leetcode.com/problems/path-sum-ii/)

<details>
<summary>Click here to see code</summary>

```python
class Solution:
    def pathSum(self, root: TreeNode, sum: int) -> List[List[int]]:
        if not root:
            return []
        
        result = []
        def _pathSum(node: TreeNode, pathSum: int, pathList: List[int]):
            if not node:
                return

            currentPathSum = node.val + pathSum
            pathList.append(node.val)
            
            nonlocal sum, result
            if not node.left and not node.right:
                if sum == currentPathSum:
                    result.append(list(pathList))
            if node.left:
                _pathSum(node.left, currentPathSum, pathList)
            if node.right:
                _pathSum(node.right, currentPathSum, pathList)

            pathList.pop()

        _pathSum(root, 0, [])
        return result
```

</details>


## 30.) Path Sum IV

Given a binary tree and a sum, find all root-to-leaf paths where each path's sum equals the given sum.

Note: A leaf is a node with no children.

**Hint:**
* Do top down traversal via a DFS, recuring down the sum along the way until the intended sum is reached.
* Note: We should pop a node once all the subtree under the node are traversed.
* Remember to use _list(pathList)_ to do a deep copy of the _pathList_ so that the `result` i.e. _List[List[int]]_ doesn't carry a pointer to the modified _pathList_.

[LeetCode link](https://leetcode.com/problems/path-sum-ii/)

<details>
<summary>Click here to see code</summary>

```python
class Solution:
    def pathSum(self, root: TreeNode, sum: int) -> List[List[int]]:
        if not root:
            return []
        
        result = []
        def _pathSum(node: TreeNode, pathSum: int, pathList: List[int]):
            if not node:
                return

            currentPathSum = node.val + pathSum
            pathList.append(node.val)
            
            nonlocal sum, result
            if not node.left and not node.right:
                if sum == currentPathSum:
                    result.append(list(pathList))
            if node.left:
                _pathSum(node.left, currentPathSum, pathList)
            if node.right:
                _pathSum(node.right, currentPathSum, pathList)

            pathList.pop()

        _pathSum(root, 0, [])
        return result
```

</details>


# Template:

## .) 


**Hint:**
* 

[LeetCode link]()

<details>
<summary>Click here to see code</summary>

```python
```

</details>