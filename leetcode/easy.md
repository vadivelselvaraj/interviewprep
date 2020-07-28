# Problems
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

## 2.) Post Order traversal of a binary tree
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

## 3.) Post Order traversal of a nary tree
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

## 5.) Pre Order traversal of a n-ary tree
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


## 6.) Inorder traversal of a binary tree
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

## 7.) Level traversal of a n-ary tree
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


## 8.) Max Depth of a binary tree
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

## 9.) Min Depth of a binary tree
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


## 10.) Max Depth of a n-ary tree
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


## 11.) Check if a binary tree is height-balanced
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

## 12.) Same tree
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

## 13.) Leaf-similar trees
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

## 14.) Single Element in a Sorted Array
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

## 15.) Minimum Distance Between BST Nodes
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

## 18.) Binary Tree Vertical Order Traversal
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

## 19.)  Maximum Width of Binary Tree
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