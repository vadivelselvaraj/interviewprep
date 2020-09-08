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

[LeetCode link](https://leetcode.com/problems/binary-tree-postorder-traversal)
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


## 3.) Pre Order traversal of a binary tree
*Hint:* Same code as postorder but throw rightChild, leftChild and then the root node into the stack.

[LeetCode link](https://leetcode.com/problems/binary-tree-preorder-traversal)
<details>
<summary>Click here to see code</summary>

```python
def preorderTraversal(self, root: TreeNode) -> List[int]:
    if not root:
        return []

    result = []
    
    stack = [(root, False)]
    while stack:
        node, subtreeTraversed = stack.pop()
        if subtreeTraversed:
            result.append(node.val)
        else:
            if node.right:
                stack.append((node.right, False))
            if node.left:
                stack.append((node.left, False))

            stack.append((node, True))
    
    return result
```

</details> 

## 4.) Pre Order traversal of a n-ary tree
*Hint:* Same code as postorder but throw rightChild, leftChild and then the root node into the stack.
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

**Test Cases to consider**
- Sum appearing in both the sides of the tree.
- Sum appearing only in left subtree.
- Sum appearing only in right subtree.
- Only one node having the sum.

**Hint:**
* Do top down traversal via a DFS, recuring down the sum along the way until the intended sum is reached.
* Key Note: We should remove the currentSum once the left and right subtrees are done.
* Remember to use _list(pathList)_ to do a deep copy of the _pathList_ so that the `result` i.e. _List[List[int]]_ doesn't carry a pointer to the modified _pathList_.

[LeetCode link](https://leetcode.com/problems/path-sum-iii/)

<details>
<summary>Click here to see code</summary>

```python
from collections import defaultdict
class Solution:
    def pathSum(self, root: TreeNode, sum: int) -> int:
        if not root:
            return 0
        
        numberOfPaths, prefixSumMap = 0, defaultdict(int)
        def countPathSum(node: TreeNode, currentSum: int):
            if not node:
                return
            
            currentSum += node.val
            nonlocal sum, prefixSumMap, numberOfPaths
            
            if currentSum == sum:
                numberOfPaths += 1

            if currentSum - sum in prefixSumMap:
                numberOfPaths += prefixSumMap[currentSum - sum]
            
            prefixSumMap[currentSum] += 1

            if node.left:
                countPathSum(node.left, currentSum)
            if node.right:
                countPathSum(node.right, currentSum)
            
            prefixSumMap[currentSum] -= 1

        countPathSum(root, 0)
        return numberOfPaths
```

</details>


## 30.) Path Sum IV

If the depth of a tree is smaller than 5, then this tree can be represented by a list of three-digits integers.

For each integer in this list:

The hundreds digit represents the depth D of this node, 1 <= D <= 4.
The tens digit represents the position P of this node in the level it belongs to, 1 <= P <= 8. The position is the same as that in a full binary tree.
The units digit represents the value V of this node, 0 <= V <= 9.
Given a list of ascending three-digits integers representing a binary tree with the depth smaller than 5, you need to return the sum of all paths from the root towards the leaves.

It's guaranteed that the given list represents a valid connected binary tree.

**Hint:**
* Construct a map of { 'depth+pos': node.val } to map the positions and the corresponding value.
* With the above, the left and right child of a node can be found at _(depth+1, 2xpos-1)_ and _(depth+1, 2xpos)_ respectively.
* Given these, traverse the nodes via DFS and add the sum.
* Note: Integer division in python is done via `123//10` and not `123/10`

[LeetCode link](https://leetcode.com/problems/path-sum-iv/)

<details>
<summary>Click here to see code</summary>

```python
class Solution:
    def pathSum(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        sum = 0
        # Construct a map of { 'depth+pos': node.val }
        # With this map, a node's left children can be found at (depth+1, 2xpos-1)
        # and right children at (depth+1, 2xpos)
        keyToNodeMap = { x//10: x%10 for x in nums }
        
        def dfs(nodeKey, sumSoFar):
            if not nodeKey:
                return
            
            sumSoFar += keyToNodeMap[nodeKey]
            nodeDepth, nodePos = divmod(nodeKey, 10)
            leftChildKey = (10 * (nodeDepth + 1)) + (2*nodePos) - 1
            rightChildKey = leftChildKey + 1
            
            nonlocal sum
            if leftChildKey not in keyToNodeMap and rightChildKey not in keyToNodeMap:
                sum += sumSoFar

            if leftChildKey in keyToNodeMap:
                dfs(leftChildKey, sumSoFar)
            if rightChildKey in keyToNodeMap:
                dfs(rightChildKey, sumSoFar)
            
            
        dfs(nums[0]//10, 0)
        return sum
```

</details>

## 31.) Construct Binary Tree from Preorder and Inorder Traversal

Given preorder and inorder traversal of a tree, construct the binary tree.

Note:
You may assume that duplicates do not exist in the tree.

**Questions to ask**
- Duplicates nodes in the tree?

**Testcases to consider**

**Hint:**
* First element in preorder list is the root.
* For each iteration, identify the left and right subtrees in the inorder list and set the root's(element currently in the preorder list) children recursively.
* Note: when the left and right inorder indices match, stop the iteration. Make sure to have a map to lookup the index of a node.

[LeetCode link](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal)

<details>
<summary>Click here to see code</summary>

```python
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        if not preorder:
            return
        
        nodeToIndexMap = { nodeVal: index for index, nodeVal in enumerate(inorder) }
        preOrderIndex = 0

        def _buildTree(leftInorderIndex, rightInorderIndex):
            if leftInorderIndex == rightInorderIndex:
                return None
            
            nonlocal preOrderIndex, nodeToIndexMap
            root = TreeNode(preorder[preOrderIndex])
            rootIndex = nodeToIndexMap[preorder[preOrderIndex]]
            
            preOrderIndex += 1
            
            root.left = _buildTree(leftInorderIndex, rootIndex)
            root.right = _buildTree(rootIndex + 1, rightInorderIndex)

            return root

        return _buildTree(0, len(inorder))
```

</details>


## 32.) Construct Binary Tree from Preorder and Inorder Traversal

Given inorder and postorder traversal of a tree, construct the binary tree.

Note:
You may assume that duplicates do not exist in the tree.

**Questions to ask**
- Duplicates nodes in the tree?

**Testcases to consider**

**Hint:**
* Last element in postorder list is the root.
* For each iteration, identify the left and right subtrees in the inorder list and set the root's(element currently in the postorder list) children recursively.
* Note:
  * when the left and right inorder indices match, stop the iteration
  * Make sure to have a map to lookup the index of a node.
  * Build right child first before the left child unlike in the preorder and inorder problem.

[LeetCode link](https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)

<details>
<summary>Click here to see code</summary>

```python
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> TreeNode:
        if not postorder:
            return
        
        nodeToIndexMap = { nodeVal: index for index, nodeVal in enumerate(inorder) }
        postOrderIndex = len(postorder) - 1

        def _buildTree(leftInorderIndex, rightInorderIndex):
            if leftInorderIndex == rightInorderIndex:
                return None
            
            nonlocal postOrderIndex, nodeToIndexMap
            root = TreeNode(postorder[postOrderIndex])
            rootInorderIndex = nodeToIndexMap[ postorder[postOrderIndex] ]
            
            postOrderIndex -= 1
            
            root.right = _buildTree(rootInorderIndex + 1, rightInorderIndex)
            root.left = _buildTree(leftInorderIndex, rootInorderIndex)

            return root

        return _buildTree(0, len(inorder))
```

</details>

## 33.) Construct Binary Tree from String

You need to construct a binary tree from a string consisting of parenthesis and integers.

The whole input represents a binary tree. It contains an integer followed by zero, one or two pairs of parenthesis. The integer represents the root's value and a pair of parenthesis contains a child binary tree with the same structure.

You always start to construct the left child node of the parent first if it exists.

**Questions to ask**
- Duplicates nodes in the tree?

**Testcases to consider**

**Hint:**
* Last element in postorder list is the root.
* For each iteration, identify the left and right subtrees in the inorder list and set the root's(element currently in the postorder list) children recursively.
* Note:
  * when the left and right inorder indices match, stop the iteration
  * Make sure to have a map to lookup the index of a node.
  * Build right child first before the left child unlike in the preorder and inorder problem.

[LeetCode link](https://leetcode.com/problems/construct-binary-tree-from-string/)

<details>
<summary>Click here to see code</summary>

```python
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> TreeNode:
        if not postorder:
            return
        
        nodeToIndexMap = { nodeVal: index for index, nodeVal in enumerate(inorder) }
        postOrderIndex = len(postorder) - 1

        def _buildTree(leftInorderIndex, rightInorderIndex):
            if leftInorderIndex == rightInorderIndex:
                return None
            
            nonlocal postOrderIndex, nodeToIndexMap
            root = TreeNode(postorder[postOrderIndex])
            rootInorderIndex = nodeToIndexMap[ postorder[postOrderIndex] ]
            
            postOrderIndex -= 1
            
            root.right = _buildTree(rootInorderIndex + 1, rightInorderIndex)
            root.left = _buildTree(leftInorderIndex, rootInorderIndex)

            return root

        return _buildTree(0, len(inorder))
```

</details>

## 34.) Construct Binary Search Tree from Preorder Traversal

Return the root node of a binary search tree that matches the given preorder traversal.

(Recall that a binary search tree is a binary tree where for every node, any descendant of node.left has a value < node.val, and any descendant of node.right has a value > node.val.  Also recall that a preorder traversal displays the value of the node first, then traverses node.left, then traverses node.right.)

It's guaranteed that for the given test cases there is always possible to find a binary search tree with the given requirements.

**Questions to ask**
- Duplicates nodes in the tree?
- Left child strictly less than parent and converse?

**Hint:**
* First element in preorder list is the root.
* Leverage BST property i.e. _node.left.val < node.val < node.right.val_. Starting from root with (-inf, inf) as the left and right ranges, traverse down the preorder list until the BST property doesn't satisfy.

[LeetCode link](https://leetcode.com/problems/construct-binary-search-tree-from-preorder-traversal/)

<details>
<summary>Click here to see code</summary>

```python
class Solution:
    def bstFromPreorder(self, preorder: List[int]) -> TreeNode:
        if not preorder:
            return None
        
        size = len(preorder)
        index = 0
        
        def buildTree(leftRange, rightRange):
            nonlocal size, preorder, index

            if index >= size:
                return None
            
            currentVal = preorder[index]
            if currentVal < leftRange or currentVal > rightRange:
                return None
            
            root = TreeNode(currentVal)
            index += 1
            
            root.left = buildTree(leftRange, currentVal - 1)
            root.right = buildTree(currentVal + 1, rightRange)
            return root

        return buildTree(-math.inf, math.inf)
```

</details>

## 35.) Construct Binary Tree from Preorder and Postorder Traversal

Return any binary tree that matches the given preorder and postorder traversals.

Values in the traversals pre and post are distinct positive integers.

**Questions to ask**

**Test Cases to consider**

**Hint:**
* Note: From a preorder and post order traversal, only a unique full binary tree can be constructed nor a binary tree.
* Partition the array into left and right subtree based on where the current preorder node is within the post order list.
*

[LeetCode link](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-postorder-traversal/solution/)

<details>
<summary>Click here to see code</summary>

```python
class Solution:
    def constructFromPrePost(self, pre: List[int], post: List[int]) -> TreeNode:
        if not pre:
            return None

        nodeToIndexMap = { node: index for index, node in enumerate(post) }
        def make(i0, i1, N):
            if N == 0: return None
            root = TreeNode(pre[i0])

            if N == 1: return root

            nonlocal nodeToIndexMap
            L = nodeToIndexMap[ pre[i0+1] ] - i1 + 1 

            root.left = make(i0 + 1, i1, L)
            root.right = make(i0 + L + 1, i1 + L, N - 1 - L)
            return root

        return make(0, 0, len(pre))
```

</details>

## 36.) Find Bottom Left Tree Value

Given a binary tree, find the leftmost value in the last row of the tree.

**Questions to ask**
- What should we return when the tree is null?
- Is it okay to return the right most element at the last depth of the tree?

**Test Cases to consider**
- Empty Tree

**Hint:**
* Do a BFS and keep updating the answer with the first element at each depth. Finally, return it.

[LeetCode link](https://leetcode.com/problems/find-bottom-left-tree-value/)

<details>
<summary>Click here to see code</summary>

```python
from collections import deque
class Solution:
    def findBottomLeftValue(self, root: TreeNode) -> int:
        if not root:
            return -1
        
        queue = deque([root])
        bottomLeftVal = None
        
        while queue:
            size = len(queue)

            for i in range(size):
                node = queue.popleft()
                if i == 0:
                    bottomLeftVal = node.val

                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            
            
        return bottomLeftVal
```

</details>

## 37.) Closest Binary Search Tree Value
Given a non-empty binary search tree and a target value, find the value in the BST that is closest to the target.

**Note:**
- Given target value is a floating point.
- You are guaranteed to have only one unique value in the BST that is closest to the target.

**Questions to ask**
- Distinct nodes?

**Test Cases to consider**

**Hint:**
* 

[LeetCode link](Closest Binary Search Tree Value)

<details>
<summary>Click here to see code</summary>

## Approach 1: O(N) time

```python
    def closestValue(self, root: TreeNode, target: float) -> int:
        if not root:
            return None
        
        closestVal, closestDiff = None, math.inf
        
        def dfs(node):
            if not node:
                return None

            nonlocal closestVal, closestDiff
            diff = abs(target - node.val)

            if diff < closestDiff:
                closestVal = node.val
                closestDiff = diff
 
            if node.left:
                dfs(node.left)
            if node.right:
                dfs(node.right)
            
        dfs(root)
        return closestVal
```

## Approach 2: O(H) time

```python
    def closestValue(self, root: TreeNode, target: float) -> int:
        if not root:
            return None
        closestVal = root.val
        t = root
        while t:
            closestVal = min(closestVal, t.val, key = lambda x: abs(target - x))
            t = t.left if target < t.val  else t.right

        return closestVal
```

</details>


## 38.) Sum of Nodes with Even-Valued Grandparent

Given a binary tree, return the sum of values of nodes with even-valued grandparent.  (A grandparent of a node is the parent of its parent, if it exists.)

If there are no nodes with an even-valued grandparent, return 0.
Question text

### Example 1:
<img src="https://assets.leetcode.com/uploads/2019/07/24/1473_ex1.png">

**Input**: root = [6,7,8,2,7,1,3,9,null,1,4,null,null,null,5]

**Output**: 18

**Explanation**: The red nodes are the nodes with even-value grandparent while the blue nodes are the even-value grandparents.

**Questions to ask**

**Test Cases to consider**

**Hint:**
* Do a BFS or a DFS passing around the parent and the grand parent. For each iteration, make the current node as parent and the current node's parent as grandparent.

[LeetCode link](https://leetcode.com/problems/sum-of-nodes-with-even-valued-grandparent/)

<details>
<summary>Click here to see code</summary>

```python
from collections import deque
class Solution:
    def sumEvenGrandparent(self, root: TreeNode) -> int:
        if not root:
            return 0
    
        result = 0
        
        queue = deque([(root, None, None)])
        while queue:
            levelSize = len(queue)
            
            for _ in range(levelSize):
                node, isParentEvenValued, isGParentEvenValued = queue.popleft()

                if isGParentEvenValued:
                    result += node.val

                currentGrandParentEvenValued = isParentEvenValued
                currentParentEvenValued = True if node.val % 2 == 0 else False

                if node.left:
                    queue.append((node.left, currentParentEvenValued,currentGrandParentEvenValued))
                if node.right:
                    queue.append((node.right, currentParentEvenValued,currentGrandParentEvenValued))

        return result
```

</details>

## 39.) Smallest Subtree with all the Deepest Nodes

Given a binary tree rooted at root, the depth of each node is the shortest distance to the root.

A node is deepest if it has the largest depth possible among any node in the entire tree.

The subtree of a node is that node, plus the set of all descendants of that node.

Return the node with the largest depth such that it contains all the deepest nodes in its subtree.

### Example 1:

**Input:** `[3,5,1,6,2,0,8,null,null,7,4]`

**Output:** `2`

<img src="https://s3-lc-upload.s3.amazonaws.com/uploads/2018/07/01/sketch1.png">

### Example 2:

**Input:** `[1,2,3,4]`

**Output:** `4`

### Example 3:

**Input:** `[1,2,3,null,4]`

**Output:** `4`

**Questions to ask**
- distinct valued tree?

**Test Cases to consider**
- `[1,2,3,null,4]` should return `4` and not `2` as `2` is the subtree with largest depth that also contains `2`.
- `[1,2,3,4]` should return `4`.

**Hint:**
* Do a bottom up approach and do the below.
    * Pass on the left or right node that has the max depth at each level.
    * If both the left and right nodes are at same depth, only then pass the parent node.

[LeetCode link](https://leetcode.com/problems/smallest-subtree-with-all-the-deepest-nodes)

<details>
<summary>Click here to see code</summary>

```python
class Solution:
    def subtreeWithAllDeepest(self, root: TreeNode) -> TreeNode:
        if not root:
            return root
    
        def _helper(node: TreeNode) -> (TreeNode, int):
            if not node:
                return (None, 0)
            
            leftMaxDepthNode, leftMaxDepth = _helper(node.left)
            rightMaxDepthNode, rightMaxDepth = _helper(node.right)
            
            if leftMaxDepth > rightMaxDepth:
                return (leftMaxDepthNode, leftMaxDepth + 1)
            elif leftMaxDepth < rightMaxDepth:
                return (rightMaxDepthNode, rightMaxDepth + 1)
            return (node, leftMaxDepth + 1)
            
            
        return _helper(root)[0]
```

</details>

## 40.) Lowest Common Ancestor of a Binary Tree

Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.

According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant of itself).”

Given the following binary tree:  root = [3,5,1,6,2,0,8,null,null,7,4]

<img src="https://assets.leetcode.com/uploads/2018/12/14/binarytree.png">

### Example 1:
**Input:** `root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1`

**Output:** `3`

**Explanation:** The LCA of nodes 5 and 1 is 3.
### Example 2:

**Input:** `root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4`

**Output:** `5`

**Explanation:** The LCA of nodes 5 and 4 is 5, since a node can be a descendant of itself according to the LCA definition.
 
### Note:

All of the nodes' values will be unique.
p and q are different and both values will exist in the binary tree.

**Questions to ask**
- A node is a descendant of itself. i.e. `p=6, p=6` should return `6` as the LCA.
**Test Cases to consider**
- Left/Right skewed tree having both the target nodes.

**Hint:**
* Do a bottom up approach returning `(num_matched_target_nodes, ancestor)` at each stage. Return the node iff 

[LeetCode link](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/)

<details>
<summary>Click here to see code</summary>

## Approach 1:

```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root:
            return root
        
        def _helper(node, x, y):
            if not node:
                return (0, None)
            
            leftMatchedCount, leftAncestor = _helper(node.left, x, y)
            if leftMatchedCount == 2:
                return (leftMatchedCount, leftAncestor)

            rightMatchedCount, rightAncestor = _helper(node.right, x, y)
            if rightMatchedCount == 2:
                return (rightMatchedCount, rightAncestor)

            currentCount = leftMatchedCount + rightMatchedCount + (node == p) + (node == q)
            
            return (currentCount, node if currentCount == 2 else None)
            
            
        return _helper(root, p, q)[1]
```
An alternate using set
```python
def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':

        result = None

        def traversal(node_root):
            nonlocal result

            if not node_root:
                return set()

            tree_left = traversal(node_root.left)
            tree_right = traversal(node_root.right)

            node_all = {node_root.val} | tree_left | tree_right
            if p.val in node_all and q.val in node_all and not result:
                result = node_root

            return node_all

        traversal(root)
        return result
```

## Approach 2: Using parent pointers

```python
class Solution:

    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """

        # Stack for tree traversal
        stack = [root]

        # Dictionary for parent pointers
        parent = {root: None}

        # Iterate until we find both the nodes p and q
        while p not in parent or q not in parent:

            node = stack.pop()

            # While traversing the tree, keep saving the parent pointers.
            if node.left:
                parent[node.left] = node
                stack.append(node.left)
            if node.right:
                parent[node.right] = node
                stack.append(node.right)

        # Ancestors set() for node p.
        ancestors = set()

        # Process all ancestors for node p using parent pointers.
        while p:
            ancestors.add(p)
            p = parent[p]

        # The first ancestor of q which appears in
        # p's ancestor set() is their lowest common ancestor.
        while q not in ancestors:
            q = parent[q]
        return q
```

```python
def lca(node0: BinaryTreeNode,
        node1: BinaryTreeNode) -> Optional[BinaryTreeNode]:
    def get_depth(node):
        depth = 0
        while node.parent:
            depth += 1
            node = node.parent
        return depth

    depth0, depth1 = map(get_depth, (node0, node1))
    # Makes node0 as the deeper node in order to simplify the code.
    if depth1 > depth0:
        node0, node1 = node1, node0

    # Ascends from the deeper node.
    depth_diff = abs(depth0 - depth1)
    while depth_diff:
        node0 = node0.parent
        depth_diff -= 1

    # Now ascends both nodes until we reach the LCA.
    while node0 is not node1:
        node0, node1 = node0.parent, node1.parent
    return node0
```

</details>

## 41.) Recover Binary Search Tree

Two elements of a binary search tree (BST) are swapped by mistake.

Recover the tree without changing its structure.

### Example 1:
```
Input: [1,3,null,null,2]

   1
  /
 3
  \
   2

Output: [3,1,null,null,2]

   3
  /
 1
  \
   2
```   
### Example 2:
```
Input: [3,1,4,null,null,2]

  3
 / \
1   4
   /
  2

Output: [2,1,4,null,null,3]

  2
 / \
1   4
   /
  3
```
### Follow up:

A solution using O(n) space is pretty straight forward.
Could you devise a constant space solution?

**Questions to ask**

**Test Cases to consider**
- swapped elements are consecutive. Example 2: [1, 3, 2, 4]
- swapped elements are not consecutive. Example 1: [3, 2, 1]

**Hint:**
* Do an inorder traversal, keeping track of the predecessor and then find the two swapped elements that can be consecutive/non-consecutive. Swap the found elements.

[LeetCode link](https://leetcode.com/problems/recover-binary-search-tree/)

<details>
<summary>Click here to see code</summary>

```python
class Solution:
    def recoverTree(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        if not root:
            return

        x, y, predecessor = None, None, None
        
        def findSwappedNodes(node: TreeNode):
            if not node:
                return

            nonlocal x, y, predecessor
            if node.left:
                findSwappedNodes(node.left)
            
            if predecessor and node.val < predecessor.val:
                x = node
                if y is None:
                    y = predecessor
                else:
                    return

            predecessor = node

            if node.right:
                findSwappedNodes(node.right)
        
        findSwappedNodes(root)
        x.val, y.val = y.val, x.val
```

## Approach 2: Using Morris Inorder Traversal(constant space)

```python
class Solution:
    def recoverTree(self, root):
        """
        :type root: TreeNode
        :rtype: void Do not return anything, modify root in-place instead.
        """
        # predecessor is a Morris predecessor. 
        # In the 'loop' cases it could be equal to the node itself predecessor == root.
        # pred is a 'true' predecessor, 
        # the previous node in the inorder traversal.
        x = y = predecessor = pred = None
        
        while root:
            # If there is a left child
            # then compute the predecessor.
            # If there is no link predecessor.right = root --> set it.
            # If there is a link predecessor.right = root --> break it.
            if root.left:       
                # Predecessor node is one step left 
                # and then right till you can.
                predecessor = root.left
                while predecessor.right and predecessor.right != root:
                    predecessor = predecessor.right
 
                # set link predecessor.right = root
                # and go to explore left subtree
                if predecessor.right is None:
                    predecessor.right = root
                    root = root.left
                # break link predecessor.right = root
                # link is broken : time to change subtree and go right
                else:
                    # check for the swapped nodes
                    if pred and root.val < pred.val:
                        y = root
                        if x is None:
                            x = pred 
                    pred = root
                    
                    predecessor.right = None
                    root = root.right
            # If there is no left child
            # then just go right.
            else:
                # check for the swapped nodes
                if pred and root.val < pred.val:
                    y = root
                    if x is None:
                        x = pred 
                pred = root
                
                root = root.right
        
        x.val, y.val = y.val, x.val
```

</details>


## 42.) Range Sum BST

Given the root node of a binary search tree, return the sum of values of all nodes with value between L and R (inclusive).

The binary search tree is guaranteed to have unique values.

## Example 1:
```
Input: root = [10,5,15,3,7,null,18], L = 7, R = 15
Output: 32
```
## Example 2:
```
Input: root = [10,5,15,3,7,13,18,1,null,6], L = 6, R = 10
Output: 23
```

### Note:

- The number of nodes in the tree is at most 10000.
- The final answer is guaranteed to be less than 2^31.

**Questions to ask**
- Sum within range?
**Test Cases to consider**

**Hint:**
* 

[LeetCode link](https://leetcode.com/problems/range-sum-of-bst/)

<details>
<summary>Click here to see code</summary>

```python
def rangeSumBST(self, root: TreeNode, L: int, R: int) -> int:
    if not root:
        return 0
    
    stack = [root]
    result = 0
    while stack:
        node = stack.pop()
        if L <= node.val <= R:
            result += node.val
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left) 
    
    return result
```

</details>

## 43.) Binary Tree Cameras

Given a binary tree, we install cameras on the nodes of the tree. 

Each camera at a node can monitor its parent, itself, and its immediate children.

Calculate the minimum number of cameras needed to monitor all nodes of the tree.

### Example 1:
<img src="https://assets.leetcode.com/uploads/2018/12/29/bst_cameras_01.png">

```
Input: [0,0,null,0,0]
Output: 1
Explanation: One camera is enough to monitor all nodes if placed as shown.
```

### Example 2:
<img src="https://assets.leetcode.com/uploads/2018/12/29/bst_cameras_02.png">

```
Input: [0,0,null,0,null,0,null,null,0]
Output: 2
Explanation: At least two cameras are needed to monitor all nodes of the tree. The above image shows one of the valid configurations of camera placement.
```
### Note:

- The number of nodes in the given tree will be in the range [1, 1000].
- Every node has value 0.

**Questions to ask**

**Test Cases to consider**

**Hint:**
* Do a postorder DFS. Pass the parent one of the below values.
  * 1: Requires monitoring
  * 2: Has a camera
  * 3: Doesn't require monitoring and doesn't have a camera

[LeetCode link](https://leetcode.com/problems/binary-tree-cameras)

<details>
<summary>Click here to see code</summary>

```python
    def minCameraCover(self, root: TreeNode) -> int:
        if not root:
            return 0
        minCover = 0
        
        # 1: Requires monitoring
        # 2: Has a camera
        # 3: Doesn't require monitoring and doesn't have a camera
        def dfs(node):
            nonlocal minCover
            
            # Null node doesn't require monitoring
            if not node:
                return 3
            
            L = dfs(node.left)
            R = dfs(node.right)
            
            # if either one of the children requires monitoring, install camera
            # at this node
            if L == 1 or R == 1:
                minCover += 1
                return 2

            # if either one of the children already has a camera, don't install
            # a camera but let the parent know.
            if L == 2 or R == 2:
                return 3

            # if the node is root, install the camera
            if node == root:
                minCover += 1
            else:
                # tell parent that it requires monitoring
                return 1
            
        dfs(root)
        return minCover
```

</details>

## 44.) Distribute coins in a binary tree

Given the root of a binary tree with N nodes, each node in the tree has node.val coins, and there are N coins total.

In one move, we may choose two adjacent nodes and move one coin from one node to another.  (The move may be from parent to child, or from child to parent.)

Return the number of moves required to make every node have exactly one coin.


### Example 1:

<img src="https://assets.leetcode.com/uploads/2019/01/18/tree1.png">

```
Input: [3,0,0]
Output: 2
Explanation: From the root of the tree, we move one coin to its left child, and one coin to its right child.
```

### Example 2:

<img src="https://assets.leetcode.com/uploads/2019/01/18/tree2.png">

```
Input: [0,3,0]
Output: 3
Explanation: From the left child of the root, we move two coins to the root [taking two moves].  Then, we move one coin from the root of the tree to the right child.
```

### Example 3:
<img src="https://assets.leetcode.com/uploads/2019/01/18/tree3.png">

```
Input: [1,0,2]
Output: 2
```
### Example 4:
<img src="https://assets.leetcode.com/uploads/2019/01/18/tree4.png">

```
Input: [1,0,0,null,3]
Output: 4
```

### Note:

- 1<= N <= 100
- 0 <= node.val <= N

**Questions to ask**

**Test Cases to consider**

**Hint:**
* Use a bottom up approach and pass the number of excessive coins from the leaf to the parent. While doing so, count the number of coins that can be transferred from a given node to its children and vice versa & accumulate this to a var.

[LeetCode link](https://leetcode.com/problems/distribute-coins-in-binary-tree/)

<details>
<summary>Click here to see code</summary>

```python
    def distributeCoins(self, root: TreeNode) -> int:
        if not root:
            return 0
        
        numberOfMoves = 0
        
        def dfs(node):
            nonlocal numberOfMoves
            if not node:
                return 0
            L = dfs(node.left)
            R = dfs(node.right)

            # 
            numberOfMoves += abs(L) + abs(R)

            # return the excess coins from this node to the parent
            return node.val + L + R - 1
            
        dfs(root)
        return numberOfMoves
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