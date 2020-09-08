# Problems
## 1.) Print Immutable Linked List in Reverse
You are given an immutable linked list, print out all values of each node in reverse with the help of the following interface:

- `ImmutableListNode:` An interface of immutable linked list, you are given the head of the list.
You need to use the following functions to access the linked list (you can't access the ImmutableListNode directly):

- `ImmutableListNode.printValue():` Print value of the current node.
- `ImmutableListNode.getNext():` Return the next node.
The input is only given to initialize the linked list internally. You must solve this problem without modifying the linked list. In other words, you must operate the linked list using only the mentioned APIs.

## Follow up:

Could you solve this problem in:

- Constant space complexity?
- Linear time complexity and less than linear space complexity?
 

## Example 1:
```
Input: head = [1,2,3,4]
Output: [4,3,2,1]
```
## Example 2:
```
Input: head = [0,-4,-1,3,-5]
Output: [-5,3,-1,-4,0]
```
## Example 3:
```
Input: head = [-2,0,6,4,4,-6]
Output: [-6,4,4,6,0,-2]
```

## Constraints:

- The length of the linked list is between `[1, 1000]`.
- The value of each node in the linked list is between `[-1000, 1000]`.

**Questions to ask**

**Test Cases to consider**
- odd and even length lists

**Hint:**
* 

[LeetCode link](https://leetcode.com/problems/print-immutable-linked-list-in-reverse/)

<details>
<summary>Click here to see code</summary>

## Approach 1: O(1) time complexity and O(n) space complexity
```python
    def printLinkedListInReverse(self, head: 'ImmutableListNode') -> None:
        if not head or head.getNext() is None:
            return
        
        def _helper(node):
            if not node:
                return
            nextNode = node.getNext()
            _helper(nextNode)
            node.printValue()
        
        _helper(head)
```

## Approach 2: o(sqrt(n)) space complexity and o(n) time complexity
```python
    def printLinkedListInReverse(self, head: 'ImmutableListNode') -> None:
        if not head:
            return
        
        def getSize(node):
            t, size = node, 0
            while t:
                t = t.getNext()
                size += 1

            return size

        def populateSubHeads(node, listSize):
            subHeadSize = int(math.sqrt(listSize))
            subHeads, count = [], 0
            t = node
            
            while t:
                if count % subHeadSize == 0:
                    subHeads.append(t)
                t = t.getNext()
                count += 1
            
            return subHeads, subHeadSize
            
            
        def reverseList(subHeads, subHeadSize):
            stack = []
            while subHeads:
                subHead = subHeads.pop()
                i = 0
                while subHead and i < subHeadSize:
                    stack.append(subHead)
                    subHead = subHead.getNext()
                    i += 1
                
                while stack:
                    stack.pop().printValue()
            
        
        listSize = getSize(head)
        subHeads, subHeadSize = populateSubHeads(head, listSize)
        return reverseList(subHeads, subHeadSize)
```

## Approach 3: O(1) space complexity and O(n^2) linear complexity
```python
    def printLinkedListInReverse(self, head: 'ImmutableListNode') -> None:
        if not head:
            return
        
        def getSize(node):
            t, size = node, 0
            while t:
                t = t.getNext()
                size += 1

            return size

        def reverseList(node, size):
            for depth in range(1, size + 1):
                t, count = node, size - depth
                while count > 0:
                    t = t.getNext()
                    count -= 1
                t.printValue()
        
        listSize = getSize(head)
        return reverseList(head, listSize)
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