# Trees

<img src='https://g.gravizo.com/svg?
digraph G {
    nodesep=0.4;
    ranksep=0.5;
    15 [label= "15;depth=0;height=4", color = blue ];
    10 [label= "10;depth=1;height=2" ];
    25 [label= "25;depth=1;height=3" ];
    8 [label= "8;depth=2;height=0", color=orange ];
    12 [label= "12;depth=2;height=1" ];
    null0 [label="null", shape=point];
    18 [label= "18;depth=2;height=2" ];
    null1 [label="null", shape=point];
    27 [label= "27;depth=2;height=0", color=orange ];
    13 [label= "13;depth=3;height=0", color=orange ];
    20 [label= "20;depth=3;height=1" ];
    19 [label= "19;depth=4;height=0", color=orange ];   
    null2 [label="null", shape=point]; 
    15 -> { 10, 25 };
    10 -> { 8, 12 };
    25 -> { 18, 27 };
    12 -> { null0, 13 };
    18 -> { null1, 20 };
    20 -> { 19, null2 };
}
'/>
### Depth
**Depth of a node** is the number of edges from the root node to the node.
**Depth of a tree** is depth of the deepest node.


### Height
**Height of a node** is the number of edges on the longest path from the node to a leaf.
**Height of a tree** is the height of root node.

### Diameter or Width
Number of nodes on the longest path between two leaves in the tree, including the nodes.
Diameter of the example tree above is 8 i.e. between nodes 13 and 19 there are 6 nodes, plus 2 including them.

### Height-Balanced Binary Tree
A binary tree in which the left and right subtrees of every node differ in height by no more than 1.

### Complete binary tree
In a complete binary tree every level, except possibly the last, is completely filled, and all nodes in the last level are as far left as possible. It can have between 1 and 2h nodes inclusive at the last level h.

<img src=https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Complete_binary2.svg/220px-Complete_binary2.svg.png />


### Full binary tree
A full binary tree (sometimes referred to as a proper or plane binary tree) is a tree in which every node has either 0 or 2 children.


<img src=https://upload.wikimedia.org/wikipedia/commons/thumb/b/b0/Full_binary.svg/220px-Full_binary.svg.png />

### Perfect Binary Tree
A perfect binary tree is a binary tree in which all interior nodes have two children and all leaves have the same depth or same level.


## Tree construction from traversals
A unique tree can be constructed from inorder and { preorder, postorder } traversals but not from just preorder and postorder traversals.

For instance, considered the left and right skewed trees below where the postorder and preorder for these two trees are the same.

<img src='https://image.slidesharecdn.com/trees-110611091544-phpapp01/95/trees-5-728.jpg?cb=1307783910' />

__preorder__:_[A,B,C]_
__postorder__:_[C,B,A]_


However, a unique full binary tree can be constructed from a preorder and postorder traversals.


# Arrays

## Substrings
There are `n*(n+1)/2)` substrings in a string of length, `n`. 
```python
def subArray(arr, n): 
  
    # Pick starting point 
    for i in range(0,n): 
  
        # Pick ending point 
        for j in range(i,n): 
  
            # Print subarray between 
            # current starting 
            # and ending points 
            for k in range(i,j+1): 
                print (arr[k],end=" ") 
  
            print ("\n",end="") 
```
## Subsequences
There are `2^n` subsequences for a string of length, `n` including the empty `''` one.
```python
def printSubsequences(arr, n) : 
  
    # Number of subsequences is (2**n -1) 
    opsize = math.pow(2, n) 
  
    # Run from counter 000..1 to 111..1 
    for counter in range( 1, (int)(opsize)) : 
        for j in range(0, n) : 
              
            # Check if jth bit in the counter 
            # is set If set then print jth  
            # element from arr[]  
            if (counter & (1<<j)) : 
                print( arr[j], end =" ") 
          
        print()
```