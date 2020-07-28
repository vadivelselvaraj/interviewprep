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