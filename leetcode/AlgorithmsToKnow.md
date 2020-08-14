# Morris Inorder Traversal

# Sweep line algorithm

This would solve orthogonal line segment intersection search i.e. Given N horizontal and vertical line segments, find how much lines intersect.


# Segment vs Range vs Interval vs Binary Indexed Trees

All these data structures are used for solving different problems:

- __Segment tree__ stores intervals, and optimized for _"which of these intervals contains a given point"_ queries.
- __Interval tree__ stores intervals as well, but optimized for _"which of these intervals overlap with a given interval"_ queries. It can also be used for point queries - similar to segment tree.
- __Range tree__ stores points, and optimized for _"which points fall within a given interval"_ queries.
- __Binary indexed tree__ stores items-count per index, and optimized for _"how many items are there between index m and n"_ queries.

## Performance / Space consumption for one dimension:

- __Segment tree__ - O(n logn) preprocessing time, O(k+logn) query time, O(n logn) space
- __Interval tree__ - O(n logn) preprocessing time, O(k+logn) query time, O(n) space
- __Range tree__ - O(n logn) preprocessing time, O(k+logn) query time, O(n) space
- __Binary Indexed tree__ - O(n logn) preprocessing time, O(logn) query time, O(n) space
(k is the number of reported results).

All data structures can be dynamic, in the sense that the usage scenario includes both data changes and queries:

- __Segment tree__ - interval can be added/deleted in O(logn) time (see here)
- __Interval tree__ - interval can be added/deleted in O(logn) time
- __Range tree__ - new points can be added/deleted in O(logn) time (see here)
- __Binary Indexed tree__ - the items-count per index can be increased in O(logn) time

Higher dimensions (d>1):

- __Segment tree__ - O(n(logn)^d) preprocessing time, O(k+(logn)^d) query time, O(n(logn)^(d-1)) space
- __Interval tree__ - O(n logn) preprocessing time, O(k+(logn)^d) query time, O(n logn) space
- __Range tree__ - O(n(logn)^d) preprocessing time, O(k+(logn)^d) query time, O(n(logn)^(d-1))) space
- __Binary Indexed tree__ - O(n(logn)^d) preprocessing time, O((logn)^d) query time, O(n(logn)^d) space