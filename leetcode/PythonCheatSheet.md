# Constants

1. **positive infinity** math.inf or Decimal('Infinity')
1. **negative infinity** -math.inf or Decimal('-Infinity')

# Operators

- **Integer division:** 23 // 2 = 11
- **Float division:** 23 / 2 = 11.5

# Strings
- Iterate over string

```python
for ch in string:
    print(ch, end=' ')

for i, ch in enumerate(string):
    print(f"index: {i}, char: {ch}")
```

- Reverse iterate a string

```python
for ch in string[::-1]:
    print(ch)
```

    (OR)

```python
for ch in reversed( range( 0, len(string) ) ):
    print(ch)
```

- Check if a character is present in a string

```python
'ai' in "aeiouAEIOU"
```

- Replace

```python
"aeioa".replace("a", "d")
```

- Check for alphabets, digits

```python
"0".isdigit()
"abc".isalpha()
```

- Concatenate strings
```python
>>> p+="2"
>>> p
'1->2'
```

- Split a string into list
```python
>>> "abc 23 45 65".split(" ")
['abc', '23', '45', '65']
>>> "abc 23 45 65".split(" ", 1)
['abc', '23 45 65']
```

# List
- Iterate

```python
```
- Add first
```python
```
- Add last
```python
```
- Delete first
```python
```
- Delete last
```python
```
- Merge lists
```python
>>> [1,2] + [3] + [4,5]
[1, 2, 3, 4, 5]
```
- sort lists with tuples
```python
>>> a = [(2, 1), (0, 3), (0, 2)]
>>> sorted(a)
[(0, 2), (0, 3), (2, 1)]
```
__Note:__ To do in-place sort, call `list.sort()`, otherwise use `sorted(list)`
- Reverse a list
```python
>>> for i in reversed(['hai', 'bye', 'cde']):
...     print(i)
... 
cde
bye
hai
```
- Check if a list is None or has zero elements in it
```python
if not aList:
    ...
```
- Find index of an element in a list
```python
>>> a = [1,2,3,3,4,5,6]
>>> a.index(3)
2
>>> a.index(4)
4
```
# Dictionary

- Iterate keys and values
```python
for k, v in enumerate(dictObject):
    print(k, v)
```
- Delete a key
```python
del dictObject[key_to_delete]
```
- Initialize dict with list, set
```python
nodeMap = defaultdict(list)
nodeMap[1].append(23)
```

# Set
- Add elements to set
```python
a = set()
a.add(2)
```
- Find if element exists in a set
```python
if 2 in a:
    print('hai')
```

# OrderedDict:
`OrderedDict` is a linked list mashed up with hash map(equivalent to `LinkedHashMap` in Java). Offers `O(1)` insert, delete and update time complexities.

# Heap
- Converting an array to a heap
```python
from heapq import heappop, heappush, heapify
l = [3,5,9,1,2,10]
heapify(l)
```
- See the max

Leetcode problems on heap [here](https://leetcode.com/tag/heap/).

# bisect(Java's TreeMap alternative)
- insort(Add elements in sorted order)
```python
>>> a = [30,40,50,60,70]
>>> bisect.insort(a, 33)
>>> a
[30, 33, 40, 50, 60, 70]
```
- 
# Custom Sort
- Sort the array based on the second element in the tuple
```python
def sortSecond(val): 
    return val[1]  
  
# list1 to demonstrate the use of sorting  
# using using second key  
list1 = [(1, 2), (3, 3), (1, 1)] 
  
# sorts the array in ascending according to  
# second element 
list1.sort(key = sortSecond)
```
- Sort multiple attributes
```python
a = [('a', 15),('d', 34),('a', 10),('d', 23),('e', 2)]
>>> a.sort()
[('a', 10), ('a', 15), ('d', 23), ('d', 34), ('e', 2)]
>>> sorted(a, key=lambda x: (x[1], x[0]))
[('e', 2), ('a', 10), ('a', 15), ('d', 23), ('d', 34)]
```
- Refer `Reorder data log files` problem [here](./arrays.md) for an advanced custom sort.

# Errors
- __TypeError:__ Raised when a function or operation is applied to an object of an incorrect type.
- __ValueError:__ Raised when a function gets an argument of correct type but improper value.

# try, except, else and finally
```python
try:
    #statements in try block
except:
    #executed when error in try block
else:
    #executed if try block is error-free
finally:
    #executed irrespective of exception occured or not
```

# Magic Methods
- **__init__(self, other):** To get called by the __new__ method.
- **__del__(self):** Destructor method.
- **__str__(self):** To get called by built-int str() method to return a string representation of a type.
- **__repr__(self):** To get called by built-int repr() method to return a machine readable representation of a type.
- **__format__(self, formatstr):** To get called by built-int string.format() method to return a new style of string.
- **__hash__(self):** To get called by built-int hash() method to return an integer.
- **__sizeof__(self):** To get called by built-int sys.getsizeof() method to return the size of an object.
- **__lt__(self, other):** To get called on comparison using < operator.
- **__le__(self, other):** To get called on comparison using <= operator.
- **__eq__(self, other):** To get called on comparison using == operator.
- **__ne__(self, other):** To get called on comparison using != operator.
- **__ge__(self, other):** To get called on comparison using >= operator.

Others [here](https://www.tutorialsteacher.com/python/magic-methods-in-python).


# Tricks
- To find the minimum of two values but based on a function, use the below.
```python
min(pred, root.val, key = lambda x: abs(target - x))
```
- Swap two elements in a list or vars without the use of a temporary var
```python
a[i], a[j] = a[j], a[i]
a, b = b, a
```
- Count the number of matches between an item and a group of items
```python
>>> (1, 2).count(3)
0
>>> ('ab', 'b').count('ab')
1
>>> ['ab', 'b', 'abc', 'ab'].count('ab')
2
```
- Adding boolean values
```python
>>> False + True + True
2
```
- Get the frequency of chars in a string or items in a list
```python
>>> Counter("abbxxxxxxxyyy")
Counter({'x': 7, 'y': 3, 'b': 2, 'a': 1})
>>> Counter([1,1,2,3,2,4,5,6])
Counter({1: 2, 2: 2, 3: 1, 4: 1, 5: 1, 6: 1})
```
- Iterate Counter object
```python
>>> for k, v in Counter("abc").items():
        print(f"{k}:{v}")
```
_Note:_ Don't use enumerate(Counter("abc")) as it treats it as an array and produces k, v as (0, a).
- Get element with max frequency
```
def majorityElement(self, nums):
    counts = collections.Counter(nums)
    return max(counts.keys(), key=counts.get)
```