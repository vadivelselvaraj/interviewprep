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

- Check for digit

```python
Character.isDigit(ch)
```
- Concatenate strings
```python
>>> p+="2"
>>> p
'1->2'
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