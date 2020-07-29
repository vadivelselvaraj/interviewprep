# Strings
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