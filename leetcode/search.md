## 1.) Word Search

Given a 2D board and a word, find if the word exists in the grid.

The word can be constructed from letters of sequentially adjacent cell, where "adjacent" cells are those horizontally or vertically neighboring. The same letter cell may not be used more than once.

### Example:
```
board =
[
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]

Given word = "ABCCED", return true.
Given word = "SEE", return true.
Given word = "ABCB", return false.
```

### Constraints:

- board and word consists only of lowercase and uppercase English - letters.
- 1 <= board.length <= 200
- 1 <= board[i].length <= 200
- 1 <= word.length <= 10^3

**Questions to ask**

**Test Cases to consider**

**Hint:**
* Use a backtracking strategy.

[LeetCode link](https://leetcode.com/problems/word-search/)

<details>
<summary>Click here to see code</summary>

Time Complexity: `O(N*4^L)`, where `N` = # of cells in the board and `L` = length of the word. When searching for a character, we've 4 choices o
```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        if not board:
            return False
    
        rowSize, colSize = len(board), len(board[0])
        visited = set()
        
        def backtrack(row, col, suffix):
            nonlocal rowSize, colSize, visited
            
            if len(suffix) == 0:
                return True
            
            if 0 <= row < rowSize and 0 <= col < colSize \
                and (row, col) not in visited \
                and suffix[0] == board[row][col]:

                visited.add((row,col))
                for newX, newY in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
                    if backtrack(row + newX, col + newY, suffix[1:]):
                        return True
                visited.remove((row, col))

            return False
        
        for i in range(rowSize):
            for j in range(colSize):
                if board[i][j] == word[0] and backtrack(i, j, word):
                    return True
        return False
```

</details>

## 2.) Word Search II

Given a 2D board and a list of words from the dictionary, find all words in the board.

Each word must be constructed from letters of sequentially adjacent cell, where "adjacent" cells are those horizontally or vertically neighboring. The same letter cell may not be used more than once in a word.

### Example:
```
Input: 
board = [
  ['o','a','a','n'],
  ['e','t','a','e'],
  ['i','h','k','r'],
  ['i','f','l','v']
]
words = ["oath","pea","eat","rain"]

Output: ["eat","oath"]
```

### Note:
- All inputs are consist of lowercase letters a-z.
- The values of words are distinct.

**Questions to ask**
- Can the same letter cell be used more than once in a word?
- Will one word be present only once?

**Test Cases to consider**
- matrix filled with `a's` and we search for `["aaa"]`.

**Hint:**
* Use backtracking with trie. Use trie to check if the words traversed so far do form a word.

[LeetCode link](https://leetcode.com/problems/word-search-ii/)

<details>
<summary>Click here to see code</summary>

```python
class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        if not board:
            return False
    
        rowSize, colSize = len(board), len(board[0])
        trie, matchedWords = defaultdict(dict), []
        
        # Insert words into trie
        for word in words:
            t = trie
            for letter in word:
                t = t.setdefault(letter, {})
            t['$'] = word

        visited = set()
        
        def backtrack(row, col, parentTrieNode):
            nonlocal rowSize, colSize, visited, matchedWords
            
            letter = board[row][col]
            currentNode = parentTrieNode[letter]
            if '$' in currentNode:
                # Found a word
                matchedWords.append(currentNode['$'])
                currentNode.pop('$')

            visited.add((row, col))
            for (newX, newY) in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
                newRow, newCol = row + newX, col + newY
                if newRow < 0 or newRow >= rowSize \
                    or newCol < 0 or newCol >= colSize \
                    or (newRow, newCol) in visited \
                    or board[newRow][newCol] not in currentNode:
                    continue
                backtrack(newRow, newCol, currentNode)
            
            # Remove from visited for backtracking
            visited.remove((row, col))
            
            # Remove word from trie, if there are 
            if not currentNode:
                parentTrieNode.pop(letter)

        
        for i in range(rowSize):
            for j in range(colSize):
                if board[i][j] in trie:
                     backtrack(i, j, trie)
        
        return matchedWords
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