```python
class TrieNode:
    def __init__(self):
        self.children = dict()
        self.isEndOfWord = False

class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TrieNode()
        
    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        t = self.root
        for character in word:
            if character not in t.children:
                t.children[character] = TrieNode()
            t = t.children[character]
        
        t.isEndOfWord = True

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        t = self.root
        for character in word:
            if character not in t.children:
                return False
            t = t.children[character]
        
        return t.isEndOfWord

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        t = self.root
        for character in prefix:
            if character not in t.children:
                return False
            t = t.children[character]
        
        return True


# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)
```