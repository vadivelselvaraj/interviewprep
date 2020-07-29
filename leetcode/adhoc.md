## 1. Reverse an integer

Given a 32-bit signed integer, reverse digits of an integer.

**Note:** Assume we are dealing with an environment which could only store integers within the 32-bit signed integer range: [−231,  231 − 1]. For the purpose of this problem, assume that your function returns 0 when the reversed integer overflows.

**Hint:**
* pop and push digits logic like `reversedNumber = reversedNumber * 10 + pop` can cause an overflow. So, we need to catch it before the overflow happens.
* If `reversedNumber = reversedNumber * 10 + pop` can cause an overflow, then `reversedNumber >= INTMAX/10`. So, the below are true.
  * If `reversedNumber > INTMAX/10`, then `reversedNumber = reversedNumber * 10 + pop` will cause an overflow.
  * If `reversedNumber == INTMAX/10`, then `reversedNumber = reversedNumber * 10 + pop` will cause an overflow iff `pop > 7`(rightmost digit of 2^31).

[LeetCode link](https://leetcode.com/problems/reverse-integer/)

<details>
<summary>Click here to see code</summary>

```python
def reverse(self, x: int) -> int:
        if x >=-9 and x <= 9:
            return x
        
        negativeSign = True if x < 0 else False
        
        signedX = -1 * x if x < 0 else x
        result, t = 0, signedX
        divider = 10
        
        leftLimit, rightLimit = (2**31), (2**31)-1
        
        while t:
            digitToAdd = t%10
            if (
                not negativeSign
                and ( ( result == (rightLimit // 10) and digitToAdd > 7 ) or result > (rightLimit // 10) )
            ) or (
                negativeSign
                and ( ( result == (leftLimit // 10) and digitToAdd > 8 ) or (result > leftLimit // 10) )
            ):
                return 0

            result = (result * divider) + digitToAdd
            t //= 10
        
        return -1 * result if negativeSign else result
```

</details>
