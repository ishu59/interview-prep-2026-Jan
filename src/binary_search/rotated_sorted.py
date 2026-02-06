'''
n this version of binary search you are trying to partition the array with the condition function. For every value of the array, you will get a True/False value for a given condition function. Moreover, you should have a monotonic function, where once you get the value to be true for a given index, you will get a true value for all indices greater than that index.

In other words, if you ran just your condition function on all values of the array (forget about the binary search part), you results array of just running the condition function, would look like this:
[F, F, F, F, T, T, T, T, T, T]
Notice, how it is monotonic. Once it becomes true, it will always be true. What you are trying to find is where that first true is.

Let's look at an example:
In the problem Find Minimum in Rotated Sorted Array. The first step is to find the rotation point. (i.e. the smallest value of the array). Suppose you are given the array,

arr = [4, 5, 6, 7, 0, 1, 2]
condition fn applied to each element in arr = [F, F, F, F, T, T, T]

Easier to see like this:
[4, 5, 6, 7, 0, 1, 2]
[F, F, F, F, T, T, T]

Our binary search will lead us to converge to the index where the 0 is stored (as explained in the above post, i.e. binary search will lead us to the minimum index where the condition function is True. Here it is the index that stores the 0 value.).

How do we write this condition?
We want to move the right index in the left direction, when the mid value is less than or equal to the current right index value

Therefore in your binary search template you should have something like this (using Python-ish syntax):
if (arr[mid] <= arr[right]): //Comment: the condition in the parenthesis is your condition function.
//You can make this a separate condition function if you would like it to be.
right = mid
else:
left = mid + 1
(I haven't posted code in Leetcode before, so I apologize for not figuring out how to put this in proper code cells)

I hope this helps!
'''