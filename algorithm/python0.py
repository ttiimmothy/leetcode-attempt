from typing import List, Optional
from collections import deque, defaultdict, OrderedDict

class TreeNode:
  def __init__(self, val=0, left=None, right=None):
    self.val = val
    self.left = left
    self.right = right

# 3
# Longest Substring Without Repeating Characters
def lengthOfLongestSubstring(s: str) -> int:
  longest = 0
  l = 0
  charSet = set()
  for r in range(len(s)):
    while s[r] in charSet:
      charSet.remove(s[l])
      l += 1
    charSet.add(s[r])
    longest = max(longest, r - l + 1)
  return longest

# 11
# Container With Most Water
def maxArea(height: List[int]) -> int:
  result, left, right = 0, 0, len(height) - 1
  while left < right:
    result = max(result, min(height[left], height[right]) * (right - left))
    if height[left] < height[right]:
      left += 1
    else:
      right -= 1
  return result

# 15
# 3Sum
def threeSum(nums: List[int]) -> List[List[int]]:
  nums.sort()
  result = []
  for i in range(0, len(nums) - 2):
    if i > 0 and nums[i] == nums[i - 1]:
      continue
    mid, high = i + 1, len(nums) - 1
    while mid < high:
      total = nums[i] + nums[mid] + nums[high]
      if total < 0:
        mid += 1
      elif total > 0:
        high -= 1
      else:
        result.append([nums[i], nums[mid], nums[high]])
        mid += 1
        high -= 1
      while mid < high and nums[mid] == nums[mid - 1]:
        mid += 1
  return result

# 39
# Combination Sum
def combinationSum(candidates: List[int], target: int) -> List[List[int]]:
  result = []
  def dfs(candidates: List[int], target: int, subArray: List[int], result: List[List[int]]):
    if target < 0:  # target is negative
      return  # back tracking
    if target == 0:
      result.append(subArray)
      return
    for i in range(len(candidates)):
      if candidates[i] <= target:
        # candidates need to be shorten after finish one iteration
        dfs(candidates[i:], target - candidates[i], subArray + [candidates[i]], result)
  # subArray initially is an empty array
  dfs(candidates, target, [], result)
  return result

# 134
# Gas Station
def canCompleteCircuit(gas: List[int], cost: List[int]) -> int:
  possibleOutcome = 0  # assume the starting point is the result
  gasTank = 0
  totalTank = 0
  for i in range(len(gas)):
    totalTank += gas[i] - cost[i]
    gasTank += gas[i] - cost[i]
    if gasTank < 0:
      possibleOutcome = i + 1
      gasTank = 0
  if totalTank >= 0:
    return possibleOutcome
  else:
    return -1

# 102
# Binary Tree Level Order Traversal
def levelOrder(root: Optional[TreeNode]):
  if root == None:
    return None
  result = []
  queue = [root]
  while len(queue) > 0:
    size = len(queue)
    temp = []
    # size will not change after the queue.append
    for _ in range(size):
      node = queue.pop(0)
      temp.append(node.val)
      if node.left:
        queue.append(node.left)
      if node.right:
        queue.append(node.right)
    result.append(temp)
  return result

# 102
# Binary Tree Level Order Traversal
def levelOrder_1(root: Optional[TreeNode]):
  result = []
  def dfs(node, level):
    if not node:
      return None
    nonlocal result
    if level == len(result):
      result.append([])
    result[level].append(node.val)
    dfs(node.left, level + 1)
    dfs(node.right, level + 1)
  dfs(root, 0)
  return result

# 200
# Number of Islands
def numIslands(grid: List[List[str]]) -> int:
  q = deque()
  count = 0
  row, col = len(grid), len(grid[0])
  def coordinates(r,c):
    return [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]
  for i in range(len(grid)):
    for j in range(len(grid[i])):
      if grid[i][j] == "1":
        grid[i][j] = "#"
        q.append((i, j))
        count += 1
        while q:
          x, y = q.popleft()
          for r,c in coordinates(x,y):
            if 0 <= r < row and 0 <= c < col and grid[r][c] == "1":
              grid[r][c] = "#"
              q.append((r, c))
  return count

# 226
# Invert Binary Tree
def invertTree(root: Optional[TreeNode]) -> Optional[TreeNode]:
  # base case and edge case
  # will keep going when it breaks one recursive function after it touches the leaf node (line 14, 15)
  if root == None:
    return None
  root.left, root.right = root.right, root.left
  invertTree(root.right)
  invertTree(root.left)
  return root

# 236
# Lowest Common Ancestor of a Binary Tree
def lowestCommonAncestor(root, p, q):
  if not root:
    return None
  if root == p or root == q:
    return root
  l = lowestCommonAncestor(root.left, p, q)
  r = lowestCommonAncestor(root.right, p, q)
  if l and r:
    return root
  return l or r

# 460
# LFU Cache
class LFUCache:
  def __init__(self, capacity: int):
    self.capacity = capacity
    self.items = defaultdict(int)
    self.freqs = defaultdict(OrderedDict)
    self.min_freq = 0 

  def update_freq(self, key, value = None):
    f = self.items[key]
    v = self.freqs[f].pop(key)
    v = value if value else v
    self.freqs[f+1][key] = v
    self.items[key] += 1
    if self.min_freq == f and not self.freqs[f]:
      self.min_freq += 1
    return v

  def get(self, key: int) -> int:
    if key in self.items:
      return self.update_freq(key)
    return -1
      
  def put(self, key: int, value: int) -> None:
    if key in self.items:
      self.update_freq(key, value)
    else:
      if len(self.items) == self.capacity:
        self.items.pop(self.freqs[self.min_freq].popitem(last=False)[0])
      self.min_freq = 1
      self.items[key] = 1
      self.freqs[1][key] = value

# 543
# Diameter of Binary Tree
def diameterOfBinaryTree(root: Optional[TreeNode]) -> int:
  diameter = 0
  # return height
  def dfs(root):
    if not root:
      return 0
    left = dfs(root.left)
    right = dfs(root.right)
    nonlocal diameter
    diameter = max(diameter, left + right)
    return max(left, right) + 1
  dfs(root)
  return diameter

# 844
# Backspace String Compare, not the optimized solution
def backspaceCompare(s: str, t: str) -> bool:
  def checking(str: str):
    stack = []
    for i in str:
      if i != "#":
        stack.append(i)
      elif len(stack) > 0:
        stack.pop()
    return stack
  return checking(s) == checking(t)

# 844-1
# Backspace String Compare
def backspaceCompare_1(s: str, t: str) -> bool:
    pS, pT = len(s) - 1, len(t) - 1
    def findNextValidChar(string: str, end: int):
      backspaceCount = 0
      while end >= 0:
        if string[end] == '#':
          backspaceCount += 1
        elif backspaceCount > 0:
          backspaceCount -= 1
        else:
          break
        end -= 1
      return end
    while pS >= 0 or pT >= 0:
      pS = findNextValidChar(s, pS)
      pT = findNextValidChar(t, pT)
      if pS < 0 and pT < 0:
        return True
      elif pS < 0 or pT < 0:
        return False
      elif s[pS] != t[pT]:
        return False
      pS -= 1
      pT -= 1
    return True

# 912
# Sort an Array
def sortArray(nums: List[int]) -> List[int]:
  def heapSort(nums: List[int]):
    n = len(nums)
    def heapify(nums: List[int], root: int, length: int):
      largest = root
      left = 2 * root + 1
      right = 2 * root + 2
      if left < length and nums[left] > nums[largest]:
        largest = left
      if right < length and nums[right] > nums[largest]:
        largest = right
      if largest != root:
        nums[largest], nums[root] = nums[root], nums[largest]
        heapify(nums, largest, length)
    for i in range(n // 2 - 1, -1, -1):
      heapify(nums, i, n)
    for i in range(n - 1, 0, -1):
      nums[0], nums[i] = nums[i], nums[0]
      heapify(nums, 0, i)
  heapSort(nums)
  return nums

# Quick sort, time complexity O(nlogn), memory complexity O(1), may cause time limit error
def quick_sort(nums, low, high):
  if low >= high:
    return
  def partition(nums, low, high):
    start = low - 1
    pivot = nums[high]
    for i in range(low, high - 1):
      if (nums[i] < pivot):
        start += 1
      nums[i], nums[start] = nums[start], nums[i]
    nums[start + 1], nums[high] = nums[high], nums[start + 1]
    return start + 1
  pivot = partition(nums, low, high)
  quick_sort(nums, low, pivot - 1)
  quick_sort(nums, pivot + 1, high)

# Merge sort, time complexity O(nlogn), memory complexity O(n)
def merge_sort(nums, low, high):
  if low < high:
    mid = low + (high - low) // 2
    merge_sort(nums, low, mid)
    merge_sort(nums, mid + 1, high)
    L = nums[low:mid + 1]
    R = nums[mid + 1:high + 1]
    i = j = 0
    k = low
    while i < len(L) and j < len(R):
      if (L[i] <= R[j]):
        nums[k] = L[i]
        i += 1
      else:
        nums[k] = R[j]
        j += 1
      k += 1
    while i < len(L):
      nums[k] = L[i]
      i += 1
      k += 1
    while j < len(R):
      nums[k] = R[j]
      j += 1
      k += 1

# Bubble sort, time complexity O(n^2), memory complexity O(1)
def bubble_sort(nums):
  n = len(nums)
  for i in range(n - 1):
    swapped = False
    for j in range(n - i - 1):
      if nums[j] > nums[j + 1]:
        nums[j], nums[j + 1] = nums[j + 1], nums[j]
        swapped = True
    if swapped == False:
      break

# Heap sort
def heap_sort(nums: List[int]):
  n = len(nums)
  def heapify(nums: List[int], node: int, length: int):
    largest = node
    left = 2 * node + 1
    right = 2 * node + 2
    if left < length and nums[left] > nums[largest]:
      largest = left
    if right < length and nums[right] > nums[largest]:
      largest = right
    if largest != node:
      nums[largest], nums[node] = nums[node], nums[largest]
      heapify(nums, largest, length)
  for i in range(n // 2 - 1, -1, -1):
    heapify(nums, i, n)
  for i in range(n - 1, 0, -1):
    nums[0], nums[i] = nums[i], nums[0]
    heapify(nums, 0, i)

numbers = [1, 9, 8, 20, 15, 17, 5, 4, 8, 3]
heap_sort(numbers)
print(numbers)
