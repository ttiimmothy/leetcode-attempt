from typing import List, Optional
from collections import deque, defaultdict, OrderedDict

class ListNode:
  def __init__(self, val=0, next=None):
    self.val = val
    self.next = next

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

# 21
# Merge Two Sorted Lists
def mergeTwoLists(list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
  result = ListNode()
  current = result
  while list1 and list2:
    if list1.val < list2.val:
      current.next = list1
      list1 = list1.next
    else:
      current.next = list2
      list2 = list2.next
    current = current.next
  current.next = list1 if list1 else list2
  return result.next

# 33
# Search in Rotated Sorted Array
def search(nums: List[int], target: int) -> int:
  l, r = 0, len(nums) - 1
  while l <= r:
    m = l + (r - l) // 2
    if nums[m] == target:
      return m
    if nums[l] <= nums[m]:
      if nums[l] <= target < nums[m]:
        r = m - 1
      else:
        l = m + 1
    else:
      if nums[m] < target <= nums[r]:
        l = m + 1
      else:
        r = m - 1
  return -1

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

# 92
# Reversed Linked List II
def reverseBetween(head, left, right):
  result = ListNode(0, head)
  current, leftPrev = head, result
  for _ in range(left - 1):
    leftPrev = current
    current = current.next
  prev = None
  for _ in range(right - left + 1):
    next = current.next
    current.next = prev
    prev = current
    current = next
  lef_Prev.next.next = current # at right + 1 posiiton
  leftPrev.next = prev # at right position
  return result.next

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

# 102-1
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

# 105
# Construct Binary Tree from Preorder and Inorder Traversal
def buildTree(preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
  if not preorder or not inorder:
    return None
  root = TreeNode(preorder[0])
  m = inorder.index(preorder[0])
  root.left = buildTree(preorder[1:m + 1], inorder[:m])
  root.right = buildTree(preorder[m + 1:], inorder[m + 1:])
  return root

# 105-1
# Construct Binary Tree from Preorder and Inorder Traversal
def buildTree1(preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
  imap = {val:i for i, val in enumerate(inorder)}
  # current put outside the recursive so it will not decrease unintentionally
  # current initialize in the build function, current will decrease unintentionally
  current = 0
  def build(l, r):
    if l > r:
      return None
    nonlocal current
    root = TreeNode(preorder[current])
    m = imap[preorder[current]]
    current += 1
    root.left = build(l, m - 1)
    root.right = build(m + 1, r)
    return root
  return build(0, len(inorder) - 1)

# 110
# Balanced Binary Tree
def isBalanced(root: Optional[TreeNode]) -> bool:
  def dfs(root):
    if not root:
      return [True, 0]
    left = dfs(root.left)
    right = dfs(root.right)
    balanced = left[0] and right[0] and abs(left[1] - right[1]) <= 1
    return [balanced, max(left[1], right[1]) + 1]
  return dfs(root)[0]

# 199
# Binary Tree Right Side View
def rightSideView(root: Optional[TreeNode]) -> List[int]:
  if root == None:
    return []
  q = [root]
  result = []
  while q:
    size = len(q)
    result.append(q[len(q) - 1].val)
    for _ in range(size):
      node = q.pop(0)
      if node.left:
        q.append(node.left)
      if node.right:
        q.append(node.right)
  return result

# 199-1
# Binary Tree Right Side View
def rightSideView_1(root: Optional[TreeNode]) -> List[int]:
  result = []
  def dfs(root, level):
    if root == None:
      return None
    nonlocal result
    if level == len(result):
      result.append(root.val)
    # need to be right first and need left because we are doing right side view
    dfs(root.right, level + 1)
    dfs(root.left, level + 1)
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

# 235
# Lowest Common Ancestor of a Binary Search Tree
def lowestCommonAncestor(root, p, q):
  while root:
    if p.val > root.val and q.val > root.val:
      root = root.right
    elif p.val < root.val and q.val < root.val:
      root = root.left
    else:
      return root

# 236
# Lowest Common Ancestor of a Binary Tree
def lowestCommonAncestor_1(root, p, q):
  if not root:
    return None
  if root == p or root == q:
    return root
  l = lowestCommonAncestor_1(root.left, p, q)
  r = lowestCommonAncestor_1(root.right, p, q)
  if l and r:
    return root
  return l or r

# 278
# First Bad Version
def firstBadVersion(n: int) -> int:
  l, r = 1, n
  while l < r:
    m = l + (r - l) // 2
    if not isBadVersion(m):
      l = m + 1
    else:
      r = m
  return l

def isBadVersion(version: int) -> bool:
  if version > 0: return True
  return False

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

# 542
# 01 Matrix
def updateMatrix(mat: List[List[int]]) -> List[List[int]]:
  m = len(mat)
  n = len(mat[0])
  max = m * n
  q = deque()
  for i in range(len(mat)):
    for j in range(len(mat[i])):
      if mat[i][j] == 0:
        q.append([i, j])
      else:
        mat[i][j] = max
  directions = [[-1,0],[1,0],[0,-1],[0,1]]
  while q:
    r, c = q.popleft()
    for i, j in directions:
      dr = r + i
      dc = c + j
      if 0 <= dr < len(mat) and 0 <= dc < len(mat[0]) and mat[dr][dc] > mat[r][c] + 1:
        q.append([dr, dc])
        mat[dr][dc] = mat[r][c] + 1
  return mat

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

# 680
# Valid Palindrome II
def validPalindrome(s: str) -> bool:
  l, r = 0, len(s) - 1
  while l < r:
    if s[l] != s[r]:
      string1 = s[l:r]
      string2 = s[l + 1:r + 1]
      return string1 == string1[::-1] or string2 == string2[::-1]
    l += 1
    r -= 1
  return True

# 680-1
# Valid Palindrome II
def validPalindrome_1(s: str) -> bool:
  l, r = 0, len(s) - 1
  def verify(s, l, r):
    while l < r:
      if s[l] != s[r]:
        return False
      l += 1
      r -= 1
    return True
  while l < r:
    if s[l] != s[r]:
      return verify(s, l + 1, r) or verify(s, l, r - 1)
    l += 1
    r -= 1
  return True

# 704
# Binary Search
def search_1(nums: List[int], target: int) -> int:
  l, r = 0, len(nums) - 1
  while l <= r:
    # lead to overflow
    # m = (l + r) // 2
    m = l + (r - l) // 2
    if target == nums[m]:
      return m
    elif target < nums[m]:
      r = m - 1
    else:
      l = m + 1
  return -1

# 733
# Flood Fill
def floodFill(image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
  if image[sr][sc] == color: return image
  def dfs(image, sr, sc, color, reference):
    if sr < 0 or sr > len(image) - 1 or sc < 0 or sc > len(image[0]) - 1:
      return
    if image[sr][sc] == reference:
      image[sr][sc] = color
      dfs(image, sr - 1, sc, color, reference)
      dfs(image, sr + 1, sc, color, reference)
      dfs(image, sr, sc - 1, color, reference)
      dfs(image, sr, sc + 1, color, reference)
  dfs(image, sr, sc, color, image[sr][sc])
  return image

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

# 981
# Time Based Key-Value Store
class TimeMap:
  def __init__(self):
    self.map = {}

  def set(self, key: str, value: str, timestamp: int) -> None:
    if key not in self.map:
      self.map[key] = []
    self.map[key].append([value, timestamp])

  def get(self, key: str, timestamp: int) -> str:
    result = ""
    if key in self.map:
      pair = self.map[key]
      l, r = 0, len(pair) - 1
      while l <= r:
        m = l + (r - l) // 2
        if pair[m][1] == timestamp:
          return pair[m][0]
        if pair[m][1] < timestamp:
          result = pair[m][0]
          l = m + 1
        else:
          r = m - 1
    return result

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

# Quick sort
def quickSort(nums, start, end):
  if start < end:
    pivot = partition(nums, start, end)
    quickSort(nums, start, pivot - 1)
    quickSort(nums, pivot + 1, end)

def partition(nums, start, end):
  pivot = nums[end]
  j = start - 1
  for i in range(start, end):
    if nums[i] < pivot:
      j += 1
      nums[i], nums[j] = nums[j], nums[i]
  nums[j + 1], nums[end] = nums[end], nums[j + 1]
  return j + 1

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

# merge sort
def mergeSort(nums:List[int], start: int, end:int):
  if start < end:
    mid = start + (end - start) // 2
    mergeSort(nums, start, mid)
    mergeSort(nums, mid + 1, end)
    l = nums[start:mid + 1]
    r = nums[mid + 1:end + 1]
    i = 0
    j = 0
    k = start
    while i < len(l) and j < len(r):
      if l[i] < r[j]:
        nums[k] = l[i]
        i += 1
        k += 1
      else:
        nums[k] = r[j]
        j += 1
        k += 1
    while i < len(l):
      nums[k] = l[i]
      i += 1
      k += 1
    while j < len(r):
      nums[k] = r[j]
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
