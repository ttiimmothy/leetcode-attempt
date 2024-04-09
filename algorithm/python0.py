from typing import List

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
