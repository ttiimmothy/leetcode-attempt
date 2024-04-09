# Container With Most Water
proc maxArea(height: seq[int]): int =
  var result, left, right = 0, 0, height.len - 1
  while left < right:
    result = max(result, min(height[left], height[right]) * (right - left))
    if height[left] < height[right]:
      inc(left)
    else:
      dec(right)
  result

# 3Sum
proc threeSum(nums: seq[int]): seq[seq[int]] =
  nums.sort()
  var result: seq[seq[int]] = @[]
  for i in 0 ..< len(nums) - 2:
    if i > 0 and nums[i] == nums[i - 1]:
      continue
    var mid = i + 1
    var high = len(nums) - 1
    while mid < high:
      let total = nums[i] + nums[mid] + nums[high]
      if total < 0:
        mid += 1
      elif total > 0:
        high -= 1
      else:
        result.add @[int](nums[i], nums[mid], nums[high])
        mid += 1
        high -= 1
      while mid < high and nums[mid] == nums[mid - 1]:
        mid += 1
  result


# Combination Sum
proc backtrack(candidates: var seq[int], target: int, result: var seq[seq[int]], temp: var seq[int], current: int) =
  if target == 0:
    result.add(temp)
  if target <= 0:
    return
  for i in current ..< candidates.len:
    temp.add(candidates[i])
    backtrack(candidates, target - candidates[i], result, temp, i)
    temp.pop()

func combinationSum(candidates: seq[int], target: int): seq[seq[int]] =
  var result: seq[seq[int]] = @[]
  var temp: seq[int] = @[]
  backtrack(candidates, target, result, temp, 0)
  result


# Gas Station
proc canCompleteCircuit(gas, cost: seq[int]): int =
  var result = 0
  var gasTank = 0
  var totalTank = 0
  for i in 0 ..< gas.len:
    totalTank += gas[i] - cost[i]
    gasTank += gas[i] - cost[i]
    if gasTank < 0:
      result = i + 1
      gasTank = 0
  if totalTank >= 0:
    return result
  else:
    return -1

# Backspace String Compare
proc findNextValidChar(str: string, end: int): int =
  var backspaceCount = 0
  while end >= 0:
    if str[end] == '#':
      inc(backspaceCount)
    elif backspaceCount > 0:
      dec(backspaceCount)
    else:
      break
    dec(end)
  return end

proc backspaceCompare(s: string, t: string): bool =
  var pS = s.len - 1
  var pT = t.len - 1
  while pS >= 0 or pT >= 0:
    pS = findNextValidChar(s, pS)
    pT = findNextValidChar(t, pT)
    if pS < 0 and pT < 0:
      return true
    elif pS < 0 or pT < 0:
      return false
    elif s[pS] != t[pT]:
      return false
    dec(pS)
    dec(pT)
  return true

# Sort an Array
proc sortArray(nums: seq[int]): seq[int] =
  heapSort(nums)
  result = nums

proc heapSort(nums: int[]) =
  let n = nums.length
  for i in (n / 2 - 1).downTo(0):
    heapify(nums, i, n)
  for i in (n - 1).downTo(0):
    nums.swap(0, i)
    heapify(nums, 0, i)

proc heapify(nums: seq[int], root: int, length: int) =
  var largest = root
  let left = 2 * root + 1
  let right = 2 * root + 2
  if left < length and nums[left] > nums[largest]:
    largest = left
  if right < length and nums[right] > nums[largest]:
    largest = right
  if largest != root:
    nums.swap(largest, root)
    heapify(nums, largest, length)
