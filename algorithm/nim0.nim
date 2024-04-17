# 11
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

# 15
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

# 39
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

# 134
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

# 460
# LFU Cache
import tables
type
  LFUCache = object
    capacity: int
    items: Table[int, int]
    freqs: Table[int, OrderedTable[int, int]]
    minFreq: int

proc newLFUCache(capacity: int): LFUCache =
  result.capacity = capacity
  initTable(result.items)
  initTable(result.freqs)
  result.minFreq = 0

proc updateFreq(cache: var LFUCache, key: int, value: int = 0): int =
  let f = cache.items.getOrDefault(key, 0)
  var v: int
  if f > 0:
    v = cache.freqs[f][key]
    cache.freqs[f].remove(key)
    v = if value != 0: value else: v
    cache.freqs[f + 1][key] = v
    inc(cache.items[key])
    if cache.minFreq == f and cache.freqs[f].isEmpty:
      inc(cache.minFreq)
  else:
    v = value
    cache.freqs[1][key] = v
    cache.items[key] = 1
    cache.minFreq = 1
  result = v

proc get(cache: var LFUCache, key: int): int =
  if cache.items.contains(key):
    result = updateFreq(cache, key)
  else:
    result = -1

proc put(cache: var LFUCache, key, value: int) =
  if cache.capacity > 0:
    if not cache.items.contains(key):
      if cache.items.len >= cache.capacity:
        let minFreqItems = cache.freqs[cache.minFreq]
        if minFreqItems.len > 0:
          let lruKey = minFreqItems.frontKey
          minFreqItems.remove(lruKey)
          cache.items.remove(lruKey)
    updateFreq(cache, key, value)

# 844
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

# 912
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
