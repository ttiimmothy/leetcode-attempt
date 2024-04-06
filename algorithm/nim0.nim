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
