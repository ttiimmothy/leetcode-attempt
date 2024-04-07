package src

import (
  "sort"
)

// Two Sum
//
//lint:ignore U1000 Function is intentionally left unused
func twoSum(nums []int, target int) []int {
  hashMap := make(map[int]int)
  for i, num := range nums {
    difference := target - num
    if index, ok := hashMap[difference]; ok {
      return []int{index, i}
    }
    hashMap[num] = i
  }
  return []int{}
}

// Container With Most Water
func maxArea(height []int) int {
  result := 0
  left, right := 0, len(height) - 1
  for left < right {
    result = max(result, (right - left) * min(height[left], height[right]))
    if height[left] < height[right] {
      left++
    } else {
      right--
    }
  }
  return result
}

// Combination Sum
//
//lint:ignore U1000 Function is intentionally left unused
func combinationSum(candidates []int, target int) [][]int {
  var result [][]int
  backtrack(candidates, target, &result, []int{}, 0)
  return result
}

func backtrack(candidates []int, target int, result *[][]int, subList []int, index int) {
  if target < 0 {
    return
  } else if target == 0 {
    temp := make([]int, len(subList))
    copy(temp, subList)
    *result = append(*result, temp)
    return
  }
  for i := index; i < len(candidates); i++ {
    subList = append(subList, candidates[i])
    backtrack(candidates, target-candidates[i], result, subList, i)
    subList = subList[:len(subList)-1]
  }
}

// 3Sum
func threeSum(nums []int) [][]int {
  sort.Ints(nums)
  result := [][]int{}
  n := len(nums)
  for i := 0; i < n - 2; i++ {
    if i > 0 && nums[i] == nums[i - 1] {
      continue
    }
    low, high := i + 1, n - 1;
    for low < high {
      threeSum := nums[i] + nums[low] + nums[high]
      if threeSum < 0 {
        low++
      } else if threeSum > 0 {
        high--
      } else {
        result = append(result, []int{nums[i], nums[low], nums[high]})
        low++
        high--
        for low < high && nums[low] == nums[low - 1] {
          low++
        }
      }
    }
  }
  return result
}

// Combination Sum II
//
//lint:ignore U1000 Function is intentionally left unused
func combinationSum2(candidates []int, target int) [][]int {
  sort.Ints(candidates)
  result := [][]int{}
  backtrack_1(candidates, target, &result, []int{}, 0)
  return result
}

func backtrack_1(candidates []int, target int, result *[][]int, subList []int, index int) {
  if target < 0 {
     return
  } else if target == 0 {
    temp := make([]int, len(subList))
    copy(temp, subList)
    *result = append(*result, temp)
    return
  }
  for i := index; i < len(candidates); i++ {
    if i > index && candidates[i] == candidates[i-1] {
      // return will break the loop, but we don't want to break the loop
      continue
    }
    subList = append(subList, candidates[i])
    backtrack_1(candidates, target-candidates[i], result, subList, i+1)
    subList = subList[:len(subList)-1]
  }
}

// Permutations II
//
//lint:ignore U1000 Function is intentionally left unused
func permuteUnique(nums []int) [][]int {
  sort.Ints(nums)
  result := [][]int{}
  visit := make([]bool, len(nums))
  backtrack_2(nums, &result, []int{}, visit)
  return result
}

func backtrack_2(nums []int, result *[][]int, temp []int, visit []bool) {
  if len(temp) == len(nums) {
    list := make([]int, len(temp))
    copy(list, temp)
    *result = append(*result, list)
    return
  }
  for i := 0; i < len(nums); i++ {
    if visit[i] || i > 0 && !visit[i - 1] && nums[i - 1] == nums[i] {
      continue
    }
    visit[i] = true
    temp = append(temp, nums[i])
    backtrack_2(nums, result, temp, visit)
    visit[i] = false
    temp = temp[:len(temp) - 1]
  }
}

// Merge Intervals
func merge(intervals [][]int) [][]int {
  sort.Slice(intervals, func(i, j int) bool{
    return intervals[i][0] < intervals[j][0]
  })
  result := [][]int{intervals[0]}
  for i := 1; i < len(intervals); i++ {
    interval := intervals[i]
    if interval[0] <= result[len(result) - 1][1] {
      if interval[1] > result[len(result) - 1][1] {
        result[len(result) - 1][1] = interval[1]
      }
    } else {
      result = append(result, interval)
    }
  }
  return result
}

// Insert Interval
func insert(intervals [][]int, newInterval []int) [][]int {
  result := [][]int{};
  for i := 0; i < len(intervals); i++ {
    if newInterval[1] < intervals[i][0] {
      result = append(result, newInterval);
      return append(result, intervals[i:]...)
    } else if newInterval[0] > intervals[i][1] {
      result = append(result, intervals[i])
    } else {
      newInterval = []int{min(newInterval[0], intervals[i][0]), max(newInterval[1], intervals[i][1])}
    }
  }
  result = append(result, newInterval)
  return result
}

// Insert Interval
func insert_1(intervals [][]int, newInterval []int) [][]int {
  result := [][]int{};
  i := 0
  for i < len(intervals) && intervals[i][1] < newInterval[0] {
    result = append(result, intervals[i])
    i++
  }
  for i < len(intervals) && intervals[i][0] <= newInterval[1] && intervals[i][1] >= newInterval[0]   {
    newInterval = []int{min(intervals[i][0], newInterval[0]), max(intervals[i][1], newInterval[1])}
    i++
  }
  result = append(result, newInterval)
  for i < len(intervals) {
    result = append(result, intervals[i])
    i++
  }
  return result
}

// Sort Colors
//
//lint:ignore U1000 Function is intentionally left unused
func sortColors(nums []int) {
  low, mid, high := 0, 0, len(nums)-1
  for mid <= high {
    if nums[mid] == 0 {
      nums[low], nums[mid] = nums[mid], nums[low]
      low++
      mid++
    } else if nums[mid] == 1 {
      mid++
    } else {
      nums[mid], nums[high] = nums[high], nums[mid]
      high--
    }
  }
}

// Gas Station
func canCompleteCircuit(gas []int, cost []int) int {
  total, current, result := 0, 0, 0
  for i := 0; i < len(gas); i++ {
    total += gas[i] - cost[i]
    current += gas[i] - cost[i]
    if current < 0 {
      result = i + 1
      current = 0
    }
  }
  if total >= 0 {
    return result
  }
  return -1
}

// Implement Queue using Stacks
//
//lint:ignore U1000 Function is intentionally left unused
type MyQueue struct {
  input, output []int
}

func Constructor() MyQueue {
  return MyQueue{}
}

func (this *MyQueue) Push(x int)  {
  this.input = append(this.input, x)
}

func (this *MyQueue) Pop() int {
  val := this.Peek()
  this.output = this.output[:len(this.output)-1]
  return val
}

func (this *MyQueue) Peek() int {
  if len(this.output) == 0 {
    for len(this.input) > 0 {
      this.output = append(this.output, this.input[len(this.input)-1])
      this.input = this.input[:len(this.input)-1]
    }
  }
  return this.output[len(this.output)-1]
}

func (this *MyQueue) Empty() bool {
  return len(this.input) == 0 && len(this.output) == 0
}

// Product of Array Except Self
//
//lint:ignore U1000 Function is intentionally left unused
func productExceptSelf(nums []int) []int {
  prefix := 1
  result := make([]int, len(nums))
  for i := 0; i < len(nums); i++ {
    result[i] = prefix
    prefix *= nums[i]
  }
  postfix := 1
  for i := len(nums) - 1; i >= 0; i-- {
    result[i] *= postfix
    postfix *= nums[i]
  }
  return result
}

// Intersection of Two Arrays
//
//lint:ignore U1000 Function is intentionally left unused
func intersection(nums1 []int, nums2 []int) []int {
  // no set in go, so use map
  hashMap := make(map[int]bool)
  for _, num := range nums1 {
    hashMap[num] = true
  }
  result := []int{}
  for _, num := range nums2 {
    if hashMap[num] {
      result = append(result, num)
      hashMap[num] = false
    }
  }
  return result
}

// Backspace String Compare
//
//lint:ignore U1000 Function is intentionally left unused
func backspaceCompare(s string, t string) bool {
  pointerS := len(s) - 1
  pointerT := len(t) - 1
  for pointerS >= 0 || pointerT >= 0 {
    pointerS = findValidCharIndex(s, pointerS)
    pointerT = findValidCharIndex(t, pointerT)
    if pointerS < 0 && pointerT < 0 {
      return true
    } else if pointerS < 0 || pointerT < 0 {
      return false
    } else if s[pointerS] != t[pointerT] {
    return false
    }
    pointerS--
    pointerT--
  }
  return true
}

func findValidCharIndex(str string, end int) int {
  backspaceCount := 0
  for end >= 0 {
    if str[end] == '#' {
      backspaceCount++
    } else if backspaceCount > 0 {
      backspaceCount--
    } else {
      break
    }
    end--
  }
  return end
}

// Sort an Array
//
//lint:ignore U1000 Function is intentionally left unused
func sortArray(nums []int) []int {
  mergeSort(nums, 0, len(nums)-1)
  return nums
}

func mergeSort(nums []int, low int, high int) {
  if low < high {
    mid := low + (high-low)/2
    mergeSort(nums, low, mid)
    mergeSort(nums, mid+1, high)
    leftNums := make([]int, mid-low+1)
    rightNums := make([]int, high-mid)
    copy(leftNums, nums[low:mid+1])
    copy(rightNums, nums[mid+1:high+1])
    i, j, k := 0, 0, low
    for i < len(leftNums) && j < len(rightNums) {
      if leftNums[i] < rightNums[j] {
        nums[k] = leftNums[i]
        i++
      } else {
        nums[k] = rightNums[j]
        j++
      }
      k++
    }
    for i < len(leftNums) {
      nums[k] = leftNums[i]
      i++
      k++
    }
    for j < len(rightNums) {
      nums[k] = rightNums[j]
      j++
      k++
    }
  }
}

// Merge sort
//
//lint:ignore U1000 Function is intentionally left unused
func mergeSortSample(nums []int, low int, high int) {
  if low < high {
    mid := low + (high-low)/2
    mergeSort(nums, low, mid)
    mergeSort(nums, mid+1, high)
    leftNums := make([]int, mid-low+1)
    rightNums := make([]int, high-mid)
    copy(leftNums, nums[low:mid+1])
    copy(rightNums, nums[mid+1:high+1])
    i, j, k := 0, 0, low
    for i < len(leftNums) && j < len(rightNums) {
      if leftNums[i] < rightNums[j] {
        nums[k] = leftNums[i]
        i++
      } else {
        nums[k] = rightNums[j]
        j++
      }
      k++
    }
    for i < len(leftNums) {
      nums[k] = leftNums[i]
      i++
      k++
    }
    for j < len(rightNums) {
      nums[k] = rightNums[j]
      j++
      k++
    }
  }
}

// Quick sort, may cause time limit error
//
//lint:ignore U1000 Function is intentionally left unused
func quickSort(nums []int, low int, high int) {
  if low < high {
    pivot := partition(nums, low, high)
    quickSort(nums, low, pivot-1)
    quickSort(nums, pivot+1, high)
  }
}

func partition(nums []int, low int, high int) int {
  start := low - 1
  pivot := nums[high]
  for i := low; i < high; i++ {
    if nums[i] < pivot {
      start++
      nums[start], nums[i] = nums[i], nums[start]
    }
  }
  nums[start+1], nums[high] = nums[high], nums[start+1]
  return start + 1
}

// Bubble sort
//
//lint:ignore U1000 Function is intentionally left unused
func bubbleSort(nums []int) {
  n := len(nums)
  for i := 0; i < n-1; i++ {
    swapped := false
    for j := 0; j < n-i-1; j++ {
      if nums[j] > nums[j+1] {
        nums[j], nums[j+1] = nums[j+1], nums[j]
        swapped = true
      }
    }
    if !swapped {
      break
    }
  }
}

// Heap sort
func heapify(nums []int, max int, length int) {
  largest := max
  left := 2*max + 1
  right := 2*max + 2
  if left < length && nums[left] > nums[largest] {
    largest = left
  }
  if right < length && nums[right] > nums[largest] {
    largest = right
  }
  if largest != max {
    nums[largest], nums[max] = nums[max], nums[largest]
    heapify(nums, largest, length)
  }
}

//lint:ignore U1000 Function is intentionally left unused
func heapSort(nums []int) {
  n := len(nums)
  for i := n/2 - 1; i >= 0; i-- {
    heapify(nums, i, n)
  }
  for i := n - 1; i > 0; i-- {
    nums[0], nums[i] = nums[i], nums[0]
    heapify(nums, 0, i)
  }
}
