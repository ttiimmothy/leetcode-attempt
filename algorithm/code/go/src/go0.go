package src

import (
  "sort"
  "math"
  "strconv"
  "unicode"
  "strings"
)

type ListNode struct {
  Val int
  Next *ListNode
}

// 1
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

// 3
// String to Integer (atoi)
func myAtoi(s string) int {
  i, result, sign := 0, 0, 1
  for i < len(s) && s[i] == ' ' {
    i++
  }
  if i < len(s) && (s[i] == '-' || s[i] == '+') {
    if (s[i] == '-') {
      sign = -1
    }
    i++
  }
  for i < len(s) && s[i] >= '0' && s[i] <='9' {
    if result > math.MaxInt32/10 || (result == math.MaxInt32/10 && int(s[i])-'0' > math.MaxInt32%10) {
      if sign == -1 {
        return math.MinInt32
      }
      return math.MaxInt32
    }
    result = result*10+int(s[i])-'0'
    i++
  }
  return result*sign
}

// 11
// Container With Most Water
//
//lint:ignore U1000 Function is intentionally left unused
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

// 15
// 3Sum
//
//lint:ignore U1000 Function is intentionally left unused
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

// 39
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

// 40
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

// 47
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

// 56
// Merge Intervals
//
//lint:ignore U1000 Function is intentionally left unused
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

// 57
// Insert Interval
//
//lint:ignore U1000 Function is intentionally left unused
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

// 57-1
// Insert Interval
//
//lint:ignore U1000 Function is intentionally left unused
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
  for i < len(intervals) && intervals[i][0] > newInterval[1] {
    result = append(result, intervals[i])
    i++
  }
  return result
}

// 75
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

// 92
// Reverse Linked List II
//
//lint:ignore U1000 Function is intentionally left unused
func reverseBetween(head *ListNode, left int, right int) *ListNode {
  result := &ListNode{Val:0, Next:head}
  leftPrev, current := result, head
  for i := 0; i < left - 1; i++ {
    leftPrev = current
    current = current.Next
  }
  node := leftPrev
  for i := 0; i < right - left + 1; i++ {
    stored := current.Next
    current.Next = leftPrev
    leftPrev = current
    current = stored
  }
  node.Next.Next = current
  node.Next = leftPrev
  return result.Next
}

// 110
// Balanced Binary Tree
//
//lint:ignore U1000 Function is intentionally left unused
type TreeNode struct {
  Val int
  Left *TreeNode
  Right *TreeNode
}
func isBalanced(root *TreeNode) bool {
  result,_ := dfs(root)
  return result
}

func dfs(node *TreeNode) (bool,int) {
  if node == nil {
    return true,0
  }
  left,depthL := dfs(node.Left)
  right,depthR := dfs(node.Right)
  difference := depthL-depthR
  if difference < 0 {
    difference = -difference
  }
  height := max(depthL,depthR)+1
  if !left || !right || difference > 1 {
    return false,height
  }
  return true,height
}

// 125
// Valid Palindrome
//
//lint:ignore U1000 Function is intentionally left unused
func isPalindrome(s string) bool {
  array := []rune{}
  for _,i := range s {
    if unicode.IsLetter(i) || unicode.IsNumber(i) {
      array = append(array, unicode.ToLower(i))
    }
  }
  left, right := 0, len(array)-1
  for left < right {
    if array[left] != array[right] {
      return false
    }
    left++
    right--
  }
  return true
}


// 125-1
// Valid Palindrome
//
//lint:ignore U1000 Function is intentionally left unused
func isPalindrome_1(s string) bool {
  left, right := 0, len(s)-1
  for left < right {
    if !unicode.IsLetter(rune(s[left])) && !unicode.IsNumber(rune(s[left])) {
      left++
    } else if !unicode.IsLetter(rune(s[right])) && !unicode.IsNumber(rune(s[right])) {
      right--
    } else if strings.ToLower(string(s[left])) != strings.ToLower(string(s[right])) {
      return false
    } else {
      left++
      right--
    }
  }
  return true
}

// 131
// Palindrome Partitioning
//
//lint:ignore U1000 Function is intentionally left unused
func partition(s string) [][]string {
  result := [][]string{}
  backtrack_3(s, &result, []string{}, 0)
  return result
}

func backtrack_3(s string, result *[][]string, temp []string, index int) {
  if index >= len(s) {
    array := make([]string, len(temp))
    copy(array, temp)
    *result = append(*result, array)
    return
  }
  for i := index; i < len(s); i++ {
    if isPalindrome_2(s, index, i) {
      temp = append(temp, s[index:i+1])
      backtrack_3(s, result, temp, i+1)
      temp = temp[:len(temp)-1]
    }
  }
}

func isPalindrome_2(s string, start int, end int) bool {
  for start < end {
    if s[start] != s[end] {
      return false
    }
    start++
    end--
  }
  return true
}

// 134
// Gas Station
//
//lint:ignore U1000 Function is intentionally left unused
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

// 146
// LRU Cache
//
//lint:ignore U1000 Function is intentionally left unused
type Node struct {
  key int
  val int
  prev *Node
  next *Node
}

type LRUCache struct {
  cacheCapacity int
  cache map[int]*Node
  left *Node
  right *Node
}

func Constructor(capacity int) LRUCache {
  ret := LRUCache{
    cacheCapacity: capacity,
    cache: make(map[int]*Node),
    left: &Node{key:0,val:0},
    right: &Node{key:0,val:0}}
  ret.left.next,ret.right.prev = ret.right,ret.left
  return ret
}

func (this *LRUCache) Get(key int) int {
  if _,ok := this.cache[key]; ok {
    this.Remove(this.cache[key])
    this.Insert(this.cache[key])
    return this.cache[key].val
  }
  return -1
}

func (this *LRUCache) Put(key int, value int)  {
  if _,ok := this.cache[key]; ok {
    this.Remove(this.cache[key])
  }
  this.cache[key] = &Node{key:key,val:value}
  this.Insert(this.cache[key])
  if len(this.cache) > this.cacheCapacity {
    lru := this.left.next
    this.Remove(lru)
    delete(this.cache, lru.key)
  }
}

func (this *LRUCache) Remove(node *Node) {
  prev, next := node.prev, node.next
  prev.next, next.prev = next, prev
}

func (this *LRUCache) Insert(node *Node) {
  prev, next := this.right.prev, this.right
  prev.next, next.prev = node, node
  node.prev, node.next = prev, next
}

// 150
// Evaluate Reverse Polish Notation
//
//lint:ignore U1000 Function is intentionally left unused
func evalRPN(tokens []string) int {
  stack := []int{}
  for _,a := range tokens {
    switch a {
    case "+":
      a, b := stack[len(stack)-1], stack[len(stack)-2]
      stack = stack[:len(stack)-2]
      stack = append(stack, a+b)
    case "-":
      a, b := stack[len(stack)-1], stack[len(stack)-2]
      stack = stack[:len(stack)-2]
      stack = append(stack, b-a)
    case "*":
      a, b := stack[len(stack)-1], stack[len(stack)-2]
      stack = stack[:len(stack)-2]
      stack = append(stack, a*b)
    case "/":
      a, b := stack[len(stack)-1], stack[len(stack)-2]
      stack = stack[:len(stack)-2]
      stack = append(stack, b/a)
    default:
      num, _ := strconv.Atoi(a)
      stack = append(stack, num)
    }
  }
  return stack[0]
}

// 155
// Min Stack
//
//lint:ignore U1000 Function is intentionally left unused
type MinStack struct {
  stack []int
  minStack []int
}

func Constructor_1() MinStack {
  return MinStack{}
}

func (this *MinStack) Push(val int)  {
  this.stack = append(this.stack, val)
  if len(this.minStack) > 0 {
    val = min(this.minStack[len(this.minStack) - 1], val)
  }
  this.minStack = append(this.minStack, val)
}

func (this *MinStack) Pop()  {
  this.stack = this.stack[:len(this.stack) - 1]
  this.minStack = this.minStack[:len(this.minStack) - 1]
}

func (this *MinStack) Top() int {
  return this.stack[len(this.stack) - 1]
}

func (this *MinStack) GetMin() int {
  return this.minStack[len(this.minStack) - 1]
}

// 200
// Number of Islands
//
//lint:ignore U1000 Function is intentionally left unused
func numIslands(grid [][]byte) int {
  result := 0
  for i := 0; i < len(grid); i++ {
    for j := 0; j < len(grid[0]); j++ {
      if grid[i][j] == '1' {
        dfs_1(grid, i, j)
        result++
      }
    }
  }
  return result
}

func dfs_1(grid [][]byte, i int, j int) {
  if i < 0 || i >= len(grid) || j < 0 || j >= len(grid[0]) || grid[i][j] != '1' {
    return
  }
  grid[i][j] = '#'
  dfs_1(grid, i-1, j)
  dfs_1(grid, i+1, j)
  dfs_1(grid, i, j-1)
  dfs_1(grid, i, j+1)
}

// 206
// Reversed Linked List
//
//lint:ignore U1000 Function is intentionally left unused
func reverseList(head *ListNode) *ListNode {
  var prev *ListNode
  current := head
  for current != nil {
    temp := current.Next
    current.Next = prev
    prev = current
    current = temp
  }
  return prev
}

// 225
// Implement Stack using Queues
//
//lint:ignore U1000 Function is intentionally left unused
type MyStack struct {
  queue []int
}

func Constructor_2() MyStack {
  return MyStack{}
}

func (this *MyStack) Push(x int)  {
  this.queue = append(this.queue, x)
  for i := 0; i < len(this.queue) - 1; i++ {
    val := this.queue[0]
    this.queue = this.queue[1:]
    this.queue = append(this.queue, val)
  }
}

func (this *MyStack) Pop() int {
  val := this.queue[0]
  this.queue = this.queue[1:]
  return val
}

func (this *MyStack) Top() int {
  return this.queue[0]
}

func (this *MyStack) Empty() bool {
  return len(this.queue) == 0
}

// 232
// Implement Queue using Stacks
//
//lint:ignore U1000 Function is intentionally left unused
type MyQueue struct {
  input, output []int
}

func Constructor_3() MyQueue {
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

// 236
// Lowest Common Ancestor of a Binary Tree
func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
  if root == nil {
    return nil
  }
  if root == p || root == q {
    return root
  }
  left := lowestCommonAncestor(root.Left, p, q)
  right := lowestCommonAncestor(root.Right, p, q)
  if left == nil {
    return right
  } else if right == nil {
    return left
  } else {
    return root
  }
}

// 238
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

// 242
// Valid Anagram
//
//lint:ignore U1000 Function is intentionally left unused
func isAnagram(s string, t string) bool {
  array := make([]int, 26)
  for _,i := range s {
    array[i-'a']++
  }
  for _,i := range t {
    array[i-'a']--
  }
  for i := 0; i < len(array); i++ {
    if array[i] != 0 {
      return false
    }
  }
  return true
}

// 242-1
// Valid Anagram
//
//lint:ignore U1000 Function is intentionally left unused
func isAnagram_1(s string, t string) bool {
  charMap := make(map[rune]int)
  for _,i := range s {
    charMap[i]++
  }
  for _,i := range t {
    if charMap[i] == 0 {
      return false
    }
    charMap[i]--
  }
  for _,i := range charMap {
    if i > 0 {
      return false
    }
  }
  return true
}

// 349
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

// 409
// Longest Palindrome
//
//lint:ignore U1000 Function is intentionally left unused
func longestPalindrome(s string) int {
  array := make([]int,128)
  odd := 0
  result := 0
  for _,a := range s {
    array[a]++
  }
  for _,i := range array {
    if i % 2 != 0 && odd == 0 {
      result += i
      odd++
    } else if i % 2 != 0 {
      result += i - 1
    } else {
      result += i
    }
  }
  return result
}

// 409-1
// Longest Palindrome
//
//lint:ignore U1000 Function is intentionally left unused
func longestPalindrome_1(s string) int {
  charMap := make(map[rune]int)
  odd := 0
  for _,a := range s {
    if _,ok := charMap[a]; ok {
      charMap[a]++
    } else {
      charMap[a] = 1
    }
    if charMap[a] % 2 != 0 {
      odd++
    } else {
      odd--
    }
  }
  if odd > 0 {
    return len(s) - odd + 1
  }
  return len(s)
}

// 460
// LFU Cache
//
//lint:ignore U1000 Function is intentionally left unused
type Node_1 struct {
  key   int
  val   int
  count int
  next  *Node_1
  prev  *Node_1
}

type LFUCache struct {
  keyMap   map[int]*Node_1
  countMap map[int]*Node_1
  capacity int
  minF     int
}

func Constructor_4(capacity int) LFUCache {
  return LFUCache{
    make(map[int]*Node_1),
    make(map[int]*Node_1),
    capacity,
    0,
  }
}

func (this *LFUCache) Get(key int) int {
  v, found := this.keyMap[key]
  if !found {
    return -1
  }
  this.count_remove(v)
  v.count += 1
  this.count_insert(v)
  return v.val
}

func (this *LFUCache) Put(key int, value int) {
  v, found := this.keyMap[key]
  if !found {
    if len(this.keyMap) >= this.capacity {
      this.evict()
    }
    v = &Node_1{
      key,
      value,
      0,
      nil,
      nil,
    }
    this.keyMap[key] = v
    this.minF = 1
  }
  v.val = value
  this.count_remove(v)
  v.count += 1
  this.count_insert(v)
}

func (this *LFUCache) count_remove(node *Node_1) {
  if node.count == 0 {
    return
  }
  dummy, _ := this.countMap[node.count]
  if node.next != nil {
    node.next.prev = node.prev
  } else {
    if dummy.next == node {
      delete(this.countMap, node.count)
      if node.count == this.minF {
        this.minF++
      }
      return
    }
    dummy.prev = node.prev
  }
  node.prev.next = node.next
}

func (this *LFUCache) count_insert(node *Node_1) {
  dummy, found := this.countMap[node.count]
  if !found {
    dummy = &Node_1{
      0, 0, 0, nil, nil,
    }
    this.countMap[node.count] = dummy
  }
  last := dummy.prev
  if last == nil {
    dummy.next = node
    node.prev = dummy
  } else {
    last.next = node
    node.prev = last
  }
  dummy.prev = node
  node.next = nil
}

func (this *LFUCache) evict() {
  dummy, _ := this.countMap[this.minF]
  next := dummy.next
  if next.next == nil {
    delete(this.countMap, this.minF)
    this.minF++
  } else {
    dummy.next = next.next
    next.next.prev = dummy
  }
  delete(this.keyMap, next.key)
}

// 844
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

// 912
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
    pivot := partition_1(nums, low, high)
    quickSort(nums, low, pivot-1)
    quickSort(nums, pivot+1, high)
  }
}

func partition_1(nums []int, low int, high int) int {
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
//
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
