package src

import "slices"

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

// Combination Sum 2
//
//lint:ignore U1000 Function is intentionally left unused
func combinationSum2(candidates []int, target int) [][]int {
	slices.Sort(candidates)
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
	var result []int
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
		L := make([]int, mid-low+1)
		R := make([]int, high-mid)
		copy(L, nums[low:mid+1])
		copy(R, nums[mid+1:high+1])
		i, j, k := 0, 0, low
		for i < len(L) && j < len(R) {
			if L[i] < R[j] {
				nums[k] = L[i]
				i++
			} else {
				nums[k] = R[j]
				j++
			}
			k++
		}
		for i < len(L) {
			nums[k] = L[i]
			i++
			k++
		}
		for j < len(R) {
			nums[k] = R[j]
			j++
			k++
		}
	}
}

// Merge Sort
//
//lint:ignore U1000 Function is intentionally left unused
func mergeSortSample(nums []int, low int, high int) {
	if low < high {
		mid := low + (high-low)/2
		mergeSortSample(nums, low, mid)
		mergeSortSample(nums, mid+1, high)
		L := make([]int, mid-low+1)
		R := make([]int, high-mid)
		copy(L, nums[low:mid+1])
		copy(R, nums[mid+1:high+1])
		i, j := 0, 0
		k := low
		for i < len(L) && j < len(R) {
			if L[i] <= R[j] {
				nums[k] = L[i]
				i++
			} else {
				nums[k] = R[j]
				j++
			}
			k++
		}
		for i < len(L) {
			nums[k] = L[i]
			i++
			k++
		}
		for j < len(R) {
			nums[k] = R[j]
			j++
			k++
		}
	}
}

// Quick Sort, may cause time limit error
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

// Bubble Sort
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

// Heap Sort
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
func HeapSort(nums []int) {
	n := len(nums)
	for i := n/2 - 1; i >= 0; i-- {
		heapify(nums, i, n)
	}
	for i := n - 1; i > 0; i-- {
		nums[0], nums[i] = nums[i], nums[0]
		heapify(nums, 0, i)
	}
}
