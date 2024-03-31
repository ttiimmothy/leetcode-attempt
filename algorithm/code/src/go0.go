package code

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
