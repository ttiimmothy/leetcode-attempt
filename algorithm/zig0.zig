const std = @import("std");

// Merge sort
pub fn mergeSort(nums: []i32, start: i32, end: i32) {
  if (start >= end) {
    return
  }
  var mid = start + (end - start) / 2;
  mergeSort(nums, start, mid);
  mergeSort(nums, mid + 1, end);
  var leftNums = nums[start..mid];
  var rightNums = nums[mid + 1..end + 1];
  var i = 0, j = 0, k = start;
  while (i < leftNums.len && j < rightNums.len) {
    if (leftNums[i] < rightNums[j]) {
      nums[k] = leftNums[i];
      i += 1;
    } else {
      nums[k] = rightNums[j];
      j += 1;
    }
    k += 1;
  }
  while (i < leftNums.len) {
    nums[k] = leftNums[i];
    i += 1;
    k += 1;
  }
  while (j < rightNums.len) {
    nums[k] = rightNums[j];
    j += 1;
    k += 1;
  }
}

// Heap sort
pub fn heapSort(nums: []i32) {
  
}

pub fn main() !void {
  std.debug.print("hello");
}
