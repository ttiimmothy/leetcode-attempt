const std = @import("std");

// Backspace String Compare
fn findNextValidChar(str: []const u8, end: &usize) -> usize {
  var backspaceCount: usize = 0;
  while (end.* >= 0) : (end -= 1) {
    if (str[end] == '#' as u8) {
      backspaceCount += 1;
    } else if (backspaceCount > 0) {
      backspaceCount -= 1;
    } else {
      break;
    }
  }
  return end;
}

fn backspaceCompare(s: []const u8, t: []const u8) -> bool {
  var pS: usize = @intCast(usize, s.len) - 1;
  var pT: usize = @intCast(usize, t.len) - 1;
  while (pS >= 0) || (pT >= 0) : (pS -= 1; pT -= 1) {
    pS = findNextValidChar(s, &pS);
    pT = findNextValidChar(t, &pT);
    if (pS < 0) && (pT < 0) {
      return true;
    } else if (pS < 0) || (pT < 0) {
      return false;
    } else if (s[pS] != t[pT]) {
      return false;
    }
  }
  return true;
}

// Sort an Array
pub fn sortArray(nums: []i32) []i32 {
  heapSort(&nums);
  return nums;
}

fn heapSort(nums: []i32) void {
  var n = nums.len;
  for (std.math.reverseStep(0, n / 2)) |i| {
    heapify(nums, i, n);
  }
  for (std.math.reverseStep(1, n)) |i| {
    nums.swap(0, i);
    heapify(nums, 0, i);
  }
}

fn heapify(nums: []i32, root i32, length i32) void {
  var largest = root;
  var left = 2 * root + 1;
  var right = 2 * root + 2;
  if (left < length && nums[left] > nums[largest]) {
    largest = left;
  }
  if (right < length && nums[right] > nums[largest]) {
    largest = right;
  }
  if (largest != root) {
    nums.swap(largest, root);
    heapify(nums, largest, length);
  }
}

// Merge sort
fn mergeSort(nums: []i32, start: i32, end: i32) {
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
fn heapSort(nums: []i32) void {
  var n = nums.len;
  for (std.math.reverseStep(0, n / 2)) |i| {
    heapify(nums, i, n);
  }
  for (std.math.reverseStep(1, n)) |i| {
    nums.swap(0, i);
    heapify(nums, 0, i);
  }
}

fn heapify(nums: []i32, root i32, length i32) void {
  var largest = root;
  var left = 2 * root + 1;
  var right = 2 * root + 2;
  if (left < length && nums[left] > nums[largest]) {
    largest = left;
  }
  if (right < length && nums[right] > nums[largest]) {
    largest = right;
  }
  if (largest != root) {
    nums.swap(largest, root);
    heapify(nums, largest, length);
  }
}

// Quick sort
fn quickSort(nums: []i32, start: i32, end: i32) void {
  if (start >= end) {
    return;
  }
  const pivot = partition(nums, start, end);
  quickSort(nums, start, pivot - 1);
  quickSort(nums, pivot + 1, end);
}

fn partition(nums: []i32, start: i32, end: i32) i32 {
  var index = start - 1;
  var pivot = nums[end];
  for (start..<end) |i| {
    if (nums[i] < pivot) {
      index += 1;
      nums.swap(index, i);
    }
  }
  nums.swap(index + 1, end);
  return index + 1;
}

pub fn main() !void {
  const nums = [_]i32{12, 11, 13, 5, 6, 7};
  const sorted = sortArray(nums);
  std.debug.print("{any}\n", .{sorted});
  std.debug.print("hello");
}
