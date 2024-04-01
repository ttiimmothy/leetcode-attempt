// Merge Intervals
pub fn merge(intervals: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
  let mut intervals = intervals;
  intervals.sort_by_key(|a| a[0]);
  let mut result = Vec::new();
  result.push(intervals[0].clone());
  for i in 1..intervals.len() {
    let interval = intervals[i].clone();
    if interval[0] <= result[result.len() - 1][1] {
      result.last_mut().unwrap()[1] = result.last_mut().unwrap()[1].max(interval[1]);
    } else {
      result.push(interval);
    }
  }
  result
}

// Merge Intervals
pub fn merge(intervals: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
  let mut intervals = intervals;
  intervals.sort_by_key(|a| a[0]);
  let mut result = Vec::new();
  let mut merged_interval = intervals[0].clone();
  for i in 1..intervals.len() {
    if intervals[i][0] <= merged_interval[1] {
      merged_interval[1] = merged_interval[1].max(intervals[i][1]);
    } else {
      result.push(merged_interval);
      merged_interval = intervals[i].clone();
    }
  }
  result.push(merged_interval);
  result
}

// Insert Interval
pub fn insert(intervals: Vec<Vec<i32>>, new_interval: Vec<i32>) -> Vec<Vec<i32>> {
  let mut i = 0;
  let mut result = Vec::new();
  while i < intervals.len() && intervals[i][1] < new_interval[0] {
    result.push(intervals[i].clone());
    i += 1;
  }
  let mut merged_interval = [new_interval[0], new_interval[1]];
  while i < intervals.len() && intervals[i][0] <= new_interval[1] {
    let interval = &intervals[i];
    merged_interval = [merged_interval[0].min(interval[0]), merged_interval[1].max(interval[1])];
    i += 1;
  }
  result.push(merged_interval.to_vec());
  while i < intervals.len() {
    result.push(intervals[i].clone());
    i += 1;
  }
  result
}

// Insert Interval
pub fn insert(intervals: Vec<Vec<i32>>, new_interval: Vec<i32>) -> Vec<Vec<i32>> {
  let mut result = Vec::new();
  let mut new_interval = new_interval;
  for i in intervals {
    if new_interval[1] < i[0] {
      result.push(new_interval);
      new_interval = i;
    } else if new_interval[0] > i[1]{
      result.push(i);
    } else {
      new_interval[0] = new_interval[0].min(i[0]);
      new_interval[1] = new_interval[1].max(i[1]);
    }
  }
  result.push(new_interval);
  result
}

// Sort an Array
pub fn sort_array(nums: Vec<i32>) -> Vec<i32> {
  let n = nums.len();
  let mut nums = nums.clone();
  Self::merge_sort(&mut nums, 0, n - 1);
  return nums;
}

pub fn merge_sort(nums: &mut Vec<i32>, start: usize, end: usize) {
  if start < end {
    let mut mid = start + (end - start) / 2;
    Self::merge_sort(nums, start, mid);
    Self::merge_sort(nums, mid + 1, end);
    let left_nums = nums[start..=mid].to_vec();
    let right_nums = nums[mid + 1..=end].to_vec();
    let (mut i, mut j, mut k) = (0, 0, start);
    while i < left_nums.len() && j < right_nums.len() {
      if left_nums[i] < right_nums[j] {
        nums[k] = left_nums[i];
        i += 1;
      } else {
        nums[k] = right_nums[j];
        j += 1;
      }
      k += 1;
    }
    while i < left_nums.len() {
      nums[k] = left_nums[i];
      i += 1;
      k += 1;
    }
    while j < right_nums.len() {
      nums[k] = right_nums[j];
      j += 1;
      k += 1;
    }
  }
}

// Merge Sort
pub fn merge_sort(nums: &mut Vec<i32>, start: usize, end: usize) {
  if start < end {
    let mid = start + (end - start) / 2;
    merge_sort(nums, start, mid);
    merge_sort(nums, mid + 1, end);
    let left_arr = nums[start..=mid].to_vec();
    let right_arr = nums[mid + 1..=end].to_vec();
    let (mut i, mut j, mut k) = (0, 0, start);
    while i < left_arr.len() && j < right_arr.len() {
      if left_arr[i] < right_arr[j] {
        nums[k] = left_arr[i];
        i += 1;
      } else {
        nums[k] = right_arr[j];
        j += 1;
      }
      k += 1;
    }
    while i < left_arr.len() {
      nums[k] = left_arr[i];
      i += 1;
      k += 1;
    }
    while j < right_arr.len() {
      nums[k] = right_arr[j];
      j += 1;
      k += 1;
    }
  }
}

// Quick Sort
pub fn quick_sort(nums: &mut Vec<i32>, start: usize, end: usize) {
  if start < end {
    let pivot = partition(nums, start, end);
    quick_sort(nums, start, pivot - 1);
    quick_sort(nums, pivot + 1, end);
  }
}

pub fn partition(nums: &mut Vec<i32>, start: usize, end: usize) -> usize {
  let mut low = start as isize - 1;
  let pivot = nums[end];
  for i in start..end {
    if nums[i] < pivot {
      low += 1;
      nums.swap(i, low as usize);
    }
  }
  nums.swap((low + 1) as usize, end);
  (low + 1) as usize
}

// Bubble Sort
pub fn bubble_sort(nums: &mut Vec<i32>) {
  let n = nums.len();
  for i in 0..n {
    let mut swapped = false;
    // n - i - 1 needs to -1 because j is compared with j+1, if there is not -1, the index will be out of bound
    for j in 0..n - i - 1 {
      if nums[j] > nums[j + 1] {
        nums.swap(j, j + 1);
        swapped = true;
      }
    }
    if !swapped {
      break;
    }
  }
}

fn main() {
  let mut nums = vec![1, 9, 8, 20, 15, 17, 5, 4, 8, 3];
  let length = nums.len();
  merge_sort(&mut nums, 0, length - 1);
  // bubble_sort(&mut nums);
  println!("{:?}", nums);
}