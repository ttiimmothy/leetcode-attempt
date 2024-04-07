// Two Sum
pub fn two_sum(nums: Vec<i32>, target: i32) -> Vec<i32> {
  use std::collections::HashMap;
  let mut map = HashMap::new();
  for (i, &num) in nums.iter().enumerate() {
    let difference = target - num;
    match map.get(&difference) {
      Some(&j) => return vec![i as i32, j as i32],
      None => map.insert(num, i)
    };
  }
  Vec::new()
}

// Container With Most Water
pub fn max_area(height: Vec<i32>) -> i32 {
  let (mut left, mut right) = (0, height.len() - 1);
  let mut result = 0;
  while left < right {
    result = result.max((right - left) as i32 * height[left].min(height[right]));
    if height[left] > height[right] {
      right -= 1;
    } else {
      left += 1;
    }
  }
  result
}

// 3Sum
pub fn three_sum(nums: Vec<i32>) -> Vec<Vec<i32>> {
  let mut nums = nums;
  nums.sort();
  let mut result = vec![];
  for i in 0..nums.len()-2 {
    if i > 0 && nums[i] == nums[i - 1] {
      continue;
    }
    let (mut low, mut high) = (i + 1, nums.len() - 1);
    while low < high {
      let three_sum = nums[i] + nums[low] + nums[high];
      if three_sum < 0 {
        low += 1
      } else if three_sum > 0 {
        high -= 1
      } else {
        result.push(vec![nums[i], nums[low], nums[high]]);
        low += 1;
        high -= 1;
        while low < high && nums[low] == nums[low - 1] {
          low += 1;
        }
      }
    }
  }
  result
}

// Combination Sum
pub fn combination_sum(candidates: Vec<i32>, target: i32) -> Vec<Vec<i32>> {
  let mut result = Vec::new();
  let mut temp = Vec::new();
  pub fn backtrack(candidates: &Vec<i32>, target: i32, result: &mut Vec<Vec<i32>>, temp: &mut Vec<i32>, index: usize) {
    if target == 0 {
      result.push(temp.clone());
      return;
    } else if target < 0 {
      return;
    }
    for i in index..candidates.len() {
      temp.push(candidates[i]);
      backtrack(candidates, target - candidates[i], result, temp, i);
      temp.pop();
    }
  }
  backtrack(&candidates, target, &mut result, &mut temp, 0);
  result
}

// Combination Sum
pub fn combination_sum(candidates: Vec<i32>, target: i32) -> Vec<Vec<i32>> {
  let mut result = Vec::new();
  let mut temp = Vec::new();
  pub fn backtrack(candidates: &[i32], target: i32, result: &mut Vec<Vec<i32>>, temp: &mut Vec<i32>) {
    if target == 0 {
      result.push(temp.clone());
      return;
    } else if target < 0 {
      return;
    }
    for i in 0..candidates.len() {
      temp.push(candidates[i]);
      backtrack(&candidates[i..], target - candidates[i], result, temp);
      temp.pop();
    }
  }
  backtrack(&candidates, target, &mut result, &mut temp);
  result
}

// Combination Sum II
pub fn combination_sum2(candidates: Vec<i32>, target: i32) -> Vec<Vec<i32>> {
  let mut candidates = candidates.clone();
  candidates.sort();
  let mut result = Vec::new();
  let mut temp = Vec::new();
  fn backtrack(candidates: &Vec<i32>, target: i32, result: &mut Vec<Vec<i32>>, temp: &mut Vec<i32>, current: usize) {
    if target == 0 {
      result.push(temp.clone());
      return;
    } else if target < 0 {
      return;
    }
    for i in current..candidates.len() {
      if i > current && candidates[i] == candidates[i - 1] {
        continue;
      }
      temp.push(candidates[i]);
      backtrack(candidates, target - candidates[i], result, temp, i + 1);
      temp.pop();
    }
  }
  backtrack(&candidates, target, &mut result, &mut temp, 0);
  result
}

// Permutations II
pub fn permute_unique(nums: Vec<i32>) -> Vec<Vec<i32>> {
  let mut nums = nums;
  nums.sort();
  let mut result = Vec::new();
  let mut temp = Vec::new();
  let mut visit = vec![false; nums.len()];
  fn permutations(nums: &[i32], result: &mut Vec<Vec<i32>>, temp: &mut Vec<i32>, visit: &mut Vec<bool>) {
    if temp.len() == nums.len() {
      result.push(temp.clone());
      return;
    }
    for i in 0..nums.len() {
      if visit[i] || i > 0 && !visit[i - 1] && nums[i - 1] == nums[i] {
        continue;
      }
      temp.push(nums[i]);
      visit[i] = true;
      permutations(nums, result, temp, visit);
      temp.pop();
      visit[i] = false;
    }
  }
  permutations(&nums, &mut result, &mut temp, &mut visit);
  result
}

// Merge Intervals
pub fn merge(intervals: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
  let mut intervals = intervals;
  intervals.sort_by(|x, y| x[0].cmp(&y[0]));
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
 
// Insert Interval
pub fn insert(intervals: Vec<Vec<i32>>, new_interval: Vec<i32>) -> Vec<Vec<i32>> {
  let mut result = Vec::new();
  let mut new_interval = new_interval;
  for i in 0..intervals.len() {
    if new_interval[1] < intervals[i][0] {
      result.push(new_interval);
      result.extend_from_slice(&intervals[i..]);
      return result;
    } else if new_interval[0] > intervals[i][1] {
      result.push(intervals[i].clone());
    } else {
      new_interval = vec![new_interval[0].min(intervals[i][0]), new_interval[1].max(intervals[i][1])];
    }
  }
  result.push(new_interval);
  result
}

// Sort Colors
pub fn sort_colors(nums: &mut Vec<i32>) {
  let (mut low, mut mid, mut high) = (0, 0, nums.len() as i32 - 1);
  while mid <= high {
    match nums[mid as usize] {
      0 => {nums.swap(low as usize,mid as usize);low+=1;mid+=1},
      2 => {nums.swap(mid as usize,high as usize);high-=1},
      _ => {mid+=1}
    }
  }
}

// Gas Station
pub fn can_complete_circuit(gas: Vec<i32>, cost: Vec<i32>) -> i32 {
  let mut total = 0;
  let mut current = 0;
  let mut result = 0;
  for i in 0..gas.len() {
    total += gas[i] - cost[i];
    current += gas[i] - cost[i];
    if current < 0 {
      current = 0;
      result = i as i32 + 1;
    }
  }
  if total < 0 {
    return -1;
  }
  result
}

// Implement Queue using Stacks
struct MyQueue {
  input: Vec<i32>,
  output: Vec<i32>,
}
impl MyQueue {
  fn new() -> Self {
    return MyQueue{
      input: Vec::new(),
      output: Vec::new(),
    }
  }
  
  fn push(&mut self, x: i32) {
    self.input.push(x);
  }
  
  fn pop(&mut self) -> i32 {
    self.peek();
    self.output.pop().unwrap()
  }
  
  fn peek(&mut self) -> i32 {
    if self.output.is_empty() {
      loop {
        match self.input.pop() {
          Some(i) => self.output.push(i),
          None => break
        }
      }
    }
    *self.output.last().unwrap()
  }
  
  fn empty(&self) -> bool {
    self.input.is_empty() && self.output.is_empty()
  }
}

// Product of Array Except Self
pub fn product_except_self(nums: Vec<i32>) -> Vec<i32> {
  let mut prefix = 1;
  let mut result = vec![0; nums.len()];
  for i in 0..nums.len() {
    result[i] = prefix;
    prefix *= nums[i];
  }
  let mut postfix = 1;
  for i in (0..nums.len()).rev() {
    result[i] *= postfix;
    postfix *= nums[i];
  }
  result
}

// Intersection of Two Arrays
pub fn intersection(nums1: Vec<i32>, nums2: Vec<i32>) -> Vec<i32> {
  use std::collections::HashSet;
  let mut hash_set = HashSet::new();
  let mut result = Vec::new();
  for i in nums1 {
    hash_set.insert(i);
  }
  for i in nums2{
    if hash_set.contains(&i) {
      result.push(i);
      hash_set.remove(&i);
    }
  }
  result
}

// Backspace String Compare
pub fn get_next_valid_character(str: String, mut end: usize) -> usize {
  let mut backspace_count = 0;
  while end > 0 {
    if let Some(c) = str.chars().nth(end - 1) {
      if c == '#' {
        backspace_count += 1;
      } else if backspace_count > 0 {
        backspace_count -= 1;
      } else {
        break;
      }
    } else {
      break; // Handle the case when 'end' is out of bounds
    }
    end -= 1;
  }
  end
}

pub fn backspace_compare(s: String, t: String) -> bool {
  let mut pS = s.len();
  let mut pT = t.len();
  while pS > 0 || pT > 0 {
    pS = Self::get_next_valid_character(s.clone(), pS);
    pT = Self::get_next_valid_character(t.clone(), pT);
    if pS == 0 && pT == 0 {
      return true;
    } else if pS == 0 || pT == 0 {
      return false;
    } else if s.chars().nth(pS - 1) != t.chars().nth(pT - 1) {
      return false;
    }
    pS -= 1;
    pT -= 1;
  }
  return true;
}

// Backspace String Compare
pub fn get_next_valid_character(str: String, mut end: usize) -> usize {
  let mut backspace_count = 0;
  while end > 0 {
    match str.chars().nth(end - 1) {
      Some(c) if c == '#' => backspace_count += 1,
      _ => {
        if backspace_count > 0 {
          backspace_count -= 1;
        } else {
          break;
        }
      }
    }
    end -= 1;
  }
  end
}

pub fn backspace_compare(s: String, t: String) -> bool {
  let mut pS = s.len();
  let mut pT = t.len();
  while pS > 0 || pT > 0 {
    pS = Self::get_next_valid_character(s.clone(), pS);
    pT = Self::get_next_valid_character(t.clone(), pT);
    if pS == 0 && pT == 0 {
      return true;
    } else if pS == 0 || pT == 0 {
      return false;
    } else if s.chars().nth(pS - 1) != t.chars().nth(pT - 1) {
      return false;
    }
    pS -= 1;
    pT -= 1;
  }
  return true;
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

// Merge sort
pub fn merge_sort(nums: &mut Vec<i32>, start: usize, end: usize) {
  if start < end {
    let mid = start + (end - start) / 2;
    Self::merge_sort(nums, start, mid);
    Self::merge_sort(nums, mid + 1, end);
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

// Quick sort
pub fn quick_sort(nums: &mut Vec<i32>, start: usize, end: usize) {
  if start < end {
    let pivot = Self::partition(nums, start, end);
    Self::quick_sort(nums, start, pivot - 1);
    Self::quick_sort(nums, pivot + 1, end);
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

// Bubble sort
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

// Heap sort
pub fn heapify(nums: &mut Vec<i32>, node: usize, length: usize) {
  let mut largest = node;
  let left = 2 * node + 1;
  let right = 2 * node + 2;
  if left < length && nums[left] > nums[largest] {
    largest = left;
  }
  if right < length && nums[right] > nums[largest] {
    largest = right;
  }
  if largest != node {
    nums.swap(node, largest);
    Self::heapify(nums, largest, length);
  }
}

pub fn heap_sort(nums: &mut Vec<i32>) {
  let n = nums.len();
  for i in (0..n / 2).rev() {
    Self::heapify(nums, i, n);
  }
  for i in (1..n).rev() {
    nums.swap(0, i);
    Self::heapify(nums, 0, i);
  }
}

fn main() {
  let mut nums = vec![1, 9, 8, 20, 15, 17, 5, 4, 8, 3];
  // let length = nums.len();
  // merge_sort(mut nums, 0, length - 1);
  heap_sort(&mut nums);
  // bubble_sort(&mut nums);
  println!("{:?}", nums);
}
