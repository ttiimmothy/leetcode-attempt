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
  while(i < intervals.len()){
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