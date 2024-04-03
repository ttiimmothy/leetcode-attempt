// Container With Most Water
public int maxArea(int[] height) {
  int left = 0, right = height.length - 1;
  int area = 0;
  while(left < right){
    int currentArea = (right - left) * Math.min(height[left], height[right]);
    area = Math.max(area, currentArea);
    if(height[left] < height[right]){
      left++;
    }else{
      right--;
    }
  }
  return area;
}

// 3Sum
public List<List<Integer>> threeSum(int[] nums) {
  List<List<Integer>> result = new ArrayList();
  Arrays.sort(nums);
  for(int i = 0; i < nums.length - 2; i++){
    if(i > 0 && nums[i] == nums[i - 1]){
      continue; // skip the same result array occurred
    }
    int left = i + 1;
    int right = nums.length - 1;
    while(left < right){
      int threeSum = nums[i] + nums[left] + nums[right];
      if(threeSum < 0){
        left++;
      }else if(threeSum > 0){
        right--;
      }else{
        result.add(Arrays.asList(nums[i], nums[left], nums[right]));
        left++;
        right--;
        while(nums[left] == nums[left - 1] && left < right){
          left++; // skip the same result array occurred
        }
        while(left < right && nums[right] == nums[right + 1]){
          right--;
        }
      }
    }
  }
  return result;
}

// Valid Parentheses
public boolean isValid(String s) {
  Stack<Character> stack = new Stack();
  for(char i : s.toCharArray()){
    if(i == '('){
      stack.push(')');
    }else if(i == '{'){
      stack.push('}');
    }else if(i == '['){
      stack.push(']');
    }else if(stack.isEmpty() || i != stack.pop()){ // the order is important, stack.isEmpty() needs to be the first condition
      return false;
    }
  }
  return stack.isEmpty();
}

// Combination Sum
public List<List<Integer>> combinationSum(int[] candidates, int target) {
  List<List<Integer>> result = new ArrayList();
  List<Integer> subList = new ArrayList();
  backtrack(0, result, subList, candidates, target);
  return result;
}

public void backtrack(int index, List<List<Integer>> result, List<Integer> subList, int[] candidates, int target) {
  if (target == 0) {
    result.add(new ArrayList(subList));
    return;
  }
  if (index >= candidates.length || target < 0) {
    return;
  }
  for (int i = index; i < candidates.length; i++) {
    if (candidates[i] <= target) {
      // backtracking
      subList.add(candidates[i]);
      backtrack(i, result, subList, candidates, target - candidates[i]);
      subList.remove(subList.size() - 1);
    }
  }
}

// Combination Sum
public List<List<Integer>> combinationSum(int[] candidates, int target) {
  List<List<Integer>> result = new ArrayList();
  List<Integer> subList = new ArrayList();
  backtrack(0, result, subList, candidates, target, 0);
  return result;
}

public void backtrack(int i, List<List<Integer>> result, List<Integer> subList, int[] candidates, int target, int total) {
  if(total == target){
    result.add(new ArrayList(subList));
    return;
  }
  if(i >= candidates.length || total > target){
    return;
  }
  subList.add(candidates[i]);
  backtrack(i, result, subList, candidates, target, total + candidates[i]);
  subList.remove(subList.size() - 1);
  backtrack(i + 1, result, subList, candidates, target, total);
}

// Combination Sum II
public List<List<Integer>> combinationSum2(int[] candidates, int target) {
  Arrays.sort(candidates);
  List<List<Integer>> result = new ArrayList();
  backtrack(candidates, target, 0, new ArrayList(), result);
  return result;
}

public void backtrack(int[] candidates, int target, int index, List<Integer> list, List<List<Integer>> result) {
  if (target == 0) {
    result.add(new ArrayList(list));
    return;
  }
  if (index >= candidates.length || target < 0) {
    return;
  }
  for (int i = index; i < candidates.length; i++) {
    if(i > index && candidates[i] == candidates[i - 1]){
      continue; // skip duplicate combinations
    }
    if (candidates[i] <= target) {
      list.add(candidates[i]);
      backtrack(candidates, target - candidates[i], i + 1, list, result); // each number in candidates may only be used once 
      list.remove(list.size() - 1);
    }
  }
}

// Permutations
public List<List<Integer>> permute(int[] nums) {
  List<List<Integer>> result = new ArrayList();
  backtrack(nums, result, new ArrayList());
  return result;
}

public void backtrack(int[] nums, List<List<Integer>> result, List<Integer> subList) {
  if(subList.size() == nums.length){
    result.add(new ArrayList(subList));
  }else{
    for(int num:nums){
      if(subList.contains(num)){
        continue;
      }
      subList.add(num);
      backtrack(nums, result, subList);
      subList.remove(subList.size() - 1);
    }
  }
}

// Merge Intervals
public int[][] merge(int[][] intervals) {
  List<int[]> result = new ArrayList();
  Arrays.sort(intervals,(a,b)->a[0]-b[0]);
  result.add(intervals[0]);
  for(int i = 0; i < intervals.length; i++){
    int[] interval = intervals[i];
    if(interval[0] <= result.get(result.size() - 1)[1]){
      result.get(result.size() - 1)[1] = Math.max(result.get(result.size() - 1)[1], interval[1]);
    }else{
      result.add(interval);
    }
  }
  return result.toArray(new int[result.size()][]);
}

// Merge Intervals
public int[][] merge(int[][] intervals) {
  Arrays.sort(intervals, (a, b) -> Integer.compare(a[0], b[0]));
  List<int[]> merged = new ArrayList();
  int[] mergedInterval = intervals[0];
  for (int i = 1; i < intervals.length; i++) {
    int[] interval = intervals[i];
    if (interval[0] <= mergedInterval[1]) {
      mergedInterval[1] = Math.max(mergedInterval[1], interval[1]);
    } else {
      merged.add(mergedInterval);
      mergedInterval = interval; 
    }
  }
  merged.add(mergedInterval);
  return merged.toArray(new int[merged.size()][]);        
}

// Insert Interval
public int[][] insert(int[][] intervals, int[] newInterval) {
  List<int[]> result = new ArrayList();
  int i = 0;
  while(i < intervals.length && intervals[i][1] < newInterval[0]){
    result.add(intervals[i]);
    i++;
  }
  while(i < intervals.length && intervals[i][0] <= newInterval[1]){
    newInterval = new int[]{Math.min(newInterval[0], intervals[i][0]), Math.max(newInterval[1], intervals[i][1])};
    i++;
  }
  result.add(newInterval);
  while(i < intervals.length){
    result.add(intervals[i]);
    i++;
  }
  return result.toArray(new int[result.size()][]);
}

// Insert Interval
public int[][] insert(int[][] intervals, int[] newInterval) {
  List<int[]> result = new ArrayList();
  for(int[] interval:intervals){
    if(newInterval[1] < interval[0]){
      result.add(newInterval);
      newInterval = interval;
    }else if(newInterval[0] > interval[1]){
      result.add(interval);
    }else{
      newInterval = new int[]{Math.min(newInterval[0], interval[0]), Math.max(newInterval[1], interval[1])};
    }
  }
  result.add(newInterval);
  return result.toArray(new int[result.size()][]);
}

// Sort Colors
public void sortColors(int[] nums) {
  int low = 0, mid = 0, high = nums.length - 1;
  while (mid <= high) {
    if (nums[mid] == 0) {
      swap(nums, low, mid);
      low++;
      mid++;
    } else if (nums[mid] == 1) {
      mid++;
    } else {
      swap(nums, mid, high);
      high--;
    }
  }
}

public void swap(int[] nums, int i, int j) {
  int temp = nums[i];
  nums[i] = nums[j];
  nums[j] = temp;
}

// Best Time to Buy and Sell Stock
public int maxProfit(int[] prices) {
  int difference = 0, left = 0, right = 0;
  while(right > left && right < prices.length){
    if(prices[left] > prices[right]){
      left = right;
    }else if(prices[right] - prices[left] > difference){
      difference = prices[right] - prices[left];
    }
    right += 1;
  }
  return difference;
}

// Best Time to Buy and Sell Stock
public int maxProfit(int[] prices) {
  int overallProfit = 0;
  int minPrice = prices[0];
  for(int i:prices){
    if(i > minPrice){
      overallProfit = Math.max(i - minPrice, overallProfit);
    }else{
      minPrice = i;
    }
  }
  return overallProfit;
}

// Best Time to Buy and Sell Stock
public int maxProfit(int[] prices) {
  int overallProfit = 0;
  int maxCurrent = 0;
  for(int i = 1; i < prices.length; i++){
    maxCurrent = Math.max(0, maxCurrent += prices[i] - prices[i - 1]);
    overallProfit = Math.max(overallProfit, maxCurrent);
  }
  return overallProfit;
}

// Gas Station
public int canCompleteCircuit(int[] gas, int[] cost) {
  int totalGas = 0, totalCost = 0;
  for(int i = 0; i < gas.length; i++){
    totalGas += gas[i];
    totalCost += cost[i];
  }
  if(totalGas < totalCost){
    return -1;
  }
  int total = 0, result = 0;
  for(int i = 0; i < gas.length; i++){
    total += gas[i] - cost[i];
    if(total < 0){
      result = i + 1;
      total = 0;
    }
  }
  return result;
}

// Evaluate Reverse Polish Notation
public int evalRPN(String[] tokens) {
  Stack<Integer> stack = new Stack();
  for(String i:tokens){
    if(i.equals("+")){
      stack.push(stack.pop() + stack.pop());
    }else if(i.equals("-")){
      int right = stack.pop();
      int left = stack.pop();
      stack.push(left - right);
    }else if(i.equals("*")){
      stack.push(stack.pop() * stack.pop());
    }else if(i.equals("/")){
      int right = stack.pop();
      int left = stack.pop();
      stack.push(left / right);
    }else{
      stack.push(Integer.parseInt(i));
    }
  }
  return stack.pop();
}

// Two Sum II - Input Array Is Sorted
public int[] twoSum(int[] numbers, int target) {
  int left = 0, right = numbers.length - 1;
  while(numbers[left] + numbers[right] != target){
    if(numbers[left] + numbers[right] < target){
      left++;
    }else{
      right--;
    }
  }
  return new int[]{left + 1,right + 1};
}

// Majority Element
public int majorityElement(int[] nums) {
  int result = 0, freq = 0;
  for(int i:nums){
    if(freq == 0){
      result = i;
      freq = 1;
    }else if(i == result){
      freq++;
    }else{
      freq--;
    }
  }
  return result;             
}

// Majority Element
public int majorityElement(int[] nums) {
  int n = nums.length;
  Map<Integer,Integer> map = new HashMap();
  for(int i:nums){
    map.put(i,map.getOrDefault(i,0) + 1);
  }
  n = n / 2;
  for(Map.Entry<Integer,Integer> entry:map.entrySet()){
    if(entry.getValue() > n){
      return entry.getKey();
    }
  }
  return 0;
}

// Contains Duplicate
public boolean containsDuplicate(int[] nums) {
  Set<Integer> set = new HashSet();
  for(int i:nums){
    if(set.contains(i)){
      return true;
    }
    set.add(i);
  }
  return false;
}

// Implement Queue using Stacks
class MyQueue {
  Stack<Integer> input = new Stack();
  Stack<Integer> output = new Stack();
  public void push(int x) {
    input.push(x);
  }

  public int pop() {
    peek();
    return output.pop();
  }

  public int peek() {
    if(output.isEmpty()){
      while(!input.isEmpty()){
        output.push(input.pop());
      }
    }
    return output.peek();
  }

  public boolean empty() {
    return input.isEmpty() && output.isEmpty();
  }
}

// Product of Array Except Self
public int[] productExceptSelf(int[] nums) {
  int[] result = new int[nums.length];
  int prefix = 1;
  for(int i = 0; i < nums.length; i++){
    result[i] = prefix;
    prefix *= nums[i];
  }
  int postfix = 1;
  for(int i = nums.length - 1; i >= 0; i--){
    result[i] *= postfix;
    postfix *= nums[i];
  }
  return result;
}

// Intersection of Two Arrays
public int[] intersection(int[] nums1, int[] nums2) {
  Set<Integer> set = new HashSet();
  for(int i:nums1){
    set.add(i);
  }
  List<Integer> list = new ArrayList();
  for (int i:nums2){
    if(set.contains(i)){
      set.remove(i);
      list.add(i);
    }
  }
  int[] result = new int[list.size()];
  for(int i = 0; i < result.length; i++){
    result[i] = list.get(i);
  }
  return result;
}

// Backspace String Compare
public boolean backspaceCompare(String s, String t) {
  int ps = s.length() - 1;
  int pt = t.length() - 1;
  // or condition is to ensure that either one is smaller than 0 will still continue the checking
  while(ps >= 0 || pt >= 0){
    ps = findValidCharIndex(s, ps);
    pt = findValidCharIndex(t, pt);
    if(ps < 0 && pt < 0){
      return true;
    }else if(ps < 0 || pt < 0){
      return false;
    }else if(s.charAt(ps) != t.charAt(pt)){
      return false;
    }
    ps--;
    pt--;
  }
  return true;
}

public int findValidCharIndex(String str, int end){
  int backspaceNum = 0;
  while(end >= 0){
    if(str.charAt(end) == '#'){
      backspaceNum++;
    }else if(backspaceNum > 0){
      backspaceNum--;
    }else{
      break;
    }
    end--;
  }
  return end;
}

// Sort an Array
public int[] sortArray(int[] nums) {
  mergeSort(nums, 0, nums.length - 1);
  return nums;
}

public void mergeSort(int[] nums, int low, int high) {
  if (low < high) {
    int mid = low + (high - low) / 2;
    mergeSort(nums, low, mid);
    mergeSort(nums, mid + 1, high);
    int n1 = mid - low + 1;
    int n2 = high - mid;
    int[] leftNums = new int[n1];
    int[] rightNums = new int[n2];
    for (int i = 0; i < n1; i++) {
      leftNums[i] = nums[low + i];
    }
    for (int i = 0; i < n2; i++) {
      rightNums[i] = nums[mid + 1 + i]; // avoid missing the last element of the input nums array by adding 1 to the index
    }
    int i = 0, j = 0, k = low;
    while (i < n1 && j < n2) {
      if (leftNums[i] < rightNums[j]) {
        nums[k] = leftNums[i];
        i++;
      }else{
        nums[k] = rightNums[j];
        j++;
      }
      k++;
    }
    while (i < n1) {
      nums[k] = leftNums[i];
      i++;
      k++;
    }
    while (j < n2) {
      nums[k] = rightNums[j];
      j++;
      k++;
    }
  }
}

// Sort an Array
public int[] sortArray(int[] nums) {
  heapSort(nums);
  return nums;
}

public void heapify(int[] nums, int maxNode, int length) {
  int largest = maxNode;
  int left = 2 * maxNode + 1;
  int right = 2 * maxNode + 2;
  if (left  < length && nums[left] > nums[largest]) {
    largest = left;
  }
  if(right < length && nums[right] > nums[largest]){
    largest = right;
  }
  if (largest != maxNode) {
    int temp = nums[largest];
    nums[largest] = nums[maxNode];
    nums[maxNode] = temp;
    heapify(nums, largest, length); // heapify the affected sub tree
  }
}

public void heapSort(int[] nums){
  int n = nums.length;
  // create max heap
  for (int i = n / 2 - 1; i >= 0; i--) {
    heapify(nums, i, n);
  }
  // extract the maximum node one by one
  // starting from n - 1 because n index is out of the array bound
  for (int i = n - 1; i > 0; i--) {
    int temp = nums[0];
    nums[0] = nums[i];
    nums[i] = temp;
    heapify(nums, 0, i);
  }
}

// Count Elements With Maximum Frequency
public int maxFrequencyElements(int[] nums) {
  int maxFrequency = 0, result = 0;
  Map<Integer,Integer> map = new HashMap();
  for(int i = 0; i < nums.length; i++){
    map.put(nums[i],map.getOrDefault(nums[i], 0) + 1);
  }
  for(int value:map.values()){
    maxFrequency = Math.max(maxFrequency,value);
  }
  for(int value:map.values()){
    if(value == maxFrequency){
      result += maxFrequency;
    }
  }
  return result;
}

// Quick sort, time complexity O(nlogn), memory complexity O(1), may cause time limit error
public void quickSort(int[] nums, int low, int high) {
  if(low < high){
    int pivot = partition(nums, low, high);
    quickSort(nums, low, pivot - 1);
    quickSort(nums, pivot + 1, high);
  }
}

public int partition(int[] nums, int low, int high) {
  int start = low - 1;
  int pivot = nums[high];
  for(int i = low; i < high; i++){
    if(nums[i] < pivot){
      start++;
      int temp = nums[i];
      nums[i] = nums[start];
      nums[start] = temp;
    }
  }
  int temp = nums[start + 1];
  nums[start + 1] = nums[high];
  nums[high] = temp;
  return start + 1;
}

// Merge sort, time complexity O(nlogn), memory complexity O(n)
public void mergeSort(int[] nums, int low, int high) {
  if (low < high) {
    int mid = low + (high - low) / 2;
    mergeSort(nums, low, mid);
    mergeSort(nums, mid + 1, high);
    int n1 = mid - low + 1;
    int n2 = high - mid;
    int[] leftArr = new int[n1];
    int[] rightArr = new int[n2];
    for (int i = 0; i < n1; i++) {
      leftArr[i] = nums[low + i];
    }
    for (int i = 0; i < n2; i++) {
      rightArr[i] = nums[mid + 1 + i]; // avoid missing the last element of the input nums array by adding 1 to the index
    }
    int i = 0, j = 0, k = low;
    while (i < n1 && j < n2) {
      if (leftArr[i] < rightArr[j]) {
        nums[k] = leftArr[i];
        i++;
      }else{
        nums[k] = rightArr[j];
        j++;
      }
      k++;
    }
    while (i < n1) {
      nums[k] = leftArr[i];
      i++;
      k++;
    }
    while (j < n2) {
      nums[k] = rightArr[j];
      j++;
      k++;
    }
  }
}

// Bubble sort, time complexity O(n^2), memory complexity O(1)
public void bubbleSort(int[] nums) {
  int n = nums.length;
  boolean swapped;
  for (int i = 0; i < n - 1; i++) {
    swapped = false;
    for (int j = 0; j < n - i - 1; j++) {
      if (nums[j] > nums[j + 1]) {
        int temp = nums[j];
        nums[j] = nums[j + 1];
        nums[j + 1] = temp;
        swapped = true;
      }
    }
    if(swapped == false){
      break; // if there is no swapping in the inner loop, that means the sorting already finishes, so breaks the loop
    }
  }
}

// Heap sort, time complexity O(nlogn), memory complexity O(1)
public void heapify(int[] nums, int maxNode, int length) {
  int largest = maxNode;
  int left = 2 * maxNode + 1;
  int right = 2 * maxNode + 2;
  if (left  < length && nums[left] > nums[largest]) {
    largest = left;
  }
  if (right < length && nums[right] > nums[largest]) {
    largest = right;
  }
  if (largest != maxNode) {
    int temp = nums[largest];
    nums[largest] = nums[maxNode];
    nums[maxNode] = temp;
    heapify(nums, largest, length); // heapify the affected sub tree
  }
}

public void heapSort(int[] nums) {
  int n = nums.length;
  // create max heap
  for (int i = n / 2 - 1; i >= 0; i--) {
    heapify(nums, i, n);
  }
  // extract the maximum node one by one
  // starting from n - 1 because n index is out of the array bound
  for (int i = n - 1; i > 0; i--) {
    int temp = nums[0];
    nums[0] = nums[i];
    nums[i] = temp;
    heapify(nums, 0, i);
  }
}

// Selection sort, time complexity: O(n^2), memory complexity: O(1)
public void selectionSort(int[] nums) {
  for (int i = 0; i < nums.length; i++) {
    int min = i;
    for (int j = i + 1; j < nums.length; j++) {
      if(nums[j] < nums[min]){
        min = j;
      }
    }
    int temp = nums[i];
    nums[i] = nums[min];
    nums[min] = temp;
  }
}
