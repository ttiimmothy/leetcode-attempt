// 1
// Two Sum
public int[] twoSum(int[] nums, int target){
  Map<Integer, Integer> preMap = new HashMap();
  for(var i = 0; i < nums.length; i++){
    var diff = target - nums[i];
    if(preMap.containsKey(diff)){
      return new int[]{preMap.get(diff),i};
    }
    preMap.put(nums[i],i);
  }
  return new int[]{};
}

// 3
// Longest Substring Without Repeating Characters
public int lengthOfLongestSubstring(String s) {
  int left = 0, result = 0;
  Set<Character> set = new HashSet<>();
  for (int right = 0; right < s.length(); right++) {
    char a = s.charAt(right);
    while (set.contains(a)) {
      set.remove(s.charAt(left));
      left++;
    }
    set.add(a);
    result = Math.max(result, right - left + 1);
  }
  return result;
}

// 5
// Longest Palindromic Substring
public String longestPalindrome(String s) {
  String result = "";
  int resultLength = 0;
  for (int i = 0; i < s.length(); i++) {
    int left = i, right = i;
    while (left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
      if (right - left + 1 > resultLength) {
        resultLength = right - left + 1;
        result = s.substring(left, right+1);
      }
      left--;
      right++;
    }
    left = i;
    right = i + 1;
    while (left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
      if (right - left + 1 > resultLength) {
        resultLength = right - left + 1;
        result = s.substring(left, right+1);
      }
      left--;
      right++;
    }
  }
  return result;
}

// 8
// String to Integer (atoi) 
public int myAtoi(String s) {
  int i = 0, result = 0, sign = 1;
  while (i < s.length() && s.charAt(i) == ' ') {
    i++;
  }
  if (i < s.length() && (s.charAt(i) == '+' || s.charAt(i) == '-')) {
    if (s.charAt(i) == '-') {
      sign = -1;
    }
    i++;
  }
  while(i < s.length() && s.charAt(i) >= '0' && s.charAt(i) <= '9') {
    if (result > Integer.MAX_VALUE / 10 || (result == Integer.MAX_VALUE / 10 && s.charAt(i) - '0' > Integer.MAX_VALUE % 10)) {
      if (sign < 0) {
        return Integer.MIN_VALUE;
      } else {
        return Integer.MAX_VALUE;
      }
    }
    result = result * 10 + (s.charAt(i) - '0');
    i++;
  }
  return result*sign;
}

// 11
// Container With Most Water
public int maxArea(int[] height) {
  int left = 0, right = height.length - 1;
  int area = 0;
  while (left < right) {
    int currentArea = (right - left) * Math.min(height[left], height[right]);
    area = Math.max(area, currentArea);
    if (height[left] < height[right]) {
      left++;
    } else {
      right--;
    }
  }
  return area;
}

// 15
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

// 19
// Remove Nth Node From End of List
public ListNode removeNthFromEnd(ListNode head, int n) {
  ListNode result = new ListNode(0, head);
  ListNode prev = result, current = head;
  for (int i = 0; i < n; i++) {
    if (current != null) {
      current = current.next;
    }
  }
  while (current != null) {
    prev = prev.next;
    current = current.next;
  }
  prev.next = prev.next.next;
  return result.next;
}

// 20
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

// 21
// Merge Two Sorted Lists
public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
  ListNode dummy = new ListNode();
  ListNode result = dummy;
  while (list1 != null && list2 != null) {
    if (list1.val < list2.val) {
      result.next = list1;
      list1 = list1.next;
    } else {
      result.next = list2;
      list2 = list2.next;
    }
    result = result.next;
  }
  if (list1 != null) {
    result.next = list1;
  } else {
    result.next = list2;
  }
  return dummy.next;
}

// 21-1
// Merge Two Sorted Lists
public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
  if (list1 == null) return list2;
  if (list2 == null) return list1;
  if (list1.val < list2.val) {
    list1.next = mergeTwoLists(list1.next, list2);
    return list1;
  } else {
    list2.next = mergeTwoLists(list1, list2.next);
    return list2;
  }
}

// 33
// Search in Rotated Sorted Array
public int search(int[] nums, int target) {
  int left = 0, right = nums.length - 1;
  while (left <= right) {
    int mid = left + (right - left) / 2;
    if (nums[mid] == target) {
      return mid;
    } 
    if (nums[left] <= nums[mid]) {
      if (nums[left] <= target && target < nums[mid]) {
        right = mid - 1;
      } else {
        left = mid + 1;
      }
    } else {
      if (nums[mid] < target && target <= nums[right]) {
        left = mid + 1;
      } else {
        right = mid - 1;
      }
    }
  }
  return -1;
}

// 39
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
  } else if (target < 0) {
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

// 39-1
// Combination Sum
public List<List<Integer>> combinationSum(int[] candidates, int target) {
  List<List<Integer>> result = new ArrayList();
  List<Integer> subList = new ArrayList();
  backtrack(0, result, subList, candidates, target, 0);
  return result;
}

public void backtrack(int i, List<List<Integer>> result, List<Integer> subList, int[] candidates, int target, int total) {
  if (total == target) {
    result.add(new ArrayList(subList));
    return;
  }
  if (i >= candidates.length || total > target) {
    return;
  }
  subList.add(candidates[i]);
  backtrack(i, result, subList, candidates, target, total + candidates[i]);
  subList.remove(subList.size() - 1);
  backtrack(i + 1, result, subList, candidates, target, total);
}

// 40
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
  } else if (target < 0) {
    return;
  }
  for (int i = index; i < candidates.length; i++) {
    if(i > index && candidates[i] == candidates[i - 1]){
      continue; // skip duplicate combinations
    }
    list.add(candidates[i]);
    backtrack(candidates, target - candidates[i], i + 1, list, result); // each number in candidates may only be used once 
    list.remove(list.size() - 1);
  }
}

// 46
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

// 46-1
// Permutations
public List<List<Integer>> permute(int[] nums) {
  List<List<Integer>> result = new ArrayList<>();
  permute(nums, new ArrayList<>(), result);
  return result;
}

public void permute(int[] nums, List<Integer> current, List<List<Integer>> result) {
  if (current.size() == nums.length) {
    result.add(current);
    return;
  }
  for (int i = 0; i < nums.length; i++) {
    if (current.contains(nums[i])) continue;
    List<Integer> newCurrent = new ArrayList<>(current);
    newCurrent.add(nums[i]);
    permute(nums, newCurrent, result);
    // don't need to run list.remove()  
  }
}

// 47
// Permutations II
public List<List<Integer>> permuteUnique(int[] nums) {
  Arrays.sort(nums);
  List<List<Integer>> result = new ArrayList();
  backtrack(nums, result, new ArrayList(), new boolean[nums.length]);
  return result;
}

public void backtrack(int[] nums, List<List<Integer>> result, List<Integer> temp, boolean[] visit) {
  if (temp.size() == nums.length) {
    result.add(new ArrayList(temp));
    return;
  }
  for (int i = 0; i < nums.length; i++){
    if (visit[i] || i > 0 && !visit[i - 1] && nums[i] == nums[i - 1]) {
      continue;
    }
    temp.add(nums[i]);
    visit[i] = true;
    backtrack(nums, result, temp, visit);
    temp.remove(temp.size() - 1);
    visit[i] = false;
  }
}

// 49
// Group Anagrams
public List<List<String>> groupAnagrams(String[] strs) {
  Map<String, List<String>> map = new HashMap<>();
  for (String str:strs) {
    char[] charArray = str.toCharArray();
    Arrays.sort(charArray);
    String sortedWords = new String(charArray);
    if (!map.containsKey(sortedWords)) {
      map.put(sortedWords, new ArrayList<>());
    }
    map.get(sortedWords).add(str);
  }
  return new ArrayList<>(map.values());
}

// 56
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

// 56-1
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

// 57
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

// 57-1
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

75
// Sort Colors
public void sortColors(int[] nums) {
  int low = 0, mid = 0, high = nums.length - 1;
  while (mid <= high) {
    if (nums[mid] == 0) {
      int temp = nums[low];
      nums[low] = nums[mid];
      nums[mid] = temp;
      low++;
      mid++;
    } else if (nums[mid] == 1) {
      mid++;
    } else {
      int temp = nums[mid];
      nums[mid] = nums[high];
      nums[high] = temp;
      high--;
    }
  }
}

// 78
// Subsets
public List<List<Integer>> subsets(int[] nums) {
  List<List<Integer>> result = new ArrayList<>();
  backtrack(nums, result, new ArrayList<>(), 0);
  return result;
}

public void backtrack(int[] nums, List<List<Integer>> result, List<Integer> temp, int index) {
  result.add(new ArrayList(temp));
  for (int i = index; i < nums.length; i++) {
    temp.add(nums[i]);
    backtrack(nums, result, temp, i + 1);
    temp.remove(temp.size() - 1);
  }
}

// 90
// Subsets II
public List<List<Integer>> subsetsWithDup(int[] nums) {
  Arrays.sort(nums);
  List<List<Integer>> result = new ArrayList<>();
  backtrack(nums, result, new ArrayList<>(), 0);
  return result;
}

public void backtrack(int[] nums, List<List<Integer>> result, List<Integer> temp, int index) {
  result.add(new ArrayList(temp));
  for (int i = index; i < nums.length; i++) {
    if (i > index && nums[i - 1] == nums[i]) {
      continue;
    }
    temp.add(nums[i]);
    backtrack(nums, result, temp, i + 1);
    temp.remove(temp.size() - 1);
  }
}

// 92
// Reverse Linked List II
public ListNode reverseBetween(ListNode head, int left, int right) {
  ListNode result = new ListNode(0, head);
  ListNode leftPrev = result;
  ListNode current = head;
  for (int i = 0; i < left - 1; i++) {
    leftPrev = current;
    current = current.next;
  }
  ListNode prev = null;
  for (int i = 0; i < right - left + 1; i++) {
    ListNode next = current.next;
    current.next = prev;
    prev = current;
    current = next;
  }
  leftPrev.next.next = current;
  leftPrev.next = prev;
  return result.next;
}

// 102
// Binary Tree Level Order Traversal
public List<List<Integer>> levelOrder(TreeNode root) {
  List<List<Integer>> result = new ArrayList<>();
  Queue<TreeNode> queue = new LinkedList<>();
  if (root == null) {
    return result;
  }
  queue.add(root);
  while (!queue.isEmpty()) {
    int size = queue.size();
    List<Integer> temp = new ArrayList<>();
    for (int i = 0; i < size; i++) {
      TreeNode node = queue.remove();
      temp.add(node.val);
      if (node.left != null) {
        queue.add(node.left);
      }
      if (node.right != null) {
        queue.add(node.right);
      }
    }
    result.add(temp);
  }
  return result;
}

// 102-1
// Binary Tree Level Order Traversal
public List<List<Integer>> levelOrder(TreeNode root) {
  List<List<Integer>> result = new ArrayList<>();
  dfs(root, result, 0);
  return result;
}

public void dfs(TreeNode node, List<List<Integer>> result, int level) {
  if (node == null) {
    return;
  }
  if (result.size() == level) {
    result.add(new ArrayList<>());
  }
  result.get(level).add(node.val);
  dfs(node.left, result, level + 1);
  dfs(node.right, result, level + 1);
}

// 104
// Maximum Depth of Bianry Tree
public int maxDepth(TreeNode root) {
  if (root == null) {
    return 0;
  }
  int left = maxDepth(root.left);
  int right = maxDepth(root.right);
  return Math.max(left, right) + 1;
}

// 104-1
// Maximum Depth of Bianry Tree
public int maxDepth(TreeNode root) {
  if (root == null) {
    return 0;
  }
  Queue<TreeNode> queue = new LinkedList<>();
  queue.add(root);
  int result = 0;
  while (!queue.isEmpty()) {
    int size = queue.size();
    for (int i = 0; i < size; i++) {
      TreeNode node = queue.remove();
      if (node.left != null) {
        queue.add(node.left);
      }
      if (node.right != null) {
        queue.add(node.right);
      }
    }
    result++;
  }
  return result;
}

// 105
// Construct Binary Tree from Preorder and Inorder Traversal
public TreeNode buildTree(int[] preorder, int[] inorder) {
  // if (preorder == null || inorder == null) // wrong condition
  if (preorder == null || inorder == null || preorder.length == 0 || inorder.length == 0) {
    return null;
  }
  TreeNode root = new TreeNode(preorder[0]);
  int m = 0;
  for (int i = 0; i < inorder.length; i++) {
    if (inorder[i] == preorder[0]) {
      m = i;
    }
  }
  root.left = buildTree(Arrays.copyOfRange(preorder, 1, m + 1), Arrays.copyOfRange(inorder, 0, m));
  root.right = buildTree(Arrays.copyOfRange(preorder, m + 1, preorder.length), Arrays.copyOfRange(inorder, m + 1, inorder.length));
  return root;
}

// 105-1
// Construct Binary Tree from Preorder and Inorder Traversal
public int current = 0;
public TreeNode buildTree(int[] preorder, int[] inorder) {
  Map<Integer, Integer> inOrderMap = new HashMap<>();
  for (int i = 0; i < inorder.length; i++) {
    inOrderMap.put(inorder[i], i);
  }
  return build(0, preorder.length - 1, preorder, inOrderMap);
}

public TreeNode build(int l, int r, int[] preorder, Map<Integer, Integer> map) {
  if (l > r) {
    return null;
  }
  int rootVal = preorder[current++];
  TreeNode root = new TreeNode(rootVal);
  int m = map.get(rootVal);
  root.left = build(l, m - 1, preorder, map);
  root.right = build(m + 1, r, preorder, map);
  return root;
}

// 110
// Balanced Binary Tree
public boolean isBalanced(TreeNode root) {
  return dfs(root) != -1;
}

public int dfs(TreeNode node) {
  if (node == null) {
    return 0;
  }
  int left = dfs(node.left);
  int right = dfs(node.right);
  if (left == -1 || right == -1 || Math.abs(left - right) > 1) {
    return -1;
  }
  return Math.max(left, right) + 1;
}

// 121
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

// 121-1
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

// 121-2
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

// 125
// Valid Palindrome
public boolean isPalindrome(String s) {
  int left = 0, right = s.length() - 1;
  while (left < right) {
    char charL = s.charAt(left);
    char charR = s.charAt(right);
    if (!Character.isLetterOrDigit(charL)) {
      left++;
      continue;
    }
    if (!Character.isLetterOrDigit(charR)) {
      right--;
      continue;
    }
    if (Character.toLowerCase(charL) != Character.toLowerCase(charR)) {
      return false;
    }
    left++;
    right--;
  }
  return true;
}

// 131
// Palindrome Patitioning
public List<List<String>> partition(String s) {
  List<List<String>> result = new ArrayList<>();
  backtrack(result, s, new ArrayList<>(), 0);
  return result;
}

public void backtrack(List<List<String>> result, String s, List<String> temp, int index) {
  if (index == s.length()) {
    result.add(new ArrayList(temp));
  }
  for (int i = index; i < s.length(); i++) {
    if (validPalindrome(s, index, i)) {
      temp.add(s.substring(index, i + 1));
      backtrack(result, s, temp, i + 1);
      temp.remove(temp.size() - 1);
    }
  }
}

public boolean validPalindrome(String s, int start, int end) {
  while (start < end) {
    if (s.charAt(start) != s.charAt(end)) {
      return false;
    }
    start++;
    end--;
  }
  return true;
}

// 134
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

// 141
// Linked List Cycle
public boolean hasCycle(ListNode head) {
  ListNode slow = head, fast = head;
  while (fast != null && fast.next != null) {
    slow = slow.next;
    fast = fast.next.next;
    if (slow == fast) {
      return true;
    }
  }
  return false;
}

// 146
// LRU Cache
class ListNode {
  int key, val;
  ListNode prev, next;
  public ListNode(int key, int val) {
    this.key = key;
    this.val = val;
  }
}

class LRUCache {
  Map<Integer, ListNode> cache;
  ListNode left, right;
  int cacheCapacity;
  public LRUCache(int capacity) {
    this.cache = new HashMap<>();
    this.left = new ListNode(0, 0);
    this.right = new ListNode(0, 0);
    this.left.next = this.right;
    this.right.prev = thisleft;
    this.cacheCapacity = capacity;
  }

  public int get(int key) {
    if (cache.containsKey(key)) {
      remove(cache.get(key));
      insert(cache.get(key));
      return cache.get(key).val;
    }
    return -1;
  }

  public void put(int key, int value) {
    if (cache.containsKey(key)) {
      remove(cache.get(key));
    }
    cache.put(key, new ListNode(key, value));
    insert(cache.get(key));
    if (cache.size() > cacheCapacity) {
      ListNode lru = left.next;
      remove(lru);
      cache.remove(lru.key);
    }
  }

  public void remove(ListNode node) {
    ListNode prev = node.prev, next = node.next;
    prev.next = next;
    next.prev = prev;
  }

  public void insert(ListNode node) {
    ListNode prev = right.prev, next = right;
    prev.next = node;
    next.prev = node;
    node.prev = prev;
    node.next = next;
  }
}

// 150
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
  return stack.get(0);
}

// 155
// Min Stack
class MinStack {
  Stack<Integer> stack;
  Stack<Integer> minStack;
  public MinStack() {
    this.stack = new Stack<>();
    this.minStack = new Stack<>();
  }

  public void push(int val) {
    stack.push(val);
    if (!minStack.isEmpty()) {
      val = Math.min(val, minStack.peek());
    }
    minStack.push(val);
  }

  public void pop() {
    stack.pop();
    minStack.pop();
  }

  public int top() {
    return stack.peek();
  }

  public int getMin() {
    return minStack.peek();
  }
}

// 167
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

// 169
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

// 169-1
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

// 199
// Binary Tree Right Side View
public List<Integer> rightSideView(TreeNode root) {
  if (root == null) {
    return new ArrayList();
  }
  LinkedList<TreeNode> q = new LinkedList<>();
  List<Integer> result = new ArrayList<>();
  q.add(root);
  while (!q.isEmpty()) {
    int size = q.size();
    result.add(q.getLast().val);
    for (int i = 0; i < size; i++) {
      // remove is getting the first element
      TreeNode node = q.remove();
      if (node.left != null) {
        q.add(node.left);
      }
      if (node.right != null) {
        q.add(node.right);
      }
    }
  }
  return result;
}

// 199-1
// Binary Tree Right Side View
public List<Integer> rightSideView(TreeNode root) {
  if (root == null) {
    return new ArrayList();
  }
  Queue<TreeNode> q = new LinkedList<>();
  List<Integer> result = new ArrayList<>();
  q.add(root);
  while (!q.isEmpty()) {
    int size = q.size();
    for (int i = 0; i < size; i++) {
      // remove is getting the first element
      TreeNode node = q.remove();
      if (i == size - 1) {
        result.add(node.val);
      }
      if (node.left != null) {
        q.add(node.left);
      }
      if (node.right != null) {
        q.add(node.right);
      }
    }
  }
  return result;
}

// 200
// Number of Islands
public int numIslands(char[][] grid) {
  int result = 0;
  for(int i = 0; i < grid.length; i++) {
    for (int j = 0; j < grid[i].length; j++) {
      if (grid[i][j] == '1' && dfs(grid, i, j)) {
        result++;
      }
    }
  }
  return result;
}

public boolean dfs(char[][] grid, int i, int j) {
  if (i < 0 || i >= grid.length || j < 0 || j >= grid[i].length || grid[i][j] != '1') {
    return false;
  }
  grid[i][j] = '#';
  dfs(grid, i - 1, j);
  dfs(grid, i + 1, j);
  dfs(grid, i, j - 1);
  dfs(grid, i, j + 1);
  return true;
}

// 206
// Reverse Linked List
public ListNode reverseList(ListNode head) {
  ListNode current = head, prev = null;
  while (current != null) {
    ListNode next = current.next;
    current.next = prev;
    prev = current;
    current = next;
  }
  return prev;
}

// 216
// Combination Sum III
public List<List<Integer>> combinationSum3(int k, int n) {
  List<List<Integer>> result = new ArrayList<>();
  backtrack(result, new ArrayList<>(), 1, k, n);
  return result;
}

public void backtrack(List<List<Integer>> result, List<Integer> temp, int index, int length, int target) {
  if (target == 0 && temp.size() == length) {
    result.add(new ArrayList(temp));
  }
  if (target <= 0 || temp.size() >= length) {
    return;
  }
  for (int i = index; i < 10; i++) {
    temp.add(i);
    backtrack(result, temp, i + 1, length, target - i);
    temp.remove(temp.size() - 1);
  }
}

// 217
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

// 225
// Implement Stack using Queues
class MyStack {
  Deque<Integer> queue;
  public MyStack() {
    this.queue = new LinkedList<>();
  }
  
  public void push(int x) {
    queue.add(x);
    int n = queue.size();
    for (int i = 0; i < n - 1; i++) {
      queue.add(queue.remove());
    }
  }
  
  public int pop() {
    return queue.remove();
  }
  
  public int top() {
    return queue.peek();
  }
  
  public boolean empty() {
    return queue.isEmpty();
  }
}

// 225-1
// Implememnt Stack using Queues
class MyStack {
  Deque<Integer> queue;
  public MyStack() {
    this.queue = new ArrayDeque<>();
  }
  
  public void push(int x) {
    queue.addLast(x);
    int n = queue.size();
    for (int i = 0; i < n - 1; i++) {
      queue.addLast(queue.removeFirst());
    }
  }
  
  public int pop() {
    return queue.removeFirst();
  }
  
  public int top() {
    return queue.peekFirst();
  }
  
  public boolean empty() {
    return queue.isEmpty();
  }
}

// 226
// Invert Binary Tree
public TreeNode invertTree(TreeNode root) {
  if (root == null) {
    return null;
  }
  TreeNode temp = root.left;
  root.left = root.right;
  root.right = temp;
  invertTree(root.left);
  invertTree(root.right);
  return root;
}

// 232
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

// 235
// Lowest Common Ancestor of a Binary Search Tree
public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
  while (root != null) {
    if (p.val > root.val && q.val > root.val) {
      root = root.right;
    } else if (p.val < root.val && q.val < root.val) {
      root = root.left;
    } else {
      return root;
    }
  }
  return root;
}

// 236
// Lowest Common Ancestor of a Binary Tree
public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
  if (root == null) {
    return null;
  }
  if (p == root || q == root) {
    return root;
  }
  TreeNode left = lowestCommonAncestor(root.left, p, q);
  TreeNode right = lowestCommonAncestor(root.right, p, q);
  if (left == null) {
    return right;
  } else if (right == null) {
    return left;
  } else {
    return root;
  }
}

// 238
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

// 242
// Valid Anagram
public boolean isAnagram(String s, String t) {
  Map<Character, Integer> map = new HashMap<>();
  for (char i:s.toCharArray()) {
    map.put(i, map.getOrDefault(i, 0) + 1);
  }
  for (char i:t.toCharArray()) {
    if (map.containsKey(i) && map.get(i) > 0) {
      map.put(i, map.get(i) - 1);
    } else {
      return false;
    }
  }
  for (int val:map.values()) {
    if (val > 0) {
      return false;
    }
  }
  return true;
}

// 242-1
// Valid Anagram
public boolean isAnagram(String s, String t) {
  int[] array = new int[26];
  for (char a:s.toCharArray()) array[a - 'a']++;
  for (char a:t.toCharArray()) array[a - 'a']--;
  for (int i:array) {
    if (i != 0) {
      return false;
    }
  }
  return true;
}

// 278
// First Bad Version
public int firstBadVersion(int n) {
  int l = 1, r = n;
  while (l < r) {
    int m = l + (r - l) / 2;
    if (!isBadVersion(m)) {
      l = m + 1;
    } else {
      r = m;
    }
  }
  return r;
}

// 278-1
// First Bad Version
public int firstBadVersion(int n) {
  int l = 1, r = n;
  while (l <= r) {
    int m = l + (r - l) / 2;
    if (!isBadVersion(m)) {
      l = m + 1;
    } else {
      if (!isBadVersion(m - 1)) {
        return m;
      } else {
        r = m - 1;
      }
    }
  }
  return 0;
}

// 349
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

// 409
// Longest Palindrome
public int longestPalindrome(String s) {
  int oddCount = 0, total = 0;
  int[] charArray = new int[128];
  for (char a:s.toCharArray()) {
    charArray[a]++;
  }
  for(int i:charArray) {
    if (i % 2 != 0 && oddCount == 0) {
      total += i;
      oddCount++;
    } else if (i % 2 != 0 && oddCount > 0) {
      total += i - 1;
    } else {
      total += i;
    }
  }
  return total;
}

// 409-1
// Longest Palindrome
public int longestPalindrome(String s) {
  Map<Character,Integer> map = new HashMap<>();
  boolean hasOdd = false;
  int longest = 0;
  for (char a:s.toCharArray()) {
    map.put(a, map.getOrDefault(a, 0) + 1);
  }
  for(int value:map.values()) {
    if (value % 2 != 0 && hasOdd) {
      longest += value - 1;
    } else if (value % 2 != 0) {
      longest += value;
      hasOdd = true;
    } else {
      longest += value;
    }
  }
  return longest;
}

// 438
// Find All Anagrams in a String
public List<Integer> findAnagrams(String s, String p) {
  int left = 0;
  int[] sArray = new int[26], pArray = new int[26];
  List<Integer> result = new ArrayList<>();
  for (char a:p.toCharArray()) {
    pArray[a - 'a']++;
  }
  for (int r = 0; r < s.length(); r++) {
    sArray[s.charAt(r) - 'a']++;
    if (r >= p.length() - 1) {
      boolean isAnagram = true;
      for (int i = 0; i < sArray.length; i++) {
        if (pArray[i] != sArray[i]) {
          isAnagram = false;
        }
      }
      if (isAnagram) {
        result.add(left);
      }
      sArray[s.charAt(left) - 'a']--;
      left++;
    }
  }
  return result;
}

// 438-1
// Find All Anagrams in a String
public List<Integer> findAnagrams(String s, String p) {
  int left = 0;
  int[] sArray = new int[26], pArray = new int[26];
  List<Integer> result = new ArrayList<>();
  for (char a:p.toCharArray()) {
    pArray[a - 'a']++;
  }
  for (int r = 0; r < s.length(); r++) {
    sArray[s.charAt(r) - 'a']++;
    if (r >= p.length() - 1) {
      if (Arrays.equals(pArray,sArray)) {
        result.add(left);
      }
      sArray[s.charAt(left) - 'a']--;
      left++;
    }
  }
  return result;
}

// 438-2
// Find All Anagrams in a String
public List<Integer> findAnagrams(String s, String p) {
  int left = 0;
  Map<Character, Integer> sMap = new HashMap<>(), pMap = new HashMap<>();
  List<Integer> result = new ArrayList<>();
  for (char a:p.toCharArray()) {
    pMap.put(a, pMap.getOrDefault(a, 0) + 1);
  }
  for (int r = 0; r < s.length(); r++) {
    sMap.put(s.charAt(r), sMap.getOrDefault(s.charAt(r), 0) + 1);
    if (r >= p.length() - 1) {
      if (sMap.equals(pMap)) {
        result.add(left);
      }
      sMap.put(s.charAt(left), sMap.get(s.charAt(left)) - 1);
      if (sMap.get(s.charAt(left)) == 0) {
        sMap.remove(s.charAt(left));
      }
      left++;
    }
  }
  return result;
}

// 460
// LFU Cache
class LFUCache {
  int capacity;
  int minFreq = 0;
  Map<Integer, Integer> keyToFreq = new HashMap<>();
  Map<Integer, Integer> keyToVal = new HashMap<>();
  Map<Integer, LinkedHashSet<Integer>> freqToLRUKeys = new HashMap<>();
  public LFUCache(int capacity) {
    this.capacity = capacity;
  }

  public int get(int key) {
    if (!keyToVal.containsKey(key))
      return -1;
    final int freq = keyToFreq.get(key);
    freqToLRUKeys.get(freq).remove(key);
    if (freq == minFreq && freqToLRUKeys.get(freq).isEmpty()) {
      freqToLRUKeys.remove(freq);
      ++minFreq;
    }
    putFreq(key, freq + 1);
    return keyToVal.get(key);
  }

  public void put(int key, int value) {
    if (capacity == 0) return;
    if (keyToVal.containsKey(key)) {
      keyToVal.put(key, value);
      get(key);
      return;
    }
    if (keyToVal.size() == capacity) {
      final int keyToEvict = freqToLRUKeys.get(minFreq).iterator().next();
      freqToLRUKeys.get(minFreq).remove(keyToEvict);
      keyToVal.remove(keyToEvict);
    }
    minFreq = 1;
    putFreq(key, minFreq);
    keyToVal.put(key, value);
  }

  public void putFreq(int key, int freq) {
    keyToFreq.put(key, freq);
    freqToLRUKeys.putIfAbsent(freq, new LinkedHashSet<>());
    freqToLRUKeys.get(freq).add(key);
  }
}

// 543
// Diameter of Binary Tree
int result = 0;
public int diameterOfBinaryTree(TreeNode root) {
  dfs(root);
  return result;
}

public int dfs(TreeNode node) {
  if (node == null) {
    return 0;
  }
  int left = dfs(node.left);
  int right = dfs(node.right);
  result = Math.max(result, left + right);
  return Math.max(left, right) + 1;
}

// 543-1
// Diameter of Binary Tree
public int diameterOfBinaryTree(TreeNode root) {
  int[] result = dfs(root);
  return result[0];
}

public int[] dfs(TreeNode node) {
  if (node == null) {
    return new int[]{0,0};
  }
  int[] left = dfs(node.left);
  int[] right = dfs(node.right);
  int[] array = new int[2];
  int maxCurrent = Math.max(left[0], right[0]);
  array[0] = Math.max(left[1] + right[1], maxCurrent);
  array[1] = Math.max(left[1], right[1]) + 1;
  return array;
}

// 680
// Valid Palindrome II
public boolean validPalindrome(String s) {
  int left = 0, right = s.length() - 1;
  while (left < right) {
    if (s.charAt(left) != s.charAt(right)) {
      return validSubPalindrome(s, left + 1, right) || validSubPalindrome(s, left, right - 1);
    } 
    left++;
    right--;
  }
  return true;
}

public boolean validSubPalindrome(String s, int left, int right) {
  while (left < right) {
    if (s.charAt(left) != s.charAt(right)) {
      return false;
    }
    left++;
    right--;
  }
  return true;
}

// 704
// Binary Search
public int search(int[] nums, int target) {
  int l = 0, r = nums.length - 1;
  while (l <= r) {
    // prevent integer overflow
    int m = l + (r - l) / 2;
    if (nums[m] == target) {
      return m;
    } else if (nums[m] < target) {
      l = m + 1;
    } else {
      r = m - 1;
    }
  }
  return -1;
}

// 704-1
// Binary Search
public int search(int[] nums, int target) {
  return helper(nums, target, 0, nums.length - 1);
}

public int helper(int[] nums, int target, int left, int right) {
  // not the same as merge sort
  // because when merge sort in an array with length of 1
  // it does not need to run the function below
  // but this function requires to run the function below
  // like nums = [5], target = 5 the case
  // prevent integer overflow
  if (left <= right) {
    int m = left + (right - left) / 2;
    if (nums[m] == target) {
      return m;
    } else if (nums[m] < target) {
      return helper(nums, target, m + 1, right);
    } else {
      return helper(nums, target, left, m - 1);
    }
  }
  return -1;
}

// 733
// Flood Fill
public int[][] floodFill(int[][] image, int sr, int sc, int color) {
  if(image[sr][sc] == color) return image;
  dfs(image, color, sr, sc, image[sr][sc]);
  return image;
}

public void dfs(int[][] image, int color, int i, int j, int reference) {
  if (i < 0 || i > image.length - 1 || j < 0 || j > image[0].length - 1) {
    return;
  }
  if (image[i][j] == reference) {
    image[i][j] = color;
    dfs(image, color, i - 1, j, reference);
    dfs(image, color, i + 1, j, reference);
    dfs(image, color, i, j - 1, reference);
    dfs(image, color, i, j + 1, reference);
  }
}

// 739
// Daily Temperatures
public int[] dailyTemperatures(int[] temperatures) {
  Stack<Integer> stack = new Stack<>();
  int n = temperatures.length;
  int[] result = new int[n];
  for (int i = 0; i < n; i++) {
    while (!stack.isEmpty() && temperatures[i] > temperatures[stack.peek()]) {
      int previousDay = stack.pop();
      result[previousDay] = i - previousDay;
    }
    stack.push(i);
  }
  return result;
}

// 844
// Backspace String Compare
public boolean backspaceCompare(String s, String t) {
  int ps = s.length() - 1;
  int pt = t.length() - 1;
  // or condition is to ensure that either one is smaller than 0 will still continue the checking
  while (ps >= 0 || pt >= 0) {
    ps = findValidCharIndex(s, ps);
    pt = findValidCharIndex(t, pt);
    if(ps < 0 && pt < 0){
      return true;
    } else if (ps < 0 || pt < 0) {
      return false;
    } else if (s.charAt(ps) != t.charAt(pt)) {
      return false;
    }
    ps--;
    pt--;
  }
  return true;
}

public int findValidCharIndex(String str, int end) {
  int backspaceNum = 0;
  while (end >= 0) {
    if (str.charAt(end) == '#') {
      backspaceNum++;
    } else if (backspaceNum > 0) {
      backspaceNum--;
    } else {
      break;
    }
    end--;
  }
  return end;
}

// 844-1
// Backspace String Compare
public boolean backspaceCompare(String s, String t) {
  return checking(s).equals(checking(t));
}

public Stack checking(String str) {
  Stack stack = new Stack<>();
  for (char ch:str.toCharArray()) {
    if (ch != '#') {
      stack.push(ch);
    } else if (stack.size() > 0) {
      stack.pop();
    }
  }
  return stack;
}

// 876
// Middle of the Linked List
public ListNode middleNode(ListNode head) {
  ListNode slow = head, fast = head;
  while (fast != null && fast.next != null) {
    slow = slow.next;
    fast = fast.next.next;
  }
  return slow;
}

// 912
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

// 912-1
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

// 981
// Time Based Key-Value Store
class Pair {
  public String value;
  public int timestamp;
  public Pair(String value, int timestamp) {
    this.value = value;
    this.timestamp = timestamp;
  }
}

class TimeMap {
  public Map<String, List<Pair>> map;
  public TimeMap() {
    map = new HashMap<>();
  }
  
  public void set(String key, String value, int timestamp) {
    if (!map.containsKey(key)) {
      map.put(key, new ArrayList());
    }
    map.get(key).add(new Pair(value, timestamp));
  }
  
  public String get(String key, int timestamp) {
    String result = "";
    if (map.containsKey(key)) {
      List<Pair> list = map.get(key);
      int l = 0, r = list.size() - 1;
      while (l <= r) {
        int m = l + (r - l) / 2;
        if (list.get(m).timestamp == timestamp) {
          return list.get(m).value;
        }
        if (list.get(m).timestamp < timestamp) {
          result = list.get(m).value;
          l = m + 1;
        } else {
          r = m - 1;
        }
      }
    }
    return result;
  }
}

// 3005
// Count Elements With Maximum Frequency
public int maxFrequencyElements(int[] nums) {
  int maxFrequency = 0, result = 0;
  Map<Integer,Integer> map = new HashMap();
  for (int i = 0; i < nums.length; i++) {
    map.put(nums[i],map.getOrDefault(nums[i], 0) + 1);
  }
  for (int value:map.values()) {
    maxFrequency = Math.max(maxFrequency,value);
  }
  for (int value:map.values()) {
    if (value == maxFrequency) {
      result += maxFrequency;
    }
  }
  return result;
}

// Quick sort, time complexity O(nlogn), memory complexity O(1), may cause time limit error
public void quickSort(int[] nums, int low, int high) {
  if (low < high) {
    int pivot = partition(nums, low, high);
    quickSort(nums, low, pivot - 1);
    quickSort(nums, pivot + 1, high);
  }
}

public int partition(int[] nums, int low, int high) {
  int start = low - 1;
  int pivot = nums[high];
  for (int i = low; i < high; i++) {
    if (nums[i] < pivot) {
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

// Quick sort
public void quickSort (int[] num, int start, int end) {
  if (start < end) {
    int pivot = partition(num, start, end);
    quickSort(num, start, pivot - 1);
    quickSort(num, pivot + 1, end);
  }
}

public int partition (int[] num, int start, int end) {
  int pivot = num[end];
  int j = start - 1;
  for (int i = start; i < end; i++) {
    if (num[i] < pivot) {
      j++;
      int temp = num[i];
      num[i] = num[j];
      num[j] = temp;
    }
  }
  int temp = num[end];
  num[end] = num[j + 1];
  num[j + 1] = temp;
  return j + 1;
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
      } else {
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

// Merge sort
public void mergeSort(int[] nums, int start, int end) {
  if (start <= end) {
    int mid = (start + end) / 2;
    mergeSort(nums, start, mid - 1);
    mergeSort(nums, mid, end);
    int n1 = mid - start;
    int n2 = end - mid + 1;
    int[] left = new int[n1];
    int[] right = new int[n2];
    for (int i = 0; i < n1; i++) {
      left[i] = nums[start + i];
    }
    for (int i = 0; i < n2; i++) {
      right[i] = nums[mid + i];
    }
    int i = 0, j = 0, k = start;
    while (i < n1 && j < n2) {
      if (left[n1] < right[n2]) {
        nums[k] = left[i];
        i++;
      } else {
        nums[k] = right[j];
        j++;
      }
      k++;
    }
    while (i < n1) {
      nums[k] = left[i];
      i++;
      k++;
    }
    while (j < n2) {
      nums[k] = right[j];
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
  if (left < length && nums[left] > nums[largest]) {
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

// Heap sort
public void sort(int[] nums) {
  int n = nums.length;
  for (int i = n / 2 - 1; i >= 0; i--) {
    heapify(nums, i, n);
  }
  // the 0-th index will be sorted, so don't need to include i = 0 case to be i >= 0
  for (int i = n - 1; i > 0; i--) {
    // put the largest number to the back one by one and extract it from the nums array
    int swap = nums[0];
    nums[0] = nums[i];
    nums[i] = swap;
    heapify(nums, 0, i);
  }
}

public void heapify(int[] nums, int root, int n) {
  int largest = root;
  int left = 2 * root + 1;
  int right = 2 * root + 2;
  if (left < n && nums[left] > nums[largest]) {
    largest = left;
  }
  if (right < n && nums[right] > nums[largest]) {
    largest = right;
  }
  if (largest != root) {
    int swap = nums[largest];
    nums[largest] = nums[root];
    nums[root] = swap;
    heapify(nums, largest, n);
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
