#include <iostream>
#include <vector>

using namespace std;

// 15
// 3Sum
vector<vector<int>> threeSum(vector<int>& nums) {
  std::sort(nums.begin(), nums.end());
  vector<vector<int>> result;
  int n = nums.size();
  for (int i = 0; i < n - 2; i++) {
    if (i > 0 && nums[i] == nums[i - 1]) continue;
    int low = i + 1, high = n - 1;
    while (low < high) {
      int total = nums[i] + nums[low] + nums[high];
      if (total < 0) {
        low++;
      } else if (total > 0) {
        high--;
      } else {
        if (total == 0) {
          result.push_back({nums[i], nums[low], nums[high]});
          low++;
          high--;
          while (low < high && nums[low] == nums[low - 1]) {
            low++;
          }
          while (low < high && nums[high] == nums[high + 1]) {
            high--;
          }
        }
      }
    }
  }
  return result;
}

// 39
// Combination Sum
void backtrack(vector<int>& candidates, int target, vector<vector<int>>& result, vector<int>& temp, int current) {
  if (target == 0) {
    result.push_back(temp);
  }
  if (target <= 0) {
    return;
  }
  for (int i = current; i < candidates.size(); i++) {
    temp.push_back(candidates[i]);
    backtrack(candidates, target - candidates[i], result, temp, i);
    temp.pop_back();
  }
}

vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
  vector<vector<int>> result;
  vector<int> temp;   
  backtrack(candidates, target, result, temp, 0);
  return result;
}

// 134
// Gas Station
int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
  int current = 0, total = 0, result = 0;
  for (int i = 0; i < gas.size(); i++) {
    total += gas[i] - cost[i];
    current += gas[i] - cost[i];
    if (current < 0) {
      current = 0;
      result = i + 1;
    }
  }
  if (total >= 0) {
    return result;
  }
  return -1;
}

// 844
// Backspace String Compare
int findNextValidChar(string str, int end) {
  int backspaceCount = 0;
  while (end >= 0) {
    if (str[end] == '#') {
      backspaceCount++;
    } else if (backspaceCount > 0) {
      backspaceCount--;
    } else {
      break;
    }
    end--;
  }
  return end;
}

bool backspaceCompare(string s, string t) {
  int pS = s.size() - 1;
  int pT = t.size() - 1;
  while (pS >= 0 || pT >= 0) {
    pS = findNextValidChar(s, pS);
    pT = findNextValidChar(t, pT);
    if (pS < 0 && pT < 0) {
      return true;
    } else if (pS < 0 || pT < 0) {
      return false;
    } else if (s[pS] != t[pT]) {
      return false;
    }
    pS--;
    pT--;
  }
  return true;
}

// 912
// Sort an Array
void mergeSort(vector<int>& nums, int start, int end) {
  if (start < end) {
    int mid = start + (end - start) / 2;
    mergeSort(nums, start, mid);
    mergeSort(nums, mid + 1, end);
    vector<int> leftNums(nums.begin() + start, nums.begin() + mid + 1);
    vector<int> rightNums(nums.begin() + mid + 1, nums.begin() + end + 1);
    int i = 0;
    int j = 0;
    int k = start;
    while (i < leftNums.size() && j < rightNums.size()) {
      if (leftNums[i] < rightNums[j]) {
        nums[k] = leftNums[i];
        i++;
      } else {
        nums[k] = rightNums[j];
        j++;
      }
      k++;
    }
    while (i < leftNums.size()) {
      nums[k] = leftNums[i];
      i++;
      k++;
    }
    while (j < rightNums.size()) {
      nums[k] = rightNums[j];
      j++;
      k++;
    }
  }
}

vector<int> sortArray(vector<int>& nums) {
  int n = nums.size();
  mergeSort(nums, 0,  n - 1);
  return nums;
}

// 912-1
// Sort an Array
void heapify(vector<int>& nums, int root, int length) {
  int largest = root;
  int left = 2 * root + 1;
  int right = 2 * root + 2;
  if (left < length && nums[left] > nums[largest]) {
    largest = left;
  }
  if (right < length && nums[right] > nums[largest]) {
    largest = right;
  }
  if (largest != root){
    swap(nums[largest], nums[root]);
    heapify(nums, largest, length);
  }
}

void heapSort(vector<int>& nums) {
  int n = nums.size();
  for (int i = n / 2 - 1; i >= 0; i--) {
    heapify(nums, i, n);
  }
  for (int i = n - 1; i > 0; i--) {
    swap(nums[0], nums[i]);
    heapify(nums, 0, i);
  }
}

vector<int> sortArray_1(vector<int>& nums) {
  heapSort(nums);
  return nums;
}

// Quick sort
int partition(vector<int>& nums, int start, int end) {
  int pivot = nums[end];
  int low = start - 1;
  for (int i = start; i < end; i++) {
    if (nums[i] < pivot) {
      low++;
      swap(nums[i], nums[low]);
    }
  }
  swap(nums[low + 1], nums[end]);
  return low + 1;
}

void quickSort(vector<int>& nums, int start, int end) {
  if (start < end) {
    int pivot = partition(nums, start, end);
    quickSort(nums, start, pivot - 1);
    quickSort(nums, pivot + 1, end);
  }
}

// Merge sort
void mergeSort_1(vector<int>& nums, int start, int end) {
  if (start < end) {
    int mid = start + (end - start) / 2;
    mergeSort_1(nums, start, mid);
    mergeSort_1(nums, mid + 1, end);
    vector<int> leftNums(nums.begin() + start, nums.begin() + mid + 1);
    vector<int> rightNums(nums.begin() + mid + 1, nums.begin() + end + 1);
    int i = 0;
    int j = 0;
    int k = start;
    while (i < leftNums.size() && j < rightNums.size()) {
      if (leftNums[i] < rightNums[j]) {
        nums[k] = leftNums[i];
        i++;
      } else {
        nums[k] = rightNums[j];
        j++;
      }
      k++;
    }
    while (i < leftNums.size()) {
      nums[k] = leftNums[i];
      i++;
      k++;
    }
    while (j < rightNums.size()) {
      nums[k] = rightNums[j];
      j++;
      k++;
    }
  }
}

// Bubble sort
void bubbleSort(vector<int>& nums) {
  int n = nums.size();
  for (int i = 0; i < n; i++) {
    bool swapped = false;
    for (int j = 0; j < n - i - 1; j++) {
      if (nums[j] > nums[j + 1]) {
        swap(nums[j], nums[j + 1]);
        swapped = true;
      }
    }
    if (!swapped) {
      break;
    }
  }
}

// Heap sort
void heapify_1(vector<int>& nums, int root, int length) {
  int largest = root;
  int left = 2 * root + 1;
  int right = 2 * root + 2;
  if (left < length && nums[left] > nums[largest]) {
    largest = left;
  }
  if (right < length && nums[right] > nums[largest]) {
    largest = right;
  }
  if (largest != root){
    swap(nums[largest], nums[root]);
    heapify_1(nums, largest, length);
  }
}

void heapSort_1(vector<int> &nums) {
  int n = nums.size();
  for (int i = n / 2 - 1; i >= 0; i--) {
    heapify_1(nums, i, n);
  }
  for (int i = n - 1; i > 0; i--) {
    swap(nums[0], nums[i]);
    heapify_1(nums, 0, i);
  }
}

int main() {
  vector<int> nums = {12, 7, 11, 8, 6, 3, 9};
  int n = nums.size();
  // mergeSort(nums, 0, n - 1);
  heapSort(nums);
  // bubbleSort(nums);
  cout << "[";
  if (n > 0) {
    cout << nums[0];
    for (int i = 0; i < n; i++) {
      cout << ", " << nums[i];
    }
  }
  cout << "]";
  return 0;
}
