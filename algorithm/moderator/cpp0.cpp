#include <iostream>
#include <vector>

using namespace std;

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

vector<int> sortArray(vector<int>& nums) {
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

void heapSort(vector<int> &nums) {
  int n = nums.size();
  for (int i = n / 2 - 1; i >= 0; i--) {
    heapify(nums, i, n);
  }
  for (int i = n - 1; i > 0; i--) {
    swap(nums[0], nums[i]);
    heapify(nums, 0, i);
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