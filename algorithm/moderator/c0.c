#include "stdio.h"
// #include "stdlib.h"
// #include "string.h"


// Merge sort
void mergeSort(int nums[], int start, int end) {
  if (start >= end) {
    return;
  }
  int mid = start + (end - start) / 2;
  mergeSort(nums, start, mid);
  mergeSort(nums, mid + 1, end);
  int n0 = mid - start + 1; 
  int n1 = end - mid;
  int leftNums[n0], rightNums[n1];
  for (int i = 0; i < n0; i++) {
    leftNums[i] = nums[start + i];
  }
  for (int i = 0; i < n1; i++) {
    rightNums[i] = nums[mid + i + 1];
  }
  int i = 0, j = 0, k = start;
  while (i < n0 && j < n1){
    if (leftNums[i] < rightNums[j]) {
      nums[k] = leftNums[i];
      i++;
    } else {
      nums[k] = rightNums[j];
      j++;
    }
    k++;
  }
  while (i < n0) {
    nums[k] = leftNums[i];
    i++;
    k++;
  }
  while (j < n1) {
    nums[k] = rightNums[j];
    j++;
    k++;
  }
}

// Heap sort
void heapify(int nums[], int root, int length) {
  int largest = root;
  int left = 2 * root + 1;
  int right = 2 * root + 2;
  if (left < length && nums[left] > nums[largest]) {
    largest = left;
  }
  if (right < length && nums[right] > nums[largest]) {
    largest = right;
  }
  if (largest != root) {
    int temp = nums[largest];
    nums[largest] = nums[root];
    nums[root] = temp;
    heapify(nums, largest, length);
  }
}

void heapSort(int nums[], int n) {
  for (int i = n / 2 - 1; i >= 0; i--) {
    heapify(nums, i, n);
  }
  for (int i = n - 1; i > 0; i--) {
    int temp = nums[0];
    nums[0] = nums[i];
    nums[i] = temp;
    heapify(nums, 0, i);
  }
}

// Quick sort
int partition (int nums[], int start, int end) {
  int low = start;
  int pivot = nums[end];
  for (int i = start; i < end; i++) {
    if (nums[i] < pivot) {
      int temp = nums[i];
      nums[i] = nums[low];
      nums[low] = temp;
      low++;
    }
  }
  int temp = nums[low];
  nums[low] = nums[end];
  nums[end] = temp;
  return low;
}

void quickSort(int nums[], int start, int end) {
  if (start >= end) {
    return;
  }
  int pivot = partition(nums, start, end);
  quickSort(nums, start, pivot - 1);
  quickSort(nums, pivot + 1, end);
}

int main() {
  printf("world break\n\n");
  int arr[] = {12, 11, 13, 5, 6, 7};
  int n = sizeof(arr) / sizeof(arr[0]);
  mergeSort(arr, 0, n - 1);
  printf("merge sort: [");
  for (int i = 0; i < n; i++) {
    printf("%d", arr[i]);
    if (i < n - 1) {
      printf(", ");
    } 
  }
  printf("]\n");
  
  int nums[] = {20, 12, 11, 21, 109, 40, 90, 13, 5, 6, 7};
  int n1 = sizeof(nums) / sizeof(nums[0]);
  heapSort(nums, n1);
  printf("heap sort: [");
  for (int i = 0; i < n1; i++) {
    printf("%d", nums[i]);
    if (i < n1 - 1) {
      printf(", ");
    } 
  }
  printf("]\n");
  
  int numsArr[] = {20, 12, 109, 11, 21, 13, 5, 6, 7};
  int n2 = sizeof(numsArr) / sizeof(numsArr[0]);
  quickSort(numsArr, 0, n2 - 1);
  printf("quick sort: [");
  for (int i = 0; i < n2; i++) {
    printf("%d", numsArr[i]);
    if (i < n2 - 1) {
      printf(", ");
    } 
  }
  printf("]\n");

  return 0;
}
