package com.github.pseudo.code.base;

import com.github.pseudo.code.base.algorithm.Java0;

import java.util.ArrayList;
import java.util.List;

public class Main {
  public static void main(String[] args) {
    System.out.println("Hello and welcome!");
    for (int i = 1; i <= 5; i++) {
      System.out.println("i = " + i);
    }
    int[] nums = new int[]{1, 9, 8, 20, 15, 17, 5, 4, 8, 3};
    Java0 java = new Java0();
    java.mergeSort(nums, 0, nums.length - 1);
    List<Integer> arrayList = toArrayList(nums);
    System.out.println(arrayList);
  }
  public static List<Integer> toArrayList(int[] intArray) {
    List<Integer> result = new ArrayList<>();
    for (int i:intArray) {
      result.add(i);
    }
    return result;
  }
}