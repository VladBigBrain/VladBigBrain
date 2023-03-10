
#include <iostream>

#include "solution.h"
int main() {
  Solution solution;
  vector<int> nums{2, 7, 11, 15};
  int target = 9;
  vector<int> result = solution.twoSum(nums, target);

  for (auto i : result) {
    std::cout << i << std::endl;
  }

  vector<int> nums2{3, 2, 4};
  int target2 = 6;

  vector<int> result2 = solution.twoSum(nums2, target2);
  for (auto i : result2) {
    std::cout << i << std::endl;
  }

  vector<int> nums3{3, 3};
  int target3 = 6;

  vector<int> result3 = solution.twoSum(nums3, target3);
  for (auto i : result3) {
    std::cout << i << std::endl;
  }

  return 0;
}