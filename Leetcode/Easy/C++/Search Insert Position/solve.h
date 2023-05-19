#include <algorithm>
#include <vector>
class Solution {
 public:
  int searchInsert(std::vector<int>& nums, int target) {
    auto it = std::lower_bound(nums.begin(), nums.end(), target);
    return std::distance(nums.begin(), it);
  }
};