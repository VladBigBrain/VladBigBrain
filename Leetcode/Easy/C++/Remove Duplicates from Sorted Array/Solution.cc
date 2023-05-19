#include <algorithm>
#include <vector>
class Solution {
 public:
  int removeElement(std::vector<int>& nums, int val) {
    while (std::any_of(nums.begin(), nums.end(),
                       [val](int x) { return x == val; })) {
      auto temp = std::find_if(nums.begin(), nums.end(),
                               [val](int x) { return x == val; });
      nums.erase(temp);
    }
    return nums.size();
  }
};