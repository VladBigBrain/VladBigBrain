#include <iostream>
#include <regex>
using namespace std;
class Solution {
 public:
  bool isPalindrome(int x) {
    if (x < 0) return false;
    string str = to_string(x);
    bool is_palindrome =
        std::equal(str.begin(), str.begin() + str.size() / 2, str.rbegin());
    return is_palindrome;
  }
};