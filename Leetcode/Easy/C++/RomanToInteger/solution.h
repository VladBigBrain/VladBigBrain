#include <iostream>
#include <regex>
using namespace std;

class Solution {
 public:
  int romanToInt(string s) { return findValue(s); }

  int findValue(string s) {
    int result{0};
    for (int i = 0; i < s.length(); i++) {
      
    }
    // cout << result << endl;
    return result;
  }

  bool CheckRoman(string s) {
    std::regex pattern("[IVXLCDM]");
    if (std::regex_match(s, pattern)) {
      return true;
    } else {
      return false;
    }
  }
};