#include "Solution.h"

int main() {
  Solution s;
  vector<string> strs = {"flower", "flow", "flight"};
  cout << s.longestCommonPrefix(strs);
  vector<string> strs1 = {"dog", "racecar", "car"};
  cout << s.longestCommonPrefix(strs1);
  return 0;
}