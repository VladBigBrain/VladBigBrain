#include "solution.h"

int main() {
  Solution s;
  cout << s.isValid("()") << endl;
  cout << s.isValid("()[]{}") << endl;
  cout << s.isValid("(]") << endl;
  cout << s.isValid("([)]") << endl;
  cout << s.isValid("{[]}") << endl;
  return 0;
}