#include <iostream>
#include <stack>
using namespace std;

class Solution {
 public:
  bool isValid(string s) {
    bool result = false;
    stack<char> st;
    for (int i = 0; i < s.length(); i++) {
      if (s[i] == '(' || s[i] == '[' || s[i] == '{')
        st.push(s[i]);
      else if (s[i] == ')' && !st.empty() && st.top() == '(')
        st.pop();
      else if (s[i] == ']' && !st.empty() && st.top() == '[')
        st.pop();
      else if (s[i] == '}' && !st.empty() && st.top() == '{')
        st.pop();
      else
        return false;
    }
    if (st.empty()) result = true;
    return result;
  }
};