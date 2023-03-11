#include <iostream>
#include <regex>
using namespace std;

class Solution {
 public:
  int romanToInt(string s) { return findValue(s); }

  int findValue(string s) {
    int result = 0;
    map<string, int> roman_map = {
        {"I", 1},   {"IV", 4},   {"V", 5},   {"IX", 9},  {"X", 10},
        {"XL", 40}, {"L", 50},   {"XC", 90}, {"C", 100}, {"CD", 400},
        {"D", 500}, {"CM", 900}, {"M", 1000}};

    for (int i = 0; i < s.length(); i++) {
      if (roman_map.find(s.substr(i, 2)) != roman_map.end()) {
        result += roman_map[s.substr(i, 2)];
        i++;
      } else {
        result += roman_map[s.substr(i, 1)];
      }
    }
    return result;
  }
};