#include <iostream>
#include <vector>
using namespace std;

class Solution {
 public:
  string longestCommonPrefix(vector<string>& strs) {
    if (strs.size() == 0) return "";
    string result{};
    // Проходим по каждоq строке
    for (int i = 0; i < strs[0].size(); i++) {
      // Проходим по каждому символу в строке
      for (int j = 0; j < strs.size(); j++) {
        // Если символы не совпадают, то возвращаем результат
        if (strs[0][i] != strs[j][i]) return result;
      }
      // Добавляем символ в результат
      result += strs[0][i];
    }

    return result;
  }
};