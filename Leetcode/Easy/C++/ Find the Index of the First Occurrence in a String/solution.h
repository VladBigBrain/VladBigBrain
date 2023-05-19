#include <algorithm>
#include <iostream>
class Solution {
public:
    int strStr(std::string haystack, std::string needle) {
    auto it = std::search(haystack.begin(), haystack.end(), needle.begin(), needle.end());
    if (it != haystack.end()) {
        return std::distance(haystack.begin(), it);
    }
    return -1;  // Needle not found in haystack
    }
};