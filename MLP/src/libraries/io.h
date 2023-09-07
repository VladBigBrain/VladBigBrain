#ifndef MLP_LIBRARIES_IO_H_
#define MLP_LIBRARIES_IO_H_

#include <fstream>
#include <iostream>
#include <limits>

class IO {
 public:
  template <typename... Args>
  static void WriteLines(std::ostream& os, Args const&... args) {
    (Write(os, args, "\n"), ...);
  }

  template <typename... Args>
  static void Write(std::ostream& os, Args const&... args) {
    ((os << args), ...);
  }

  template <typename... Args>
  static auto Read(std::istream& is, Args&... args) -> bool {
    bool result = true;

    ((result &= (bool)(is >> args)), ...);

    return result;
  }
};

#endif  // MLP_LIBRARIES_IO_H_
