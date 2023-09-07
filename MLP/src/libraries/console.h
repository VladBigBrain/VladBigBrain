#ifndef MLP_LIBRARIES_CONSOLE_H_
#define MLP_LIBRARIES_CONSOLE_H_

#include <iostream>

#include "io.h"

class Console {
 public:
  template <typename... Args>
  static void WriteLine(Args const&... args) {
    IO::WriteLines(std::cout, args...);
  }

  template <typename... Args>
  static void Write(Args const&... args) {
    IO::Write(std::cout, args...);
  }

  template <typename Type, typename... PrefixArgs>
  static auto Read(PrefixArgs const&... args) -> Type {
    Type value{};
    Write(args...);

    IO::Read(std::cin, value);
    Clear();

    return value;
  }

  template <typename Type, typename... PrefixArgs>
  static auto ReadUntil(PrefixArgs const&... args) -> Type {
    Type value;
    Write(args...);

    while (!IO::Read(std::cin, value)) Write("Retrying...\n", args...);

    return value;
  }

  static void Clear() {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }
};

#endif  // MLP_LIBRARIES_CONSOLE_H_
