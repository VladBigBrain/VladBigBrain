#ifndef LIBRARIES_LOGGER_H_
#define LIBRARIES_LOGGER_H_

#include <fstream>
#include <iostream>
namespace s21 {
    
class Logger {
 public:
  Logger() { file.open("log.txt", std::ofstream::out | std::ofstream::app); }
  Logger(const Logger&) = delete;
  auto operator=(const Logger&) -> Logger& = delete;
  Logger(Logger&&) = delete;
  auto operator=(Logger&&) -> Logger& = delete;
  ~Logger() = default;
  
  void LogToFile(const std::string& msg) { file << msg << std::endl; }
 private:
  std::ofstream file;
};

}  // namespace s21

#endif  // LIBRARIES_LOGGER_H_
