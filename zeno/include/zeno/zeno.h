#pragma once


#include <string>
#include <memory>
#include <vector>
#include <variant>
#include <optional>
#include <sstream>
#include <array>
#include <map>
#include <set>





namespace zeno {


class Exception : public std::exception {
private:
  std::string msg;
public:
  ZENO_API Exception(std::string const &msg) noexcept;
  ZENO_API ~Exception() noexcept;
  ZENO_API char const *what() const noexcept;
};


using IValue = std::variant<std::string, int, float>;


}
