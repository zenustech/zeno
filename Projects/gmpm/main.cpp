#include "zensim/types/Iterator.h"
#include "zensim/tpls/fmt/format.h"

int main() {
  using namespace zs;
  for (auto &&[x, y] : ndrange<2>(2))
    fmt::print("{}, {}\n", x, y);
  return 0;
}
