#include "Matrix.hpp"

#include <stdexcept>

#include "zensim/Logger.hpp"

namespace zs {

  template struct YaleSparseMatrix<f32, i32>;
  template struct YaleSparseMatrix<f64, i32>;
}  // namespace zs