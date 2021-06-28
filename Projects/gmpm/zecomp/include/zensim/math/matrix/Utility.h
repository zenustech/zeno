#pragma once

namespace zs {

  namespace math {
    template <typename T> constexpr int sgn(const T n) noexcept {
      return n < -1e-10 ? -1 : n > 1e-10;
      // return n < 0.0 ? -1 : n > 0.0;
      // return n == 0 ? 0 : 1;
    }

    template <typename T> constexpr void swap(T *a, T *b) {
      T tmp = *a;
      *a = *b;
      *b = tmp;
    }

    template <typename T> constexpr T abs(const T n) noexcept { return n > 0 ? n : -n; }

    template <typename T> constexpr void orthogonalVector(const T *input, T *output) noexcept {
      T abs_x = abs(input[0]), abs_y = abs(input[1]), abs_z = abs(input[2]);
      if (abs_x < abs_y) {
        if (abs_x < abs_z)
          output[0] = 0, output[1] = input[2], output[2] = -input[1];
        else
          output[0] = input[1], output[1] = -input[0], output[2] = 0;
      } else {
        if (abs_y < abs_z)
          output[0] = -input[2], output[1] = 0, output[2] = input[0];
        else
          output[0] = input[1], output[1] = -input[0], output[2] = 0;
      }
    }

    template <typename T> constexpr void cross(const T *operand1, const T *operand2, T *output) {
      const T &x1 = operand1[0];
      const T &y1 = operand1[1];
      const T &z1 = operand1[2];
      const T &x2 = operand2[0];
      const T &y2 = operand2[1];
      const T &z2 = operand2[2];
      output[0] = y1 * z2 - y2 * z1;
      output[1] = x2 * z1 - x1 * z2;
      output[2] = x1 * y2 - x2 * y1;
    }

    template <typename T, typename TT>
    constexpr void vectorCopy(const T scale, const T *v, TT *res) {
      res[0] = v[0] * scale;
      res[1] = v[1] * scale;
      res[2] = v[2] * scale;
    }

  }  // namespace math

}  // namespace zs
