#pragma once
#include <chrono>
#include <cmath>
#include <random>

#include "zensim/math/Rotation.hpp"
#include "zensim/math/Vec.h"

namespace zs {

  /**
      Random number generator. T must be floating point.
  */
  template <typename T> struct RandomNumber {
  public:
    std::mt19937 generator;

    RandomNumber(unsigned s = 123) : generator{s} {}

    /**
       Reset the random seed.
    */
    void resetSeed(T s) noexcept { generator.seed(s); }

    /**
       Reset seed using time
    */
    void resetSeedUsingTime() noexcept {
      auto s = std::chrono::high_resolution_clock::now().time_since_epoch().count();
      generator.seed(s);
    }

    /**
       Random real number from 0 to 1
    */
    T randReal() noexcept {
      std::uniform_real_distribution<T> distribution((T)0, (T)1);
      return distribution(generator);
    }

    /**
       Random real number from an interval
    */
    T randReal(T a, T b) noexcept {
      std::uniform_real_distribution<T> distribution(a, b);
      return distribution(generator);
    }

    /**
       Random integer number from a to b inclusive, i.e. in the both closed
       interval [a,b]
    */
    int randInt(int a, int b) noexcept {
      std::uniform_int_distribution<> distribution(a, b);
      return distribution(generator);
    }

    /**
      Random vector in box
      */
    template <int dim>
    vec<T, dim> randInBox(const vec<T, dim> &minCorner, const vec<T, dim> &maxCorner) noexcept {
      vec<T, dim> r{};
      for (int i = 0; i < dim; i++) r[i] = randReal(minCorner[i], maxCorner[i]);
      return r;
    }

    /**
       Random barycentric weights
    */
    template <int dim> vec<T, dim> randomBarycentricWeights() noexcept {
      vec<T, dim> r{};
      T sum;
      do {
        sum = 0;
        for (int i = 0; i < dim - 1; i++) sum += (r[i] = randReal());
      } while (sum > 1);
      r(dim - 1) = 1 - sum;
      return r;
    }

    /**
      Random vector in ball
      */
    template <int dim> vec<T, dim> randInBall(const vec<T, dim> &center, T radius) noexcept {
      vec<T, dim> minCorner = center - radius, maxCorner = center + radius, r{};
      do {
        r = randInBox(minCorner, maxCorner);
      } while ((r - center).l2NormSqr() > radius * radius);
      return r;
    }

    /**
       Random rotation matrix
    */
    void randRotation(vec<T, 2, 2> &R) noexcept {
      constexpr T pi = std::acos(-T(1));
      T theta = randReal(0, 2 * pi);
      T c = std::cos(theta);
      T s = std::sin(theta);
      R(0, 0) = c, R(0, 1) = -s, R(1, 0) = s, R(1, 1) = c;
    }

    /**
      Random rotation matrix
      */
    void randRotation(vec<T, 3, 3> &R) noexcept {
      std::normal_distribution<T> n;
      vec<T, 4> q(n(generator), n(generator), n(generator), n(generator));
      q = q.normalized();
      R = Rotation<T, 3>::quaternion2matrix(q);
    }

    /**
      Fill with uniform random numbers
    */
    template <class MultiVec> MultiVec fill(T a = 0, T b = 1) noexcept {
      MultiVec r{};
      for (std::size_t i = 0; i < MultiVec::extent; ++i) r.val(i) = randReal(a, b);
      return r;
    }

    /**
      Fill with uniform random numbers
    */
    template <class MultiVec> void fill(MultiVec &r, T a = 0, T b = 1) noexcept {
      for (std::size_t i = 0; i < MultiVec::extent; ++i) r.val(i) = randReal(a, b);
    }

    /**
      Fill with random integers
    */
    template <class MultiVec> MultiVec fillInt(int a = 0, int b = 1) noexcept {
      MultiVec r{};
      for (std::size_t i = 0; i < MultiVec::extent; ++i) r.val(i) = randInt(a, b);
      return r;
    }

    /**
      Fill with random integers
    */
    template <class MultiVec> void fillInt(MultiVec &r, int a = 0, int b = 1) noexcept {
      for (std::size_t i = 0; i < MultiVec::extent; ++i) r.val(i) = randInt(a, b);
    }
  };

}  // namespace zs