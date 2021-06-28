#pragma once
#include <omp.h>
#include <algorithm>
#include <fstream>

#include "zensim/container/DenseGrid.hpp"
#include "zensim/math/RandomNumber.hpp"
#include "zensim/math/Vec.h"
// #include <taskflow/taskflow.hpp>
#include "zensim/execution/Concurrency.h"

namespace zs {

  template <typename T, int dim> struct PoissonDisk {
    using TV = vec<T, dim>;
    using IV = vec<int, dim>;

    RandomNumber<T> rnd{123};
    T minDistance{0};
    TV minCorner{}, maxCorner{};
    int maxAttempts{30};
    bool periodic{false};

    void setDistanceByPpc(T dx, T ppc) noexcept {
      T v = std::pow((double)dx, dim) / ppc;
      if constexpr (dim == 2)
        minDistance = std::sqrt(v * ((double)2 / 3));
      else if constexpr (dim == 3)
        minDistance = std::pow(v * ((double)13 / 18), (double)1 / 3);
    }

    TV generateRandomPointAroundAnnulus(const TV &center) noexcept {
      while (true) {
        TV v{};
        rnd.fill(v, -1, 1);
        T mag2 = v.l2NormSqr();
        if (mag2 >= 0.25 && mag2 <= 1) return v * (minDistance * 2) + center;
      }
      return TV{1};
    }
    IV worldToIndexSpace(const TV &X, T h) const {
      IV idx{};
      for (int d = 0; d < dim; ++d) idx[d] = std::floor((X[d] - minCorner[d]) / h);
      return idx;
    }
    /**
    Return true if the distance between the candidate point and any other points
    in samples are sufficiently far away (at least _minDistance away), and false
    otherwise.
     */
    bool checkDistance(const vec<T, 2> &point, const T h,
                       const DenseGrid<int, int, 2> &background_grid,
                       const std::vector<std::array<T, 2>> &samples) const {
      if constexpr (dim == 2) {
        IV index = worldToIndexSpace(point, h);
        // Check if we are outside of the background_grid. If so, return false
        for (int d = 0; d < 2; ++d) {
          if (index(d) < 0 || index(d) >= background_grid.domain(d)) return false;
        }
        // If there is already a particle in that cell, return false
        if (background_grid(index) != -1) return false;
        T min_distance_sqr = minDistance * minDistance;
        IV local_min_index = index - 2;
        IV local_max_index = index + 3;
        // If not periodic, clamp local_min_index and local_max_index to the size
        // of the background grid
        if (!periodic) {
          for (int d = 0; d < 2; ++d) {
            local_min_index[d] = std::max(0, local_min_index[d]);
            local_max_index[d] = std::min(background_grid.domain(d), local_max_index[d]);
          }
        }
        // Create local_box for iterator purposes
        if (!periodic)
          for (index[0] = local_min_index[0]; index[0] < local_max_index[0]; ++index[0])
            for (index[1] = local_min_index[1]; index[1] < local_max_index[1]; ++index[1]) {
              if (background_grid(index) == -1) continue;
              TV x = point - TV::from_array(samples[background_grid(index)]);
              if (x.l2NormSqr() < min_distance_sqr) return false;
            }
        else {
          for (index[0] = local_min_index[0]; index[0] < local_max_index[0]; ++index[0])
            for (index[1] = local_min_index[1]; index[1] < local_max_index[1]; ++index[1]) {
              IV local_index = index;
              // Need to shift point in MultiArray if one of the indices is
              // negative or greater than background_grid.size
              TV shift = TV::zeros();
              for (int d = 0; d < 2; ++d) {
                // If it.index < 0 update local index to all the way down to the
                // right/top/back If there is a point in that MultiArray index, we
                // need to shift that point to the left/bottom/front
                if (index[d] < 0) {
                  local_index[d]
                      = local_index[d] % background_grid.domain(d) + background_grid.domain(d);
                  shift[d] = minCorner[d] - maxCorner[d];
                }
                // If it.index(d) >= background_grid(d) update local index to all
                // the way down to the left/bottom/front If there is a point in
                // that MultiArray index, we need to shift that point to the left
                // right/top/back
                else if (index[d] >= background_grid.domain(d)) {
                  local_index[d] = local_index[d] % background_grid.domain(d);
                  shift[d] = maxCorner[d] - minCorner[d];
                }
              }
              if (background_grid(local_index) == -1) continue;
              TV x = point - (TV::from_array(samples[background_grid(local_index)]) + shift);
              if (x.l2NormSqr() < min_distance_sqr) return false;
            }
        }
        return true;
      } else
        return true;
    }
    /**
      Faster sampling based on particles-1000k.dat file. The function assumes that
      this file is saved in Data/MpmParticles/particles-1000k.dat. _minDistance =
      mimimum distance between particles. A few information about
      particles-1000k.dat:
        - Samples count: 1001436
        - _minDistance = 1	_h=0.57735
        - min_point =      -60 -59.9999 -59.9998
        - max_point = 59.9999      60 59.9999
      The file is generated by Projects/pdsampler/pdsampler.cpp.
       */
    template <typename Predicate> decltype(auto) sample(Predicate &&feasible) {
      std::vector<std::array<T, dim>> samples{};
      if constexpr (dim == 3) {
        // Figure out the number of offsets
        TV scaled_ref_box_length{120.0, 120.0, 120.0};
        scaled_ref_box_length = scaled_ref_box_length * minDistance;
        TV sideLength = maxCorner - minCorner;
        IV offset_number;
        for (int d = 0; d < dim; ++d)
          offset_number[d] = std::ceil(sideLength[d] / scaled_ref_box_length[d]) + 1;
        // Read std vector
        std::ifstream is((std::string{AssetDirPath} + "MpmParticles/particles-1000k.dat"),
                         std::ios::in | std::ios::binary);
        if (!is) throw std::runtime_error("particle-1000k.dat file not found!");
        std::size_t cnt, tmp;
        is.read((char *)&cnt, sizeof(std::size_t));
        auto estimate = cnt * (sideLength.prod() / scaled_ref_box_length.prod());
        samples.reserve(estimate);

        is.read((char *)&tmp, sizeof(std::size_t));  ///< neglect this
        // Read from file as float
        std::vector<vec<float, dim>> data(cnt);
        is.read((char *)data.data(), cnt * sizeof(vec<float, dim>));

#if 1
        puts("begin parallel sampling!");
        const auto nworkers = std::thread::hardware_concurrency();
        omp_set_num_threads(nworkers);
        std::vector<std::vector<std::array<T, dim>>> localSamples(nworkers);
#  pragma omp parallel for
        for (int id = 0; id < nworkers; id++) {
          localSamples[id].reserve(estimate / nworkers);
          for (int i = id; i < cnt; i += nworkers) {
            const auto &new_point_read = data[i];
            TV new_point, offset_center, offset_new_point;
            for (int d = 0; d < dim; ++d)
              new_point[d] = (minDistance * new_point_read[d]) + minCorner[d];
            IV index{};
            for (index[0] = 0; index[0] < offset_number[0]; ++index[0])
              for (index[1] = 0; index[1] < offset_number[1]; ++index[1])
                for (index[2] = 0; index[2] < offset_number[2]; ++index[2]) {
                  for (int d = 0; d < dim; ++d)
                    offset_center[d] = (index[d] /*+ (T).5*/) * scaled_ref_box_length[d];
                  offset_new_point = new_point + offset_center;
                  bool inside = true;
                  for (int d = 0; d < dim; ++d)
                    if (offset_new_point[d] < minCorner[d] || offset_new_point[d] > maxCorner[d])
                      inside = false;
                  if (feasible(offset_new_point) && inside)
                    localSamples[id].emplace_back(std::array<T, 3>{
                        offset_new_point[0], offset_new_point[1], offset_new_point[2]});
                }
          }
        }
        for (int id = 0; id < nworkers; id++)
          samples.insert(samples.end(), localSamples[id].begin(), localSamples[id].end());
        puts("done parallel sampling!");
#else
        for (std::size_t i = 0; i < cnt; ++i) {
          const auto &new_point_read = data[i];
          TV new_point, offset_center, offset_new_point;
          for (int d = 0; d < dim; ++d)
            new_point[d] = (minDistance * new_point_read[d]) + minCorner[d];
          IV index{};
          for (index[0] = 0; index[0] < offset_number[0]; ++index[0])
            for (index[1] = 0; index[1] < offset_number[1]; ++index[1])
              for (index[2] = 0; index[2] < offset_number[2]; ++index[2]) {
                for (int d = 0; d < dim; ++d)
                  offset_center[d] = (index[d] /*+ (T).5*/) * scaled_ref_box_length[d];
                offset_new_point = new_point + offset_center;
                bool inside = true;
                for (int d = 0; d < dim; ++d)
                  if (offset_new_point[d] < minCorner[d] || offset_new_point[d] > maxCorner[d])
                    inside = false;
                if (feasible(offset_new_point) && inside)
                  samples.emplace_back(std::array<T, 3>{offset_new_point[0], offset_new_point[1],
                                                        offset_new_point[2]});
              }
        }
#endif
        printf(
            "[PoissonDiskSampling]\tcnt: %d, minDistance: %f, sidelen: %f, "
            "%f, %f, scaledlen: %f, %f, %f\n",
            (int)samples.size(), minDistance, sideLength[0], sideLength[1], sideLength[2],
            scaled_ref_box_length[0], scaled_ref_box_length[1], scaled_ref_box_length[2]);

      } else if (dim == 2) {
        const T h = minDistance / std::sqrt((T)dim);
        /**
          Set up background grid
          dx should be bounded by _minDistance / sqrt(dim)
          background_grid is a MultiArray which keeps tracks the indices of points
          in samples. the value of background_grid is initialized to be -1,
          meaning that there is no particle in that cell
          */
        TV cell_numbers_candidate = maxCorner - minCorner;
        IV cell_numbers;
        for (int d = 0; d < dim; ++d) cell_numbers[d] = std::ceil(cell_numbers_candidate[d] / h);

        DenseGrid<int, int, dim> grid{cell_numbers, -1};
        printf("len(%d, %d) coord(%d, %d) index(%d) val(%d)\n", cell_numbers[0], cell_numbers[1], 2,
               3, grid.offset(vec<int, dim>{2, 3}), grid(vec<int, dim>{2, 3}));
        // Set up active list
        std::vector<int> active_list{};
        {
          // Generate a random point within the range and append it to samples and
          // active list
          TV first_point = (T).5 * (maxCorner + minCorner);
          while (!feasible(first_point))
            for (int d = 0; d < dim; ++d) {
              T r = rnd.randReal();
              first_point[d] = minCorner[d] + (maxCorner[d] - minCorner[d]) * r;
            }
          samples.push_back(first_point.to_array());
          grid(worldToIndexSpace(first_point, h)) = 0;
          active_list.emplace_back(0);
        }
        /**
          While active_list is non-zero, do step 2 in Bridson's proposed algorithm
          */
        while (active_list.size()) {
          // Get a random index from the active list and find the point
          // corresponding to it
          int random_index = rnd.randInt(0, active_list.size() - 1);
          const TV &current_point = TV::from_array(samples[active_list[random_index]]);
          // Swap random index with the last element in the active list so that we
          // can pop_back if found_at_least_one is false at the end of this
          // procedure
          std::iter_swap(active_list.begin() + random_index, active_list.end() - 1);
          // Generate up to _maxAttempts points in the annulus of radius _h and 2h
          // around current_point
          bool found_at_least_one = false;
          for (int i = 0; i < maxAttempts; ++i) {
            TV new_point = generateRandomPointAroundAnnulus(current_point);
            // If periodic and new_point is outside of the _minCorner, _maxCorner
            // shift it to be inside
            if (periodic)
              for (int d = 0; d < dim; ++d) {
                if (new_point[d] < minCorner[d])
                  new_point[d] += maxCorner[d] - minCorner[d];
                else if (new_point[d] > maxCorner[d])
                  new_point[d] -= maxCorner[d] - minCorner[d];
              }

            if (!feasible(new_point)) continue;
            if (checkDistance(new_point, h, grid, samples)) {
              found_at_least_one = true;
              // Add new_point to samples
              samples.push_back(new_point.to_array());
              int index = (int)samples.size() - 1;
              // Add new_point to active list
              active_list.push_back(index);
              // Update background_grid
              grid(worldToIndexSpace(new_point, h)) = index;
            }
          }
          // If not found at least one, remove random_index from active list
          if (!found_at_least_one)  // active_list.erase(active_list.begin()+random_index);
            active_list.pop_back();
        }
      }
      return samples;
    }

  protected:
  };

}  // namespace zs
