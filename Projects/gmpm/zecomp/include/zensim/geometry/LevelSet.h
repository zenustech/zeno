#pragma once
#include <fstream>
#include <type_traits>

#include "LevelSetInterface.h"
#include "VdbLevelSet.h"
// #include "zensim/execution/Concurrency.h"
#include "zensim/math/Vec.h"
#include "zensim/memory/Allocator.h"
#include "zensim/types/RuntimeStructurals.hpp"

namespace zs {

  /***************************************************************************/
  /****************************** height field *******************************/
  /***************************************************************************/

  template <typename DataType, typename Tn_ = int> struct HeightField
      : LevelSetInterface<HeightField<DataType, Tn_>, DataType, 3> {
    using T = DataType;
    using Tn = Tn_;
    static constexpr int dim = 3;
    using TV = vec<T, dim>;
    using IV = vec<Tn, dim>;

    using Arena = vec<T, 2, 2, 2>;
    // using VecArena = vec<TV, 2, 2, 2>;

    template <typename Allocator>
    HeightField(Allocator &&allocator, T dx, Tn const &x, Tn const &y, Tn const &z)
        : _dx{dx}, _extent{x, y, z}, _min{TV::zeros()} {
      _field = (T *)allocator.allocate((std::size_t)x * z * sizeof(T));
    }

    constexpr void setDx(T dx) noexcept { _dx = dx; }
    constexpr void setOffset(T dx, T dy, T dz) noexcept { _min = TV{dx, dy, dz}; }

    template <typename Allocator = heap_allocator>
    void constructFromTxtFile(const std::string &fn, Allocator &&allocator = heap_allocator{}) {
      std::ifstream is(fn, std::ios::in);
      if (!is.is_open()) {
        printf("%s not found!\n", fn.c_str());
        return;
      }
      for (int x = 0; x < _extent(0); ++x)
        for (int z = 0; z < _extent(2); ++z) is >> entry(x, z);
      is.close();
    }

    constexpr auto &entry(Tn const &x, Tn const &z) noexcept { return _field[x * _extent(2) + z]; }
    constexpr auto const &entry(Tn const &x, Tn const &z) const noexcept {
      return _field[x * _extent(2) + z];
    }
    constexpr bool inside(const IV &X) const noexcept {
      if (X(0) >= _extent(0) || X(1) >= _extent(1) || X(2) >= _extent(2)) return false;
      return true;
    }
    template <std::size_t d, typename Field>
    constexpr auto trilinear_interop(const TV &diff, const Field &arena) const noexcept {
      if constexpr (d == dim - 1) {
        return linear_interop(diff(d), arena(0), arena(1));
      } else
        return linear_interop(diff(d), trilinear_interop<d + 1>(diff, arena[0]),
                              trilinear_interop<d + 1>(diff, arena[1]));
      return (T)1;
    }

    constexpr T getSignedDistance(const TV &X) const noexcept {
      /// world to local
      Arena arena{};
      IV loc{};
      for (int d = 0; d < dim; ++d) loc(d) = gcem::floor((X(d) - _min(d)) / _dx);
      TV diff = (X - _min) / _dx - loc;
      {
        for (Tn dx = 0; dx < 2; dx++)
          for (Tn dy = 0; dy < 2; dy++)
            for (Tn dz = 0; dz < 2; dz++) {
              if (inside(IV{loc(0) + dx, loc(1) + dy, loc(2) + dz})) {
                T h = entry(loc(0) + dx, loc(2) + dz);
                arena(dx, dy, dz) = (loc(1) + dy) * _dx - h;
              } else
                arena(dx, dy, dz) = 2 * _dx;
            }
      }
      return trilinear_interop<0>(diff, arena);
    }
    constexpr TV getNormal(const TV &X) const noexcept { return TV{0, 1, 0}; }
    constexpr TV getMaterialVelocity(const TV &X) const noexcept { return TV::zeros(); }
    constexpr decltype(auto) getBoundingBox() const noexcept {
      return std::make_tuple(_min, _min + _extent * _dx);
    }

    IV _extent;
    T _dx;

  private:
    T *_field;
    TV _min;
  };

  /***************************************************************************/
  /**************************** general levelset *****************************/
  /***************************************************************************/
  template <int Dim, typename T, typename... Tn> using levelset_snode
      = ds::snode_t<ds::dynamic_decorator,
                    ds::dynamic_domain<0, tuple<Tn...>, std::index_sequence_for<Tn...>>,
                    tuple<T, T>, vseq_t<1, Dim>  ///< signed distances and its gradient
                    >;
  template <typename T, typename... Tn> using levelset_instance
      = ds::instance_t<ds::dense, levelset_snode<sizeof...(Tn), T, Tn...>>;

  template <typename DataType, typename IndexTypes> struct LevelSetImpl;
  template <typename DataType, int dim, typename Tn = int> using LevelSet
      = LevelSetImpl<DataType, typename gen_seq<dim>::template uniform_types_t<tseq_t, Tn>>;

  //#define ENABLE_VELOCITY

  template <typename DataType, typename... Tn> struct LevelSetImpl<DataType, tseq_t<Tn...>>
      : LevelSetInterface<LevelSetImpl<DataType, tseq_t<Tn...>>, DataType, sizeof...(Tn)>,
        levelset_instance<DataType, Tn...> {
    using T = DataType;
    using Ti = std::common_type_t<Tn...>;
    static constexpr int dim = sizeof...(Tn);
    using TV = vec<T, dim>;
    using IV = vec<Ti, dim>;
    using Arena = vec<T, (is_same_v<Tn, Tn> ? 2 : 1)...>;
    using VecArena = vec<TV, (is_same_v<Tn, Tn> ? 2 : 1)...>;

    using ls = levelset_instance<DataType, Tn...>;

    constexpr ls &self() noexcept { return static_cast<ls &>(*this); }
    constexpr ls const &self() const noexcept { return static_cast<ls const &>(*this); }

    template <typename Allocator, typename... Dims>
    constexpr auto buildInstance(Allocator &&allocator, Dims &&...dims) {
      using namespace ds;
      auto dec = dynamic_decorator{soa, compact, alloc_ahead};
      /// not sure why this domain type could cause deduction problem
      dynamic_domain<0, tuple<Tn...>, std::index_sequence_for<Tn...>> dom{
          wrapv<0>{}, (int)std::forward<Dims>(dims)...};
      auto node = snode{dec, dom, zs::make_tuple(wrapt<T>{}, wrapt<T>{}), vseq_t<1, dim>{}};
      auto inst = instance{wrapv<dense>{}, zs::make_tuple(node)};
      inst.alloc(allocator, (std::size_t)512);
      return inst;
    }
    constexpr auto defaultInstance() {
      using namespace ds;
      auto dec = dynamic_decorator{};
      /// not sure why this domain type could cause deduction problem
      dynamic_domain<0, tuple<Tn...>, std::index_sequence_for<Tn...>> dom{};
      auto node = snode{dec, dom, zs::make_tuple(wrapt<T>{}, wrapt<T>{}), vseq_t<1, dim>{}};
      auto inst = instance{wrapv<dense>{}, zs::make_tuple(node)};
      return inst;
    }

    constexpr LevelSetImpl(T dx = 1.f) : ls{defaultInstance()}, _dx{dx} {}
    template <typename Allocator, typename... Dims, enable_if_t<sizeof...(Dims) == dim> = 0>
    constexpr LevelSetImpl(Allocator &&allocator, Dims &&...dims)
        : ls{buildInstance(std::forward<Allocator>(allocator), std::forward<Dims>(dims)...)},
          _dx{1} {}
    template <typename Allocator, typename... Dims, enable_if_t<sizeof...(Dims) == dim> = 0>
    constexpr LevelSetImpl(Allocator &&allocator, T dx, Dims &&...dims)
        : ls{buildInstance(std::forward<Allocator>(allocator), std::forward<Dims>(dims)...)},
          _dx{dx} {}

    constexpr void setDx(T dx) noexcept { _dx = dx; }
    template <typename Allocator = heap_allocator>
    void constructFromVdbFile(const std::string &fn, Allocator &&allocator = heap_allocator{}) {
      const auto res = readPhiFromVdbFile(fn, _dx);
      const auto &g = res.template get<0>();
      _min = res.template get<1>();
      _max = res.template get<2>();

      if constexpr (dim == 2)
        self() = buildInstance(allocator, g.domain(0), g.domain(1));
      else if constexpr (dim == 3)
        self() = buildInstance(allocator, g.domain(0), g.domain(1), g.domain(2));
      else
        ZS_UNREACHABLE;

      const auto dom = self().node().get_extents();
      if constexpr (dim == 2) {
#pragma omp parallel for
        for (int x = 0; x < dom(0); ++x)
          for (int y = 0; y < dom(1); ++y) {
            self()(wrapv<0>{}, x, y) = g({x, y});
          }
      } else if constexpr (dim == 3) {
#pragma omp parallel for
        for (int x = 0; x < dom(0); ++x)
          for (int y = 0; y < dom(1); ++y)
            for (int z = 0; z < dom(2); ++z) {
              self()(wrapv<0>{}, x, y, z) = g({x, y, z});
            }
      }
    }
    template <typename Allocator = heap_allocator>
    void constructPhiVelFromVdbFile(const std::string &fn,
                                    Allocator &&allocator = heap_allocator{}) {
      const auto res = readPhiVelFromVdbFile(fn, _dx);
      const auto &g = res.template get<0>();
      const auto &vg = res.template get<1>();
      _min = res.template get<2>();
      _max = res.template get<3>();

      if constexpr (dim == 2)
        self() = buildInstance(allocator, g.domain(0), g.domain(1));
      else if constexpr (dim == 3)
        self() = buildInstance(allocator, g.domain(0), g.domain(1), g.domain(2));
      else
        ZS_UNREACHABLE;

      const auto dom = self().node().get_extents();
      if constexpr (dim == 2) {
#pragma omp parallel for
        for (int x = 0; x < dom(0); ++x)
          for (int y = 0; y < dom(1); ++y) {
            self()(wrapv<0>{}, x, y) = g({x, y});
            const auto &vel = vg({x, y});
            for (int d = 0; d < 2; ++d) self()(wrapv<1>{}, d, x, y) = vel(d);
          }
      } else if constexpr (dim == 3) {
#pragma omp parallel for
        for (int x = 0; x < dom(0); ++x)
          for (int y = 0; y < dom(1); ++y)
            for (int z = 0; z < dom(2); ++z) {
              self()(wrapv<0>{}, x, y, z) = g({x, y, z});
              const auto &vel = vg({x, y, z});
              for (int d = 0; d < 3; ++d) self()(wrapv<1>{}, d, x, y, z) = vel(d);
            }
      }
    }
    template <std::size_t d, typename Field>
    constexpr auto trilinear_interop(const TV &diff, const Field &arena) const noexcept {
      if constexpr (d == dim - 1)
        return linear_interop(diff(d), arena(0), arena(1));
      else
        return linear_interop(diff(d), trilinear_interop<d + 1>(diff, arena[0]),
                              trilinear_interop<d + 1>(diff, arena[1]));
      return (T)1;
    }

    constexpr T getSignedDistance(const TV &X) const noexcept {
      /// world to local
      Arena arena{};
      IV loc{};
      for (int d = 0; d < dim; ++d) loc(d) = gcem::floor((X(d) - _min(d)) / _dx);
      TV diff = (X - _min) / _dx - loc;
      if constexpr (dim == 2) {
        for (Ti dx = 0; dx < 2; dx++)
          for (Ti dy = 0; dy < 2; dy++)
            if (self().node().inside(IV{loc(0) + dx, loc(1) + dy}))
              arena(dx, dy) = self()(wrapv<0>{}, loc(0) + dx, loc(1) + dy);
            else
              arena(dx, dy) = 2 * _dx;
      } else if constexpr (dim == 3) {
        for (Ti dx = 0; dx < 2; dx++)
          for (Ti dy = 0; dy < 2; dy++)
            for (Ti dz = 0; dz < 2; dz++)
              if (self().node().inside(IV{loc(0) + dx, loc(1) + dy, loc(2) + dz}))
                arena(dx, dy, dz) = self()(wrapv<0>{}, loc(0) + dx, loc(1) + dy, loc(2) + dz);
              else
                arena(dx, dy, dz) = 2 * _dx;
      } else
        ZS_UNREACHABLE;
      return trilinear_interop<0>(diff, arena);
    }
    constexpr TV getNormal(const TV &X) const noexcept {
      TV diff{}, v1{}, v2{};
      T eps = (T)1e-6;
      /// compute a local partial derivative
      for (int i = 0; i < dim; i++) {
        v1 = X;
        v2 = X;
        v1(i) = X(i) + eps;
        v2(i) = X(i) - eps;
        diff(i) = (getSignedDistance(v1) - getSignedDistance(v2)) / (eps + eps);
      }
      return diff.normalized();
    }
    constexpr TV getMaterialVelocity(const TV &X) const noexcept {
#ifdef ENABLE_VELOCITY
      /// world to local
      VecArena arena{};
      IV loc{};
      for (int d = 0; d < dim; ++d) loc(d) = gcem::floor((X(d) - _min(d)) / _dx);
      TV diff = (X - _min) / _dx - loc;
      // printf("diff [%f, %f, %f]\n", diff(0), diff(1), diff(2));
      if constexpr (dim == 2) {
        for (Ti dx = 0; dx < 2; dx++)
          for (Ti dy = 0; dy < 2; dy++)
            if (self().node().inside(IV{loc(0) + dx, loc(1) + dy})) {
              TV v{};
              for (int d = 0; d < 2; ++d) v(d) = self()(wrapv<1>{}, d, loc(0) + dx, loc(1) + dy);
              arena(dx, dy) = v;
            } else
              arena(dx, dy) = TV::zeros();
      } else if constexpr (dim == 3) {
        for (Ti dx = 0; dx < 2; dx++)
          for (Ti dy = 0; dy < 2; dy++)
            for (Ti dz = 0; dz < 2; dz++)
              if (self().node().inside(IV{loc(0) + dx, loc(1) + dy, loc(2) + dz})) {
                TV v{};
                for (int d = 0; d < 3; ++d)
                  v(d) = self()(wrapv<1>{}, d, loc(0) + dx, loc(1) + dy, loc(2) + dz);
                arena(dx, dy, dz) = v;
              } else
                arena(dx, dy, dz) = TV::zeros();
      } else
        ZS_UNREACHABLE;
      return trilinear_interop<0>(diff, arena);
#else
      return TV::zeros();
#endif
    }
    constexpr decltype(auto) getBoundingBox() const noexcept { return std::make_tuple(_min, _max); }

  private:
    TV _min, _max;
    T _dx;
  };

}  // namespace zs
