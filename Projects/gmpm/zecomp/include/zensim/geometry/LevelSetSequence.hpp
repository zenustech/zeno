#pragma once
#include <type_traits>

#include "LevelSet.h"
#include "LevelSetInterface.h"
#include "SparseLevelSet.hpp"
#include "zensim/container/RingBuffer.hpp"

namespace zs {

  template <int WindowSize, typename DataType, int dim_, typename Tn = int> struct LevelSetSequence
      : RingBuffer<LevelSet<DataType, dim_, Tn>, WindowSize> {
    static constexpr int dim = dim_;
    using ls_t = LevelSet<DataType, dim, Tn>;
    using base_t = RingBuffer<ls_t, WindowSize>;
    using T = typename ls_t::T;
    using TV = typename ls_t::TV;

    template <typename Allocator = heap_allocator>
    constexpr void loadVdbFile(const std::string &fn, T dx,
                               Allocator &&allocator = heap_allocator{}) {
      base_t::push_back();
      ls_t &ls = base_t::back();
      ls.setDx(dx);
      ls.constructPhiVelFromVdbFile(fn, std::forward<Allocator>(allocator));
      return;
    }
    constexpr void roll() { base_t::pop(); }

    constexpr ls_t &ls(int index) noexcept { return (*this)[index]; }
    constexpr ls_t const &ls(int index) const noexcept { return (*this)[index]; }

    constexpr void setDt(T dtIn) noexcept { dt = dtIn; }
    constexpr void setAlpha(T ratio) noexcept { alpha = ratio; }
    constexpr T getSignedDistance(const TV &X) const noexcept {
      TV v = (ls(0).getMaterialVelocity(X) + ls(1).getMaterialVelocity(X)) / 2;
      TV X0 = X - alpha * dt * v, X1 = X + (1 - alpha) * dt * v;
      return (1 - alpha) * ls(0).getSignedDistance(X0) + alpha * ls(1).getSignedDistance(X1);
    }
    constexpr TV getNormal(const TV &X) const noexcept {
      TV v = (ls(0).getMaterialVelocity(X) + ls(1).getMaterialVelocity(X)) / 2;
      TV X0 = X - alpha * dt * v, X1 = X + (1 - alpha) * dt * v;
      return (1 - alpha) * ls(0).getNormal(X0) + alpha * ls(1).getNormal(X1);
    }
    constexpr TV getMaterialVelocity(const TV &X) const noexcept {
      TV v = (ls(0).getMaterialVelocity(X) + ls(1).getMaterialVelocity(X)) / 2;
      TV X0 = X - alpha * dt * v, X1 = X + (1 - alpha) * dt * v;
      return (1 - alpha) * ls(0).getMaterialVelocity(X0) + alpha * ls(1).getMaterialVelocity(X1);
    }
    constexpr decltype(auto) getBoundingBox() const noexcept { return ls(0).getBoundingBox(); }

  protected:
    T dt, alpha;  ///< [0, 1]
  };

  template <typename LS> struct LevelSetWindow {
    using ls_t = LS;
    static constexpr int dim = ls_t::dim;
    using T = typename ls_t::value_type;
    using TV = typename ls_t::TV;
    static_assert(std::is_floating_point_v<T>, "");

    constexpr LevelSetWindow(T dt = 0, T alpha = 0) : stepDt{dt}, alpha{alpha} {}
    ~LevelSetWindow() = default;

    /// push
    template <typename LsT, enable_if_t<std::is_assignable_v<ls_t, LsT>> = 0> void push(LsT &&ls) {
      st = std::move(ed);
      ed = FWD(ls);
      alpha = 0;
    }
    /// frame dt
    void setDeltaT(T time) { stepDt = time; }
    /// advance
    void advance(T ratio) { alpha += ratio; }

    ls_t st;
    ls_t ed;
    T stepDt, alpha;
  };

  template <execspace_e, typename LevelSetWindowT, typename = void> struct LevelSetWindowProxy;

  template <execspace_e Space, typename LevelSetWindowT>
  struct LevelSetWindowProxy<Space, LevelSetWindowT>
      : LevelSetInterface<LevelSetWindowProxy<Space, LevelSetWindowT>, typename LevelSetWindowT::T,
                          LevelSetWindowT::dim> {
    using ls_t = typename LevelSetWindowT::ls_t;
    static constexpr int dim = LevelSetWindowT::dim;
    using T = typename LevelSetWindowT::T;
    using TV = typename LevelSetWindowT::TV;

    constexpr LevelSetWindowProxy() = default;
    ~LevelSetWindowProxy() = default;
    explicit constexpr LevelSetWindowProxy(LevelSetWindowT &ls)
        : st{ls.st.self()}, ed{ls.ed.self()}, stepDt{ls.stepDt}, alpha{ls.alpha} {}

    constexpr T getSignedDistance(const TV &X) const noexcept {
      TV v = (st.getMaterialVelocity(X) + ed.getMaterialVelocity(X)) / 2;
      TV X0 = X - alpha * stepDt * v, X1 = X + (1 - alpha) * stepDt * v;
      return (1 - alpha) * st.getSignedDistance(X0) + alpha * ed.getSignedDistance(X1);
    }
    constexpr TV getNormal(const TV &X) const noexcept {
      TV v = (st.getMaterialVelocity(X) + ed.getMaterialVelocity(X)) / 2;
      TV X0 = X - alpha * stepDt * v, X1 = X + (1 - alpha) * stepDt * v;
      return (1 - alpha) * st.getNormal(X0) + alpha * ed.getNormal(X1);
    }
    constexpr TV getMaterialVelocity(const TV &X) const noexcept {
      TV v = (st.getMaterialVelocity(X) + ed.getMaterialVelocity(X)) / 2;
      TV X0 = X - alpha * stepDt * v, X1 = X + (1 - alpha) * stepDt * v;
      return (1 - alpha) * st.getMaterialVelocity(X0) + alpha * ed.getMaterialVelocity(X1);
    }
    constexpr decltype(auto) getBoundingBox() const noexcept { return st.getBoundingBox(); }

    SparseLevelSetProxy<Space, ls_t> st, ed;
    T stepDt, alpha;
  };

  template <execspace_e ExecSpace, typename LS>
  constexpr decltype(auto) proxy(LevelSetWindow<LS> &levelset) {
    return LevelSetWindowProxy<ExecSpace, LevelSetWindow<LS>>{levelset};
  }

}  // namespace zs
