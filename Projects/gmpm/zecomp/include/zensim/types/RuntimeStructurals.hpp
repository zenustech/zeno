#pragma once
/// reference:
/// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2020/p2043r0.pdf

#include <numeric>
#include <type_traits>
#include <utility>

#include "Function.h"
#include "Property.h"
#include "zensim/Platform.hpp"
#include "zensim/math/Vec.h"
#include "zensim/meta/ControlFlow.h"
#include "zensim/meta/Sequence.h"
#include "zensim/resource/Resource.h"
#include "zensim/types/Polymorphism.h"
#include "zensim/types/SmallVector.hpp"
#include "zensim/types/Tuple.h"

namespace zs {

  namespace ds {

    /******************************************************************/
    /**                        domain signature                       */
    /**  extent, offset                                               */
    /******************************************************************/

    /******************************************************************/
    /**                        static domain                          */
    /******************************************************************/
    template <typename VSeq, typename> struct static_domain_impl;

    template <auto... Ns, typename Map> struct static_domain_impl<vseq_t<Ns...>, Map> {
      static constexpr auto N = sizeof...(Ns);
      using extents = typename vseq_t<Ns...>::template shuffle<Map>;
      using Tn = typename extents::Tn;
      using weights = typename extents::template scan<multiplies<Tn>, 2>;
      static constexpr auto extent() noexcept { return (Ns * ...); }
      static constexpr auto extent(unsigned dim) noexcept {
        constexpr Tn arr[] = {Ns...};
        return arr[dim];
      }
      static constexpr auto get_extents() noexcept { return make_vec(Ns...); }
      template <typename Tn>
      constexpr bool inside(const vec<Tn, sizeof...(Ns)> &coord) const noexcept {
        return extents::map_reduce(
            [&coord](std::size_t I, const auto &N) { return coord(I) < N && coord(I) >= 0; },
            logical_and<bool>{});
      }
      template <typename Tn>
      constexpr auto offset(const vec<Tn, sizeof...(Ns)> &coord) const noexcept {
        return weights::map_reduce([&coord](const auto &i, const auto &n) { return coord[i] * n; },
                                   plus<Tn>{});
      }
    };

    template <auto... Ns> using static_domain
        = static_domain_impl<vseq_t<Ns...>, typename gen_seq<sizeof...(Ns)>::ascend>;

    /******************************************************************/
    /**                        dynamic domain                         */
    /******************************************************************/
    template <std::size_t Cate, typename Index, typename Map> struct dynamic_domain;

    template <typename... Tn, std::size_t... Is>
    struct dynamic_domain<0, tuple<Tn...>, std::index_sequence<Is...>> {
      static constexpr auto N = sizeof...(Tn);
      using id = tuple<Tn...>;
      constexpr dynamic_domain() noexcept : domain{(Is >= 0 ? 1 : 0)...} {}
      template <typename... Tn_> constexpr dynamic_domain(wrapv<0>, Tn_ &&...dims) noexcept
          : domain{std::forward<Tn_>(dims)...} {}

      constexpr auto extent() const noexcept {
        return domain.reduce([](auto const &...ns) { return (ns * ...); });
      }
      template <typename Tn_> constexpr auto extent(Tn_ dim) const noexcept {
        std::size_t sizes[N]{};
        ((sizes[Is] = zs::get<Is>(domain)), ...);
        return sizes[dim];
      }
      constexpr auto get_extents() const noexcept { return make_vec(zs::get<Is>(domain)...); }
      template <typename Tn_> constexpr bool inside(const vec<Tn_, N> &coord) const noexcept {
        return ((coord[Is] >= 0 && coord[Is] < zs::get<Is>(domain)) && ...);
      }
      template <typename Tn_> constexpr auto offset(const vec<Tn_, N> &coord) const noexcept {
        auto const weights
            = domain.shuffle(std::index_sequence<Is...>{})
                  .excl_suffix_scan([](auto const &a, auto const &b) { return a * b; }, 1);
        auto const c = id{static_cast<zs::tuple_element_t<Is, id>>(coord[Is])...};
        return weights.reduce([](auto const &a, auto const &b) { return a * b; },
                              [](auto const &...ns) { return (ns + ...); }, c);
      }

      id domain;
    };

    template <typename... Tn, std::size_t... Is>
    struct dynamic_domain<1, tuple<Tn...>, std::index_sequence<Is...>> {
      static constexpr auto N = sizeof...(Tn);
      using id = tuple<Tn...>;
      constexpr dynamic_domain() noexcept
          : domain{(Is >= 0 ? 1 : 0)...}, weights{(Is >= 0 ? 1 : 0)...} {}
      template <typename... Tn_> constexpr dynamic_domain(wrapv<1>, Tn_ &&...dims) noexcept
          : domain{std::forward<Tn_>(dims)...},
            weights{domain.shuffle(std::index_sequence<Is...>{})
                        .excl_suffix_scan([](auto const &a, auto const &b) { return a * b; }, 1)} {}

      constexpr auto extent() const noexcept {
        return domain.reduce([](auto const &...ns) { return (ns * ...); });
      }
      template <typename Tn_> constexpr auto extent(Tn_ dim) const noexcept {
        std::size_t sizes[N]{};
        ((sizes[Is] = zs::get<Is>(domain)), ...);
        return sizes[dim];
      }
      constexpr auto get_extents() const noexcept { return make_vec(zs::get<Is>(domain)...); }
      template <typename Tn_> constexpr bool inside(const vec<Tn_, N> &coord) const noexcept {
        return ((coord[Is] >= 0 && coord[Is] < zs::get<Is>(domain)) && ...);
      }
      template <typename Tn_> constexpr auto offset(const vec<Tn_, N> &coord) const noexcept {
        auto const c = id{coord[Is]...};
        return weights.reduce([](auto const &a, auto const &b) { return a * b; },
                              [](auto const &...ns) { return (ns + ...); }, c);
      }

      id domain, weights;
    };

    template <auto Cate, typename... Tn> dynamic_domain(wrapv<Cate>, Tn &&...ns)
        -> dynamic_domain<Cate, tuple<std::decay_t<Tn>...>, std::index_sequence_for<Tn...>>;

    /******************************************************************/
    /**                        uniform domain                         */
    /******************************************************************/
    template <std::size_t Cate, typename Ti, auto dim, typename Map> struct uniform_domain;

    template <typename Ti, auto dim_, std::size_t... Is>
    struct uniform_domain<0, Ti, dim_, std::index_sequence<Is...>> {
      static constexpr auto N = dim_;
      using id = vec<Ti, N>;
      constexpr uniform_domain() noexcept : domain{(Is >= 0 ? 1 : 0)...} {}
      template <typename... Tn_> constexpr uniform_domain(wrapv<0>, Tn_ &&...dims) noexcept
          : domain{static_cast<Ti>(std::forward<Tn_>(dims))...} {}

      constexpr auto extent() const noexcept { return domain.prod(); }
      template <typename Tn_> constexpr auto extent(Tn_ dim) const noexcept { return domain[dim]; }
      constexpr auto get_extents() const noexcept { return domain; }

      template <typename Tn_> constexpr bool inside(vec<Tn_, N> const &coord) const noexcept {
        return ((coord[Is] >= 0 && coord[Is] < domain(Is)) && ...);
      }
      template <typename Tn_> constexpr auto offset(vec<Tn_, N> const &coord) const noexcept {
        Ti weight = 1, res = 0;
        for (int d = N - 1; d >= 0; weight *= domain(d), --d) res += weight * coord[d];
        return res;
      }

      id domain;
    };

    template <auto Cate, typename... Tn> uniform_domain(wrapv<Cate>, Tn &&...ns)
        -> uniform_domain<Cate, std::common_type_t<Tn...>, sizeof...(Tn),
                          std::index_sequence_for<Tn...>>;

    /******************************************************************/
    /**                          decorator                            */
    /**  padding, layout, etc...                                      */
    /******************************************************************/
    /// for snode
    enum padding_policy : char { compact = 0, sum_pow2_align = 1, max_pow2_align = 2 };
    template <padding_policy v> struct padding_policy_t : wrapv<v> {};
    template <padding_policy v, enable_if_t<(v >= compact && v <= max_pow2_align)> = 0>
    padding_policy_t<v> snode_policy() {
      return padding_policy_t<v>{};
    }

    enum layout_policy : char { aos = 10, soa = 11 };
    template <layout_policy v> struct layout_policy_t : wrapv<v> {};
    template <layout_policy v, enable_if_t<(v >= aos && v <= soa)> = 0>
    layout_policy_t<v> snode_policy() {
      return layout_policy_t<v>{};
    }

    enum alloc_policy : char { alloc_ahead = 100, alloc_on_demand = 101 };
    template <alloc_policy v> struct alloc_policy_t : wrapv<v> {};
    template <alloc_policy v, enable_if_t<(v >= alloc_ahead && v <= alloc_on_demand)> = 0>
    alloc_policy_t<v> snode_policy() {
      return alloc_policy_t<v>{};
    }

    struct default_decorator {
      static constexpr layout_policy layoutp{layout_policy::aos};
      static constexpr padding_policy paddingp{padding_policy::compact};
      static constexpr alloc_policy allocp{alloc_policy::alloc_ahead};
    };
    template <layout_policy layoutIn = default_decorator::layoutp,
              padding_policy paddingIn = default_decorator::paddingp,
              alloc_policy allocIn = default_decorator::allocp>
    struct static_decorator {
      static constexpr layout_policy layoutp = layoutIn;
      static constexpr padding_policy paddingp = paddingIn;
      static constexpr alloc_policy allocp = allocIn;
    };
    struct dynamic_decorator {
      layout_policy layoutp{default_decorator::layoutp};
      padding_policy paddingp{default_decorator::paddingp};
      alloc_policy allocp{default_decorator::allocp};
    };

    template <typename> struct vsetter;
    template <padding_policy paddingIn> struct vsetter<padding_policy_t<paddingIn>>
        : public virtual default_decorator {
      static constexpr padding_policy paddingp = paddingIn;
    };
    template <layout_policy layoutIn> struct vsetter<layout_policy_t<layoutIn>>
        : public virtual default_decorator {
      static constexpr layout_policy layoutp = layoutIn;
    };
    template <alloc_policy allocIn> struct vsetter<alloc_policy_t<allocIn>>
        : public virtual default_decorator {
      static constexpr alloc_policy allocp = allocIn;
    };
    template <std::size_t, typename Base> class type_discriminator : public virtual Base {};
    template <typename Setter1 = default_decorator, typename Setter2 = default_decorator,
              typename Setter3 = default_decorator>
    struct decorator_compositor : public type_discriminator<1, Setter1>,
                                  public type_discriminator<2, Setter2>,
                                  public type_discriminator<3, Setter3> {};

    template <typename> struct decorator_extracter;
    template <typename DecorationSetter> struct decorator_extracter {
      using type = static_decorator<DecorationSetter::layoutp, DecorationSetter::paddingp,
                                    DecorationSetter::allocp>;
    };
    template <auto... Settings> using decorations = typename decorator_extracter<
        decorator_compositor<vsetter<decltype(snode_policy<Settings>())>...>>::type;

    /******************************************************************/
    /**                       snode signature                         */
    /**   get_attrib (by const), get_channel, get_element             */
    /**                       snode parameter                         */
    /**   decorator, domain, attribs, channel counts                  */
    /**   size, child_seq, count_seq, attrib_offset, element_stride   */
    /**   dynamic in count_seq, decorator, */
    /******************************************************************/
    /// inherit from all types
    // these specific types (fundamental, vec) are automatically treated as valueT
    template <typename T> constexpr bool auto_value_type() {
      return std::is_fundamental<std::decay_t<T>>::value || is_vec<std::decay_t<T>>::value;
    }
    template <typename T> using snode_attrib_wrapper
        = conditional_t<auto_value_type<T>(), wrapt<std::decay_t<T>>, std::decay_t<T>>;

    template <typename T> struct snode_attrib_unwrapper { using type = std::decay_t<T>; };
    template <typename T> struct snode_attrib_unwrapper<wrapt<T>> {
      using type = conditional_t<auto_value_type<std::decay_t<T>>(), std::decay_t<T>,
                                 wrapt<std::decay_t<T>>>;
    };

    template <typename T> struct extract_snode_type;
    template <typename T> struct extract_snode_type : T {
      static constexpr auto value = false;
      using type = T;
      using type::alignment;
    };
    template <typename T> struct extract_snode_type<wrapt<T>> {
      static constexpr auto value = true;
      using type = T;
    };

    template <typename T> constexpr std::size_t snode_size(T &&snode) noexcept {
      if constexpr (extract_snode_type<std::decay_t<T>>::value)
        return sizeof(typename extract_snode_type<std::decay_t<T>>::type);
      /// reserved for future variant-channel snode
      // else if constexpr (is_variant<T>::value)
      //  return match([](auto &&node) { return snode_size(node); })(snode);
      else
        return snode.size();
      return 0;
    }
    /// variant-channel snode should take in a node as in *snode_size*
    template <typename T> constexpr std::size_t snode_alignment() noexcept {
      using Snode = T;
      if constexpr (Snode::value)
        return alignof(typename Snode::type);
      else
        return Snode::alignment();
      return 1;
    }

    /// decorator & domain always flexible
    /// separate definitions for channel counts & child snodes due to tuple issues
    template <typename Decorator, typename Domain, typename Attribs, typename Cnts,
              typename Indices>
    struct snode;

    template <typename Decorator, typename Domain, typename Attribs, typename Cnts> using snode_t
        = snode<Decorator, Domain, Attribs, Cnts, std::make_index_sequence<Attribs::tuple_size>>;

    /// alignment solution
    /// 1. https://stackoverflow.com/questions/24788262/why-alignment-is-power-of-2
    /// 2.
    /// https://stackoverflow.com/questions/54384522/will-sizeof-always-be-a-multiple-of-alignof
    /// compute an internal mapping to channels (types) during compile time, whose
    /// alignments are in descending order

    /// dynamic attrib + static channel count
    template <typename Decorator, typename Domain, typename... Ts, auto... Ns, std::size_t... Is>
    struct snode<Decorator, Domain, tuple<Ts...>, vseq_t<Ns...>, std::index_sequence<Is...>>
        : Decorator, Domain, tuple<snode_attrib_wrapper<Ts>...>, vseq_t<Ns...> {
      static_assert(sizeof...(Ts) == sizeof...(Ns), "[snode] dimension mismatch");
      using domain = Domain;
      using Decorator::allocp;
      using Decorator::layoutp;
      using Decorator::paddingp;
      using Domain::extent;
      using Domain::N;
      using Domain::offset;

      using attribs = tuple<snode_attrib_wrapper<Ts>...>;
      template <std::size_t I> using attrib_type = extract_snode_type<zs::tuple_element_t<I, attribs>>;
      using channel_counts = vseq_t<Ns...>;

      constexpr auto &self() noexcept { return (attribs &)(*this); }
      constexpr auto const &self() const noexcept { return (attribs const &)(*this); }

      static constexpr std::size_t alignment() noexcept {
        static_assert(
            (((snode_alignment<attrib_type<Is>>() & (snode_alignment<attrib_type<Is>>() - 1)) == 0)
             && ...),
            "child snode alignment(s) not power of two!");
        std::size_t res{1};
        const std::size_t aligns[] = {snode_alignment<attrib_type<Is>>()...};
        for (const auto &n : aligns) res = n > res ? n : res;
        // res = std::lcm(res, n);
        return res;
      }
      constexpr std::size_t align() const noexcept {
        if (allocp == alloc_on_demand)
          return alignof(void *);
        else
          return alignment();
        return static_cast<std::size_t>(1);
      }

      static constexpr auto channel_mapping() noexcept {
        constexpr int n = sizeof...(Ts);
        using Vec = vec<std::size_t, n>;
        Vec alignments{snode_alignment<attrib_type<Is>>()...};
        Vec indices{Is...};
        std::size_t tmp{};
        /// bubble sort
        for (int i = 0; i < n - 1; ++i)
          for (int j = 0; j < n - i - 1; ++j)
            if (alignments(j) < alignments(j + 1)) {
              tmp = alignments(j);
              alignments(j) = alignments(j + 1);
              alignments(j + 1) = tmp;
              tmp = indices(j);
              indices(j) = indices(j + 1);
              indices(j + 1) = tmp;
            }
        for (int i = 0; i < n; ++i) alignments(indices(i)) = i;
        return zs::make_tuple(indices, alignments);
      }
      static constexpr auto chmap = channel_mapping().template get<1>();
      static constexpr auto chsrc = channel_mapping().template get<0>();

      template <auto I> constexpr auto &child(wrapv<I>) noexcept {
        return zs::get<(std::size_t)I>(self());
      }
      template <auto I> constexpr auto const &child(wrapv<I>) const noexcept {
        return zs::get<(std::size_t)I>(self());
      }

      constexpr std::size_t channel_count() const noexcept { return (Ns + ...); }

      template <std::size_t I> constexpr std::size_t attrib_size() const noexcept {
        return snode_size(zs::get<I>(self()));
      }

      constexpr auto ordered_attrib_sizes() const noexcept {
        return zs::make_tuple(attrib_size<chsrc(Is)>()...);
      }

      constexpr std::size_t element_size() const noexcept {
        std::size_t ebytes{((attrib_size<Is>() * Ns) + ...)};
        std::size_t alignbytes = alignment();
        ebytes = (ebytes + alignbytes - 1) / alignbytes * alignbytes;
        return ebytes;
      }
      constexpr std::size_t element_storage_size() const noexcept {
        switch (allocp) {
          case alloc_ahead:
            return element_size();
          case alloc_on_demand:
            return sizeof(void *) * channel_count();
          default:
            ; // static_assert(false, "should not be here!");
        }
        return static_cast<std::size_t>(0);
      }

      constexpr std::size_t size() const noexcept { return extent() * element_storage_size(); }

      /// related to access
      template <std::size_t I> constexpr std::size_t element_stride() const noexcept {
        switch (allocp) {
          case alloc_ahead:
            return (layoutp == aos ? element_storage_size() : attrib_size<I>());
          case alloc_on_demand:
            return sizeof(void *) * (layoutp == aos ? channel_count() : 1);
          default:
            ; // static_assert(false, "should not be here!");
        }
        return static_cast<std::size_t>(0);
      }

      template <std::size_t I, typename Ti = char>
      constexpr std::size_t channel_offset(Ti chno = 0) const noexcept {
        std::size_t attrib_offset = 0;
        constexpr std::size_t chn_counts[] = {select_value<chsrc(Is), channel_counts>::value...};
        {
          const std::size_t attrib_offsets[] = {attrib_size<chsrc(Is)>()...};
          for (int i = 0; i < (int)chmap(I); ++i)
            attrib_offset += attrib_offsets[i] * chn_counts[i];
        }
        if constexpr (is_same_v<Decorator, dynamic_decorator>) {
          switch (allocp) {
            case alloc_ahead:
              return (layoutp == aos ? 1 : extent()) * (attrib_offset + attrib_size<I>() * chno);
            case alloc_on_demand:
              return (layoutp == aos ? 1 : extent()) * (attrib_offset + chno)
                     * sizeof(void *);  // ?
            default:
              ; // static_assert(false, "should not be here!");
          }
        } else {
          if constexpr (allocp == alloc_ahead)
            return (layoutp == aos ? 1 : extent()) * (attrib_offset + attrib_size<I>() * chno);
          else if constexpr (allocp == alloc_on_demand)
            return (layoutp == aos ? 1 : extent()) * (attrib_offset + chno) * sizeof(void *);  // ?
        }
        return static_cast<std::size_t>(0);
      }

      template <auto I, typename Ti, typename... Tis>
      constexpr uintptr_t element_offset(wrapv<I>, Ti chno, Tis &&...coords) const noexcept {
        return channel_offset<I>(chno)
               + offset(make_vec(std::forward<Tis>(coords)...)) * element_stride<I>();
      }
    };

    /// dynamic attrib + dynamic channel count
    template <typename Decorator, typename Domain, typename... Ts, typename... Tn,
              std::size_t... Is>
    struct snode<Decorator, Domain, tuple<Ts...>, tuple<Tn...>, std::index_sequence<Is...>>
        : Decorator, Domain, tuple<snode_attrib_wrapper<Ts>...>, tuple<Tn...> {
      // static_assert(sizeof...(Ts) == sizeof...(Tn), "[snode] dimension mismatch");
      using domain = Domain;
      using Decorator::allocp;
      using Decorator::layoutp;
      using Decorator::paddingp;
      using Domain::extent;
      using Domain::N;
      using Domain::offset;

      using attribs = tuple<snode_attrib_wrapper<Ts>...>;
      template <std::size_t I> using attrib_type = extract_snode_type<zs::tuple_element_t<I, attribs>>;
      using channel_counts = tuple<Tn...>;

      constexpr auto &self() noexcept { return (attribs &)(*this); }
      constexpr auto const &self() const noexcept { return (attribs const &)(*this); }

      static constexpr std::size_t alignment() noexcept {
        static_assert(
            (((snode_alignment<attrib_type<Is>>() & (snode_alignment<attrib_type<Is>>() - 1)) == 0)
             && ...),
            "child snode alignment(s) not power of two!");
        std::size_t res{1};
        const std::size_t aligns[] = {snode_alignment<attrib_type<Is>>()...};
        for (const auto &n : aligns) res = n > res ? n : res;
        // res = std::lcm(res, n);
        return res;
      }
      constexpr std::size_t align() const noexcept {
        if (allocp == alloc_on_demand)
          return alignof(void *);
        else
          return alignment();
        return static_cast<std::size_t>(1);
      }

      static constexpr auto channel_mapping() noexcept {
        constexpr int n = sizeof...(Ts);
        using Vec = vec<std::size_t, n>;
        Vec alignments{snode_alignment<attrib_type<Is>>()...};
        Vec indices{Is...};
        std::size_t tmp{};
        /// bubble sort
        for (int i = 0; i < n - 1; ++i)
          for (int j = 0; j < n - i - 1; ++j)
            if (alignments(j) < alignments(j + 1)) {
              tmp = alignments(j);
              alignments(j) = alignments(j + 1);
              alignments(j + 1) = tmp;
              tmp = indices(j);
              indices(j) = indices(j + 1);
              indices(j + 1) = tmp;
            }
        for (int i = 0; i < n; ++i) alignments(indices(i)) = i;
        return zs::make_tuple(indices, alignments);
      }
      static constexpr auto chmap = channel_mapping().template get<1>();
      static constexpr auto chsrc = channel_mapping().template get<0>();

      template <auto I> constexpr auto &child(wrapv<I>) noexcept {
        return zs::get<(std::size_t)I>(self());
      }
      template <auto I> constexpr auto const &child(wrapv<I>) const noexcept {
        return zs::get<(std::size_t)I>(self());
      }

      constexpr std::size_t channel_count() const noexcept {
        return channel_counts::reduce([](const auto &...ns) { return (ns + ...); });
      }

      template <std::size_t I> constexpr std::size_t attrib_size() const noexcept {
        return snode_size(zs::get<I>(self()));
      }

      constexpr auto ordered_attrib_sizes() const noexcept {
        return zs::make_tuple(attrib_size<chsrc(Is)>()...);
      }

      constexpr std::size_t element_size() const noexcept {
        std::size_t ebytes{channel_counts::reduce(
            multiplies<std::size_t>{}, [](const auto &...ns) { return (ns + ...); },
            ordered_attrib_sizes())};
        std::size_t alignbytes = alignment();
        ebytes = (ebytes + alignbytes - 1) / alignbytes * alignbytes;
        return ebytes;
      }
      constexpr std::size_t element_storage_size() const noexcept {
        if constexpr (is_same_v<Decorator, dynamic_decorator>) {
          switch (allocp) {
            case alloc_ahead:
              return element_size();
            case alloc_on_demand:
              return sizeof(void *) * channel_count();
            default:
              ; // static_assert(false, "should not be here!");
          }
        } else {
          if constexpr (allocp == alloc_ahead)
            return element_size();
          else if constexpr (allocp == alloc_on_demand)
            return sizeof(void *) * channel_count();
        }
        return static_cast<std::size_t>(0);
      }

      constexpr std::size_t size() const noexcept { return extent() * element_storage_size(); }

      /// related to access
      template <std::size_t I> constexpr std::size_t element_stride() const noexcept {
        if constexpr (is_same_v<Decorator, dynamic_decorator>) {
          switch (allocp) {
            case alloc_ahead:
              return (layoutp == aos ? element_storage_size() : attrib_size<I>());
            case alloc_on_demand:
              return sizeof(void *) * (layoutp == aos ? channel_count() : 1);
            default:
              ; // static_assert(false, "should not be here!");
          }
        } else {
          if constexpr (allocp == alloc_ahead)
            return (layoutp == aos ? element_storage_size() : attrib_size<I>());
          else if constexpr (allocp == alloc_on_demand)
            return sizeof(void *) * (layoutp == aos ? channel_count() : 1);
        }
        return static_cast<std::size_t>(0);
      }

      template <std::size_t I, typename Ti = char>
      constexpr std::size_t channel_offset(Ti chno = 0) const noexcept {
        std::size_t attrib_offset = 0;
        {
          const std::size_t attrib_offsets[] = {attrib_size<chsrc(Is)>()...};
          const std::size_t chn_counts[]
              = {(std::size_t)(((channel_counts &)(*this)).template get<chsrc(Is)>())...};
          for (int i = 0; i < (int)chmap(I); ++i)
            attrib_offset += attrib_offsets[i] * chn_counts[i];
        }
        if constexpr (is_same_v<Decorator, dynamic_decorator>) {
          switch (allocp) {
            case alloc_ahead:
              return (layoutp == aos ? 1 : extent()) * (attrib_offset + attrib_size<I>() * chno);
            case alloc_on_demand:
              return (layoutp == aos ? 1 : extent()) * (attrib_offset + chno)
                     * sizeof(void *);  // ?
            default:
              ; // static_assert(false, "should not be here!");
          }
        } else {
          if constexpr (allocp == alloc_ahead)
            return (layoutp == aos ? 1 : extent()) * (attrib_offset + attrib_size<I>() * chno);
          else if constexpr (allocp == alloc_on_demand)
            return (layoutp == aos ? 1 : extent()) * (attrib_offset + chno) * sizeof(void *);  // ?
        }
        return static_cast<std::size_t>(0);
      }

      template <auto I, typename Ti, typename... Tis>
      constexpr uintptr_t element_offset(wrapv<I>, Ti chno, Tis &&...coords) const noexcept {
        return channel_offset<I>(chno)
               + offset(make_vec(std::forward<Tis>(coords)...)) * element_stride<I>();
      }
    };

    /// CTAD only available since C++17!
    template <typename Decorator, typename Domain, typename... Ts, auto... Ns>
    snode(Decorator, Domain, tuple<Ts...>, vseq_t<Ns...>)
        -> snode<Decorator, Domain, tuple<typename snode_attrib_unwrapper<Ts>::type...>,
                 vseq_t<Ns...>, std::index_sequence_for<Ts...>>;

    template <typename Decorator, typename Domain, typename... Ts, typename... Tn>
    snode(Decorator, Domain, tuple<Ts...>, tuple<Tn...>)
        -> snode<Decorator, Domain, tuple<typename snode_attrib_unwrapper<Ts>::type...>,
                 tuple<std::decay_t<Tn>...>, std::index_sequence_for<Ts...>>;

    /******************************************************************/
    /**                      instance signature                       */
    /**   operator(), get_accessor, custom utils                      */
    /******************************************************************/
    enum instance_type : char {
      dense = 0,  ///< building block
      general = 1,
      dynamic = 2,
      hash = 3,
      bvh = 4
    };
    template <typename Derived, typename Snodes, typename Indices> struct instance_interface;
    template <instance_type, typename, typename> struct instance;

    template <instance_type I, typename... Snodes> instance(wrapv<I>, tuple<Snodes...>)
        -> instance<I, type_seq<Snodes...>, std::index_sequence_for<Snodes...>>;

    template <typename Derived, typename... Snodes, std::size_t... Is>
    struct instance_interface<Derived, type_seq<Snodes...>, std::index_sequence<Is...>> {
      constexpr auto &self() noexcept { return static_cast<Derived &>(*this); }
      constexpr auto const &self() const noexcept { return static_cast<Derived const &>(*this); }
      template <auto I = 0>
      constexpr typename type_seq<Snodes...>::template type<I> &node() noexcept {
        return self().template get<I>();
      }
      template <auto I = 0>
      constexpr typename type_seq<Snodes...>::template type<I> const &node() const noexcept {
        return self().template get<I>();
      }

      constexpr auto snode_sizes() const noexcept {
        return zs::make_tuple(snode_size(self().template get<Is>())...);
      }
      template <auto I = 0> constexpr bool illegal_alignment(const std::size_t &alignment) {
        return alignment % node<I>().align() != 0;
      }
      /// pmr-compliant
      template <typename Allocator> void alloc(Allocator allocator, std::size_t alignment) {
        auto nodesizes = snode_sizes();
        if ((illegal_alignment<Is>(alignment) || ...))
          // throw std::bad_alloc("misalignment during snode instance allocation!");
          throw std::bad_alloc();
        ((zs::get<Is>(self().handles) = allocator.allocate(zs::get<Is>(nodesizes), alignment)),
         ...);
      }
      template <typename Allocator>
      constexpr void dealloc(Allocator allocator, std::size_t alignment) {
        auto nodesizes = snode_sizes();
        ((allocator.deallocate((void *)zs::get<Is>(self().handles), zs::get<Is>(nodesizes),
                               alignment)),
         ...);
      }
      /// umpire-compliant
      // find an alignment that satisfies all child snodes
      constexpr std::size_t maxAlignment() const noexcept {
        std::size_t alignment = 0;
        {
          const std::size_t alignments[] = {node<Is>().align()...};
          for (const auto &align : alignments) alignment = align > alignment ? align : alignment;
        }
        return alignment;
      }
      constexpr void alloc(GeneralAllocator &allocator) {
        auto nodesizes = snode_sizes();
        ((zs::get<Is>(self().handles) = allocator.allocate(zs::get<Is>(nodesizes))), ...);
      }
      constexpr void alloc(GeneralAllocator &&allocator) {
        auto nodesizes = snode_sizes();
        ((zs::get<Is>(self().handles) = allocator.allocate(zs::get<Is>(nodesizes))), ...);
      }
      constexpr void dealloc() {
        auto &rm = get_resource_manager();
        ((rm.deallocate((void *)zs::get<Is>(self().handles))), ...);
      }
      template <typename... Handles> constexpr void assign(Handles &&...ptrs) {
        ((zs::get<Is>(self().handles) = std::forward<Handles>(ptrs)), ...);
      }
      constexpr void assign(tuple<std::enable_if_t<Is >= 0, void *>...> ptrs) {
        self().handles = ptrs;
      }
      template <auto I = 0> constexpr auto at(wrapv<I>) const noexcept {
        return instance{wrapv<dense>{}, zs::make_tuple(zs::get<I>(self()))};
      }
      constexpr auto getHandles() noexcept { return self().handles; }
    };

    /// dense, the most fundamental one
    template <typename Snode> struct instance<dense, type_seq<Snode>, index_seq<0>>
        : wrapv<dense>,
          tuple<Snode>,
          instance_interface<instance<dense, type_seq<Snode>, index_seq<0>>, type_seq<Snode>,
                             index_seq<0>> {
      instance() = default;
      ~instance() = default;
      explicit instance(wrapv<dense>, const tuple<Snode> &tup) : tuple<Snode>{tup} {}
      using base_t = instance_interface<instance<dense, type_seq<Snode>, index_seq<0>>,
                                        type_seq<Snode>, index_seq<0>>;
      using base_t::node;
      template <std::size_t I>
      using attrib_t = typename Snode::template attrib_type<I>::type;
      template <std::size_t I>
      static constexpr auto attrib_v() noexcept {
        return Snode::template attrib_type<I>::value;
      }

      template <auto I = 0> constexpr std::uintptr_t address() const noexcept {
        return (uintptr_t)zs::get<I>(handles);
      }
      template <auto I, typename Ti, typename... Tis,
                enable_if_all<attrib_v<I>(),
                              sizeof...(Tis) == Snode::N> = 0>
      constexpr std::uintptr_t address(wrapv<I>, Ti chno, Tis &&...coords) const noexcept {
        auto ele_offset = node().element_offset(wrapv<I>{}, chno, std::forward<Tis>(coords)...);
        return (uintptr_t)zs::get<0>(handles) + ele_offset;
      }
      template <auto I, typename Ti, typename... Tis,
                enable_if_all<attrib_v<I>(),
                              sizeof...(Tis) == Snode::N> = 0>
      constexpr std::uintptr_t address_offset(wrapv<I>, Ti chno, Tis &&...coords) const noexcept {
        return node().element_offset(wrapv<I>{}, chno, std::forward<Tis>(coords)...);
      }

      /// full index query
      template <auto I, typename Ti, typename... Tis,
                enable_if_all<attrib_v<I>(),
                              sizeof...(Tis) == Snode::N> = 0>
      constexpr auto &operator()(wrapv<I>, Ti chno, Tis &&...coords) noexcept {
        auto ele_offset = node().element_offset(wrapv<I>{}, chno, std::forward<Tis>(coords)...);
        return *reinterpret_cast<attrib_t<I> *>(
            (uintptr_t)zs::get<0>(handles) + ele_offset);
      }
      template <auto I, typename Ti, typename... Tis,
                enable_if_all<attrib_v<I>(),
                              sizeof...(Tis) == Snode::N> = 0>
      constexpr const auto &operator()(wrapv<I>, Ti chno, Tis &&...coords) const noexcept {
        auto ele_offset = node().element_offset(wrapv<I>{}, chno, std::forward<Tis>(coords)...);
        return *reinterpret_cast<
            attrib_t<I> const *>(
            (uintptr_t)zs::get<0>(handles) + ele_offset);
      }
      template <auto I, typename Ti, typename... Tis,
                enable_if_all<!attrib_v<I>(),
                              sizeof...(Tis) == Snode::N> = 0>
      constexpr auto operator()(wrapv<I>, Ti chno, Tis &&...coords) const noexcept {
        auto ele_offset = node().element_offset(wrapv<I>{}, chno, std::forward<Tis>(coords)...);
        auto ch = node().child(wrapv<I>{});
        auto inst = instance<dense, type_seq<std::decay_t<decltype(ch)>>, std::index_sequence<0>>{
            wrapv<dense>{}, zs::make_tuple(ch)};
        inst.assign((void *)((uintptr_t)zs::get<0>(handles) + ele_offset));
        return inst;
      }
      /// without sub-channel
      template <auto I, typename... Tis,
                enable_if_all<attrib_v<I>(),
                              sizeof...(Tis) == Snode::N> = 0>
      constexpr auto &operator()(wrapv<I>, Tis &&...coords) noexcept {
        auto ele_offset = node().element_offset(wrapv<I>{}, 0, std::forward<Tis>(coords)...);
        return *reinterpret_cast<attrib_t<I> *>((uintptr_t)zs::get<0>(handles) + ele_offset);
      }
      template <auto I, typename... Tis,
                enable_if_all<attrib_v<I>(),
                              sizeof...(Tis) == Snode::N> = 0>
      constexpr const auto &operator()(wrapv<I>, Tis &&...coords) const noexcept {
        auto ele_offset = node().element_offset(wrapv<I>{}, 0, std::forward<Tis>(coords)...);
        return *reinterpret_cast<attrib_t<I> const *>((uintptr_t)zs::get<0>(handles) + ele_offset);
      }
      template <auto I, typename... Tis,
                enable_if_all<!attrib_v<I>(),
                              sizeof...(Tis) == Snode::N> = 0>
      constexpr auto operator()(wrapv<I>, Tis &&...coords) const noexcept {
        auto ele_offset = node().element_offset(wrapv<I>{}, 0, std::forward<Tis>(coords)...);
        auto ch = node().child(wrapv<I>{});
        auto inst = instance<dense, type_seq<std::decay_t<decltype(ch)>>, std::index_sequence<0>>{
            wrapv<dense>{}, zs::make_tuple(ch)};
        inst.assign((void *)((uintptr_t)zs::get<0>(handles) + ele_offset));
        return inst;
      }
      /// without attribute number
      template <typename Ti, typename... Tis,
                enable_if_all<attrib_v<0>(),
                              !is_value_wrapper<Ti>::value, sizeof...(Tis) == Snode::N> = 0>
      constexpr auto &operator()(Ti chno, Tis &&...coords) noexcept {
        auto ele_offset = node().element_offset(wrapv<0>{}, chno, std::forward<Tis>(coords)...);
        return *reinterpret_cast<attrib_t<0> *>(
            (uintptr_t)zs::get<0>(handles) + ele_offset);
      }
      template <typename Ti, typename... Tis,
                enable_if_all<attrib_v<0>(),
                              !is_value_wrapper<Ti>::value, sizeof...(Tis) == Snode::N> = 0>
      constexpr const auto &operator()(Ti chno, Tis &&...coords) const noexcept {
        auto ele_offset = node().element_offset(wrapv<0>{}, chno, std::forward<Tis>(coords)...);
        return *reinterpret_cast<
            attrib_t<0> const *>(
            (uintptr_t)zs::get<0>(handles) + ele_offset);
      }
      template <typename Ti, typename... Tis,
                enable_if_all<!attrib_v<0>(),
                              !is_value_wrapper<Ti>::value, sizeof...(Tis) == Snode::N> = 0>
      constexpr auto operator()(Ti chno, Tis &&...coords) const noexcept {
        auto ele_offset = node().element_offset(wrapv<0>{}, chno, std::forward<Tis>(coords)...);
        auto ch = node().child(wrapv<0>{});
        auto inst = instance<dense, type_seq<std::decay_t<decltype(ch)>>, std::index_sequence<0>>{
            wrapv<dense>{}, zs::make_tuple(ch)};
        inst.assign((void *)((uintptr_t)zs::get<0>(handles) + ele_offset));
        return inst;
      }
      /// without both attribute number & subchannel number
      template <typename... Tis, enable_if_all<attrib_v<0>(),
                                               sizeof...(Tis) == Snode::N> = 0>
      constexpr auto &operator()(Tis &&...coords) noexcept {
        auto ele_offset = node().element_offset(wrapv<0>{}, 0, std::forward<Tis>(coords)...);
        return *reinterpret_cast<attrib_t<0> *>((uintptr_t)zs::get<0>(handles) + ele_offset);
      }
      template <typename... Tis, enable_if_all<attrib_v<0>(),
                                               sizeof...(Tis) == Snode::N> = 0>
      constexpr const auto &operator()(Tis &&...coords) const noexcept {
        auto ele_offset = node().element_offset(wrapv<0>{}, 0, std::forward<Tis>(coords)...);
        return *reinterpret_cast<attrib_t<0> const *>((uintptr_t)zs::get<0>(handles) + ele_offset);
      }
      template <typename... Tis, enable_if_all<!attrib_v<0>(),
                                               sizeof...(Tis) == Snode::N> = 0>
      constexpr auto operator()(Tis &&...coords) const noexcept {
        auto ele_offset = node().element_offset(wrapv<0>{}, 0, std::forward<Tis>(coords)...);
        auto ch = node().child(wrapv<0>{});
        auto inst = instance<dense, type_seq<std::decay_t<decltype(ch)>>, std::index_sequence<0>>{
            wrapv<dense>{}, zs::make_tuple(ch)};
        inst.assign((void *)((uintptr_t)zs::get<0>(handles) + ele_offset));
        return inst;
      }
      /// linear index
      template <typename Ti, bool isval = attrib_v<0>(), enable_if_t<isval> = 0>
      constexpr auto &operator[](Ti &&coord) noexcept {
        auto ele_offset = node().element_offset(wrapv<0>{}, 0, FWD(coord));
        return *reinterpret_cast<attrib_t<0> *>((uintptr_t)zs::get<0>(handles) + ele_offset);
      }
      template <typename Ti, bool isval = attrib_v<0>(), enable_if_t<isval> = 0>
      constexpr const auto &operator[](Ti &&coord) const noexcept {
        auto ele_offset = node().element_offset(wrapv<0>{}, 0, FWD(coord));
        return *reinterpret_cast<attrib_t<0> const *>((uintptr_t)zs::get<0>(handles) + ele_offset);
      }
      template <typename Ti, bool isnode = !attrib_v<0>(), enable_if_t<isnode> = 0>
      constexpr auto operator[](Ti &&coord) const noexcept {
        auto ele_offset = node().element_offset(wrapv<0>{}, 0, FWD(coord));
        auto ch = node().child(wrapv<0>{});
        auto inst = instance<dense, type_seq<std::decay_t<decltype(ch)>>, std::index_sequence<0>>{
            wrapv<dense>{}, zs::make_tuple(ch)};
        inst.assign((void *)((uintptr_t)zs::get<0>(handles) + ele_offset));
        return inst;
      }

      tuple<void *> handles{nullptr};
    };

    /// general
    template <typename... Snodes, std::size_t... Is>
    struct instance<general, type_seq<Snodes...>, std::index_sequence<Is...>>
        : wrapv<general>,
          tuple<Snodes...>,
          instance_interface<instance<general, type_seq<Snodes...>, std::index_sequence<Is...>>,
                             type_seq<Snodes...>, std::index_sequence<Is...>> {
      instance() = default;
      ~instance() = default;
      explicit instance(wrapv<general>, const tuple<Snodes...> &tup) : tuple<Snodes...>{tup} {}
      tuple<std::enable_if_t<Is >= 0> *...> handles{(Is >= 0 ? nullptr : nullptr)...};
    };

    template <instance_type it, typename... Snodes> using instance_t
        = instance<it, type_seq<Snodes...>, std::index_sequence_for<Snodes...>>;

  }  // namespace ds

  using PropertyTag = tuple<SmallString, int>;

  inline auto select_properties(const std::vector<PropertyTag> &props,
                                const std::vector<SmallString> &names) {
    std::vector<PropertyTag> ret(0);
    for (auto &&name : names)
      for (auto &&prop : props)
        if (prop.template get<0>() == name) {
          ret.push_back(prop);
          break;
        }
    return ret;
  }

}  // namespace zs
