#pragma once

#include <type_traits>
#include <utility>

#include "../TypeAlias.hpp"
#include "Functional.h"
#include "zensim/Reflection.h"

namespace zs {

  /** type decorations */

  /// cvref
  template <class T> struct remove_cvref {
    using type = std::remove_cv_t<std::remove_reference_t<T>>;
  };
  template <class T> using remove_cvref_t = typename remove_cvref<T>::type;

  /// vref
  template <class T> struct remove_vref {
    using type = std::remove_volatile_t<std::remove_reference_t<T>>;
  };
  template <class T> using remove_vref_t = typename remove_vref<T>::type;

  /// https://zh.cppreference.com/w/cpp/utility/tuple/make_tuple
  /// decay+unref
  template <class T> struct unwrap_refwrapper { using type = T; };
  template <class T> struct unwrap_refwrapper<std::reference_wrapper<T>> { using type = T &; };
  template <class T> using special_decay_t =
      typename unwrap_refwrapper<typename std::decay_t<T>>::type;

  template <typename T> struct wrapt { using type = T; };
  /// wrap at most 1 layer
  template <typename T> struct wrapt<wrapt<T>> { using type = T; };
  template <auto N> using wrapv = std::integral_constant<decltype(N), N>;

  /// indexable type list to avoid recursion
  namespace type_impl {
    template <std::size_t, typename T> struct indexed_type { using type = T; };
    template <typename, typename... Ts> struct indexed_types;

    template <std::size_t... Is, typename... Ts>
    struct indexed_types<std::index_sequence<Is...>, Ts...> : indexed_type<Is, Ts>... {};

    // use pointer rather than reference as in taocpp! [incomplete type error]
    template <std::size_t I, typename T> indexed_type<I, T> extract_type(indexed_type<I, T> *);
  }  // namespace type_impl

  /******************************************************************/
  /** declaration: monoid_op, gen_seq, gather */
  /******************************************************************/

  /// sequence manipulation declaration
  template <typename, typename> struct gather;
  template <typename Indices, typename ValueSeq> using gather_t =
      typename gather<Indices, ValueSeq>::type;

  /// generate index sequence declaration
  template <typename> struct gen_seq_impl;
  template <std::size_t N> using gen_seq = gen_seq_impl<std::make_index_sequence<N>>;

  /// seq + op
  template <typename... Ts> struct type_seq;
  template <auto... Ns> struct value_seq;
  template <typename> struct tseq;
  template <typename> struct vseq {};
  template <auto... Ns> using vseq_t = vseq<value_seq<Ns...>>;
  template <typename... Ts> using tseq_t = tseq<type_seq<Ts...>>;

  template <typename... Seqs> struct concat;
  template <typename... Seqs> using concat_t = typename concat<Seqs...>::type;

  /******************************************************************/
  /** definition: monoid_op, type_seq, value_seq, gen_seq, gather */
  /******************************************************************/

  template <typename Tn, Tn N> using integral_v = std::integral_constant<Tn, N>;
  template <typename> struct is_integral_constant : std::false_type {};
  template <typename T, T v> struct is_integral_constant<integral_v<T, v>> : std::true_type {};
  template <std::size_t N> using index_v = std::integral_constant<std::size_t, N>;
  /// static uniform non-types
  template <typename Tn, Tn... Ns> using integral_seq = std::integer_sequence<Tn, Ns...>;
  template <auto... Ns> using index_seq = std::index_sequence<Ns...>;

  template <std::size_t... Is> struct gen_seq_impl<std::index_sequence<Is...>> {
    using signed_size_t = std::make_signed_t<std::size_t>;
    template <auto N0 = 0, auto Step = 1> using arithmetic
        = std::index_sequence<static_cast<std::size_t>(
            static_cast<signed_size_t>(N0)
            + static_cast<signed_size_t>(Is) * static_cast<signed_size_t>(Step))...>;
    using ascend = arithmetic<0, 1>;
    using descend = arithmetic<sizeof...(Is) - 1, -1>;
    template <auto J> using uniform = integral_seq<decltype(J), (Is == Is ? J : J)...>;
    template <auto J> using uniform_vseq = vseq_t<(Is == Is ? J : J)...>;
    template <template <typename...> typename T, typename Arg> using uniform_types_t
        = T<std::enable_if_t<Is >= 0, Arg>...>;
    template <template <auto...> typename T, auto Arg> using uniform_values_t
        = T<(Is >= 0 ? Arg : 0)...>;
  };

  /// tseq
  template <typename... Ts> struct type_seq {
    using indices = std::index_sequence_for<Ts...>;
    static constexpr auto count = sizeof...(Ts);
    using tseq = zs::tseq<type_seq<Ts...>>;
    template <std::size_t I> using type = typename decltype(type_impl::extract_type<I>(
        std::add_pointer_t<type_impl::indexed_types<indices, Ts...>>{}))::type;
  };

  /// select type by index
  template <std::size_t I, typename TypeSeq> using select_type = typename TypeSeq::template type<I>;
  template <std::size_t I, typename... Ts> using select_indexed_type
      = select_type<I, type_seq<Ts...>>;

  /** sequence transformations */
  template <typename, typename> struct tseqop_impl;
  template <std::size_t... Is, typename... Ts>
  struct tseqop_impl<std::index_sequence<Is...>, type_seq<Ts...>> {
    using indices = std::index_sequence_for<Ts...>;
#if 0
  template <template <typename> auto Pred>
  static constexpr auto all() noexcept {
    return (Pred<Ts> && ...);
  }
#endif
    /// convert
    template <template <typename> typename Unary> using to_vseq = vseq_t<Unary<Ts>::value...>;
    template <template <std::size_t, typename> typename Unary> using convert
        = tseq_t<Unary<Is, Ts>...>;
    /// shuffle_convert
    template <template <std::size_t, typename> typename, typename> struct shuffle_convert_impl;
    template <template <std::size_t, typename> typename Unary, auto... Js>
    struct shuffle_convert_impl<Unary, vseq_t<Js...>> {
      using type = tseq_t<Unary<Js, Ts>...>;
    };
    template <template <std::size_t, typename> typename Unary, std::size_t... Js>
    struct shuffle_convert_impl<Unary, std::index_sequence<Js...>>
        : shuffle_convert_impl<Unary, vseq_t<Js...>> {};
    template <template <std::size_t, typename> typename Unary, typename Indices>
    using shuffle_convert = typename shuffle_convert_impl<Unary, Indices>::type;
    /// shuffle
    template <typename> struct shuffle_impl;
    template <auto... Js> struct shuffle_impl<vseq_t<Js...>> {
      using type = tseq_t<typename type_seq<Ts...>::template type<Js>...>;
    };
    template <std::size_t... Js> struct shuffle_impl<std::index_sequence<Js...>>
        : shuffle_impl<vseq_t<Js...>> {};
    template <typename Indices> using shuffle = typename shuffle_impl<Indices>::type;
  };

  template <typename> struct tseqop;
  template <typename... Ts> struct tseqop<type_seq<Ts...>>
      : tseqop_impl<std::index_sequence_for<Ts...>, type_seq<Ts...>> {};

  template <typename... Ts> struct tseq<type_seq<Ts...>> : type_seq<Ts...>,
                                                           tseqop<type_seq<Ts...>> {
    using types = type_seq<Ts...>;
    using types::count;
    using op = tseqop<type_seq<Ts...>>;
    using op::convert;
    using op::shuffle;
    using op::shuffle_convert;
  };

  /// vseq
  template <auto... Ns> struct value_seq : tseq_t<std::integral_constant<decltype(Ns), Ns>...> {
    static constexpr auto count = sizeof...(Ns);
    using Tn = std::common_type_t<decltype(Ns)...>;
    using iseq = integral_seq<Tn, (Tn)Ns...>;
    using vseq = zs::vseq_t<Ns...>;
  };

  /// select (constant integral) value (integral_constant<T, N>) by index
  template <std::size_t I, typename ValueSeq> using select_value =
      typename ValueSeq::template type<I>;
  template <std::size_t I, auto... Ns> using select_indexed_value
      = select_value<I, value_seq<Ns...>>;

  /** sequence transformations */
  template <typename, typename> struct vseqop_impl;
  template <std::size_t... Is, auto... Ns>
  struct vseqop_impl<std::index_sequence<Is...>, value_seq<Ns...>> {
    using indices = std::make_index_sequence<sizeof...(Ns)>;
    /// monoid calculation
    template <typename BinaryOp> static constexpr auto reduce(BinaryOp &&) noexcept {
      return monoid_op<BinaryOp>{}(Ns...);
    }
    template <typename UnaryOp, typename BinaryOp>
    static constexpr auto reduce(const UnaryOp &uop, BinaryOp &&) noexcept {
      return monoid_op<BinaryOp>{}(uop(Ns)...);
    }
    template <typename UnaryOp, typename BinaryOp>
    static constexpr auto map_reduce(const UnaryOp &uop, BinaryOp &&) noexcept {
      return monoid_op<BinaryOp>{}(uop(Is, Ns)...);
    }
    /// component wise operation
    template <typename BinaryOp, typename Seq> struct component_wise_impl;
    template <typename BinaryOp, auto... Ms> struct component_wise_impl<BinaryOp, vseq_t<Ms...>> {
      using type = vseq_t<(BinaryOp{}(Ns, Ms))...>;
    };
    template <typename BinaryOp, typename Seq> using compwise =
        typename component_wise_impl<BinaryOp, Seq>::type;
    /// map (Op(i), index_sequence)
    template <typename MonoidOp, typename Js> struct map_impl;
    template <typename MapOp, std::size_t... Js>
    struct map_impl<MapOp, std::index_sequence<Js...>> {
      /// gather-style mapping
      using type = vseq_t<MapOp{}(Js, Ns...)...>;  ///< J is the target index
    };
    template <typename MapOp, std::size_t N> using map =
        typename map_impl<MapOp, std::make_index_sequence<N>>::type;
    /// shuffle
    template <typename> struct shuffle_impl;
    template <auto... Js> struct shuffle_impl<vseq_t<Js...>> {
      using type = vseq_t<(value_seq<Ns...>::template type<Js>::value)...>;
    };
    template <std::size_t... Js> struct shuffle_impl<std::index_sequence<Js...>>
        : shuffle_impl<vseq_t<Js...>> {};
    template <typename Indices> using shuffle = typename shuffle_impl<Indices>::type;
    /// transform
    template <typename UnaryOp> using transform = vseq_t<UnaryOp{}(Ns)...>;
    /// for_each
    template <typename UnaryOp> static constexpr auto for_each(UnaryOp &&op) {
      return (op(Ns), ...);
    }
    /// scan
    template <std::size_t, typename, typename> struct scan_element;
    template <std::size_t J, typename BinaryOp, std::size_t... Js>
    struct scan_element<J, BinaryOp, std::index_sequence<Js...>> {
      using T = decltype(monoid_op<BinaryOp>::e);
      // excl/incl prefix/suffix
      static constexpr T value(std::size_t I) noexcept {
        constexpr T values[]
            = {vseq_t<(Js < J ? Ns : monoid_op<BinaryOp>::e)...>::reduce(BinaryOp{}),
               vseq_t<(Js <= J ? Ns : monoid_op<BinaryOp>::e)...>::reduce(BinaryOp{}),
               vseq_t<(Js > J ? Ns : monoid_op<BinaryOp>::e)...>::reduce(BinaryOp{}),
               vseq_t<(Js >= J ? Ns : monoid_op<BinaryOp>::e)...>::reduce(BinaryOp{})};
        return values[I];
      }
    };
    template <typename BinaryOp, std::size_t Cate = 0> using scan
        = vseq_t<scan_element<Is, BinaryOp, indices>::value(Cate)...>;
  };
  template <std::size_t... Is, typename Tn, Tn... Ns>
  struct vseqop_impl<std::index_sequence<Is...>, integral_seq<Tn, Ns...>>
      : vseqop_impl<std::index_sequence<Is...>, value_seq<Ns...>> {};

  template <typename> struct vseqop;
  template <auto... Ns> struct vseqop<value_seq<Ns...>>
      : vseqop_impl<std::make_index_sequence<sizeof...(Ns)>, value_seq<Ns...>> {};
  template <typename Tn, Tn... Ns> struct vseqop<integral_seq<Tn, Ns...>>
      : vseqop_impl<std::make_index_sequence<sizeof...(Ns)>, integral_seq<Tn, Ns...>> {};

  template <auto... Ns> struct vseq<value_seq<Ns...>> : value_seq<Ns...>, vseqop<value_seq<Ns...>> {
    using vals = value_seq<Ns...>;
    using typename vals::iseq;
    using typename vals::Tn;
    using vals::count;
    using op = vseqop<value_seq<Ns...>>;
    using op::compwise;
    using op::for_each;
    using op::map;
    using op::reduce;
    using op::scan;
    using op::shuffle;
    using op::transform;
  };
  template <typename Tn, Tn... Ns> struct vseq<integral_seq<Tn, Ns...>> : vseq<value_seq<Ns...>> {};

  /** utilities */

  /// concat
  template <typename... Seqs> struct concat {
    static constexpr auto length = (... + Seqs::count);
    using indices = typename gen_seq<length>::ascend;
    using counts = vseq_t<Seqs::count...>;
    using outer =
        typename counts::template scan<plus<std::size_t>, 1>::template map<count_leq, length>;
    using inner = typename vseq<indices>::template compwise<
        minus<std::size_t>,
        typename counts::template scan<plus<std::size_t>, 0>::template shuffle<outer>>;
    using type = typename tseq_t<Seqs...>::template shuffle<outer>::template shuffle_convert<
        select_type, typename inner::iseq>;
  };

  /// uniform value sequence
  template <std::size_t... Is, typename T, T... Ns>
  struct gather<std::index_sequence<Is...>, integral_seq<T, Ns...>> {
    using type = integral_seq<T, select_indexed_value<Is, Ns...>{}...>;
  };
  /// non uniform value sequence
  template <std::size_t... Is, auto... Ns>
  struct gather<std::index_sequence<Is...>, value_seq<Ns...>> {
    using type = value_seq<(select_value<Is, value_seq<Ns...>>::value)...>;
  };

  /** type identification */

  /// variadic type template parameters
  template <typename T, template <typename...> class Ref> struct is_type_specialized
      : std::false_type {};
  template <template <typename...> class Ref, typename... Ts>
  struct is_type_specialized<Ref<Ts...>, Ref> : std::true_type {};

  /// variadic non-type template parameters
  template <typename T, template <auto...> class Ref> struct is_value_specialized
      : std::false_type {};
  template <template <auto...> class Ref, auto... Args>
  struct is_value_specialized<Ref<Args...>, Ref> : std::true_type {};

  /// static sequence identification
  template <typename T> struct is_tseq : std::false_type {};
  template <typename... Ts> struct is_tseq<tseq_t<Ts...>> : std::true_type {};
  template <typename T> struct is_vseq : std::false_type {};
  template <auto... Ns> struct is_vseq<vseq_t<Ns...>> : std::true_type {};
  template <typename T> struct is_type_wrapper : std::false_type {};
  template <typename T> struct is_type_wrapper<wrapt<T>> : std::true_type {};
  template <typename T> struct is_value_wrapper : std::false_type {};
  template <auto N> struct is_value_wrapper<wrapv<N>> : std::true_type {};

  /** direct operations on sequences */
  template <typename> struct seq_tail { using type = index_seq<>; };
  template <std::size_t I, std::size_t... Is> struct seq_tail<index_seq<I, Is...>> {
    using type = index_seq<Is...>;
  };
  template <typename Seq> using seq_tail_t = typename seq_tail<Seq>::type;

#define ZS_HAS_MEMBER(MEMBER)                                               \
  template <typename T> struct has_MEMBER {                                 \
  private:                                                                  \
    template <typename U> static std::false_type test(...);                 \
    template <typename U> static std::true_type test(decltype(&U::MEMBER)); \
                                                                            \
  public:                                                                   \
    static constexpr bool value = decltype(test<T>(0))();                   \
  };

  /** placeholder */
  namespace placeholders {
    using placeholder_type = unsigned;
    constexpr auto _0 = std::integral_constant<placeholder_type, 0>{};
    constexpr auto _1 = std::integral_constant<placeholder_type, 1>{};
    constexpr auto _2 = std::integral_constant<placeholder_type, 2>{};
    constexpr auto _3 = std::integral_constant<placeholder_type, 3>{};
    constexpr auto _4 = std::integral_constant<placeholder_type, 4>{};
    constexpr auto _5 = std::integral_constant<placeholder_type, 5>{};
    constexpr auto _6 = std::integral_constant<placeholder_type, 6>{};
    constexpr auto _7 = std::integral_constant<placeholder_type, 7>{};
    constexpr auto _8 = std::integral_constant<placeholder_type, 8>{};
    constexpr auto _9 = std::integral_constant<placeholder_type, 9>{};
    constexpr auto _10 = std::integral_constant<placeholder_type, 10>{};
    constexpr auto _11 = std::integral_constant<placeholder_type, 11>{};
    constexpr auto _12 = std::integral_constant<placeholder_type, 12>{};
    constexpr auto _13 = std::integral_constant<placeholder_type, 13>{};
    constexpr auto _14 = std::integral_constant<placeholder_type, 14>{};
    constexpr auto _15 = std::integral_constant<placeholder_type, 15>{};
    constexpr auto _16 = std::integral_constant<placeholder_type, 16>{};
    constexpr auto _17 = std::integral_constant<placeholder_type, 17>{};
    constexpr auto _18 = std::integral_constant<placeholder_type, 18>{};
    constexpr auto _19 = std::integral_constant<placeholder_type, 19>{};
    constexpr auto _20 = std::integral_constant<placeholder_type, 20>{};
    constexpr auto _21 = std::integral_constant<placeholder_type, 21>{};
    constexpr auto _22 = std::integral_constant<placeholder_type, 22>{};
    constexpr auto _23 = std::integral_constant<placeholder_type, 23>{};
    constexpr auto _24 = std::integral_constant<placeholder_type, 24>{};
    constexpr auto _25 = std::integral_constant<placeholder_type, 25>{};
    constexpr auto _26 = std::integral_constant<placeholder_type, 26>{};
    constexpr auto _27 = std::integral_constant<placeholder_type, 27>{};
    constexpr auto _28 = std::integral_constant<placeholder_type, 28>{};
    constexpr auto _29 = std::integral_constant<placeholder_type, 29>{};
    constexpr auto _30 = std::integral_constant<placeholder_type, 30>{};
    constexpr auto _31 = std::integral_constant<placeholder_type, 31>{};
  }  // namespace placeholders
  using place_id = placeholders::placeholder_type;

}  // namespace zs
