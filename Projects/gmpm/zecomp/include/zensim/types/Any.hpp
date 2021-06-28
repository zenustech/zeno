
namespace zs {

  struct in_place_t {
    explicit in_place_t() = default;
  };

  inline constexpr in_place_t in_place{};

  template <typename _Tp> struct in_place_type_t { explicit in_place_type_t() = default; };

  template <typename _Tp> inline constexpr in_place_type_t<_Tp> in_place_type{};

  template <size_t _Idx> struct in_place_index_t { explicit in_place_index_t() = default; };

  template <size_t _Idx> inline constexpr in_place_index_t<_Idx> in_place_index{};

  template <typename> struct __is_in_place_type_impl : std::false_type {};

  template <typename _Tp> struct __is_in_place_type_impl<in_place_type_t<_Tp>> : std::true_type {};

  template <typename _Tp> struct __is_in_place_type : public __is_in_place_type_impl<_Tp> {};

  // copied from std
  class Any {
    // Holds either pointer to a heap object or the contained object itself.
    union _Storage {
      constexpr _Storage() : _M_ptr{nullptr} {}

      // Prevent trivial copies of this type, buffer might hold a non-POD.
      _Storage(const _Storage&) = delete;
      _Storage& operator=(const _Storage&) = delete;

      void* _M_ptr;
      std::aligned_storage<sizeof(_M_ptr), alignof(void*)>::type _M_buffer;
    };

    enum _Op { _Op_access, _Op_get_type_info, _Op_clone, _Op_destroy, _Op_xfer };

    union _Arg {
      void* _M_obj;
      const std::type_info* _M_typeinfo;
      Any* _M_any;
    };

    // _S_manage from either _Manager_external/ _Manager_internal
    void (*_M_manager)(_Op, const Any*, _Arg*);
    _Storage _M_storage;

    template <typename _Tp> friend void* __any_caster(const Any* __any);

    // store as internal object if _Tp is nothrow move constructible & fits the storage
    template <typename _Tp, typename _Safe = std::is_nothrow_move_constructible<_Tp>,
              bool _Fits = (sizeof(_Tp) <= sizeof(_Storage)) && (alignof(_Tp) <= alignof(_Storage))>
    using _Internal = std::integral_constant<bool, _Safe::value && _Fits>;
    // Manage in-place contained object.
    template <typename _Tp> struct _Manager_internal {  // uses small-object optimization
      static void _S_manage(_Op __which, const Any* __anyp, _Arg* __arg);

      template <typename _Up> static void _S_create(_Storage& __storage, _Up&& __value) {
        void* __addr = &__storage._M_buffer;
        ::new (__addr) _Tp(std::forward<_Up>(__value));
      }

      template <typename... _Args> static void _S_create(_Storage& __storage, _Args&&... __args) {
        void* __addr = &__storage._M_buffer;
        ::new (__addr) _Tp(std::forward<_Args>(__args)...);
      }
    };
    // Manage external contained object.
    template <typename _Tp> struct _Manager_external {  // creates contained object on the heap
      static void _S_manage(_Op __which, const Any* __anyp, _Arg* __arg);

      template <typename _Up> static void _S_create(_Storage& __storage, _Up&& __value) {
        __storage._M_ptr = new _Tp(std::forward<_Up>(__value));
      }
      template <typename... _Args> static void _S_create(_Storage& __storage, _Args&&... __args) {
        __storage._M_ptr = new _Tp(std::forward<_Args>(__args)...);
      }
    };
    template <typename _Tp> using _Manager
        = conditional_t<_Internal<_Tp>::value, _Manager_internal<_Tp>, _Manager_external<_Tp>>;

    template <typename _Tp, typename _VTp = std::decay_t<_Tp>> using _Decay_if_not_any
        = enable_if_t<!is_same_v<_VTp, Any>, _VTp>;

    /// Emplace with an object created from @p __args as the contained object.
    template <typename _Tp, typename... _Args, typename _Mgr = _Manager<_Tp>>
    void __do_emplace(_Args&&... __args) {
      reset();
      _Mgr::_S_create(_M_storage, std::forward<_Args>(__args)...);
      _M_manager = &_Mgr::_S_manage;
    }

    /// Emplace with an object created from @p __il and @p __args as
    /// the contained object.
    template <typename _Tp, typename _Up, typename... _Args, typename _Mgr = _Manager<_Tp>>
    void __do_emplace(std::initializer_list<_Up> __il, _Args&&... __args) {
      reset();
      _Mgr::_S_create(_M_storage, __il, std::forward<_Args>(__args)...);
      _M_manager = &_Mgr::_S_manage;
    }

    // _Res if _Tp is copy constructible & _Tp constructible from _Args
    template <typename _Res, typename _Tp, typename... _Args>
    using __any_constructible = std::enable_if<
        __and_<std::is_copy_constructible<_Tp>, std::is_constructible<_Tp, _Args...>>::value, _Res>;

    template <typename _Tp, typename... _Args> using __any_constructible_t =
        typename __any_constructible<bool, _Tp, _Args...>::type;

    template <typename _VTp, typename... _Args> using __emplace_t =
        typename __any_constructible<_VTp&, _VTp, _Args...>::type;

  public:
    // construct/destruct

    /// Default constructor, creates an empty object.
    constexpr Any() noexcept : _M_manager(nullptr) {}

    /// Copy constructor, copies the state of @p __other
    Any(const Any& __other) {
      if (!__other.has_value())
        _M_manager = nullptr;
      else {
        _Arg __arg;
        __arg._M_any = this;
        __other._M_manager(_Op_clone, &__other, &__arg);
      }
    }

    /**
     * @brief Move constructor, transfer the state from @p __other
     *
     * @post @c !__other.has_value() (this postcondition is a GNU extension)
     */
    Any(Any&& __other) noexcept {
      if (!__other.has_value())
        _M_manager = nullptr;
      else {
        _Arg __arg;
        __arg._M_any = this;
        __other._M_manager(_Op_xfer, &__other, &__arg);
      }
    }

    /// Construct with a copy of @p __value as the contained object.
    template <typename _Tp, typename _VTp = _Decay_if_not_any<_Tp>, typename _Mgr = _Manager<_VTp>,
              std::enable_if_t<std::is_copy_constructible<_VTp>::value
                                   && !__is_in_place_type<_VTp>::value,
                               bool> = true>
    Any(_Tp&& __value) : _M_manager(&_Mgr::_S_manage) {
      _Mgr::_S_create(_M_storage, std::forward<_Tp>(__value));
    }

    /// Construct with an object created from @p __args as the contained object.
    template <typename _Tp, typename... _Args, typename _VTp = decay_t<_Tp>,
              typename _Mgr = _Manager<_VTp>, __any_constructible_t<_VTp, _Args&&...> = false>
    explicit Any(in_place_type_t<_Tp>, _Args&&... __args) : _M_manager(&_Mgr::_S_manage) {
      _Mgr::_S_create(_M_storage, std::forward<_Args>(__args)...);
    }

    /// Construct with an object created from @p __il and @p __args as
    /// the contained object.
    template <typename _Tp, typename _Up, typename... _Args, typename _VTp = decay_t<_Tp>,
              typename _Mgr = _Manager<_VTp>,
              __any_constructible_t<_VTp, initializer_list<_Up>, _Args&&...> = false>
    explicit Any(in_place_type_t<_Tp>, initializer_list<_Up> __il, _Args&&... __args)
        : _M_manager(&_Mgr::_S_manage) {
      _Mgr::_S_create(_M_storage, __il, std::forward<_Args>(__args)...);
    }

    /// Destructor, calls @c reset()
    ~Any() { reset(); }

    // assignments

    /// Copy the state of another object.
    Any& operator=(const Any& __rhs) {
      *this = Any(__rhs);
      return *this;
    }

    /**
     * @brief Move assignment operator
     *
     * @post @c !__rhs.has_value() (not guaranteed for other implementations)
     */
    Any& operator=(Any&& __rhs) noexcept {
      if (!__rhs.has_value())
        reset();
      else if (this != &__rhs) {
        reset();
        _Arg __arg;
        __arg._M_any = this;
        __rhs._M_manager(_Op_xfer, &__rhs, &__arg);
      }
      return *this;
    }

    /// Store a copy of @p __rhs as the contained object.
    template <typename _Tp>
    std::enable_if_t<is_copy_constructible<_Decay_if_not_any<_Tp>>::value, Any&> operator=(
        _Tp&& __rhs) {
      *this = Any(std::forward<_Tp>(__rhs));
      return *this;
    }

    /// Emplace with an object created from @p __args as the contained object.
    template <typename _Tp, typename... _Args>
    __emplace_t<decay_t<_Tp>, _Args...> emplace(_Args&&... __args) {
      using _VTp = decay_t<_Tp>;
      __do_emplace<_VTp>(std::forward<_Args>(__args)...);
      Any::_Arg __arg;
      this->_M_manager(Any::_Op_access, this, &__arg);
      return *static_cast<_VTp*>(__arg._M_obj);
    }

    /// Emplace with an object created from @p __il and @p __args as
    /// the contained object.
    template <typename _Tp, typename _Up, typename... _Args>
    __emplace_t<decay_t<_Tp>, initializer_list<_Up>, _Args&&...> emplace(initializer_list<_Up> __il,
                                                                         _Args&&... __args) {
      using _VTp = decay_t<_Tp>;
      __do_emplace<_VTp, _Up>(__il, std::forward<_Args>(__args)...);
      Any::_Arg __arg;
      this->_M_manager(Any::_Op_access, this, &__arg);
      return *static_cast<_VTp*>(__arg._M_obj);
    }

    // modifiers

    /// If not empty, destroy the contained object.
    void reset() noexcept {
      if (has_value()) {
        _M_manager(_Op_destroy, this, nullptr);
        _M_manager = nullptr;
      }
    }

    /// Exchange state with another object.
    void swap(Any& __rhs) noexcept {
      if (!has_value() && !__rhs.has_value()) return;

      if (has_value() && __rhs.has_value()) {
        if (this == &__rhs) return;

        Any __tmp;
        _Arg __arg;
        __arg._M_any = &__tmp;
        __rhs._M_manager(_Op_xfer, &__rhs, &__arg);
        __arg._M_any = &__rhs;
        _M_manager(_Op_xfer, this, &__arg);
        __arg._M_any = this;
        __tmp._M_manager(_Op_xfer, &__tmp, &__arg);
      } else {
        Any* __empty = !has_value() ? this : &__rhs;
        Any* __full = !has_value() ? &__rhs : this;
        _Arg __arg;
        __arg._M_any = __empty;
        __full->_M_manager(_Op_xfer, __full, &__arg);
      }
    }

    // observers

    /// Reports whether there is a contained object or not.
    bool has_value() const noexcept { return _M_manager != nullptr; }

#if __cpp_rtti
    /// The @c typeid of the contained object, or @c typeid(void) if empty.
    const type_info& type() const noexcept {
      if (!has_value()) return typeid(void);
      _Arg __arg;
      _M_manager(_Op_get_type_info, this, &__arg);
      return *__arg._M_typeinfo;
    }
#endif

    template <typename _Tp> static constexpr bool __is_valid_cast() {
      return __or_<is_reference<_Tp>, is_copy_constructible<_Tp>>::value;
    }
  };

}  // namespace zs