

template <class T, class ...Ts>
struct callback {
    struct Base {
        virtual T operator()(Ts ...ts) const = 0;
    };

    template <class F>
    struct Impl : Base {
        F f;
        Impl(F const &f) : f(f) {}
        virtual T operator()(Ts ...ts) const override {
            return f(std::forward<Ts>(ts)...);
        }
    };

    std::unique_ptr<Base> p;

    callback() = default;

    template <class F>
    callback(F const &f)
        : p(std::make_unique<Impl<F>>(f))
    {}

    T operator()(Ts ...ts) const {
        return (*p)(std::forward<Ts>(ts)...);
    }
};
