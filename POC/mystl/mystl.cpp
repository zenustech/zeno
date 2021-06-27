#include <memory>
#include <cstdio>

template <class T, class D = std::default_delete<T>>
struct copiable_unique_ptr : std::unique_ptr<T, D> {
    using std::unique_ptr<T, D>::unique_ptr;

    copiable_unique_ptr(std::unique_ptr<T, D> &&o)
        : std::unique_ptr<T, D>(std::move(o)) {
    }

    copiable_unique_ptr(copiable_unique_ptr const &o)
        : std::unique_ptr<T, D>(std::make_unique<T>(
            static_cast<T const &>(*o))) {
    }

    operator std::unique_ptr<T, D> &() { return *this; }
    operator std::unique_ptr<T, D> const &() const { return *this; }
};


template <class T, class D>
copiable_unique_ptr(std::unique_ptr<T, D> &&o) -> copiable_unique_ptr<T, D>;

struct MyClass {
    int i = -1;
    MyClass(int i) : i(i) {}
    MyClass(MyClass const &o) : i(o.i) {
        printf("copying MyClass...\n");
    }
};

int main(void)
{
    copiable_unique_ptr<MyClass> x = std::make_unique<MyClass>(2);
    auto y = x;
    printf("%d\n", y->i);
    return 0;
}
