#include <memory>
#include <cstdio>

struct Void {
    virtual ~Void() = default;
};

template <class T>
struct Interface : Void {
    virtual T *get() = 0;
};

template <class T>
struct Endpoint : Interface<T> {
    T t;

    virtual T *get() override { return &t; }
};

struct Base {
};

struct Derived : Base {
};

template <>
struct Endpoint<Derived> : Interface<Base>, Interface<Derived> {
    Derived t;

    virtual Derived *get() override { return &t; }
};

template <class T>
T *family_cast(Void *p) {
    auto q = dynamic_cast<Interface<T> *>(p);
    return q ? &q->t : nullptr;
}

int main() {
    Void *p = new Endpoint<Derived>;
    auto q = family_cast<Base>(p);
    printf("%p\n", q);
    return 0;
}
