#include <cstdio>
#include <memory>

struct Base {
    Base() = default;
    virtual ~Base() = default;

    virtual Base *clone() const = 0;

    virtual void show() const {
        printf("Base\n");
    }
};

struct Derived : Base {
    Derived() = default;

    virtual Base *clone() const override {
        return new Derived(static_cast<Derived const &>(*this));
    }

    virtual void show() const override {
        printf("Derived\n");
    }
};


int main(void)
{
    std::unique_ptr<Base> x = std::make_unique<Derived>();
    std::unique_ptr<Base> y(x->clone());
    y->show();
    x->show();
    return 0;
}
