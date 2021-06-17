#include <cstdio>
#include <memory>


struct AnyObject {
    virtual ~AnyObject() = default;

    virtual AnyObject *clone() const = 0;
    virtual void show() const = 0;
};

template <class T>
struct PtrObject : AnyObject {
    std::unique_ptr<T> tp;

    explicit PtrObject(std::unique_ptr<T> &&tp) : tp(std::move(tp)) {}

    virtual AnyObject *clone() const override {
        return new PtrObject<T>(std::make_unique<T>(static_cast<T const &>(*tp)));
    }

    virtual void show() const override {
        printf("PtrObject: %s\n", typeid(T).name());
    }
};

template <class T>
struct ImplObject : AnyObject {
    T t;

    template <class Ts...>
    explicit ImplObject(Ts const &... ts) : t(ts...) {}

    virtual AnyObject *clone() const override {
        return new ImplObject<T>(static_cast<T const &>(t));
    }

    virtual void show() const override {
        printf("ImplObject: %s\n", typeid(T).name());
    }
};


struct VDBGrid {
    virtual void help() const = 0;
};

struct VDBFloatGrid : VDBGrid {
    VDBFloatGrid() = default;

    VDBFloatGrid(VDBFloatGrid const &) {
        printf("VDBFloatGrid copy!\n");
    }

    virtual void help() const override {
        printf("VDBFloatGrid!\n");
    }
};


int main(void)
{
    std::unique_ptr<VDBGrid> up = std::make_unique<VDBFloatGrid>();
    AnyObject *p = new PtrObject<VDBGrid>( std::move(up) );
    p->show();
    auto q = p->clone();
    q->show();
    return 0;
}
