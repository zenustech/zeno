#include <memory>
#include <cstdio>

#define ZS_ZTD_PIMPL_DECL(Class, ...) \
struct Class { \
    struct Self; \
    std::shared_ptr<Self> self; \
    Class() = default; \
    Class(Class const &) = default; \
    Class &operator=(Class const &) = default; \
    Class(Class &&) = default; \
    Class &operator=(Class &&) = default; \
    inline Class(std::shared_ptr<Self> const &self) : self(self) {} \
    __VA_ARGS__ \
}

#define ZS_ZTD_PIMPL_SELF(Class, ...) \
struct Class::Self


#define ZS_ZTD_PIMPL_FD(Ret, func, ...) \
    Ret func(__VA_ARGS__);

#define ZS_ZTD_PIMPL_FI(Ret, func, ...) \
    Ret Class::func(__VA_ARGS__) { return self->func(); }


ZS_ZTD_PIMPL_DECL(Animal
                  void say();
                  );


ZS_ZTD_PIMPL_SELF(Animal) {
    int age;

    void say() {
        std::printf("Animal age=%d\n", age);
    }
};

int main() {
}
