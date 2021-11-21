#include <cstdio>

namespace geom {
    struct point {
    };

    void print(point) {
        printf("geom::print\n");
    }

    template <class T>
    void print2(T) {
        printf("geom::print2\n");
    }

    template <class T, class S>
    void print3(T, S) {
        printf("geom::print3\n");
    }
};

namespace pixl {
    struct point {
    };

    void print(point) {
        printf("pixl::print\n");
    }

    template <class T>
    void print2(T) {
        printf("pixl::print2\n");
    }

    template <class T, class S>
    void print3(T, S) {
        printf("geom::print3\n");
    }
};

int main() {
    geom::point x, y;
    print(x);
    print2(x);
    print3(x, y);
}
