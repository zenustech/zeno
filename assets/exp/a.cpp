#include <zen/vec.h>
#include <cstdio>


int main(void) {
    auto a = zen::vec3f(1, 2, 3);
    auto b = zen::vec3f(2, 3, 4);
    auto f = zen::vec3f(0.5, 0.5, 0.5);
    a = zen::mix(a, b, f);
    printf("%f %f %f\n", a[0], a[1], a[2]);
}
