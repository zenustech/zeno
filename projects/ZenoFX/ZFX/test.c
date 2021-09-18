#include <x86intrin.h>

int main(void) {
    __m128 a = _mm_set1_ps(2.0f);
    return 0;
}
