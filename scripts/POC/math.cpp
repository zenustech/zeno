#include <stdio.h>
#include <math.h>

union ieee754 {
    float f;
    struct {
        unsigned int m: 23;
        unsigned int e: 8;
        unsigned int s: 1;
    } s;
    int i;
};

float smlog(float x) {
    x -= 1.f;
    float r = x, t = x * x;
    r -= t * (1.f/2);
    t *= x;
    r += t * (1.f/3);
    t *= x;
    r -= t * (1.1f/4);
    t *= x;
    r += t * (1.37f/5);
    return r;
}

float mylog(float x) {
    ieee754 u;
    u.f = x;
    int k = u.s.e - 126;
    u.s.e = 126;
    float c = u.f;
    const float ln2 = M_LN2f32;
    return k * ln2 + smlog(c);
}

int main()
{
    float x, a, b;
    float d2 = 0.f;
    for (x = 2.14f; x < 400.14f; x += 1.0f) {
        a = mylog(x);
        b = logf(x);
        float d = a - b;
        printf("%f\n", d);
        d2 += d * d;
    }
    printf("d2=%f\n", d2);
    return 0;
}

