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

static float smlog(float x) {
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
    c -= 1.f;
    const float ln2 = M_LN2;
    float r = c + k * ln2, t = c * c;
    r -= t * (1.f/2);
    t *= c;
    r += t * (1.f/3);
    t *= c;
    r -= t * (1.1f/4);
    t *= c;
    r += t * (1.37f/5);
    return r;
}

static float etable[64];

static int init_etable() {
    for (int i = 0; i < 64; i++) {
        etable[i] = expf((float)i);
    }
}

static int _init_etable = init_etable();

float myexp(float x) {
    float y = fabs(x);
    int i = (int)floorf(y);
    float k = y - i;
    float r = 1.f;
    float t = k * 0.499f;
    r += t;
    t *= k * (1.05f/3);
    r += t;
    t *= k * (0.94f/4);
    r += t;
    r = 1.f + k * r;
    //r = expf(k);
    r *= etable[i];
    if (x < 0)
        r = 1.f / r;
    return r;
}

int main()
{
    /*printf("%f\n", logf(3.14f));
    printf("%f\n", mylog(3.14f));
    printf("%f\n", expf(-3.14f));
    printf("%f\n", myexp(-3.14f));
    printf("%f\n", expf(3.14f));
    printf("%f\n", myexp(3.14f));*/
    float x, a, b;
    float d2 = 0.f;
    for (x = 0.1f; x < 10.14f; x *= 1.1f) {
        a = myexp(x);
        b = expf(x);
        float d = (a - b) / b;
        printf("%f %f %f\n", a, b, d);
        d2 += d * d;
    }
    printf("d2=%f\n", d2);
    return 0;
}

