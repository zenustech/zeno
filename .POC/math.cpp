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
        //printf("%.20ff,\n", etable[i]);
    }
    return 0;
}

static int _init_etable = init_etable();

float myexp(float x) {
    float y = fabs(x);
    int i = (int)floorf(y);
    float k = y - i;
    float r = 1.f;
    float t = k * 0.4993f;
    r += t;
    t *= k * (1.0515f/3);
    r += t;
    t *= k * (0.949f/4);
    r += t;
    r = 0.99984f + k * r;
    r *= etable[i];
    if (x < 0)
        r = 1.f / r;
    return r;
}

float mysin(float x) {
    float y = M_PI - fmod(x, M_PI*2);
    float z = fabs(y);
    float z2 = z * z;
    float r = 1.f;
    float t = z2 * (1.f/6);
    r -= t;
    t *= z2 * (1.f/20);
    r += t;
    t *= z2 * (1.f/48);
    r -= t;
    t *= z2 * (1.f/72);
    r += t;
    r *= z;
    return y > 0 ? r : -r;
}

int main()
{
    /*printf("%f\n", logf(3.14f));
    printf("%f\n", mylog(3.14f));
    printf("%f\n", expf(-3.14f));
    printf("%f\n", myexp(-3.14f));
    printf("%f\n", expf(3.14f));
    printf("%f\n", myexp(3.14f));*/
    printf("%f\n", sinf(0.52f));
    printf("%f\n", mysin(0.54f));
    printf("%f\n", sinf(-0.14f));
    printf("%f\n", mysin(-0.14f));
    printf("%f\n", sinf(5.14f));
    printf("%f\n", mysin(5.14f));
    /*float x, a, b, d2;
    d2 = 0.f;
    for (x = 0.5f; x < 50.14f; x *= 1.2f) {
        a = mylog(x);
        b = logf(x);
        float d = (a - b);
        //printf("%f %f %f\n", a, b, d);
        d2 += d * d;
    }
    printf("log stddev=%f\n", sqrtf(d2));
    d2 = 0.f;
    for (x = 0.1f; x < 10.14f; x *= 1.1f) {
        a = myexp(x);
        b = expf(x);
        float d = (a - b) / b;
        //printf("%f %f %f\n", a, b, d);
        d2 += d * d;
    }
    printf("exp stddev=%f\n", sqrtf(d2));*/
    return 0;
}

