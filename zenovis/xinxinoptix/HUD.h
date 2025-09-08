#include "zxxglslvec.h"

// Simple 7-segment number drawer
// Shows frame count as digits

// segment encoding: a b c d e f g
const int segs[10] = {
    0b0111111, // 0
    0b0000110, // 1
    0b1011011, // 2
    0b1001111, // 3
    0b1100110, // 4
    0b1101101, // 5
    0b1111101, // 6
    0b0000111, // 7
    0b1111111, // 8
    0b1101111  // 9
};

inline float rectSDF(vec2 p, vec2 c, vec2 s) {
    vec2 d = abs(p-c)-s;
    d = max(d,0.0);
    return length(d);
}

// draw one digit at local uv (0..1)
inline float drawDigit(int d, vec2 uv){
    float m = 0.12;
    float t = 0.14;
    float aa = 0.01;
    int bits = segs[d];

    // centers and sizes
    vec2 hSize = vec2(0.5-2.0*m, t*0.6);
    vec2 vSize = vec2(t*0.6, 0.5-2.0*m-hSize.y);

    vec2 aC = vec2(0.5,1.0-m-hSize.y*0.5);
    vec2 gC = vec2(0.5,0.5);
    vec2 dC = vec2(0.5,m+hSize.y*0.5);

    vec2 bC = vec2(1.0-m-vSize.x*0.5,0.75);
    vec2 cC = vec2(1.0-m-vSize.x*0.5,0.25);
    vec2 fC = vec2(m+vSize.x*0.5,0.75);
    vec2 eC = vec2(m+vSize.x*0.5,0.25);

    float maskA = 1.0 - smoothstep(0.0, aa, rectSDF(uv,aC,hSize));
    float maskG = 1.0 - smoothstep(0.0, aa, rectSDF(uv,gC,hSize));
    float maskD = 1.0 - smoothstep(0.0, aa, rectSDF(uv,dC,hSize));
    float maskB = 1.0 - smoothstep(0.0, aa, rectSDF(uv,bC,vSize));
    float maskC = 1.0 - smoothstep(0.0, aa, rectSDF(uv,cC,vSize));
    float maskF = 1.0 - smoothstep(0.0, aa, rectSDF(uv,fC,vSize));
    float maskE = 1.0 - smoothstep(0.0, aa, rectSDF(uv,eC,vSize));

    float segOn = 0.0;
    if((bits & 1)!=0) segOn += maskA;
    if((bits & 2)!=0) segOn += maskB;
    if((bits & 4)!=0) segOn += maskC;
    if((bits & 8)!=0) segOn += maskD;
    if((bits & 16)!=0) segOn += maskE;
    if((bits & 32)!=0) segOn += maskF;
    if((bits & 64)!=0) segOn += maskG;

    return clamp(segOn,0.0,1.0);
}

__device__ inline float drawLetterM(vec2 uv) {
    float aa = 0.02f;
    float mask = 0.f;

    // left vertical
    mask += 1.0f - smoothstep(0.0f, aa, rectSDF(uv, vec2(0.15,0.5), vec2(0.08,0.5)));
    // middle vertical
    mask += 1.0f - smoothstep(0.0f, aa, rectSDF(uv, vec2(0.5,0.5), vec2(0.08,0.5)));
    // right vertical
    mask += 1.0f - smoothstep(0.0f, aa, rectSDF(uv, vec2(0.85,0.5), vec2(0.08,0.5)));
    // top connector bar
    mask += 1.0f - smoothstep(0.0f, aa, rectSDF(uv, vec2(0.5,0.85), vec2(0.35,0.08)));

    return clamp(mask,0.f,1.f);
}

__device__ inline float drawLetterS(vec2 uv) {
    // crude S: three horizontal + two vertical
    float aa = 0.02f;
    float mask = 0.f;
    mask += 1.0f - smoothstep(0.0f, aa, rectSDF(uv, vec2(0.5,0.9), vec2(0.5,0.1)));
    mask += 1.0f - smoothstep(0.0f, aa, rectSDF(uv, vec2(0.5,0.5), vec2(0.5,0.1)));
    mask += 1.0f - smoothstep(0.0f, aa, rectSDF(uv, vec2(0.5,0.1), vec2(0.5,0.1)));
    mask += 1.0f - smoothstep(0.0f, aa, rectSDF(uv, vec2(0.05,0.7), vec2(0.05,0.2)));
    mask += 1.0f - smoothstep(0.0f, aa, rectSDF(uv, vec2(0.95,0.3), vec2(0.05,0.2)));
    return clamp(mask,0.f,1.f);
}

__forceinline__ __device__ float3 operator*(float a, uchar3 b)
{
    return { a * b.x, a * b.y, a * b.z };
}

__forceinline__ __device__ uchar3 mix(uchar3 a, uchar3 b, float c) {
    auto re = (1.f-c) * a + c * b;
    return {
        (unsigned char)re.x,
        (unsigned char)re.y,
        (unsigned char)re.z,
    };
}

__forceinline__ __device__ int digitLength(int v) {
    if (v == 0) return 1;
    return (int)floorf(log10f((float)abs(v))) + 1;
}

inline void drawHUD(uchar3* fragColor, uint16_t value, const vec2 uv) {
    // parameters
    if (uv.x > 1.0f || uv.y > 1.0f) return;
    int maxLength = digitLength(value);

    float digitW = 0.12;
    float spacing = 0.02;

    float mask = 0.0;

    auto index = (int)( uv.x / (digitW + spacing) );
    vec2 local_uv = uv;

    if (index<maxLength) {

        local_uv.x = ( uv.x - index * (digitW + spacing) ) / digitW;

        int d = (value / int(powf(10.0f,float(maxLength-1-index)))) % 10;
        mask = max(mask, drawDigit(d,local_uv));
    } else if (index==maxLength) {
        local_uv.x = ( uv.x - index * (digitW + spacing) ) / (digitW + spacing);
        local_uv.y *= 1.5f;
        mask = drawLetterM(local_uv);
    } else if (index==maxLength+1){
        local_uv.x = ( uv.x - index * (digitW + spacing) ) / (digitW * 0.5);
        local_uv.y *= 1.5f;
        mask = drawLetterS(local_uv);
    } else {
        return;
    }

    uchar3 bg = *fragColor;
    bg.x *= 0.5f;
    bg.y *= 0.5f;
    bg.z *= 0.5f;
    uchar3 fg = uchar3{ 255,255,0 };
    uchar3 re = mix(bg,fg,mask);

    *fragColor = re;
}