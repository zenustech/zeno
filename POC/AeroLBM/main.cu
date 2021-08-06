// https://www.slideserve.com/lars/3d-simulation-of-particle-motion-in-lid-driven-cavity-flow-by-mrt-lbm
#include "helper_cuda.h"
#include "helper_math.h"
#include <cassert>
#include <cstdio>
#include <cmath>
#include <GL/glut.h>

template <int NX, int NY, int NZ, class T>
struct volume {
    T *grid;

    void allocate() {
        size_t size = NX * NY * NZ;
        checkCudaErrors(cudaMallocManaged(&grid, size * sizeof(T)));
    }

    void free() {
        checkCudaErrors(cudaFree(grid));
    }

    __host__ __device__ T &at(int i, int j, int k) const {
        return grid[i + j * NX + k * NX * NY];
    }

    __host__ __device__ auto &at(int c, int i, int j, int k) const {
        return at(i, j, k)[c];
    }
};

template <int NX, int NY, int NZ, class T, int N>
struct volume_soa {
    T *grid;

    void allocate() {
        size_t size = NX * NY * NZ * N;
        checkCudaErrors(cudaMallocManaged(&grid, size * sizeof(T)));
    }

    void free() {
        checkCudaErrors(cudaFree(grid));
    }

    __host__ __device__ T &at(int c, int i, int j, int k) const {
        return grid[i + j * NX + k * NX * NY + c * NX * NY * NZ];
    }
};

#define GSL(_, start, end) \
    int _ = (start) + blockDim._ * blockIdx._ + threadIdx._; \
    _ < (end); _ += blockDim._ * gridDim._


static inline __constant__ const int directions[][3] = {{0,0,0},{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1},{1,1,1},{-1,-1,-1},{1,1,-1},{-1,-1,1},{1,-1,1},{-1,1,-1},{-1,1,1},{1,-1,-1}};
static inline __constant__ const float weights[] = {2.f/9.f, 1.f/9.f, 1.f/9.f, 1.f/9.f, 1.f/9.f, 1.f/9.f, 1.f/9.f,1.f/72.f, 1.f/72.f, 1.f/72.f, 1.f/72.f, 1.f/72.f, 1.f/72.f, 1.f/72.f, 1.f/72.f};
[[maybe_unused]] static inline __constant__ const int inverse_index[] = {0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13};

static_assert(sizeof(weights) / sizeof(weights[0]) == 15);

static inline const float niu = 0.005f;
static inline const float tau = 3.f * niu + 0.5f;
static inline const float inv_tau = 1.f / tau;

template <int NX, int NY, int NZ>
struct LBM {
    volume<NX, NY, NZ, float4> vel;
    volume_soa<NX, NY, NZ, float, 16> f_new;
    volume_soa<NX, NY, NZ, float, 16> f_old;
    volume<NX, NY, NZ, uint8_t> active;

    void allocate() {
        vel.allocate();
        f_new.allocate();
        f_old.allocate();
        active.allocate();
    }

    __device__ float f_eq(int q, int x, int y, int z) {
        float4 v = vel.at(x, y, z);
        float eu = v.x * directions[q][0]
            + v.y * directions[q][1] + v.z * directions[q][2];
        float uv = v.x * v.x + v.y * v.y + v.z * v.z;
        float term = 1.f + 3.f * eu + 4.5f * eu * eu - 1.5f * uv;
        float feq = weights[q] * v.w * term;
        return feq;
    }
};

template <int NX, int NY, int NZ>
__global__ void initialize1(LBM<NX, NY, NZ> lbm, int type) {
    for (GSL(z, 0, NZ)) for (GSL(y, 0, NY)) for (GSL(x, 0, NX)) {
        lbm.vel.at(x, y, z) = make_float4(0.f, 0.f, 0.f, 1.f);
        lbm.active.at(x, y, z) = 0;
        if (type == 0) {  // hires grid
            if (x <= NX / 2 + 2) lbm.active.at(x, y, z) = 1;
        } else if (type == 1) {  // lores grid
            if (x >= NX / 2) lbm.active.at(x, y, z) = 1;
        }
    }
}

template <int NX, int NY, int NZ>
__global__ void initialize2(LBM<NX, NY, NZ> lbm) {
    for (GSL(z, 0, NZ)) for (GSL(y, 0, NY)) for (GSL(x, 0, NX)) {
        for (int q = 0; q < 15; q++) {
            float f = lbm.f_eq(q, x, y, z);
            lbm.f_new.at(q, x, y, z) = f;
            lbm.f_old.at(q, x, y, z) = f;
        }
    }
}

template <int NX, int NY, int NZ>
void initialize(LBM<NX, NY, NZ> lbm, int type) {
    initialize1<<<dim3(NX / 8, NY / 8, NZ / 8), dim3(8, 8, 8)>>>(lbm, type);
    initialize2<<<dim3(NX / 8, NY / 8, NZ / 8), dim3(8, 8, 8)>>>(lbm);
}

template <int NX, int NY, int NZ>
__global__ void synchi2lo1(LBM<NX*2, NY*2, NZ*2> hi, LBM<NX, NY, NZ> lo) {
    for (GSL(z, 0, NZ)) for (GSL(y, 0, NY)) for (GSL(x, 0, NX)) {
        if (!hi.active.at(x * 2, y * 2, z * 2) || !lo.active.at(x, y, z)) continue;
        for (int dz = 0; dz < 2; dz++) for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++) {
            lo.vel.at(x, y, z) += hi.vel.at(x * 2 + dx, y * 2 + dy, z * 2 + dz);
            for (int q = 0; q < 15; q++) {
                lo.f_old.at(q, x, y, z) += hi.f_old.at(q, x * 2 + dx, y * 2 + dy, z * 2 + dz);
            }
        }
        lo.vel.at(x, y, z) /= 8.f;
        for (int q = 0; q < 15; q++) {
            lo.f_old.at(q, x, y, z) /= 8.f;
        }
    }
}

template <int NX, int NY, int NZ>
__global__ void synchi2lo2(LBM<NX*2, NY*2, NZ*2> hi, LBM<NX, NY, NZ> lo) {
    for (GSL(z, 0, NZ)) for (GSL(y, 0, NY)) for (GSL(x, 0, NX)) {
        if (!hi.active.at(x * 2, y * 2, z * 2) || !lo.active.at(x, y, z)) continue;
        for (int q = 0; q < 15; q++) {
            int xmd = x - 1;
            int ymd = y;
            int zmd = y;
            lo.f_old.at(q, x, y, z) = lo.f_eq(q, x, y, z)
                - lo.f_eq(q, xmd, ymd, zmd) + lo.f_old.at(q, xmd, ymd, zmd);
        }
    }
}

template <int NX, int NY, int NZ>
void synchi2lo(LBM<NX*2, NY*2, NZ*2> hi, LBM<NX, NY, NZ> lo) {
    synchi2lo1<<<dim3(NX / 8, NY / 8, NZ / 8), dim3(8, 8, 8)>>>(hi, lo);
    //synchi2lo2<<<dim3(NX / 8, NY / 8, NZ / 8), dim3(8, 8, 8)>>>(hi, lo);
}

template <int NX, int NY, int NZ, class T>
__device__ auto trilerp(T const &t, int x, int y, int z, int dx, int dy, int dz) {
    float c0 = 0.85f;
    float c1 = 1.f - c0;
    float x0 = dx ? c1 : c0;
    float x1 = dx ? c0 : c1;
    float y0 = dy ? c1 : c0;
    float y1 = dy ? c0 : c1;
    float z0 = dz ? c1 : c0;
    float z1 = dz ? c0 : c1;
    int x_ = x, y_ = y, z_ = z;
    if (x_ < NX - 1) x_++;
    if (y_ < NY - 1) y_++;
    if (z_ < NZ - 1) z_++;
    if (!dx && x > 0) x--;
    if (!dy && y > 0) y--;
    if (!dz && z > 0) z--;
    return x0 * y0 * z0 * t(x, y, z)
         + x0 * y0 * z1 * t(x, y, z_)
         + x0 * y1 * z0 * t(x, y_, z)
         + x0 * y1 * z1 * t(x, y_, z_)
         + x1 * y0 * z0 * t(x_, y, z)
         + x1 * y0 * z1 * t(x_, y, z_)
         + x1 * y1 * z0 * t(x_, y_, z)
         + x1 * y1 * z1 * t(x_, y_, z_);
}

template <int NX, int NY, int NZ>
__global__ void synclo2hi1(LBM<NX*2, NY*2, NZ*2> hi, LBM<NX, NY, NZ> lo) {
    for (GSL(z, 0, NZ)) for (GSL(y, 0, NY)) for (GSL(x, 0, NX)) {
        if (!hi.active.at(x * 2, y * 2, z * 2) || !lo.active.at(x, y, z)) continue;
        for (int dz = 0; dz < 2; dz++) for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++) {
            hi.vel.at(x * 2 + dx, y * 2 + dy, z * 2 + dz) = trilerp<NX, NY, NZ>(
                    [&](auto x, auto y, auto z) { return lo.vel.at(x, y, z); },
                    x, y, z, dx, dy, dz);
            for (int q = 0; q < 15; q++) {
                hi.f_old.at(q, x * 2 + dx, y * 2 + dy, z * 2 + dz) = trilerp<NX, NY, NZ>(
                    [&](auto x, auto y, auto z) { return lo.f_old.at(q, x, y, z); },
                    x, y, z, dx, dy, dz);
            }
        }
    }
}

/*template <int NX, int NY, int NZ>
__global__ void synclo2hi2(LBM<NX*2, NY*2, NZ*2> hi, LBM<NX, NY, NZ> lo) {
    for (GSL(z, 0, NZ*2)) for (GSL(y, 0, NY*2)) for (GSL(x, 0, NX*2)) {
        if (!lo.active.at(x / 2, y / 2, z / 2) || !hi.active.at(x, y, z)) continue;
        for (int q = 0; q < 15; q++) {
            auto l = hi.f_eq(q, x + 2, y, z) - hi.f_old.at(q, x + 2, y, z);
            auto r = hi.f_eq(q, x + 1, y, z) - hi.f_old.at(q, x + 1, y, z);
            hi.f_old.at(q, x, y, z) = hi.f_eq(q, x, y, z) - (l + r) * 0.5f;
        }
    }
}*/

template <int NX, int NY, int NZ>
void synclo2hi(LBM<NX*2, NY*2, NZ*2> hi, LBM<NX, NY, NZ> lo) {
    synclo2hi1<<<dim3(NX / 8, NY / 8, NZ / 8), dim3(8, 8, 8)>>>(hi, lo);
    //synclo2hi2<<<dim3(NX*2 / 8, NY*2 / 8, NZ*2 / 8), dim3(8, 8, 8)>>>(hi, lo);
}

template <int NX, int NY, int NZ>
__global__ void substep1(LBM<NX, NY, NZ> lbm) {
    //for (GSL(z, 1, NZ - 1)) for (GSL(y, 1, NY - 1)) for (GSL(x, 1, NX - 1)) {
    for (GSL(z, 0, NZ)) for (GSL(y, 0, NY)) for (GSL(x, 0, NX)) {
        if (!lbm.active.at(x, y, z)) continue;
        for (int q = 0; q < 15; q++) {
            //int mdx = x - directions[q][0];
            //int mdy = y - directions[q][1];
            //int mdz = z - directions[q][2];
            int mdx = (x - directions[q][0] + NX) % NX;
            int mdy = (y - directions[q][1] + NY) % NY;
            int mdz = (z - directions[q][2] + NZ) % NZ;
            lbm.f_new.at(q, x, y, z) = lbm.f_old.at(q, mdx, mdy, mdz)
                * (1.f - inv_tau) + lbm.f_eq(q, mdx, mdy, mdz) * inv_tau;
        }
    }
}

/*__global__ void substep11(LBM lbm) {
    for (GSL(z, 0, NZ)) for (GSL(y, 0, NY)) for (GSL(x, 0, NX)) {
        self.feq
    }
}

__global__ void substep12(LBM lbm) {
    //for (GSL(z, 1, NZ - 1)) for (GSL(y, 1, NY - 1)) for (GSL(x, 1, NX - 1)) {
    for (GSL(z, 0, NZ)) for (GSL(y, 0, NY)) for (GSL(x, 0, NX)) {
        for (int q = 0; q < 15; q++) {
            //int mdx = x - directions[q][0];
            //int mdy = y - directions[q][1];
            //int mdz = z - directions[q][2];
            int mdx = (x - directions[q][0] + NX) % NX;
            int mdy = (y - directions[q][1] + NY) % NY;
            int mdz = (z - directions[q][2] + NZ) % NZ;
            [[maybe_unused]] int iq = inverse_index[q];
            lbm.f_new.at(q, x, y, z) = lbm.f_old.at(q, mdx, mdy, mdz);
            //lbm.f_new.at(q, x, y, z) = lbm.f_old.at(q, mdx, mdy, mdz)
                // * (1.f - inv_tau) + lbm.f_eq(q, mdx, mdy, mdz) * inv_tau;
        }
    }
}*/

template <int NX, int NY, int NZ>
__global__ void substep2(LBM<NX, NY, NZ> lbm) {
    //for (GSL(z, 1, NZ - 1)) for (GSL(y, 1, NY - 1)) for (GSL(x, 1, NX - 1)) {
    for (GSL(z, 0, NZ)) for (GSL(y, 0, NY)) for (GSL(x, 0, NX)) {
        if (!lbm.active.at(x, y, z)) continue;
        float m = 0.f;
        float vx = 0.f, vy = 0.f, vz = 0.f;
        for (int q = 0; q < 15; q++) {
            float f = lbm.f_new.at(q, x, y, z);
            lbm.f_old.at(q, x, y, z) = f;
            vx += f * directions[q][0];
            vy += f * directions[q][1];
            vz += f * directions[q][2];
            m += f;
        }
        float mscale = 1.f / fmaxf(m, 1e-6f);
        vx *= mscale; vy *= mscale; vz *= mscale;
        lbm.vel.at(x, y, z) = make_float4(vx, vy, vz, m);
    }
}

//__device__ void applybc(LBM lbm, at

template <int NX, int NY, int NZ>
__global__ void applybc1(LBM<NX, NY, NZ> lbm) {
    for (GSL(z, 1, NZ - 1)) for (GSL(y, 1, NY - 1)) {
    //for (GSL(z, 0, NZ)) for (GSL(y, 0, NY)) {
        lbm.vel.at(0, y, z) = lbm.vel.at(1, y, z);
        lbm.vel.at(0, y, z).x = 0.15f;
        lbm.vel.at(0, y, z).y = 0.f;
        lbm.vel.at(0, y, z).z = 0.f;
        for (int q = 0; q < 15; q++) {
            lbm.f_old.at(q, 0, y, z) = lbm.f_eq(q, 0, y, z) - lbm.f_eq(q, 1, y, z) + lbm.f_old.at(q, 1, y, z);
        }
        lbm.vel.at(NX - 1, y, z) = lbm.vel.at(NX - 2, y, z);
        for (int q = 0; q < 15; q++) {
            lbm.f_old.at(q, NX - 1, y, z) = lbm.f_eq(q, NX - 1, y, z) - lbm.f_eq(q, NX - 2, y, z) + lbm.f_old.at(q, NX - 2, y, z);
        }
    }
}

template <int NX, int NY, int NZ>
__global__ void applybc2(LBM<NX, NY, NZ> lbm) {
    for (GSL(z, 0, NZ)) for (GSL(x, 0, NX)) {
        lbm.vel.at(x, 0, z) = lbm.vel.at(x, 1, z);
        lbm.vel.at(x, 0, z).x = 0.f;
        lbm.vel.at(x, 0, z).y = 0.f;
        lbm.vel.at(x, 0, z).z = 0.f;
        for (int q = 0; q < 15; q++) {
            lbm.f_old.at(q, x, 0, z) = lbm.f_eq(q, x, 0, z) - lbm.f_eq(q, x, 1, z) + lbm.f_old.at(q, x, 1, z);
        }
        lbm.vel.at(x, NY - 1, z) = lbm.vel.at(x, NY - 2, z);
        lbm.vel.at(x, NY - 1, z).x = 0.f;
        lbm.vel.at(x, NY - 1, z).y = 0.f;
        lbm.vel.at(x, NY - 1, z).z = 0.f;
        for (int q = 0; q < 15; q++) {
            lbm.f_old.at(q, x, NY - 1, z) = lbm.f_eq(q, x, NY - 1, z) - lbm.f_eq(q, x, NY - 2, z) + lbm.f_old.at(q, x, NY - 2, z);
        }
    }
}

template <int NX, int NY, int NZ>
__global__ void applybc3(LBM<NX, NY, NZ> lbm) {
    for (GSL(y, 0, NY)) for (GSL(x, 0, NX)) {
        lbm.vel.at(x, y, 0) = lbm.vel.at(x, y, 1);
        lbm.vel.at(x, y, 0).x = 0.f;
        lbm.vel.at(x, y, 0).y = 0.f;
        lbm.vel.at(x, y, 0).z = 0.f;
        for (int q = 0; q < 15; q++) {
            lbm.f_old.at(q, x, y, 0) = lbm.f_eq(q, x, y, 0) - lbm.f_eq(q, x, y, 1) + lbm.f_old.at(q, x, y, 1);
        }
        lbm.vel.at(x, y, NZ - 1) = lbm.vel.at(x, y, NZ - 2);
        lbm.vel.at(x, y, NZ - 1).x = 0.f;
        lbm.vel.at(x, y, NZ - 1).y = 0.f;
        lbm.vel.at(x, y, NZ - 1).z = 0.f;
        for (int q = 0; q < 15; q++) {
            lbm.f_old.at(q, x, y, NZ - 1) = lbm.f_eq(q, x, y, NZ - 1) - lbm.f_eq(q, x, y, NZ - 2) + lbm.f_old.at(q, x, y, NZ - 2);
        }
    }
}

template <int NX, int NY, int NZ>
__global__ void applybc4(LBM<NX, NY, NZ> lbm) {
    for (GSL(z, 0, NZ)) for (GSL(y, 0, NY)) for (GSL(x, 0, NX)) {
        if (!lbm.active.at(x, y, z)) continue;
        float fx = x * 2.f / NY - 1.f;
        float fy = y * 2.f / NY - 1.f;
        float fz = z * 2.f / NZ - 1.f;
        if (fx * fx + fy * fy + fz * fz >= .065f) {
            continue;
        }
        lbm.vel.at(x, y, z).x = 0.f;
        lbm.vel.at(x, y, z).y = 0.f;
        lbm.vel.at(x, y, z).z = 0.f;
    }
}

template <int NX, int NY, int NZ>
void substep(LBM<NX, NY, NZ> lbm) {
    substep1<<<dim3(NX / 8, NY / 8, NZ / 8), dim3(8, 8, 8)>>>(lbm);
    substep2<<<dim3(NX / 8, NY / 8, NZ / 8), dim3(8, 8, 8)>>>(lbm);
    applybc1<<<dim3(1, NY / 16, NZ / 16), dim3(1, 16, 16)>>>(lbm);
    applybc2<<<dim3(NX / 16, 1, NZ / 16), dim3(16, 1, 16)>>>(lbm);
    applybc3<<<dim3(NX / 16, NY / 16, 1), dim3(16, 16, 1)>>>(lbm);
    applybc4<<<dim3(NX / 16, NY / 16, NZ / 16), dim3(8, 8, 8)>>>(lbm);
}

#define NNX 512
#define NNY 128
#define NNZ 128

template <int NX, int NY, int NZ>
__global__ void render1(float *pixels, LBM<NX, NY, NZ> lbm, int chan) {
    for (GSL(y, 0, NNY)) for (GSL(x, 0, NNX)) {
        float4 v = trilerp<NX, NY, NZ>([&] (auto x, auto y, auto z) {
                return lbm.vel.at(x, y, z);
        }, x * NX / NNX, y * NY / NNY, NZ / 2, x * (NNX / NX), y % (NNY / NY), 0);
        //float val = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
        //float val = 4.f * sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
        //float val = v.x * 4.f;
        float val = v.w * 0.5f;
        pixels[(y * NNX + x) * 4 + chan] = val;
    }
}

template <int NX, int NY, int NZ>
void render(float *pixels, LBM<NX, NY, NZ> lbm, int chan) {
    render1<<<dim3(NNX / 16, NNY / 16, 1), dim3(16, 16, 1)>>>(pixels, lbm, chan);
}

LBM<NNX/2, NNY/2, NNZ/2> lbm;
LBM<NNX/4, NNY/4, NNZ/4> lbm2;
float *pixels;

void initFunc() {
    checkCudaErrors(cudaMallocManaged(&pixels, 4 * NNX * NNY * sizeof(float)));
    lbm.allocate();
    initialize(lbm, 0);
    lbm2.allocate();
    initialize(lbm2, 1);
}

void renderFunc() {
    synchi2lo(lbm, lbm2);
    substep(lbm2);
    synclo2hi(lbm, lbm2);
    substep(lbm);
    substep(lbm);

    render(pixels, lbm, 0);
    render(pixels, lbm2, 1);
    checkCudaErrors(cudaDeviceSynchronize());
    printf("R:%f\n", pixels[(NNY / 2 * NNX + NNX * 1 / 4) * 4 + 0]);
    printf("G:%f\n", pixels[(NNY / 2 * NNX + NNX * 3 / 4) * 4 + 1]);
}

void displayFunc() {
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawPixels(NNX, NNY, GL_RGBA, GL_FLOAT, pixels);
    glFlush();
}

#define ITV 20
void timerFunc(int unused) {
    renderFunc();
    glutPostRedisplay();
    glutTimerFunc(ITV, timerFunc, 0);
}

void keyboardFunc(unsigned char key, int x, int y) {
    if (key == 27)
        exit(0);
}

int main(int argc, char **argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_SINGLE | GLUT_RGBA);
    glutInitWindowPosition(100, 100);
    glutInitWindowSize(NNX, NNY);
    glutCreateWindow("GLUT Window");
    glutDisplayFunc(displayFunc);
    glutKeyboardFunc(keyboardFunc);
    initFunc();
    renderFunc();
    glutTimerFunc(ITV, timerFunc, 0);
    glutMainLoop();
}
