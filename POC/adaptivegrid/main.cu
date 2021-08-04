#include "helper_cuda.h"
#include "helper_math.h"
#include <cassert>
#include <cstdio>
#include <cmath>
#include <GL/glut.h>

static int counter = 0;

template <int X, int Y, class T = float>
struct volume {
    T *grid;

    void allocate() {
        size_t size = X * Y;
        checkCudaErrors(cudaMallocManaged(&grid, size * sizeof(T)));
    }

    void free() {
        checkCudaErrors(cudaFree(grid));
    }

    __host__ __device__ T &at(int i, int j) const {
        return grid[i + j * X];
    }

    __host__ __device__ auto &at(int c, int i, int j) const {
        return at(i, j)[c];
    }
};

#define GSL(_, start, end) \
    int _ = (start) + blockDim._ * blockIdx._ + threadIdx._; \
    _ < (end); _ += blockDim._ * gridDim._


static inline const float ka = 2.0f;
static inline const float ga = 0.2f;

template <int X, int Y>
struct DOM {
    volume<X, Y, float> pos;
    volume<X, Y, float> vel;
    volume<X, Y, uint8_t> mask;
    volume<X, Y, uint8_t> active;

    void allocate() {
        pos.allocate();
        vel.allocate();
        mask.allocate();
        active.allocate();
    }

    __device__ float laplacian(int i, int j) const {
        auto const dx = 10.f / X;
        return (-4 * pos.at(i, j) + pos.at(i, j - 1) + pos.at(i, j + 1)
            + pos.at(i + 1, j) + pos.at(i - 1, j)) / (4 * dx * dx);
    }
};

template <int X, int Y>
__global__ void initialize1(DOM<X, Y> dom, int type) {
    for (GSL(y, 0, Y)) for (GSL(x, 0, X)) {
        dom.vel.at(x, y) = 0.f;
        dom.pos.at(x, y) = 0.f;
        dom.mask.at(x, y) = 0;
        float fx = x * 2.f / X - 1.f;
        float fy = y * 2.f / Y - 1.f;
        float f2 = fx * fx + fy * fy;
        if (f2 < 0.1f || x == 0 || x == X - 1 || y == 0 || y == Y - 1) {
            dom.mask.at(x, y) = 1;
        }
        if (type == 0) {  // hi grid
            dom.active.at(x, y) = (fx < +0.05f);
        } else {  // lo grid
            dom.active.at(x, y) = (fx > -0.05f);
        }
    }
}

template <int X, int Y>
void initialize(DOM<X, Y> dom, int type) {
    initialize1<<<dim3(X / 16, Y / 16, 1), dim3(16, 16, 1)>>>(dom, type);
}

template <int X, int Y>
__global__ void substep1(DOM<X, Y> dom) {
    for (GSL(y, 0, Y)) for (GSL(x, 0, X)) {
        auto const dt = 5.f / X;
        if (!dom.active.at(x, y)) continue;
        if (dom.mask.at(x, y) != 0)
            continue;
        float acc = ka * dom.laplacian(x, y) - ga * dom.vel.at(x, y);
        dom.vel.at(x, y) += acc * dt;
    }
}

template <int X, int Y>
__global__ void substep2(DOM<X, Y> dom) {
    for (GSL(y, 0, Y)) for (GSL(x, 0, X)) {
        auto const dt = 5.f / X;
        if (!dom.active.at(x, y)) continue;
        dom.pos.at(x, y) += dom.vel.at(x, y) * dt;
    }
}

template <int X, int Y>
__global__ void substep3(DOM<X, Y> dom, float height) {
    for (GSL(y, 0, Y)) for (GSL(x, 0, X)) {
        if (!dom.active.at(x, y)) continue;
        float fx = x * 2.f / X - .25f;
        float fy = y * 2.f / Y - .25f;
        float f2 = fx * fx + fy * fy;
        if (f2 < 0.01f) {
            dom.pos.at(x, y) = height;
        }
    }
}

template <int X, int Y>
void substep(DOM<X, Y> dom) {
    substep1<<<dim3(X / 16, Y / 16, 1), dim3(16, 16, 1)>>>(dom);
    substep2<<<dim3(X / 16, Y / 16, 1), dim3(16, 16, 1)>>>(dom);
    float height = 1.0f * sinf(counter * 0.08f);
    substep3<<<dim3(X / 16, Y / 16, 1), dim3(16, 16, 1)>>>(dom, height);
}

template <int X, int Y>
__global__ void upper1(volume<X * 2, Y * 2> hi, volume<X, Y> lo,
    volume<X * 2, Y * 2, uint8_t> hi_active, volume<X, Y, uint8_t> lo_active) {
    for (GSL(y, 0, Y)) for (GSL(x, 0, X)) {
        if (!lo_active.at(x, y) || !hi_active.at(x * 2, y * 2)) continue;
        float val = lo.at(x, y);
        for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++) {
            hi.at(x * 2 + dx, y * 2 + dy) = val;
        }
    }
}

template <int X, int Y>
void upper(volume<X * 2, Y * 2> hi, volume<X, Y> lo,
    volume<X * 2, Y * 2, uint8_t> hi_active, volume<X, Y, uint8_t> lo_active) {
    upper1<<<dim3(X / 16, Y / 16, 1), dim3(16, 16, 1)>>>(hi, lo,
        hi_active, lo_active);
}

template <int X, int Y>
__global__ void lower1(volume<X * 2, Y * 2> hi, volume<X, Y> lo,
    volume<X * 2, Y * 2, uint8_t> hi_active, volume<X, Y, uint8_t> lo_active) {
    for (GSL(y, 0, Y)) for (GSL(x, 0, X)) {
        if (!lo_active.at(x, y) || !hi_active.at(x * 2, y * 2)) continue;
        float val = 0.f;
        for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++) {
            val += hi.at(x * 2 + dx, y * 2 + dy);
        }
        lo.at(x, y) = val * 0.25f;
    }
}

template <int X, int Y>
void lower(volume<X * 2, Y * 2> hi, volume<X, Y> lo,
    volume<X * 2, Y * 2, uint8_t> hi_active, volume<X, Y, uint8_t> lo_active) {
    lower1<<<dim3(X / 16, Y / 16, 1), dim3(16, 16, 1)>>>(hi, lo,
        hi_active, lo_active);
}

template <int X, int Y>
void lower(DOM<X * 2, Y * 2> hi, DOM<X, Y> lo) {
    lower(hi.pos, lo.pos, hi.active, lo.active);
    lower(hi.vel, lo.vel, hi.active, lo.active);
}

template <int X, int Y>
void upper(DOM<X * 2, Y * 2> hi, DOM<X, Y> lo) {
    upper(hi.pos, lo.pos, hi.active, lo.active);
    upper(hi.vel, lo.vel, hi.active, lo.active);
}

#define NX 512
#define NY 512
DOM<NX / 1, NY / 1> dom;
DOM<NX / 2, NY / 2> dom2;
float *pixels;

void initFunc() {
    checkCudaErrors(cudaMallocManaged(&pixels, NX * NY * sizeof(float)));
    dom.allocate();
    dom2.allocate();
    initialize(dom, 0);
    initialize(dom2, 1);
}

void stepFunc() {
    substep(dom);
    substep(dom);
    lower(dom, dom2);
    substep(dom2);
    upper(dom, dom2);
    counter++;
}

template <int X, int Y>
__global__ void render1(float *pixels, DOM<X, Y> dom, float scale) {
    for (GSL(y, 0, NY)) for (GSL(x, 0, NX)) {
        float val = dom.pos.at(x * X / NX, y * Y / NY) * scale;
        pixels[y * NX + x] = 0.5f + 0.5f * val;
    }
}

template <int X, int Y>
void render(float *pixels, DOM<X, Y> dom, float scale) {
    render1<<<dim3(NX / 16, NY / 16, 1), dim3(16, 16, 1)>>>(pixels, dom, scale);
}

void renderFunc() {
    if (counter % 200 < 100) {
        render(pixels, dom, 1.f);
    } else {
        render(pixels, dom2, 1.f);
    }
    checkCudaErrors(cudaDeviceSynchronize());
    /*printf("03:%f\n", pixels[0 * N + 3]);
    printf("30:%f\n", pixels[3 * NX + 0]);
    printf("33:%f\n", pixels[3 * NX + 3]);*/
}

void displayFunc() {
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawPixels(NX, NY, GL_RED, GL_FLOAT, pixels);
    glFlush();
}

#define ITV 2
void timerFunc(int unused) {
    stepFunc();
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
    glutInitWindowSize(NX, NY);
    glutCreateWindow("GLUT Window");
    glutDisplayFunc(displayFunc);
    glutKeyboardFunc(keyboardFunc);
    initFunc();
    renderFunc();
    glutTimerFunc(ITV, timerFunc, 0);
    glutMainLoop();
}
