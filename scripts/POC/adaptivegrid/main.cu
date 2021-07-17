#include "helper_cuda.h"
#include "helper_math.h"
#include <cassert>
#include <cstdio>
#include <cmath>
#include <GL/glut.h>

#define NX 512
#define NY 512

template <class T>
struct volume {
    T *grid;

    void allocate() {
        size_t size = NX * NY;
        checkCudaErrors(cudaMallocManaged(&grid, size * sizeof(T)));
    }

    void free() {
        checkCudaErrors(cudaFree(grid));
    }

    __host__ __device__ T &at(int i, int j) const {
        return grid[i + j * NX];
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
static inline const float dx = 0.02f;
static inline const float dt = 0.01f;

struct DOM {
    volume<float> pos;
    volume<float> vel;
    volume<uint8_t> mask;

    void allocate() {
        pos.allocate();
        vel.allocate();
    }

    __device__ float laplacian(int i, int j) const {
        return (-4 * pos.at(i, j) + pos.at(i, j - 1) + pos.at(i, j + 1)
            + pos.at(i + 1, j) + pos.at(i - 1, j)) / (4 * dx * dx);
    }
};

__global__ void initialize1(DOM dom) {
    for (GSL(y, 0, NY)) for (GSL(x, 0, NX)) {
        dom.vel.at(x, y) = 0.f;
        dom.pos.at(x, y) = 0.f;
    }
}

void initialize(DOM dom) {
    initialize1<<<dim3(NX / 16, NY / 16, 1), dim3(16, 16, 1)>>>(dom);
}

__global__ void substep1(DOM dom) {
    for (GSL(y, 0, NY)) for (GSL(x, 0, NX)) {
        float acc = ka * dom.laplacian(x, y) - ga * dom.vel.at(x, y);
        dom.vel.at(x, y) += acc * dt;
    }
}

__global__ void substep2(DOM dom) {
    for (GSL(y, 0, NY)) for (GSL(x, 0, NX)) {
        dom.pos.at(x, y) += dom.vel.at(x, y) * dt;
    }
}

void substep(DOM dom) {
    substep1<<<dim3(NX / 16, NY / 16, 1), dim3(16, 16, 1)>>>(dom);
    substep2<<<dim3(NX / 16, NY / 16, 1), dim3(16, 16, 1)>>>(dom);
}

DOM dom;
float *pixels;

void initFunc() {
    checkCudaErrors(cudaMallocManaged(&pixels, NX * NY * sizeof(float)));
    dom.allocate();
    initialize(dom);
}

void stepFunc() {
    substep(dom);
}

__global__ void render(float *pixels, DOM dom) {
    for (GSL(y, 0, NY)) for (GSL(x, 0, NX)) {
        pixels[y * NX + x] = 0.5f + dom.pos.at(x, y);
    }
}

void renderFunc() {
    render<<<dim3(NX / 16, NY / 16, 1), dim3(16, 16, 1)>>>(pixels, dom);
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

#define ITV 100
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
