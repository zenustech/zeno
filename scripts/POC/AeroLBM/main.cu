#include "helper_cuda.h"
#include "helper_math.h"
#include <cassert>
#include <cstdio>
#include <cmath>


const float niu = 0.005f;
const float tau = 3.f * niu + 0.5f;
const float inv_tau = 1.f / tau;

int directions[][3] = {{0,0,0},{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1},{1,1,1},{-1,-1,-1},{1,1,-1},{-1,-1,1},{1,-1,1},{-1,1,-1},{-1,1,1},{1,-1,-1}};
float weights[] = {2.f/9.f, 1.f/9.f, 1.f/9.f, 1.f/9.f, 1.f/9.f, 1.f/9.f, 1.f/9.f,1.f/72.f, 1.f/72.f, 1.f/72.f, 1.f/72.f, 1.f/72.f, 1.f/72.f, 1.f/72.f, 1.f/72.f};


#define N 8

struct volume {
    float *grid;

    void allocate() {
        size_t size = N * N * N;
        checkCudaErrors(cudaMallocManaged(&grid, size * sizeof(float)));
    }

    void free() {
        checkCudaErrors(cudaFree(grid));
    }

    __host__ __device__ float &at(int i, int j, int k) const {
        return grid[i + j * N + k * N * N];
    }
};

#define GSL(x, nx) \
    int x = blockDim.x * blockIdx.x + threadIdx.x; \
    x < nx; x += blockDim.x * gridDim.x

__global__ void fill(volume vol) {
    for (GSL(z, N)) {
        for (GSL(y, N)) {
            for (GSL(x, N)) {
                vol.at(x, y, z) = float(x) / N;
            }
        }
    }
}


int main(void)
{
    volume vol;
    vol.allocate();

    fill<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(vol);

    checkCudaErrors(cudaDeviceSynchronize());

    for (int i = 0; i < N; i++) {
        printf("%f\n", vol.at(i, 0, 0));
    }

    return 0;
}


/*
void displayFunc() {
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawPixels(N, N, GL_RED, GL_FLOAT, pixels);
    glFlush();
}

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
    glutInitWindowSize(N, N);
    glutCreateWindow("GLUT Window");
    glutDisplayFunc(displayFunc);
    glutKeyboardFunc(keyboardFunc);
    initFunc();
    renderFunc();
    glutTimerFunc(ITV, timerFunc, 0);
    glutMainLoop();
}
*/
