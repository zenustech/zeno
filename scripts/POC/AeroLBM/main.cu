#include "helper_cuda.h"
#include "helper_math.h"
#include <cassert>
#include <cstdio>
#include <cmath>



#define N 8

template <class T>
struct volume {
    T *grid;

    void allocate() {
        size_t size = N * N * N;
        checkCudaErrors(cudaMallocManaged(&grid, size * sizeof(T)));
    }

    void free() {
        checkCudaErrors(cudaFree(grid));
    }

    __host__ __device__ T &at(int i, int j, int k) const {
        return grid[i + j * N + k * N * N];
    }
};

#define GSL(x, nx) \
    int x = blockDim.x * blockIdx.x + threadIdx.x; \
    x < nx; x += blockDim.x * gridDim.x


static inline __constant__ const int directions[][3] = {{0,0,0},{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1},{1,1,1},{-1,-1,-1},{1,1,-1},{-1,-1,1},{1,-1,1},{-1,1,-1},{-1,1,1},{1,-1,-1}};
static inline __constant__ const float weights[] = {2.f/9.f, 1.f/9.f, 1.f/9.f, 1.f/9.f, 1.f/9.f, 1.f/9.f, 1.f/9.f,1.f/72.f, 1.f/72.f, 1.f/72.f, 1.f/72.f, 1.f/72.f, 1.f/72.f, 1.f/72.f, 1.f/72.f};

template <class T>
__global__ void _that_initialize(T that) {
    that._initialize();
}

struct lbm {
    static inline const float niu = 0.005f;
    static inline const float tau = 3.f * niu + 0.5f;
    static inline const float inv_tau = 1.f / tau;

    volume<float> rho;
    volume<float4> vel;

    void allocate() {
        rho.allocate();
        vel.allocate();
    }

    __device__ float f_eq(int q, int x, int y, int z) {
        float m = rho.at(x, y, z);
        float4 v = vel.at(x, y, z);
        float eu = v.x * directions[q][0]
            + v.y * directions[q][1] + v.z * directions[q][2];
        float uv = v.x * v.x + v.y * v.y + v.z * v.z;
        float term = 1.f + 3.f * eu + 4.5f * eu * eu - 1.5f * uv;
        float feq = weights[q] * m * term;
        return feq;
    }

    __device__ void _initialize() {
        for (GSL(z, N)) {
            for (GSL(y, N)) {
                for (GSL(x, N)) {
                    rho.at(x, y, z) = float(x) / N;
                }
            }
        }
    }

    void initialize(dim3 grid_dim, dim3 block_dim) {
        _that_initialize<<<grid_dim, block_dim>>>(*this);
    }

    /*int mdx = (x - directions[q][0] + N) % N;
    int mdy = (y - directions[q][1] + N) % N;
    int mdz = (z - directions[q][2] + N) % N;
    f_new.at(q, x, y, z) = f_old.at(q, mdx, mdy, mdz)
        * (1.f - inv_tau) + f_eq(q, mdx, mdy, mdz) * inv_tau;*/
};



int main(void)
{
    lbm lbm;
    lbm.allocate();

    lbm.initialize(dim3(1, 1, 1), dim3(1, 1, 1));

    checkCudaErrors(cudaDeviceSynchronize());

    for (int i = 0; i < N; i++) {
        printf("%f\n", lbm.rho.at(i, 0, 0));
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
