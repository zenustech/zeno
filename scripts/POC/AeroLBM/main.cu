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

    __host__ __device__ auto &at(int c, int i, int j, int k) const {
        return at(i, j, k)[c];
    }
};

#define GSL(x, nx) \
    int x = blockDim.x * blockIdx.x + threadIdx.x; \
    x < nx; x += blockDim.x * gridDim.x


static inline __constant__ const int directions[][3] = {{0,0,0},{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1},{1,1,1},{-1,-1,-1},{1,1,-1},{-1,-1,1},{1,-1,1},{-1,1,-1},{-1,1,1},{1,-1,-1}};
static inline __constant__ const float weights[] = {2.f/9.f, 1.f/9.f, 1.f/9.f, 1.f/9.f, 1.f/9.f, 1.f/9.f, 1.f/9.f,1.f/72.f, 1.f/72.f, 1.f/72.f, 1.f/72.f, 1.f/72.f, 1.f/72.f, 1.f/72.f, 1.f/72.f};

__global__ void fuck() {
    printf("FUCK NVIDIA\n");
}

#define GOSTUB(name, gx, gy, gz, bx, by, bz) \
    __host__ void name( \
        dim3 grid_dim = {gx, gy, gz}, dim3 block_dim = {bx, by, bz}) { \
        golaunch<<<grid_dim, block_dim>>> \
        (*this, [] __device__ (auto that, auto &&...ts) { \
            that.name(std::forward(ts)...); \
        }); \
    }

#define CUSTUB(name) \
        golaunch<<<GridDim{}, BlockDim{}>>> \
        (*this, [] __device__ (auto that) { \
            that.name<GridDim, BlockDim>(); \
        });

template <int X = 1, int Y = 1, int Z = 1>
struct Dim {
    static constexpr int x = X;
    static constexpr int y = Y;
    static constexpr int z = Z;

    operator dim3() const {
        return {x, y, z};
    }
};

template <class T, class F>
__global__ void golaunch(T t, F f) {
    printf("FUCK\n");
    f(&t);
}

struct lbm {
    static inline const float niu = 0.005f;
    static inline const float tau = 3.f * niu + 0.5f;
    static inline const float inv_tau = 1.f / tau;

    volume<float4> vel;
    //volume<float[16]> f_new;
    //volume<float[16]> f_old;

    void allocate() {
        vel.allocate();
        //f_new.allocate();
        //f_old.allocate();
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

    __device__ void _initialize() {
        for (GSL(z, N)) for (GSL(y, N)) for (GSL(x, N)) {
            vel.at(x, y, z).w = float(x) / N;
        }
    }

    __host__ void initialize() {
        golaunch<<<1, 1>>>
        (*this, [] __device__ (auto *that) {
            that->_initialize();
        });
    }

    /*__device__ void substep1() {
        for (GSL(z, N)) for (GSL(y, N)) for (GSL(x, N)) {
            for (int q = 0; q < 15; q++) {
                int mdx = (x - directions[q][0] + N) % N;
                int mdy = (y - directions[q][1] + N) % N;
                int mdz = (z - directions[q][2] + N) % N;
                f_new.at(q, x, y, z) = f_old.at(q, mdx, mdy, mdz)
                    * (1.f - inv_tau) + f_eq(q, mdx, mdy, mdz) * inv_tau;
            }
        }
    }
    GOSTUB(substep1, N / 8, N / 8, N / 8, 8, 8, 8);

    __device__ void substep2() {
        for (GSL(z, N)) for (GSL(y, N)) for (GSL(x, N)) {
            float m = 0.f;
            float vx = 0.f, vy = 0.f, vz = 0.f;
            for (int q = 0; q < 15; q++) {
                float f = f_new.at(q, x, y, z);
                f_old.at(q, x, y, z) = f;
                vx += f * directions[q][0];
                vy += f * directions[q][1];
                vz += f * directions[q][2];
                m += f;
            }
            float mscale = 1.f / fmaxf(m, 1e-6f);
            vx /= mscale; vy /= mscale; vz /= mscale;
            vel.at(x, y, z) = make_float4(vx, vy, vz, m);
        }
    }
    GOSTUB(substep2, N / 8, N / 8, N / 8, 8, 8, 8);*/
};



int main(void)
{
    lbm lbm;
    lbm.allocate();

    lbm.initialize();

    checkCudaErrors(cudaDeviceSynchronize());

    for (int i = 0; i < N; i++) {
        //float4 v = lbm.vel.at(i, 0, 0);
        //float vn = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
        float val = lbm.vel.at(i, 0, 0).w;
        printf("%f\n", val);
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
