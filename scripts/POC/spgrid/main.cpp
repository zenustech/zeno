#include <type_traits>
#include <iostream>
#include <cstring>
#include <cmath>
#include <chrono>
#include <tuple>
#include <vector>
#include <array>
#include <cassert>
#include <omp.h>
#include "common_utils.h"
#include "iterator_utils.h"
#include <GL/gl.h>
#include <GL/glut.h>
#include "SPGrid.h"

using namespace bate::spgrid;

#define ITV 200
#define N 256
float pixels[N * N];

SPFloat16Grid<N> f_old;
SPFloat16Grid<N> f_new;
SPFloatGrid<N> rho;
SPFloat4Grid<N> vel;

const float niu = 0.005f;
const float tau = 3.f * niu + 0.5f;
const float inv_tau = 1.f / tau;

std::array<int, 3> directions[] = {{0,0,0},{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1},{1,1,1},{-1,-1,-1},{1,1,-1},{-1,-1,1},{1,-1,1},{-1,1,-1},{-1,1,1},{1,-1,-1}};
float weights[] = {2.f/9.f, 1.f/9.f, 1.f/9.f, 1.f/9.f, 1.f/9.f, 1.f/9.f, 1.f/9.f,1.f/72.f, 1.f/72.f, 1.f/72.f, 1.f/72.f, 1.f/72.f, 1.f/72.f, 1.f/72.f, 1.f/72.f};

void initFunc() {
    #pragma omp parallel for
    for (int z = 0; z < N; z++) {
        for (int y = 0; y < N; y++) {
            for (int x = 0; x < N; x++) {
                float fx = float(x) / N * 2 - 1;
                float fy = float(y) / N * 2 - 1;
                float fz = float(z) / N * 2 - 1;
                float val = fx * fx + fy * fy + fz * fz;
                if (val < 1.0f)
                    dens.set(x, y, z, val);
            }
        }
    }
}

void stepFunc() {
    #pragma omp parallel for
    for (int zz = 0; zz < N; zz += dens.MaskScale) {
        for (int yy = 0; yy < N; yy += dens.MaskScale) {
            for (int xx = 0; xx < N; xx += dens.MaskScale) {
                if (!dens.is_active(xx, yy, zz))
                    continue;
                for (int z = zz; z < zz + dens.MaskScale; z++) {
                    for (int y = yy; y < yy + dens.MaskScale; y++) {
                        for (int x = xx; x < xx + dens.MaskScale; x++) {
                            auto ax = dens.direct_get(x + 1, y, z);
                            auto ay = dens.direct_get(x, y + 1, z);
                            auto az = dens.direct_get(x, y, z + 1);
                            auto bx = dens.direct_get(x - 1, y, z);
                            auto by = dens.direct_get(x, y - 1, z);
                            auto bz = dens.direct_get(x, y, z - 1);
                            auto co = dens.direct_get(x, y, z);
                            auto val = ax + ay + az + bx + by + bz;
                            val *= 1 / 6.f;
                            dens_tmp.set(x, y, z, val);
                        }
                    }
                }
            }
        }
    }
    #pragma omp parallel for
    for (int zz = 0; zz < N; zz += dens.MaskScale) {
        for (int yy = 0; yy < N; yy += dens.MaskScale) {
            for (int xx = 0; xx < N; xx += dens.MaskScale) {
                if (!dens.is_active(xx, yy, zz))
                    continue;
                for (int z = zz; z < zz + dens.MaskScale; z++) {
                    for (int y = yy; y < yy + dens.MaskScale; y++) {
                        for (int x = xx; x < xx + dens.MaskScale; x++) {
                            auto val = dens_tmp.get(x, y, z);
                            dens.direct_set(x, y, z, val);
                        }
                    }
                }
            }
        }
    }
}

void renderFunc() {
    int z = N / 2;
    #pragma omp parallel for
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            float acc = dens.get(x, y, z);
            //int v = *(int *)dens.pointer(0, x, y, z);
            //if (v) printf("%x\n", v);
            pixels[y * N + x] = acc;
        }
    }
}

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
