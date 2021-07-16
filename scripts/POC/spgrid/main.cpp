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

#define N 256
float pixels[N * N];
SPMasked<SPFloatGrid<N>> dens;

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
    for (int z = 0; z < N; z++) {
        for (int y = 0; y < N; y++) {
            for (int x = 0; x < N; x++) {
                dens.get(x, y, z);
            }
        }
    }
}

void displayFunc() {
    glClear(GL_COLOR_BUFFER_BIT);

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

    glDrawPixels(N, N, GL_RED, GL_FLOAT, pixels);
    glFlush();
}

void idleFunc() {
    glutPostRedisplay();
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
    //glutIdleFunc(idleFunc);
    glutDisplayFunc(displayFunc);
    glutKeyboardFunc(keyboardFunc);
    initFunc();
    glutMainLoop();
}
