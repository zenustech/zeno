#include <math.h>
#include <stdio.h>
#include <string.h>
#include <GL/glut.h>

#define NX 512
#define NY 128

float velx[NX * NY];
float vely[NX * NY];
float mass[NX * NY];
int mask[NX * NY];

float f_old[9][NX * NY];
float f_new[9][NX * NY];

int delx[9];
int dely[9];

int dirx[9] = {1,0,-1,0,1,-1,-1,1,0};
int diry[9] = {0,1,0,-1,1,1,-1,-1,0};
int inverse_index[9] = {2,3,0,1,6,7,4,5,8};
float weis[9] = {1.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0,4.0/9.0};

const float niu = 0.01f;
const float inv_tau = 1.f / (3.0f * niu + 0.5f);

float f_eq(int q, int p) {
    float m = mass[p];
    float u = velx[p];
    float v = vely[p];
    float eu = u * dirx[q] + v * diry[q];
    float uv = u * u + v * v;
    float term = 1.f + 3.f * eu + 4.5f * eu * eu - 1.5f * uv;
    float feq = weis[q] * m * term;
    return feq;
}

void initialize() {
    for (int y = 0; y < NY; y++) for (int x = 0; x < NX; x++) {
        int p = y * NX + x;
        mass[p] = 1.f;
        velx[p] = 0.1f;
        vely[p] = 0.f;
        if (x == 0 || y == 0 || x == NX - 1 || y == NY - 1) {
            mask[p] = 1;
        }
        int dx = x - NY * 3 / 4;
        int dy = y - NY / 2;
        if (dx * dx + dy * dy < NY * NY / 50) {
            //mask[p] = 0;
        }
    }
    for (int y = 0; y < NY; y++) for (int x = 0; x < NX; x++) {
        int p = y * NX + x;
        for (int q = 0; q < 9; q++) {
            int ps = ((y + dely[q]) % NY) * NX + (x + delx[q]) % NX;
            float feq = f_eq(q, p);
            f_new[q][ps] = feq;
            f_old[q][ps] = feq;
        }
    }
}

void compute_macro() {
#pragma omp parallel for
    for (int y = 0; y < NY; y++) for (int x = 0; x < NX; x++) {
        int p = y * NX + x;
        if (mask[p] > 0) continue;
        float m = 0.f;
        float u = 0.f;
        float v = 0.f;
        for (int q = 0; q < 9; q++) {
            int ps = ((y + dely[q]) % NY) * NX + (x + delx[q]) % NX;
            float f = f_new[q][ps];
            f_old[q][ps] = f;
            u += f * dirx[q];
            v += f * diry[q];
            m += f;
        }
        float fac = 1.f / fmaxf(m, 1e-6f);
        mass[p] = m;
        velx[p] = u * fac;
        vely[p] = v * fac;
    }
}

void do_collide() {
#pragma omp parallel for
    for (int y = 0; y < NY; y++) for (int x = 0; x < NX; x++) {
        int p = y * NX + x;
        if (mask[p] > 0) continue;
        for (int q = 0; q < 9; q++) {
            int ps = (y + dely[q]) % NY * NX + (x + delx[q]) % NX;
            f_new[q][ps] = f_old[q][ps] * (1.f - inv_tau) + f_eq(q, p) * inv_tau;
        }
    }
}

void do_stream() {
    memcpy(f_old, f_new, sizeof(f_new));
    for (int q = 0; q < 9; q++) {
        delx[q] = (delx[q] - dirx[q] + NX) % NX;
        dely[q] = (dely[q] - diry[q] + NY) % NY;
    }
}

void fixup_bc(int x, int y, int xm, int ym) {
    int p = y * NX + x;
    int pm = ym * NX + xm;
    for (int q = 0; q < 9; q++) {
        int ps = ((y + dely[q]) % NY) * NX + (x + delx[q]) % NX;
        int psm = ((ym + dely[q]) % NY) * NX + (xm + delx[q]) % NX;
        f_old[q][ps] = f_eq(q, p) - f_eq(q, pm) + f_old[q][psm];
    }
    mass[p] = mass[pm];
}

void bounce_bc(int x, int y, int q) {
    int ps = ((y + dely[q]) % NY) * NX + (x + delx[q]) % NX;
    int iq = inverse_index[q];
    int iqs = ((y + dely[iq]) % NY) * NX + (x + delx[iq]) % NX;
    f_old[q][ps] = f_old[iq][iqs];
}

void apply_bc() {
    for (int y = 0; y < NY; y++) for (int x = 0; x < 1; x++) {
        int p = y * NX + x;
        velx[p] = 0.1f;
        vely[p] = 0.0f;
        //fixup_bc(x, y, x + 1, y);
    }
    for (int y = 0; y < NY; y++) for (int x = NX - 2; x < NX - 1; x++) {
        int p = y * NX + x;
        int pm = y * NX + x + 1;
        velx[p] = velx[pm];
        vely[p] = vely[pm];
    }
    for (int y = 1; y < 2; y++) for (int x = 0; x < NX; x++) {
        int p = y * NX + x;
        velx[p] = 0.0f;
        vely[p] = 0.0f;
        bounce_bc(x, y, 1);
    }
    for (int y = NY - 2; y < NY - 1; y++) for (int x = 0; x < NX; x++) {
        int p = y * NX + x;
        velx[p] = 0.0f;
        vely[p] = 0.0f;
        bounce_bc(x, y, 3);
    }
#pragma omp parallel for
    for (int y = 0; y < NY; y++) for (int x = 0; x < NX; x++) {
        int p = y * NX + x;
        int dx = x - NY * 3 / 4;
        int dy = y - NY / 2;
        if (dx * dx + dy * dy < NY * NY / 50) {
            velx[p] = 0.0f;
            vely[p] = 0.0f;
            //fixup_bc(x, y, x + (dx > 0 ? 1 : -1), y + (dy > 0 ? 1 : -1));
        }
    }
}

float pixels[NX * NY];

void do_render() {
#pragma omp parallel for
    for (int y = 0; y < NY; y++) for (int x = 0; x < NX; x++) {
        int p = y * NX + x;
        float u = velx[p];
        float v = vely[p];
        float m = mass[p];
        (void)u, (void)v, (void)m;
        float c = 4.f * sqrtf(u * u + v * v);
        //float c = .5f * m;
        pixels[p] = c;
    }
    printf("%f\n", pixels[3 * NX + 3]);
}

void initFunc() {
    initialize();
}

void renderFunc() {
    for (int s = 0; s < 30; s++) {
        do_collide();
        do_stream();
        compute_macro();
        apply_bc();
    }
    do_render();
}

void displayFunc() {
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawPixels(NX, NY, GL_RED, GL_FLOAT, pixels);
    glFlush();
}

#define ITV 0
void timerFunc(int unused) {
    (void)unused;
    renderFunc();
    glutPostRedisplay();
    glutTimerFunc(ITV, timerFunc, 0);
}

void keyboardFunc(unsigned char key, int x, int y) {
    (void)x;
    (void)y;
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
