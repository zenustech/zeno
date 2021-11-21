#include <iostream>
#include <cstring>
#include <cmath>
#include <chrono>
#include <tuple>
#include <vector>
#include <cassert>

using std::cout;
using std::endl;
#define show(x) (cout << #x "=" << (x) << endl)


#define N 512
#define B 8

float src[N][N][N];

int main(void)
{
    memset(src, 0, N * N * N * sizeof(float));

    auto t0 = std::chrono::steady_clock::now();

#if 0
    #pragma omp parallel for
    for (int i = B; i < N-B; i += B) for (int j = B; j < N-B; j += B) for (int k = B; k < N-B; k += B) {
        static_assert(B % 2 == 0);
        float buf[2*B][2*B][2*B];

        for (int di = 0; di < 2*B; di++) for (int dj = 0; dj < 2*B; dj++) for (int dk = 0; dk < 2*B; dk++) {
            buf[di][dj][dk] = src[i + di - B/2][j + dj + B/2][k + dk + B/2];
        }

        for (int t = 0; t < B/2; t++) {
            for (int di = t; di < 2*B-t; di++) for (int dj = t; dj < 2*B-t; dj++) for (int dk = t; dk < 2*B-t; dk++) {
                buf[di][dj][dk] = buf[di+1][dj][dk] + buf[di-1][dj][dk] + buf[di][dj+1][dk] + buf[di][dj-1][dk] + buf[di][dj][dk+1] + buf[di][dj][dk-1];
            }
        }

        for (int di = 0; di < B; di++) for (int dj = 0; dj < B; dj++) for (int dk = 0; dk < B; dk++) {
            src[i + di][j + dj][k + dk] = buf[di + B/2][dj + B/2][dk + B/2];
        }
    }
#else
    #pragma omp parallel for simd
    for (int i = 1; i < N-1; i++) for (int j = 1; j < N-1; j++) for (int k = 1; k < N-1; k++) {
        src[i][j][k] = src[i+1][j][k] + src[i-1][j][k] + src[i][j+1][k] + src[i][j-1][k] + src[i][j][k+1] + src[i][j][k-1];
    }
#endif

    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    cout << ms << " ms" << endl;

    return 0;
}
