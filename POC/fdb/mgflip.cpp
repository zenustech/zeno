#include <cstring>
#include <cstdio>
#include <utility>
#include "vec.h"
#include "timer.h"
#include "ndgrid.h"
#include "vdbio.h"

using namespace zinc;

#define range(x, x0, x1) (uint32_t x = x0; x < x1; x++)

template <class T, size_t N>
void zeroinit(Grid<T, N> &v) {
    ZINC_PRETTY_TIMER;
    std::memset(v.m_data, 0, N * N * N * sizeof(T));
}

template <size_t M, class T, size_t N>
void smooth(Grid<T, N> &v, Grid<T, N> const &f) {
    ZINC_PRETTY_TIMER;
    for range(phase, 0, M) {
#pragma omp parallel for
        for range(z, 0, N) {
            for range(y, 0, N) {
                for range(x, 0, N) {
                    if ((x + y + z) % 2 != phase % 2)
                        continue;
                    v(x, y, z) = (
                          f(x, y, z)
                        + v(x+1, y, z)
                        + v(x, y+1, z)
                        + v(x, y, z+1)
                        + v(x-1, y, z)
                        + v(x, y-1, z)
                        + v(x, y, z-1)
                        ) / 6;
                }
            }
        }
    }
}

template <class T, size_t N>
void residual(Grid<T, N> &r, Grid<T, N> const &v, Grid<T, N> const &f) {
    ZINC_PRETTY_TIMER;
#pragma omp parallel for
    for range(z, 0, N) {
        for range(y, 0, N) {
            for range(x, 0, N) {
                r(x, y, z) = (
                      f(x, y, z)
                    + v(x+1, y, z)
                    + v(x, y+1, z)
                    + v(x, y, z+1)
                    + v(x-1, y, z)
                    + v(x, y-1, z)
                    + v(x, y, z-1)
                    - v(x, y, z) * 6
                    );
            }
        }
    }
}

template <class T, size_t N>
[[nodiscard]] T loss(Grid<T, N> const &v, Grid<T, N> const &f) {
    T res = 0;
    for range(z, 0, N) {
        for range(y, 0, N) {
            for range(x, 0, N) {
                T val = f(x, y, z)
                    + v(x+1, y, z)
                    + v(x, y+1, z)
                    + v(x, y, z+1)
                    + v(x-1, y, z)
                    + v(x, y-1, z)
                    + v(x, y, z-1)
                    - v(x, y, z) * 6;
                res = fmaxf(res, fabsf(val));
                //res += val * val;
            }
        }
    }
    return res;
}

template <class T, size_t N>
void restrict(Grid<T, N/2> &w, Grid<T, N> const &v) {
    ZINC_PRETTY_TIMER;
#pragma omp parallel for
    for range(z, 0, N/2) {
        for range(y, 0, N/2) {
            for range(x, 0, N/2) {
                w(x, y, z) = (
                      v(x*2, y*2, z*2)
                    + v(x*2+1, y*2, z*2)
                    + v(x*2, y*2+1, z*2)
                    + v(x*2+1, y*2+1, z*2)
                    + v(x*2, y*2, z*2+1)
                    + v(x*2+1, y*2, z*2+1)
                    + v(x*2, y*2+1, z*2+1)
                    + v(x*2+1, y*2+1, z*2+1)
                    );
            }
        }
    }
}

template <class T, size_t N>
void prolongate(Grid<T, N*2> &w, Grid<T, N> const &v) {
    ZINC_PRETTY_TIMER;
#pragma omp parallel for
    for range(z, 0, N*2) {
        for range(y, 0, N*2) {
            for range(x, 0, N*2) {
                w(x, y, z) += v(x/2, y/2, z/2) * 0.5f;
            }
        }
    }
}

template <size_t M, size_t N0 = M, class T, size_t N>
void vcycle(Grid<T, N> &v, Grid<T, N> const &f) {
    if constexpr (N <= M) {
        smooth<M>(v, f);

    } else {
        smooth<M>(v, f);

        Grid<T, N> r;
        residual(r, v, f);

        Grid<T, N/2> r2;
        restrict(r2, r);

        Grid<T, N/2> e2;
        zeroinit(e2);
        vcycle<M>(e2, r2);

        prolongate(v, e2);
        smooth<M>(v, f);
    }
}


template <class T>
struct Points {
    std::vector<T> m_data;

    void resize(size_t n) {
        return m_data.resize(n);
    }

    size_t size() const {
        return m_data.size();
    }

    [[nodiscard]] auto &operator()(uint32_t i) {
        return m_data[i];
    }

    [[nodiscard]] auto const &operator()(uint32_t i) const {
        return m_data[i];
    }
};

template <class T, size_t N>
T bilerp(Grid<T, N> const &f, vec3f const &p) {
    vec3i i(floor(p));
    vec3f k = p - i;

    auto c000 = f(i[0], i[1], i[2]);
    auto c100 = f(i[0]+1, i[1], i[2]);
    auto c010 = f(i[0], i[1]+1, i[2]);
    auto c110 = f(i[0]+1, i[1]+1, i[2]);
    auto c001 = f(i[0], i[1], i[2]+1);
    auto c101 = f(i[0]+1, i[1], i[2]+1);
    auto c011 = f(i[0], i[1]+1, i[2]+1);
    auto c111 = f(i[0]+1, i[1]+1, i[2]+1);

    return mix(
            mix(
                mix(c000, c100, k[0]),
                mix(c010, c110, k[0]),
                k[1]),
            mix(
                mix(c001, c101, k[0]),
                mix(c011, c111, k[0]),
                k[1]),
            k[2]
            );
}

template <class T, size_t N>
void advect(Grid<T, N> &dst, Grid<T, N> const &src, Grid<vec3f, N> const &vel) {
    ZINC_PRETTY_TIMER;
#pragma omp parallel for
    for range(z, 0, N) {
        for range(y, 0, N) {
            for range(x, 0, N) {
                vec3f p(x, y, z);
                p -= bilerp(vel, p);
                dst(x, y, z) = bilerp(src, p);
            }
        }
    }
}

struct Domain {
    inline static constexpr size_t N = 64;

    Grid<float, N> pressure;
    Grid<float, N> neg_vel_div;
    Grid<vec3f, N> velocity;
    Grid<vec3f, N> new_velocity;
    Grid<vec3f, N> color;
    Grid<vec3f, N> new_color;

    Domain() {
        zeroinit(pressure);
        zeroinit(velocity);
        zeroinit(color);
        for range(z, 0, N) {
            for range(y, 0, N) {
                for range(x, 0, N) {
                    if (x < N/2 && x > N/4 && y < N/2 && y > N/4 && z < N/2 && z > N/4)
                        color(x, y, z)[0] = 0.5f;
                    if (x > N/2 && x < N*3/4 && y > N/2 && y < N*3/4 && z > N/2 && z < N*3/4)
                        color(x, y, z)[1] = 0.5f;
                }
            }
        }
        for range(z, 0, N) {
            for range(y, 0, N) {
                for range(x, 0, N) {
                    if (x < N/2 && x > N/4 && y < N/2 && y > N/4 && z < N/2 && z > N/4)
                        velocity(x, y, z)[0] = 0.5f;
                    if (x > N/2 && x < N*3/4 && y > N/2 && y < N*3/4 && z > N/2 && z < N*3/4)
                        velocity(x, y, z)[0] = -0.5f;
                }
            }
        }
    }

    void calc_neg_vel_div() {
        ZINC_PRETTY_TIMER;
#pragma omp parallel for
        for range(z, 0, N) {
            for range(y, 0, N) {
                for range(x, 0, N) {
                    neg_vel_div(x, y, z) = 0.25f * (
                          velocity(x, y, z)[0] - velocity(x-1, y, z)[0]
                        + velocity(x, y, z)[1] - velocity(x, y-1, z)[1]
                        + velocity(x, y, z)[2] - velocity(x, y, z-1)[2]
                        );
                }
            }
        }
    }

    void solve_possion_eqn() {
        ZINC_PRETTY_TIMER;
        vcycle<16>(pressure, neg_vel_div);
        //smooth<N>(pressure, neg_vel_div);
        printf("loss: %f\n", loss(pressure, neg_vel_div));
    }

    void subtract_gradient() {
        ZINC_PRETTY_TIMER;
#pragma omp parallel for
        for range(z, 0, N) {
            for range(y, 0, N) {
                for range(x, 0, N) {
                    velocity(x, y, z)[0] -= pressure(x, y, z) - pressure(x+1, y, z);
                    velocity(x, y, z)[1] -= pressure(x, y, z) - pressure(x, y+1, z);
                    velocity(x, y, z)[2] -= pressure(x, y, z) - pressure(x, y, z+1);
                }
            }
        }
    }

    void advect_quantities() {
        advect(new_color, color, velocity);
        advect(new_velocity, velocity, velocity);
        color.swap(new_color);
        velocity.swap(new_velocity);
    }

    void dump_file(int frame) {
        char path[1024];
        sprintf(path, "/tmp/%06d.vdb", frame);
        writevdb(path,
                [&] (vec3I p) -> vec3f {
                    return color(p);
                }, {N, N, N});
    }

    void substep() {
        calc_neg_vel_div();
        solve_possion_eqn();
        subtract_gradient();
        advect_quantities();
    }
};

int main() {
    Domain dom;

    for (int i = 0; i < 250; i++) {
        dom.dump_file(i);
        dom.substep();
    }

    return 0;
}
