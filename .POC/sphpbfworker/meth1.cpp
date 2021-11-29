#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/concurrent_vector.h>
#include <tbb/concurrent_unordered_map.h>
#include <zeno/zbb/auto_profiler.h>
#include <glm/vec2.hpp>
#include <glm/glm.hpp>
#include <numbers>
#include <tuple>

static constexpr float poly6_factor = 315.f / 64.f / std::numbers::pi;
static constexpr float spiky_grad_factor = -45.f / std::numbers::pi;
static constexpr float corr_deltaQ_coeff = 0.3f;
static constexpr float corrK = 0.001f;

float poly6_value(float s, float h)
{
    if (0 < s && s < h) {
        float x = (h * h - s * s) / (h * h * h);
        return poly6_factor * x * x * x;
    } else {
        return 0;
    }
}

glm::vec2 spiky_gradient(glm::vec2 const &r, float h)
{
    float r_len = length(r);
    if (0 < r_len && r_len < h) {
        float x = (h - r_len) / (h * h * h);
        float g_factor = spiky_grad_factor * x * x;
        return r * g_factor / r_len;
    } else {
        return {0, 0};
    }
}

float compute_scorr(glm::vec2 const &d, float h)
{
    float x = poly6_value(length(d), h);
    x /= poly6_value(corr_deltaQ_coeff * h, h);
    x = x * x;
    x = x * x;
    return -corrK * x;
}

std::vector<glm::vec2> pos;
std::vector<glm::vec2> vel;

static constexpr float dx = 0.01f;
static constexpr float dt = 0.01f;
static constexpr float inv_dx = 1.f / dx;
static constexpr float neigh_radius = 0.86f * dx;
static constexpr float lambda_epsilon = 100.f;

struct ivec2_hasher {
    constexpr int operator()(glm::ivec2 const &u) const {
        return (73856093 * u.x) ^ (19349663 * u.y);// ^ (83492791 * u.z);
    }
};

tbb::concurrent_unordered_multimap<glm::ivec2, size_t, ivec2_hasher> space_lut;

void prologue()
{
    zbb::auto_profiler _("prologue");

    space_lut.clear();

    {
        zbb::auto_profiler _("prologue0");
        tbb::parallel_for
        ( size_t{0}, pos.size()
        , [&] (size_t i) {
            auto pos_i = pos[i];
            auto vel_i = vel[i];
            pos[i] = pos_i + vel_i * dt;
            vel[i] = pos_i;
        });
    }

    {
        zbb::auto_profiler _("prologue1");
        tbb::parallel_for
        ( size_t{0}, pos.size()
        , [&] (size_t i) {
            glm::ivec2 cell(pos[i] * inv_dx);
            space_lut.emplace(cell, i);
        });
    }
}

void substep()
{
    zbb::auto_profiler _("substep");

    std::vector<float> lam(pos.size());

    {
        zbb::auto_profiler _("substep0");
        tbb::parallel_for
        ( size_t{0}, pos.size()
        , [&] (size_t i) {
            glm::vec2 grad_i(0.f);
            float sum_gradient_sqr = 0.f;
            float density_constraint = -1.f;

            glm::ivec2 cell(pos[i] * inv_dx);
            auto [beg, end] = space_lut.equal_range(cell);
            for (auto it = beg; it != end; ++it) {
                size_t j = it->second;
                auto pos_ji = pos[j] - pos[i];
                if (dot(pos_ji, pos_ji) > neigh_radius * neigh_radius)
                    continue;

                auto grad_j = spiky_gradient(pos_ji, dx);
                grad_i += grad_j;
                sum_gradient_sqr += dot(grad_j, grad_j);
                density_constraint += poly6_value(length(pos_ji), dx);
            }

            sum_gradient_sqr += dot(grad_i, grad_i);
            lam[i] = -density_constraint / (sum_gradient_sqr + lambda_epsilon);
        });
    }

    std::vector<glm::vec2> pos_new(pos.size());

    {
        zbb::auto_profiler _("substep1");
        tbb::parallel_for
        ( size_t{0}, pos.size()
        , [&] (size_t i) {
            glm::vec2 pos_delta_i(0.f);

            glm::ivec2 cell(pos[i] * inv_dx);
            auto [beg, end] = space_lut.equal_range(cell);
            for (auto it = beg; it != end; ++it) {
                size_t j = it->second;
                auto pos_ji = pos[j] - pos[i];
                if (dot(pos_ji, pos_ji) > neigh_radius * neigh_radius)
                    continue;

                auto scorr_ij = compute_scorr(pos_ji, dx);
                auto grad_j = spiky_gradient(pos_ji, dx);
                pos_delta_i += (lam[i] + lam[j] + scorr_ij) * grad_j;
            }
            pos_new[i] = pos[i] + pos_delta_i;
        });
    }

    pos = std::move(pos_new);
}

void epilogue()
{
    zbb::auto_profiler _("epilogue");

    tbb::parallel_for
    ( size_t{0}, pos.size()
    , [&] (size_t i) {
        vel[i] = (pos[i] - vel[i]) / dt;
    });
}

void pbfstep()
{
    zbb::auto_profiler _("total");
    prologue();
    for (int i = 0; i < 5; i++) {
        substep();
    }
    epilogue();
}

int main()
{
    constexpr size_t N = 1024*4;
    for (int i = 0; i < N; i++) {
        pos.emplace_back(drand48(), drand48());
        vel.emplace_back(0.f, 0.f);
    }

    pbfstep();

    return 0;
}
