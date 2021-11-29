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

std::vector<size_t> neigh_index;
std::vector<std::pair<size_t, size_t>> neigh_list;

void prologue()
{
    zbb::auto_profiler _("prologue");

    neigh_index.clear();
    neigh_list.clear();

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

    struct ivec2_hasher {
        constexpr int operator()(glm::ivec2 const &u) const {
            return (73856093 * u.x) ^ (19349663 * u.y);// ^ (83492791 * u.z);
        }
    };

    tbb::concurrent_unordered_multimap<glm::ivec2, size_t, ivec2_hasher> expand_lut;

    {
        tbb::concurrent_unordered_multimap<glm::ivec2, size_t, ivec2_hasher> cell_lut;

        {
            zbb::auto_profiler _("prologue1");
            tbb::parallel_for
            ( size_t{0}, pos.size()
            , [&] (size_t i) {
                glm::ivec2 cell(pos[i] * inv_dx);
                cell_lut.emplace(cell, i);
            });
        }

        {
            zbb::auto_profiler _("prologue1.5");
            tbb::parallel_for_each
            ( cell_lut.begin(), cell_lut.end()
            , [&] (std::pair<glm::ivec2, size_t> const &cell) {
                for (int cy = -1; cy <= 1; cy++) for (int cx = -1; cx <= 1; cx++) {
                    auto newcell = cell.first + glm::ivec2(cx, cy);
                    expand_lut.emplace(newcell, cell.second);
                }
            });
        }
    }

    tbb::concurrent_vector<size_t> nei_index;
    std::vector<std::pair<tbb::concurrent_vector<size_t>::iterator, size_t>> nei_list(pos.size());

    {
        zbb::auto_profiler _("prologue2");
        tbb::parallel_for
        ( size_t{0}, pos.size()
        , [&] (size_t i) {
            glm::ivec2 cell(pos[i] * inv_dx);

            std::vector<size_t> res;
            res.reserve(16);
            auto [b, e] = expand_lut.equal_range(cell);
            for (auto it = b; it != e; ++it) {
                size_t j = it->second;
                if (i && j || distance(pos[i], pos[j]) < neigh_radius)
                    res.push_back(i);
            }

            size_t nbucket = res.size();
            auto it = nei_index.grow_by(nbucket);
            std::copy(res.begin(), res.end(), it);
            nei_list[i] = std::make_pair(std::move(it), nbucket);
        });
    }

    neigh_list.resize(pos.size());

    {
        zbb::auto_profiler _("prologue3");
        neigh_index.resize(nei_index.size());
        tbb::parallel_for
        ( size_t{0}, pos.size()
        , [&] (size_t i) {
            neigh_index[i] = nei_index[i];
        });
    }

    {
        zbb::auto_profiler _("prologue4");
        tbb::parallel_for
        ( size_t{0}, pos.size()
        , [&] (size_t i) {
            size_t index = nei_list[i].first - nei_index.begin();
            neigh_list[i] = std::make_pair(index, nei_list[i].second);
        });
    }
}

void substep()
{
    zbb::auto_profiler _("substep");

    std::vector<float> lam(pos.size());

    tbb::parallel_for
    ( size_t{0}, pos.size()
    , [&] (size_t i) {
        glm::vec2 grad_i(0.f);
        float sum_gradient_sqr = 0.f;
        float density_constraint = -1.f;

        auto [beg, num] = neigh_list[i];
        for (size_t idx = beg; idx < num; idx++) {
            size_t j = neigh_index[idx];
            auto pos_ji = pos[i] - pos[j];
            auto grad_j = spiky_gradient(pos_ji, dx);
            grad_i += grad_j;
            sum_gradient_sqr += dot(grad_j, grad_j);
            density_constraint += poly6_value(length(pos_ji), dx);
        }

        sum_gradient_sqr += dot(grad_i, grad_i);
        lam[i] = -density_constraint / (sum_gradient_sqr + lambda_epsilon);
    });

    std::vector<glm::vec2> pos_new(pos.size());

    tbb::parallel_for
    ( size_t{0}, pos.size()
    , [&] (size_t i) {
        glm::vec2 pos_delta_i(0.f);
        auto [beg, num] = neigh_list[i];
        for (size_t idx = beg; idx < num; idx++) {
            size_t j = neigh_index[idx];
            auto lam_j = lam[j];
            auto pos_ji = pos[j] - pos[i];
            auto scorr_ij = compute_scorr(pos_ji, dx);
            auto grad_j = spiky_gradient(pos_ji, dx);
            pos_delta_i += (lam[i] + lam[j] + scorr_ij) * grad_j;
        }
        pos_new[i] = pos[i] + pos_delta_i;
    });

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
