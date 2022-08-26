#include "PBD.cuh"
#include "zensim/execution/ExecutionPolicy.hpp"

namespace zeno {

void PBDSystem::preSolve(zs::CudaExecutionPolicy &pol) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    pol(range(numDofs), [vtemp = proxy<space>({}, vtemp), extForce = extForce, dt = dt] ZS_LAMBDA(int i) mutable {
        auto x = vtemp.template pack<3>("x", i);
        auto v = vtemp.template pack<3>("v", i);
        vtemp.template tuple<3>("xpre", i) = x;
        v += extForce * dt; //extForce here is actually accel
        auto xpre = x;
        x += v * dt;
        // project
        if (x[1] < 0) {
            x = xpre;
            x[1] = 0;
        }
        vtemp.template tuple<3>("x", i) = x;
    });
}

void PBDSystem::solveEdge(zs::CudaExecutionPolicy &pol) {
    constexpr T edgeCompliance = 100;
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    T alpha = edgeCompliance / dt / dt;
    zeno::vec3f grads{0, 0, 0};
#if 0
    for (int i = 0; i < numEdges; i++) {
        int id0 = edge[i][0];
        int id1 = edge[i][1];

        grads = pos[id0] - pos[id1];
        float Len = sqrt(grads[0] * grads[0] + grads[1] * grads[1] + grads[2] * grads[2]);
        grads /= Len;
        float C = Len - restLen[i];
        float w = invMass[id0] + invMass[id1];
        float s = -C / (w + alpha);

        pos[id0] += grads * s * invMass[id0];
        pos[id1] += grads * (-s * invMass[id1]);
    }
#endif
}
void PBDSystem::solveVolume(zs::CudaExecutionPolicy &pol) {
    constexpr T volumeCompliance = 0;
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    float alphaVol = volumeCompliance / dt / dt;
    vec3 gradsVol[4] = {vec3(0, 0, 0), vec3(0, 0, 0), vec3(0, 0, 0), vec3(0, 0, 0)};

#if 0
    for (int i = 0; i < numTets; i++) {
        vec4i id = vec4i(-1, -1, -1, -1);

        for (int j = 0; j < 4; j++)
            id[j] = tet[i][j];

        gradsVol[0] = cross((pos[id[3]] - pos[id[1]]), (pos[id[2]] - pos[id[1]]));
        gradsVol[1] = cross((pos[id[2]] - pos[id[0]]), (pos[id[3]] - pos[id[0]]));
        gradsVol[2] = cross((pos[id[3]] - pos[id[0]]), (pos[id[1]] - pos[id[0]]));
        gradsVol[3] = cross((pos[id[1]] - pos[id[0]]), (pos[id[2]] - pos[id[0]]));

        float w = 0.0;
        for (int j = 0; j < 4; j++)
            w += invMass[id[j]] * (length(gradsVol[j])) * (length(gradsVol[j]));

        float vol = tetVolume(pos, tet, i);
        float C = (vol - restVol[i]) * 6.0;
        float s = -C / (w + alphaVol);

        for (int j = 0; j < 4; j++)
            pos[tet[i][j]] += gradsVol[j] * s * invMass[id[j]];
    }
#endif
}

void PBDSystem::postSolve(zs::CudaExecutionPolicy &pol) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    pol(range(numDofs), [vtemp = proxy<space>({}, vtemp), dt = dt] ZS_LAMBDA(int i) mutable {
        auto x = vtemp.template pack<3>("x", i);
        auto xpre = vtemp.template pack<3>("xpre", i);
        auto v = (x - xpre) / dt;
        vtemp.template tuple<3>("v", i) = v;
    });
}

struct StepPBDSystem : INode {
    void apply() override {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        auto A = get_input<PBDSystem>("ZSPBDSystem");
        auto dt = get_input2<float>("dt");

        auto cudaPol = cuda_exec();
        A->reinitialize(cudaPol, dt);
        for (int steps = 0; steps != A->solveIterCap; ++steps) {
            A->preSolve(cudaPol);
            A->solveEdge(cudaPol);
            A->solveVolume(cudaPol);
            A->postSolve(cudaPol);
        }

        set_output("ZSPBDSystem", A);
    }
};

ZENDEFNODE(StepPBDSystem, {{
                               "ZSPBDSystem",
                               {"float", "dt", "0.01"},
                           },
                           {"ZSPBDSystem"},
                           {},
                           {"PBD"}});

} // namespace zeno