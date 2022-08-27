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

    for (auto &&prim : prims) {
        auto &eles = prim.getEles();
        pol(range(eles.size()), [vtemp = proxy<space>({}, vtemp), eles = proxy<space>({}, eles), vOffset = prim.vOffset,
                                 dt = dt, alphaVol] ZS_LAMBDA(int ei) mutable {
            auto id = eles.template pack<4>("inds", ei).template reinterpret_bits<int>() + vOffset;
            vec3 gradsVol[4];
            T ms[4];
            auto x0 = vtemp.template pack<3>("x", id[0]);
            auto x1 = vtemp.template pack<3>("x", id[1]);
            auto x2 = vtemp.template pack<3>("x", id[2]);
            auto x3 = vtemp.template pack<3>("x", id[3]);
            ms[0] = 1 / vtemp("m", id[0]);
            ms[1] = 1 / vtemp("m", id[1]);
            ms[2] = 1 / vtemp("m", id[2]);
            ms[3] = 1 / vtemp("m", id[3]);
            gradsVol[0] = (x3 - x1).cross(x2 - x1);
            gradsVol[1] = (x2 - x0).cross(x3 - x0);
            gradsVol[2] = (x3 - x0).cross(x1 - x0);
            gradsVol[3] = (x1 - x0).cross(x2 - x0);

            T w = 0;
            for (int j = 0; j != 4; ++j)
                w += zs::sqr(gradsVol[j].length()) * ms[j];

            T vol = zs::abs((x1 - x0).cross(x2 - x0).dot(x3 - x0)) / 6;
            T C = (vol - eles("rv", ei)) * 6;
            T s = -C / (w + alphaVol);

            for (int j = 0; j != 4; ++j)
                for (int d = 0; d != 3; ++d)
                    atomic_add(exec_cuda, &vtemp("x", d, id[j]), gradsVol[j][d] * s * ms[j]);
        });
    }
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

void PBDSystem::writebackPositionsAndVelocities(zs::CudaExecutionPolicy &pol) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    for (auto &primHandle : prims) {
        auto &verts = primHandle.getVerts();
        // update velocity and positions
        pol(zs::range(verts.size()), [vtemp = proxy<space>({}, vtemp), verts = proxy<space>({}, verts), dt = dt,
                                      vOffset = primHandle.vOffset] __device__(int vi) mutable {
            verts.tuple<3>("x", vi) = vtemp.pack<3>("x", vOffset + vi);
            verts.tuple<3>("v", vi) = vtemp.pack<3>("v", vOffset + vi);
        });
    }
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
        A->writebackPositionsAndVelocities(cudaPol);

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