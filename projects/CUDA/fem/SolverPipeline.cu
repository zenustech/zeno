#include "../Utils.hpp"
#include "Solver.cuh"

namespace zeno {

void IPCSystem::newtonKrylov(zs::CudaExecutionPolicy &pol) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;

    /// optimizer
    for (int newtonIter = 0; newtonIter != PNCap; ++newtonIter) {
// check constraints
#if 0
        if (!BCsatisfied) {
            computeConstraints(cudaPol, "xn");
            auto cr = constraintResidual(cudaPol, true);
            if (areConstraintsSatisfied(cudaPol)) {
                fmt::print("satisfied cons res [{}] at newton iter [{}]\n", cr, newtonIter);
                // A.checkDBCStatus(cudaPol);
                // getchar();
                projectDBC = true;
                BCsatisfied = true;
            }
            fmt::print(fg(fmt::color::alice_blue), "newton iter {} cons residual: {}\n", newtonIter, cr);
        }
#endif
        // PRECOMPUTE
        // GRAD, HESS, P
        // ROTATE GRAD, APPLY CONSTRAINTS, PROJ GRADIENT
        // PREPARE P
        // CG SOLVE
        // ROTATE BACK
        // CHECK PN CONDITION
        // LINESEARCH
        // UPDATE RULE
    }
}

struct AdvanceIPCSystem : INode {
    void apply() override {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        auto A = get_input<IPCSystem>("ZSIPCSystem");

        auto cudaPol = zs::cuda_exec();

        int nSubsteps = get_input2<int>("num_substeps");
        auto dt = get_input2<float>("dt");

        A->reinitialize(cudaPol, dt);
        for (int subi = 0; subi != nSubsteps; ++subi) {
            A->advanceSubstep(cudaPol, (typename IPCSystem::T)1 / nSubsteps);

            int numFricSolve = s_enableFriction ? 2 : 1;
        for_fric:
            A->newtonKrylov(cudaPol);
            if (--numFricSolve > 0)
                goto for_fric;

            A->updateVelocities(cudaPol);
        }
        // update velocity and positions
        A->updatePositionsAndVelocities(cudaPol);

        set_output("ZSIPCSystem", A);
    }
};

ZENDEFNODE(AdvanceIPCSystem, {{
                                  "ZSIPCSystem",
                                  {"int", "num_substeps", "1"},
                                  {"float", "dt", "0.01"},
                              },
                              {"ZSIPCSystem"},
                              {},
                              {"FEM"}});
} // namespace zeno