#include <memory>
#include <mutex>
#include <vector>

// zeno basics
#include <zeno/DictObject.h>
#include <zeno/ListObject.h>
#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/logger.h>
#include <zeno/utils/UserData.h>
#include <zeno/utils/fileio.h>
#include <zeno/zeno.h>

// zpc
#include "zensim/container/Bht.hpp"
#include "zensim/math/matrix/SparseMatrix.hpp"
#include "zensim/math/matrix/SparseMatrixOperations.hpp"
#if 1
#include "zensim/graph/ConnectedComponents.hpp"
#endif
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/zpc_tpls/fmt/color.h"

#include "RigidTest.h"

// convex decomposition
#include <VHACD/inc/VHACD.h>
#include <hacdHACD.h>
#include <hacdICHull.h>
#include <hacdVector.h>

namespace zeno {

struct BulletGlueRigidBodies : zeno::INode {
    virtual void apply() override {
        using namespace zs;
        constexpr auto space = execspace_e::openmp;
        auto pol = omp_exec();

        // std::vector<std::shared_ptr<BulletObject>>
        auto rbs = get_input<ListObject>("rbList")->get<BulletObject>();
        const auto nrbs = rbs.size();

        auto glueList = get_input<ListObject>("glueListVec2i")->getLiterial<vec2i>();
        auto ncons = glueList.size();

        std::vector<int> is(ncons), js(ncons);
        pol(zip(is, js, glueList), [](auto &i, auto &j, const auto &ij) {
            i = ij[0];
            j = ij[1];
        });
        SparseMatrix<int, true> spmat{(int)nrbs, (int)nrbs};
        spmat.build(pol, (int)nrbs, (int)nrbs, range(is), range(js), true_c);

        std::vector<int> fas(nrbs);
        union_find(pol, spmat, range(fas));
        bht<int, 1, int> tab{spmat.get_allocator(), nrbs};
        tab.reset(pol, true);
        pol(range(nrbs), [&fas, tab = proxy<space>(tab)](int vi) mutable {
            auto fa = fas[vi];
            while (fa != fas[fa])
                fa = fas[fa];
            fas[vi] = fa;
            tab.insert(fa);
        });

        auto ncompounds = tab.size();
        fmt::print("{} rigid bodies, {} compounds.\n", nrbs, ncompounds);

        // mass
        std::vector<float> cpdMasses(ncompounds);
        pol(enumerate(rbs), [&cpdMasses, &fas, tab = proxy<space>(tab)](int rbi, const auto &rb) {
            auto fa = fas[rbi];
            auto compId = tab.query(fa);
            atomic_add(exec_omp, &cpdMasses[compId], rb->body->getMass());
        });
        std::vector<float> mass(1, 0.f);
        reduce(pol, std::begin(cpdMasses), std::end(cpdMasses), mass.begin(), 0.f);

        fmt::print("total mass: {}\n", mass[0]);

        // assemble shapes
        std::vector<std::mutex> comLocks(ncompounds);
        std::vector<std::unique_ptr<btCompoundShape>> cpds(ncompounds);
        pol(range(nrbs), [&, tab = proxy<space>(tab)](int rbi) mutable {
            std::unique_ptr<btRigidBody> &bodyPtr = rbs[rbi]->body;
            auto fa = fas[rbi];
            auto compId = tab.query(fa);
            {
                std::lock_guard<std::mutex> guard(comLocks[compId]);
                auto &cpdPtr = cpds[compId];
                if (cpdPtr.get() == nullptr)
                    cpdPtr = std::make_unique<btCompoundShape>();
                cpdPtr->addChildShape(bodyPtr->getCenterOfMassTransform(), bodyPtr->getCollisionShape());
            }
        });

        // compound rigidbodies
        auto rblist = std::make_shared<ListObject>();
        rblist->arr.resize(ncompounds);
        auto shapelist = std::make_shared<ListObject>();
        shapelist->arr.resize(ncompounds);
        pol(zip(cpdMasses, cpds, rblist->arr, shapelist->arr),
            [&](auto mass, auto &shape, auto &cpdBody, auto &cpdShape) {
                btTransform trans;
                trans.setIdentity();
#if 1
                auto tmp = std::make_shared<BulletCollisionShape>(std::move(shape));
                cpdShape = tmp;
                cpdBody = std::make_shared<BulletObject>(mass, trans, tmp);
#else
            btVector3 localInertia;
            shape->calculateLocalInertia(mass, localInertia);

            auto myMotionState = std::make_unique<btDefaultMotionState>(trans);
            btRigidBody::btRigidBodyConstructionInfo ci(mass, myMotionState.get(), shape.get(), localInertia);
            cpdBody = std::make_unique<btRigidBody>(ci);
#endif
            });

        rblist->userData().set("compoundShapes", shapelist);

        set_output("compoundList", rblist);
    }
};

ZENDEFNODE(BulletGlueRigidBodies, {
                                      {
                                          "rbList",
                                          "glueListVec2i",
                                      },
                                      {
                                          "compoundList",
                                      },
                                      {},
                                      {"Bullet"},
                                  });

} // namespace zeno