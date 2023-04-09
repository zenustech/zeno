#include <memory>
#include <mutex>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/transform.hpp>

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
        std::vector<int> isRbCompound(nrbs);
        union_find(pol, spmat, range(fas));
        bht<int, 1, int> tab{spmat.get_allocator(), nrbs};
        tab.reset(pol, true);
        pol(range(nrbs), [&fas, &isRbCompound, tab = proxy<space>(tab)](int vi) mutable {
            auto fa = fas[vi];
            while (fa != fas[fa])
                fa = fas[fa];
            fas[vi] = fa;
            if (tab.insert(fa) < 0) // already inserted
                isRbCompound[vi] = true;
        });

        auto ncompounds = tab.size();
        /// @note output BulletObject list
        auto rblist = std::make_shared<ListObject>();
#if 1
        rblist->arr.resize(ncompounds);
#else
        rblist->arr.resize(nrbs);
#endif
        // determine compound or not
        // pass on rbs that are does not belong in any compound
        std::vector<int> isCompound(ncompounds);
        pol(range(nrbs),
            [&isCompound, &isRbCompound, &fas, &rbs, tab = proxy<space>(tab), &rblist = rblist->arr](int rbi) mutable {
                auto isRbCpd = isRbCompound[rbi];
                auto fa = fas[rbi];
                auto compId = tab.query(fa);
#if 1
                if (isRbCpd)
                    isCompound[compId] = 1;
                else
                    rblist[compId] = rbs[rbi];
#else
                    rblist[rbi] = rbs[rbi];
#endif
            });
        fmt::print("{} rigid bodies, {} groups (incl compounds).\n", nrbs, ncompounds);

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

        // prep PrimList for each compound
        std::vector<std::shared_ptr<ListObject>> primLists(ncompounds);
        std::vector<std::unique_ptr<btCompoundShape>> btCpdShapes(ncompounds);
        pol(zip(primLists, btCpdShapes), [](auto &primPtr, auto &cpdShape) {
            primPtr = std::make_shared<ListObject>();
            cpdShape = std::make_unique<btCompoundShape>();
        });

        // assemble shapes
        std::vector<std::mutex> comLocks(ncompounds);
        pol(range(nrbs), [&, tab = proxy<space>(tab)](int rbi) mutable {
            std::unique_ptr<btRigidBody> &bodyPtr = rbs[rbi]->body;
            auto fa = fas[rbi];
            auto compId = tab.query(fa);
            if (isCompound[compId]) {
                std::lock_guard<std::mutex> guard(comLocks[compId]);
                auto &cpdPtr = btCpdShapes[compId];
                auto &primList = primLists[compId];
                btTransform trans;
                if (bodyPtr && bodyPtr->getMotionState()) {
                    bodyPtr->getMotionState()->getWorldTransform(trans);
                } else {
                    trans = static_cast<btCollisionObject *>(bodyPtr.get())->getWorldTransform();
                }
                cpdPtr->addChildShape(trans, bodyPtr->getCollisionShape());
#if 0
                auto correspondingPrim = rbs[rbi]->userData().get<PrimitiveObject>("prim");
                primList->arr.push_back(correspondingPrim);
#else
                primList->arr.push_back(rbs[rbi]->userData().get("prim"));
#endif
            }
        });

        // assemble true compound shapes/rigidbodies
        auto shapelist = std::make_shared<ListObject>();
        shapelist->arr.resize(ncompounds);
        pol(zip(isCompound, cpdMasses, btCpdShapes, primLists, rblist->arr, shapelist->arr),
            [&](bool isCpd, auto mass, auto &btShape, auto primList, auto &cpdBody, auto &cpdShape) {
                if (isCpd) {
                    btTransform trans;
                    trans.setIdentity();
#if 1
                    auto tmp = std::make_shared<BulletCollisionShape>(std::move(btShape));
                    cpdShape = tmp;
                    // list of PrimitiveObject, corresponding with each CompoundShape children
                    tmp->userData().set("prim", primList);
                    cpdBody = std::make_shared<BulletObject>(mass, trans, tmp);
#else
            btVector3 localInertia;
            shape->calculateLocalInertia(mass, localInertia);

            auto myMotionState = std::make_unique<btDefaultMotionState>(trans);
            btRigidBody::btRigidBodyConstructionInfo ci(mass, myMotionState.get(), shape.get(), localInertia);
            cpdBody = std::make_unique<btRigidBody>(ci);
#endif
                }
            });

        rblist->userData().set("compoundShapes", shapelist);

        set_output("compoundList", rblist);
        // set_output("compoundList", get_input("rbList"));
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

#define DEBUG_CPD 0

struct BulletUpdateCpdChildPrimTrans : zeno::INode {
    virtual void apply() override {
        using namespace zs;
        constexpr auto space = execspace_e::openmp;
        auto pol = omp_exec();

#if DEBUG_CPD
        auto centerlist = std::make_shared<ListObject>();
        auto minlist = std::make_shared<ListObject>();
        auto maxlist = std::make_shared<ListObject>();
#endif

        // std::vector<std::shared_ptr<BulletObject>>
        auto cpdList = get_input<ListObject>("compoundList");
        auto cpdBodies = cpdList->get<BulletObject>();
        const auto ncpds = cpdBodies.size();

        bool hasVisualPrimlist = cpdList->userData().has("visualPrimlist");
        std::shared_ptr<ListObject> primlist;
        std::vector<std::shared_ptr<PrimitiveObject>> visPrims;
        if (hasVisualPrimlist) {
            primlist = cpdList->userData().get<ListObject>("visualPrimlist");
            visPrims = primlist->get<PrimitiveObject>();
        } else {
            primlist = std::make_shared<ListObject>();
            cpdList->userData().set("visualPrimlist", primlist);
        }
        int no = 0;
        for (int cpi = 0; cpi != ncpds; ++cpi) {
            auto &cpdBody = cpdBodies[cpi]; // shared_ptr<BulletObject>
            auto &cpdShape = cpdBody->colShape;
            // std::vector<shared_ptr<ListObject>>
            btCompoundShape *btcpdShape = nullptr;
            if (auto shape = cpdShape->shape.get(); shape->isCompound())
                btcpdShape = (btCompoundShape *)shape;
            /// @note regular rbs
            if (btcpdShape == nullptr) {
                auto prim = cpdBody->userData().get<PrimitiveObject>("prim");
                btTransform rbTrans;
                if (cpdBody->body->getMotionState())
                    cpdBody->body->getMotionState()->getWorldTransform(rbTrans);
                else
                    rbTrans = static_cast<btCollisionObject *>(cpdBody->body.get())->getWorldTransform();

                auto translate = vec3f(other_to_vec<3>(rbTrans.getOrigin()));
                auto rotation = vec4f(other_to_vec<4>(rbTrans.getRotation()));
                glm::mat4 matTrans = glm::translate(glm::vec3(translate[0], translate[1], translate[2]));
                glm::quat myQuat(rotation[3], rotation[0], rotation[1], rotation[2]);
                glm::mat4 matQuat = glm::toMat4(myQuat);

                // clone
                std::shared_ptr<PrimitiveObject> visPrim;
                if (hasVisualPrimlist)
                    visPrim = visPrims[no++];
                else
                    visPrim = std::make_shared<PrimitiveObject>(*prim);
                auto matrix = matTrans * matQuat;
                const auto &pos = prim->attr<zeno::vec3f>("pos");
                auto &dstPos = visPrim->attr<zeno::vec3f>("pos");

                auto mapplypos = [](const glm::mat4 &matrix, const glm::vec3 &vector) {
                    auto vector4 = matrix * glm::vec4(vector, 1.0f);
                    return glm::vec3(vector4) / vector4.w;
                };
#pragma omp parallel for
                for (int i = 0; i < pos.size(); i++) {
                    auto p = zeno::vec_to_other<glm::vec3>(pos[i]);
                    p = mapplypos(matrix, p);
                    dstPos[i] = zeno::other_to_vec<3>(p);
                }

                if (!hasVisualPrimlist)
                    primlist->arr.push_back(visPrim);

#if DEBUG_CPD
                centerlist->arr.push_back(
                    std::make_shared<NumericObject>(other_to_vec<3>(cpdBody->body->getCenterOfMassPosition())));
                btVector3 aabbMin, aabbMax;
                cpdBody->body->getAabb(aabbMin, aabbMax);
                minlist->arr.push_back(std::make_shared<NumericObject>(other_to_vec<3>(aabbMin)));
                maxlist->arr.push_back(std::make_shared<NumericObject>(other_to_vec<3>(aabbMax)));
#endif

                continue;
            }
            /// @note true compounds
            auto cpdPrimlist = cpdShape->userData().get<ListObject>("prim");
            auto cpdPrims = cpdPrimlist->get<PrimitiveObject>();
            if (cpdPrims.size() != btcpdShape->getNumChildShapes())
                throw std::runtime_error(
                    fmt::format("the number of child shapes [{}] and prim objs [{}] in compound [{}] mismatch!",
                                btcpdShape->getNumChildShapes(), cpdPrimlist->arr.size(), cpi));

            btTransform cpdTrans;
            if (cpdBody && cpdBody->body->getMotionState()) {
                cpdBody->body->getMotionState()->getWorldTransform(cpdTrans);
            } else {
                cpdTrans = static_cast<btCollisionObject *>(cpdBody->body.get())->getWorldTransform();
            }

#if DEBUG_CPD
            btVector3 aabbMin, aabbMax;
            cpdBody->body->getAabb(aabbMin, aabbMax);
            minlist->arr.push_back(std::make_shared<NumericObject>(other_to_vec<3>(aabbMin)));
            maxlist->arr.push_back(std::make_shared<NumericObject>(other_to_vec<3>(aabbMax)));
            centerlist->arr.push_back(std::make_shared<NumericObject>(other_to_vec<3>(cpdTrans.getOrigin())));
#endif

            for (int rbi = 0; rbi != btcpdShape->getNumChildShapes(); ++rbi) {
                auto child = btcpdShape->getChildShape(rbi);
                auto transform = btcpdShape->getChildTransform(rbi);

                transform.mult(cpdTrans, transform);

                auto translate = vec3f(other_to_vec<3>(transform.getOrigin()));
                auto rotation = vec4f(other_to_vec<4>(transform.getRotation()));
                glm::mat4 matTrans = glm::translate(glm::vec3(translate[0], translate[1], translate[2]));
                glm::quat myQuat(rotation[3], rotation[0], rotation[1], rotation[2]);
                glm::mat4 matQuat = glm::toMat4(myQuat);

                auto prim = cpdPrims[rbi];

                std::shared_ptr<PrimitiveObject> visPrim;
                if (hasVisualPrimlist)
                    visPrim = visPrims[no++];
                else
                    visPrim = std::make_shared<PrimitiveObject>(*prim);

                // clone
                prim = std::make_shared<PrimitiveObject>(*prim);
                auto matrix = matTrans * matQuat;
                const auto &pos = prim->attr<zeno::vec3f>("pos");
                auto &dstPos = visPrim->attr<zeno::vec3f>("pos");

                auto mapplypos = [](const glm::mat4 &matrix, const glm::vec3 &vector) {
                    auto vector4 = matrix * glm::vec4(vector, 1.0f);
                    return glm::vec3(vector4) / vector4.w;
                };
#pragma omp parallel for
                for (int i = 0; i < pos.size(); i++) {
                    auto p = zeno::vec_to_other<glm::vec3>(pos[i]);
                    p = mapplypos(matrix, p);
                    dstPos[i] = zeno::other_to_vec<3>(p);
                }

                if (!hasVisualPrimlist)
                    primlist->arr.push_back(visPrim);
            }
        }

        set_output("primList", primlist);
#if DEBUG_CPD
        set_output("centerList", centerlist);
        set_output("minList", minlist);
        set_output("maxList", maxlist);
#endif
    }
};

ZENDEFNODE(BulletUpdateCpdChildPrimTrans, {
                                              {
                                                  "compoundList",
                                              },
                                              {
                                                  "primList",
#if DEBUG_CPD
                                                  "centerList",
                                                  "minList",
                                                  "maxList",
#endif
                                              },
                                              {},
                                              {"Bullet"},
                                          });

} // namespace zeno