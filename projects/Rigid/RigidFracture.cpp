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

template <typename T>
std::shared_ptr<T> get_attrib(std::shared_ptr<IObject> obj, const std::string &key) {
    return obj->userData().get<T>(key);
}
template <typename T>
void set_attrib(std::shared_ptr<IObject> obj, const std::string &key, std::shared_ptr<T> value) {
    obj->userData().set(key, std::move(value));
}
template <typename T>
T get_value(std::shared_ptr<IObject> obj, const std::string &key) {
    return obj->userData().get<NumericObject>(key)->get<T>();
}
template <typename T>
void set_value(std::shared_ptr<IObject> obj, const std::string &key, const T &value) {
    auto v = std::make_shared<NumericObject>(value);
    obj->userData().set(key, std::move(v));
}

#define DEBUG_CPD 1

struct BulletGlueRigidBodies : zeno::INode {
    virtual void apply() override {
        using namespace zs;
        constexpr auto space = execspace_e::openmp;
        auto pol = omp_exec();

#if DEBUG_CPD
        auto centerlist = std::make_shared<ListObject>();
        auto locallist = std::make_shared<ListObject>();
        auto linklist = std::make_shared<ListObject>();
#endif
        // std::vector<std::shared_ptr<BulletObject>>
        auto rbs = get_input<ListObject>("rbList")->get<BulletObject>();
        const auto nrbs = rbs.size();

        auto glueList = get_input<ListObject>("glueListVec2i")->getLiterial<vec2i>();
        auto ncons = glueList.size();

        /// @brief construct compound topo
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
        /// @note the output BulletObject list
        auto rblist = std::make_shared<ListObject>();
#if 1
        rblist->arr.resize(ncompounds);
#else
        rblist->arr.resize(nrbs);
#endif
        /// @note isolated rigid bodies are delegated to this BulletObject list here!
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

        /// @brief construct compounds
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

        /// @note assemble shapes, masses
        std::vector<std::mutex> comLocks(ncompounds);
        std::vector<std::vector<btScalar>> cpdChildMasses(ncompounds);
        // std::vector<btVector3> cpdOrigins(ncompounds);
        // std::vector<btVector3> cpdOriginRefs(ncompounds);
        // for (auto v : cpdOrigins) v.setZero();

#if DEBUG_CPD
        centerlist->arr.resize(nrbs);
        locallist->arr.resize(nrbs);
        linklist->arr.resize(nrbs);
#endif
        pol(range(nrbs), [&, tab = proxy<space>(tab)](int rbi) mutable {
            std::unique_ptr<btRigidBody> &bodyPtr = rbs[rbi]->body;
            auto fa = fas[rbi];
            auto compId = tab.query(fa);

#if DEBUG_CPD
            centerlist->arr[rbi] = std::make_shared<NumericObject>(other_to_vec<3>(bodyPtr->getCenterOfMassPosition()));
#endif
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
                cpdChildMasses[compId].push_back(bodyPtr->getMass());
                // cpdOrigins[compId] += (bodyPtr->getMass() * bodyPtr->getCenterOfMassPosition());

                primList->arr.push_back(rbs[rbi]->userData().get("prim"));
            }
        });
        //pol(zip(cpdOrigins, cpdMasses), [](auto &weightedOrigin, float weight) { weightedOrigin /= weight; });

        /// @note compute compound principal transforms once all children done inserted
        std::vector<btTransform> cpdTransforms(ncompounds);
        std::vector<btVector3> cpdInertia(ncompounds);

#if DEBUG_CPD
        int grbi = 0;
        for (int cpi = 0; cpi != ncompounds; ++cpi) {
            if (isCompound[cpi]) {
                auto &cpdShape = btCpdShapes[cpi];
                btTransform principalTrans;
                btVector3 inertia;
                cpdShape->calculatePrincipalAxisTransform(cpdChildMasses[cpi].data(), principalTrans, inertia);
                for (int rbi = 0; rbi != cpdShape->getNumChildShapes(); ++rbi) {
                    auto chTrans = cpdShape->getChildTransform(rbi);

                    auto loc = std::make_shared<NumericObject>(
                        other_to_vec<3>((principalTrans.inverse() * chTrans).getOrigin()));
                    locallist->arr[grbi] = loc;

                    auto prim = std::make_shared<PrimitiveObject>();
                    prim->resize(2);
                    prim->attr<vec3f>("pos")[0] =
                        safe_dynamic_cast<NumericObject>(centerlist->arr[grbi]).get()->template get<zeno::vec3f>();
                    prim->attr<vec3f>("pos")[1] = loc.get()->template get<zeno::vec3f>();
                    prim->lines.resize(1);
                    prim->lines[0][0] = 0;
                    prim->lines[0][1] = 1;

                    linklist->arr[grbi] = std::move(prim);
                    grbi++;
                }
            } else {
                locallist->arr[grbi] =
                    std::make_shared<NumericObject>(other_to_vec<3>(rbs[grbi]->body->getCenterOfMassPosition()));
                grbi++;
            }
        }
#endif
        pol(zip(isCompound, cpdTransforms, cpdInertia, btCpdShapes, cpdChildMasses),
            [](bool isCpd, btTransform &principalTrans, btVector3 &inertia, auto &cpdShape, auto &cpdMasses) {
                if (isCpd) {
                    cpdShape->calculatePrincipalAxisTransform(cpdMasses.data(), principalTrans, inertia);
#if 0
                    originRef = principal.getOrigin();
                    fmt::print(fg(fmt::color::red),
                               "compound[{}] computed origin <{}, {}, {}>, ref origin <{}, {}, {}>\n", i, originChk[0],
                               originChk[1], originChk[2], originRef[0], originRef[1], originRef[2]);
#endif
                }
            });
        /// @note adjust compound children transforms according to the principal compound transforms
        pol(zip(isCompound, cpdTransforms, btCpdShapes), [](bool isCpd, const auto &principalTrans, auto &cpdShape) {
            if (isCpd) {
#if 1
                for (int rbi = 0; rbi != cpdShape->getNumChildShapes(); ++rbi)
                    cpdShape->updateChildTransform(rbi, principalTrans.inverse() * cpdShape->getChildTransform(rbi));
#else
                auto newCpd = std::make_unique<btCompoundShape>();
                for (int rbi = 0; rbi != cpdShape->getNumChildShapes(); ++rbi) {
                    newCpd->addChildShape(cpdShape->getChildTransform(rbi) * principalTrans.inverse(),
                                          cpdShape->getChildShape(rbi));
                }
                cpdShape = std::move(newCpd);
#endif
            }
        });

        // assemble true compound shapes/rigidbodies
        auto shapelist = std::make_shared<ListObject>();
        shapelist->arr.resize(ncompounds);
        pol(zip(isCompound, cpdMasses, cpdTransforms, cpdInertia, btCpdShapes, primLists, rblist->arr, shapelist->arr),
            [&](bool isCpd, auto cpdMass, const auto &cpdTransform, const auto &inertia, auto &btShape, auto primList,
                auto &cpdBody, auto &cpdShape) {
                if (isCpd) {
                    auto tmp = std::make_shared<BulletCollisionShape>(std::move(btShape));
                    cpdShape = tmp;
                    // list of PrimitiveObject, corresponding with each CompoundShape children
                    tmp->userData().set("prim", primList);
                    // cpdBody = std::make_shared<BulletObject>(cpdMass, cpdTransform, tmp);
                    cpdBody = std::make_shared<BulletObject>(cpdMass, cpdTransform, inertia, tmp);
                }
            });

        rblist->userData().set("compoundShapes", shapelist);

        set_output("compoundList", rblist);
#if DEBUG_CPD
        set_output("centerList", centerlist);
        set_output("localList", locallist);
        set_output("linkList", linklist);
#endif
        // set_output("compoundList", get_input("rbList"));
    }
};

#if DEBUG_CPD
ZENDEFNODE(BulletGlueRigidBodies, {
                                      {
                                          "rbList",
                                          "glueListVec2i",
                                      },
                                      {
                                          "compoundList",
                                          "centerList",
                                          "localList",
                                          "linkList",
                                      },
                                      {},
                                      {"Bullet"},
                                  });
#else
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
#endif

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
                pol(range(pos.size()), [&pos, &dstPos, &matrix, &mapplypos](int i) {
                    auto p = zeno::vec_to_other<glm::vec3>(pos[i]);
                    p = mapplypos(matrix, p);
                    dstPos[i] = zeno::other_to_vec<3>(p);
                });

                if (!hasVisualPrimlist)
                    primlist->arr.push_back(visPrim);

#if DEBUG_CPD
                centerlist->arr.push_back(std::make_shared<NumericObject>(other_to_vec<3>(rbTrans.getOrigin())));
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
            centerlist->arr.push_back(std::make_shared<NumericObject>(other_to_vec<3>(cpdTrans.getOrigin())));
            btVector3 aabbMin, aabbMax;
            cpdBody->body->getAabb(aabbMin, aabbMax);
            minlist->arr.push_back(std::make_shared<NumericObject>(other_to_vec<3>(aabbMin)));
            maxlist->arr.push_back(std::make_shared<NumericObject>(other_to_vec<3>(aabbMax)));
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
                pol(range(pos.size()), [&pos, &dstPos, &matrix, &mapplypos](int i) {
                    auto p = zeno::vec_to_other<glm::vec3>(pos[i]);
                    p = mapplypos(matrix, p);
                    dstPos[i] = zeno::other_to_vec<3>(p);
                });

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

#if DEBUG_CPD
ZENDEFNODE(BulletUpdateCpdChildPrimTrans, {
                                              {
                                                  "compoundList",
                                              },
                                              {
                                                  "primList",
                                                  "centerList",
                                                  "minList",
                                                  "maxList",
                                              },
                                              {},
                                              {"Bullet"},
                                          });
#else
ZENDEFNODE(BulletUpdateCpdChildPrimTrans, {
                                              {
                                                  "compoundList",
                                              },
                                              {
                                                  "primList",
                                              },
                                              {},
                                              {"Bullet"},
                                          });
#endif

struct BulletMakeConstraintRelationship : zeno::INode {
    virtual void apply() override {
        auto constraintName = get_param<std::string>("constraintName");
        auto constraintType = get_param<std::string>("constraintType");
        auto obj1 = get_input<BulletObject>("obj1");
        auto iter = get_input2<int>("iternum");
        std::shared_ptr<BulletConstraintRelationship> cons;
        if (has_input("obj2")) {
            auto obj2 = get_input<BulletObject>("obj2");
            cons = std::make_shared<BulletConstraintRelationship>(obj1.get(), obj2.get(), constraintName, iter,
                                                                  constraintType);
        } else {
            cons = std::make_shared<BulletConstraintRelationship>(obj1.get(), constraintName, iter, constraintType);
        }
        set_output("constraint_relationship", std::move(cons));
    }
};

ZENDEFNODE(BulletMakeConstraintRelationship,
           {
               {"obj1", "obj2", {"int", "iternum", "100"}},
               {"constraint_relationship"},
               {{"enum Glue Hard Soft Fixed ConeTwist Gear Generic6Dof Generic6DofSpring "
                 "Generic6DofSpring2 Hinge Hinge2 Point2Point Slider Universal",
                 "constraintName", "Fixed"},
                {"enum position rotation all", "constraintType", "position"}},
               {"Bullet"},
           });

struct BulletMaintainCompoundsAndConstraints : zeno::INode {
    virtual void apply() override {
        using namespace zs;
        constexpr auto space = execspace_e::openmp;
        auto pol = omp_exec();

        /// @note simple (non-compound) rigid bodies
        auto rbs = get_input<ListObject>("rbList")->get<BulletObject>();
        const auto nrbs = rbs.size();

        /// @note constraint relationships
        auto relationships = get_input<ListObject>("constraint_relationships")->get<BulletConstraintRelationship>();
        const auto ncons = relationships.size();

        ///
        ///
        /// @brief filter constraints (glue + non-glue)
        ///
        ///
        /// @note does not allow duplicate entries
        auto comp = [](const std::shared_ptr<BulletObject> &a, const std::shared_ptr<BulletObject> &b) {
            return (std::uintptr_t)a.get() < (std::uintptr_t)b.get();
        };
        std::sort(std::begin(rbs), std::end(rbs), comp);
        auto find_id = [&rbs, nrbs](const BulletObject *target) {
            int st = 0, ed = nrbs - 1, mid;
            while (ed >= st) {
                mid = st + (ed - st) / 2;
                if (target == rbs[mid].get())
                    return mid;
                if (target < rbs[mid].get())
                    ed = mid - 1;
                else
                    st = mid + 1;
            }
            return -1;
        };

        /// @brief construct compound topo
        std::vector<int> is, js;
        is.reserve(ncons);
        js.reserve(ncons);
        std::vector<int> consIs(ncons);
        std::vector<int> consJs(ncons); // might contain -1 when unary constraints exist
        for (std::size_t k = 0; k != ncons; ++k) {
            auto &rel = relationships[k];
            auto i = find_id(rel->rb0);
            auto j = find_id(rel->rb1);
            consIs[k] = i;
            consJs[k] = j;
            if (rel->isGlueConstraint()) {
                is.push_back(i);
                js.push_back(j);
            }
        }

        SparseMatrix<int, true> spmat{(int)nrbs, (int)nrbs};
        spmat.build(pol, (int)nrbs, (int)nrbs, range(is), range(js), true_c);

        std::vector<int> fas(nrbs);
        // check if the rigid body is with an compound (more than one rigid body)
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
        // map rigid body indices to target compound indices
        std::vector<int> rbDstCompId(nrbs);
        pol(range(nrbs), [&fas, &rbDstCompId, tab = proxy<space>(tab)](int rbi) mutable {
            auto fa = fas[rbi];
            auto compId = tab.query(fa);
            rbDstCompId[rbi] = compId;
        });

        auto ncompounds = tab.size();
        /// @note the output BulletObject list
        auto rblist = std::make_shared<ListObject>();
        rblist->arr.resize(ncompounds);

        /// @note isolated rigid bodies are delegated to this BulletObject list here!
        // determine compound or not pass on rbs that are does not belong in any compound
        std::vector<int> isCompound(ncompounds);
        pol(range(nrbs), [&isCompound, &isRbCompound, &rbDstCompId, &fas, &rbs, tab = proxy<space>(tab),
                          &rblist = rblist->arr](int rbi) mutable {
            auto isRbCpd = isRbCompound[rbi];
            auto compId = rbDstCompId[rbi];
            if (isRbCpd)
                isCompound[compId] = 1;
            else
                rblist[compId] = rbs[rbi];
        });
        fmt::print("{} rigid bodies, {} groups (incl compounds).\n", nrbs, ncompounds);

        std::vector<int> consMarks(ncons + 1); // 0: discard, 1: preserve
        pol(range(ncons),
            [&consMarks, &relationships, &rbDstCompId, &consIs, &consJs, &fas, tab = proxy<space>(tab)](int k) mutable {
                auto &rel = relationships[k];
                if (!rel->isGlueConstraint()) {
                    if (rel->isUnaryConstraint())
                        consMarks[k] = 1;
                    else {
                        auto compI = rbDstCompId[consIs[k]];
                        auto compJ = rbDstCompId[consJs[k]];
                        consMarks[k] = compI != compJ;
                    }
                } else
                    consMarks[k] = 0;
            });
        std::vector<int> consOffsets(ncons + 1);
        exclusive_scan(pol, std::begin(consMarks), std::end(consMarks), std::begin(consOffsets));
        // filter actual constraints in effect
        auto numPreservedCons = consOffsets[ncons];
        std::vector<BulletConstraintRelationship *> gatheredCons(numPreservedCons);
        std::vector<int> gatheredConsIs(numPreservedCons), gatheredConsJs(numPreservedCons);
        std::vector<int> gatheredConsCompIs(numPreservedCons), gatheredConsCompJs(numPreservedCons);
        pol(range(ncons), [&](int k) {
            if (consMarks[k]) {
                auto dst = consOffsets[k];
                gatheredCons[dst] = relationships[k].get();
                gatheredConsIs[dst] = consIs[k];
                gatheredConsJs[dst] = consJs[k];
                gatheredConsCompIs[dst] = rbDstCompId[consIs[k]];
                gatheredConsCompJs[dst] = rbDstCompId[consJs[k]];
            }
        });

        ///
        ///
        /// @brief compounds
        ///
        ///
        /// @brief construct compounds
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

        /// @note assemble shapes, masses
        std::vector<std::mutex> comLocks(ncompounds);
        std::vector<std::vector<btScalar>> cpdChildMasses(ncompounds);
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
                cpdChildMasses[compId].push_back(bodyPtr->getMass());
                // cpdOrigins[compId] += (bodyPtr->getMass() * bodyPtr->getCenterOfMassPosition());

                primList->arr.push_back(rbs[rbi]->userData().get("prim"));
            }
        });

        /// @note compute compound principal transforms once all children done inserted
        std::vector<btTransform> cpdTransforms(ncompounds);
        std::vector<btVector3> cpdInertia(ncompounds);
        pol(zip(isCompound, cpdTransforms, cpdInertia, btCpdShapes, cpdChildMasses),
            [](bool isCpd, btTransform &principalTrans, btVector3 &inertia, auto &cpdShape, auto &cpdMasses) {
                if (isCpd) {
                    cpdShape->calculatePrincipalAxisTransform(cpdMasses.data(), principalTrans, inertia);
                }
            });
        /// @note adjust compound children transforms according to the principal compound transforms
        pol(zip(isCompound, cpdTransforms, btCpdShapes), [](bool isCpd, const auto &principalTrans, auto &cpdShape) {
            if (isCpd) {
                for (int rbi = 0; rbi != cpdShape->getNumChildShapes(); ++rbi)
                    cpdShape->updateChildTransform(rbi, principalTrans.inverse() * cpdShape->getChildTransform(rbi));
            }
        });

        // assemble true compound shapes/rigidbodies
        auto shapelist = std::make_shared<ListObject>();
        shapelist->arr.resize(ncompounds);
        pol(zip(isCompound, cpdMasses, cpdTransforms, cpdInertia, btCpdShapes, primLists, rblist->arr, shapelist->arr),
            [&](bool isCpd, auto cpdMass, const auto &cpdTransform, const auto &inertia, auto &btShape, auto primList,
                auto &cpdBody, auto &cpdShape) {
                if (isCpd) {
                    auto tmp = std::make_shared<BulletCollisionShape>(std::move(btShape));
                    cpdShape = tmp;
                    // list of PrimitiveObject, corresponding with each CompoundShape children
                    tmp->userData().set("prim", primList);
                    // cpdBody = std::make_shared<BulletObject>(cpdMass, cpdTransform, tmp);
                    cpdBody = std::make_shared<BulletObject>(cpdMass, cpdTransform, inertia, tmp);
                }
            });

        rblist->userData().set("compoundShapes", shapelist);

        ///
        ///
        /// @brief constraints
        ///
        ///
        /// @brief generate bulletconstraints
        /// @note the output BulletConstraint list
        auto conslist = std::make_shared<ListObject>();
        conslist->arr.resize(numPreservedCons);
        // numPreservedCons, gatheredCons, gatheredConsIs, gatheredConsJs, gatheredConsCompIs, gatheredConsCompJs

        set_output("compoundList", rblist);
    }
};

ZENDEFNODE(BulletMaintainCompoundsAndConstraints, {
                                                      {
                                                          "rbList",
                                                          "constraint_relationships",
                                                      },
                                                      {
                                                          "compoundList",
                                                      },
                                                      {},
                                                      {"Bullet"},
                                                  });

} // namespace zeno