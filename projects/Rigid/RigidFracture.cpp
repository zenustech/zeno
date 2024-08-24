#include <any>
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

void BulletObject::printDynamics(std::any clr_) const {
    fmt::color clr = std::any_cast<fmt::color>(clr_);
    auto trans = getWorldTransform();
    auto origin = trans.getOrigin();
    auto lin = body->getLinearVelocity();
    auto ang = body->getAngularVelocity();
    std::string tag = isCompound() ? "cpd" : "rb";
    fmt::print(fg(clr), "{}[{}]: origin <{}, {}, {}>; linv <{}, {}, {}>; angv <{}, {}, {}>;\n", tag, (void *)this,
               origin[0], origin[1], origin[2], lin[0], lin[1], lin[2], ang[0], ang[1], ang[2]);
}
void BulletObject::printChildDynamics(int rbi, std::any clr_) const {
    fmt::color clr = std::any_cast<fmt::color>(clr_);
    auto cpdTrans = getWorldTransform();
    auto ang = body->getAngularVelocity();
    if (isCompound()) {
        btCompoundShape *btcpdShape = (btCompoundShape *)colShape->shape.get();
        auto chTrans = btcpdShape->getChildTransform(rbi);
        auto lin = body->getVelocityInLocalPoint(chTrans.getOrigin());

        btTransform trans;
        trans.mult(cpdTrans, chTrans);
        auto origin = trans.getOrigin();

        fmt::print(
            fg(clr),
            "rbShape[{}]: origin <{}, {}, {}>; (<{}, {}, {}>::<{}, {}, {}>) linv <{}, {}, {}>; angv <{}, {}, {}>;\n",
            (void *)btcpdShape->getChildShape(rbi), origin[0], origin[1], origin[2], cpdTrans.getOrigin()[0],
            cpdTrans.getOrigin()[1], cpdTrans.getOrigin()[2], chTrans.getOrigin()[0], chTrans.getOrigin()[1],
            chTrans.getOrigin()[2], lin[0], lin[1], lin[2], ang[0], ang[1], ang[2]);
    }
}

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
void set_value(IObject *obj, const std::string &key, const T &value) {
    auto v = std::make_shared<NumericObject>(value);
    obj->userData().set(key, std::move(v));
}
template <typename T>
void set_value(std::shared_ptr<IObject> obj, const std::string &key, const T &value) {
    set_value(obj.get(), key, value);
}

// moment of inertia
btMatrix3x3 getMoI(const btMatrix3x3 &rot, const btVector3 &inertia) {
    btMatrix3x3 S{};
    S.setIdentity();
    S[0][0] = inertia[0];
    S[1][1] = inertia[1];
    S[2][2] = inertia[2];
    return rot * S * rot.transpose();
}
btMatrix3x3 getMoI(const btMatrix3x3 &rot, const btVector3 &i, btScalar mass, const btVector3 &ci,
                   const btVector3 &cc) {
    btMatrix3x3 tensor(0, 0, 0, 0, 0, 0, 0, 0, 0);

    btMatrix3x3 j = rot.transpose();
    j[0] *= i[0];
    j[1] *= i[1];
    j[2] *= i[2];
    j = rot * j;

    //add inertia tensor
    tensor[0] += j[0];
    tensor[1] += j[1];
    tensor[2] += j[2];

    //compute inertia tensor of pointmass at cc
    btVector3 o = ci - cc;
    btScalar o2 = o.length2();
    j[0].setValue(o2, 0, 0);
    j[1].setValue(0, o2, 0);
    j[2].setValue(0, 0, o2);
    j[0] += o * -o.x();
    j[1] += o * -o.y();
    j[2] += o * -o.z();

    //add inertia tensor of pointmass
    tensor[0] += mass * j[0];
    tensor[1] += mass * j[1];
    tensor[2] += mass * j[2];

    return tensor;
}
btVector3 vecInv(const btVector3 &v) {
    btVector3 ret(1, 1, 1);
    ret = ret / v;
    return ret;
}

#define DEBUG_CPD 1
#define DEBUG_DISPLAY 0

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
                btTransform trans = bodyPtr->getWorldTransform();
                cpdPtr->addChildShape(trans, bodyPtr->getCollisionShape());
                cpdChildMasses[compId].push_back(bodyPtr->getMass());
                // cpdOrigins[compId] += (bodyPtr->getMass() * bodyPtr->getCenterOfMassPosition());

                primList->push_back(rbs[rbi]->userData().get("prim"));
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
        pol(zip(isCompound, cpdMasses, cpdTransforms, cpdInertia, btCpdShapes, primLists, rblist->arr),
            [&](bool isCpd, auto cpdMass, const auto &cpdTransform, const auto &inertia, auto &btShape, auto primList,
                auto &cpdBody) {
                if (isCpd) {
                    auto tmp = std::make_shared<BulletCollisionShape>(std::move(btShape));
                    // list of PrimitiveObject, corresponding with each CompoundShape children
                    tmp->userData().set("prim", primList);
                    // cpdBody = std::make_shared<BulletObject>(cpdMass, cpdTransform, tmp);
                    cpdBody = std::make_shared<BulletObject>(cpdMass, cpdTransform, inertia, tmp);
                }
            });

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

        std::shared_ptr<ListObject> initialPrimList;
        auto translationList = std::make_shared<ListObject>();
        auto rotationList = std::make_shared<ListObject>();
#if DEBUG_CPD
        auto centerlist = std::make_shared<ListObject>();
        auto minlist = std::make_shared<ListObject>();
        auto maxlist = std::make_shared<ListObject>();
#endif

        // std::vector<std::shared_ptr<BulletObject>>
        auto cpdList = get_input<ListObject>("compoundList");
        auto cpdBodies = cpdList->get<BulletObject>();
        const auto ncpds = cpdBodies.size();

        // bool hasVisualPrimlist = cpdList->userData().has("visualPrimlist");
        bool hasVisualPrimlist = false;
        std::shared_ptr<ListObject> primlist;
        std::vector<std::shared_ptr<PrimitiveObject>> visPrims;
        if (hasVisualPrimlist) {
            // initialPrimList = cpdList->userData().get<ListObject>("initialPrimlist");
            primlist = cpdList->userData().get<ListObject>("visualPrimlist");
            visPrims = primlist->get<PrimitiveObject>();
        } else {
            initialPrimList = std::make_shared<ListObject>();
            cpdList->userData().set("initialPrimlist", initialPrimList);
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
                btTransform rbTrans = cpdBody->getWorldTransform();

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

                if (!hasVisualPrimlist) {
                    initialPrimList->push_back(prim);
                    primlist->push_back(visPrim);
                }

                translationList->push_back(std::make_shared<NumericObject>(translate));
                rotationList->push_back(std::make_shared<NumericObject>(rotation)); // x, y, z, w
#if DEBUG_CPD
                centerlist->push_back(std::make_shared<NumericObject>(other_to_vec<3>(rbTrans.getOrigin())));
                btVector3 aabbMin, aabbMax;
                cpdBody->body->getAabb(aabbMin, aabbMax);
                minlist->push_back(std::make_shared<NumericObject>(other_to_vec<3>(aabbMin)));
                maxlist->push_back(std::make_shared<NumericObject>(other_to_vec<3>(aabbMax)));
#endif

                continue;
            }

            /// @note true compounds
            auto cpdPrimlist = cpdShape->userData().get<ListObject>("prim");
            auto cpdPrims = cpdPrimlist->get<PrimitiveObject>();
            if (cpdPrims.size() != btcpdShape->getNumChildShapes())
                throw std::runtime_error(
                    fmt::format("the number of child shapes [{}] and prim objs [{}] in compound [{}] mismatch!",
                                btcpdShape->getNumChildShapes(), cpdPrimlist->size(), cpi));

            btTransform cpdTrans = cpdBody->getWorldTransform();

#if DEBUG_CPD
            centerlist->push_back(std::make_shared<NumericObject>(other_to_vec<3>(cpdTrans.getOrigin())));
            btVector3 aabbMin, aabbMax;
            cpdBody->body->getAabb(aabbMin, aabbMax);
            minlist->push_back(std::make_shared<NumericObject>(other_to_vec<3>(aabbMin)));
            maxlist->push_back(std::make_shared<NumericObject>(other_to_vec<3>(aabbMax)));
#endif

#if DEBUG_CPD
            puts("newly updated compounds\' states");
            cpdBody->printDynamics(fmt::color::violet);
#endif

            for (int rbi = 0; rbi != btcpdShape->getNumChildShapes(); ++rbi) {
                auto child = btcpdShape->getChildShape(rbi);
                auto transform = btcpdShape->getChildTransform(rbi);

#if DEBUG_CPD
                if (rbi < 2)
                    cpdBody->printChildDynamics(rbi, fmt::color::pink);
#endif

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

                if (!hasVisualPrimlist) {
                    initialPrimList->push_back(prim);
                    primlist->push_back(visPrim);
                }

                translationList->push_back(std::make_shared<NumericObject>(translate));
                rotationList->push_back(std::make_shared<NumericObject>(rotation)); // x, y, z, w
            }
        }

        set_output("primList", primlist);
        set_output("initialPrimList", initialPrimList);
        set_output("translationList", translationList);
        set_output("rotationList", rotationList);
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
                                                  "initialPrimList",
                                                  "translationList",
                                                  "rotationList",
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
               {"obj1", "obj2", {gParamType_Int, "iternum", "100"}},
               {"constraint_relationship"},
               {{"enum Glue Hard Soft Fixed ConeTwist Gear Generic6Dof Generic6DofSpring "
                 "Generic6DofSpring2 Hinge Hinge2 Point2Point Slider Universal",
                 "constraintName", "Fixed"},
                {"enum position rotation all", "constraintType", "position"}},
               {"Bullet"},
           });

struct BulletObjectSetVel : zeno::INode {
    virtual void apply() override {
        auto obj = get_input<BulletObject>("object");
        auto body = obj->body.get();

        if (has_input("linearVel")) {
            auto v = get_input2<zeno::vec3f>("linearVel");
            body->setLinearVelocity(vec_to_other<btVector3>(v));
        }
        if (has_input("angularVel")) {
            auto v = get_input2<zeno::vec3f>("angularVel");
            body->setAngularVelocity(vec_to_other<btVector3>(v));
        }

        set_output("object", std::move(obj));
    }
};

ZENDEFNODE(BulletObjectSetVel, {
                                   {"object", "linearVel", "angularVel"},
                                   {"object"},
                                   {},
                                   {"Bullet"},
                               });

struct BulletMaintainCompoundsAndConstraints : zeno::INode {
    virtual void apply() override {
#if DEBUG_CPD
        static int iters = 0;
#endif
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
#if DEBUG_CPD
            if (false) {
                if (rel->isGlueConstraint() && (i > 10 && j > 10)) {
                    if (i < 0 || j < 0)
                        throw std::runtime_error("negative coords from constraint relationships!");
                    is.push_back(i);
                    js.push_back(j);
                }
            } else {
                if (rel->isGlueConstraint()) {
                    if (i < 0 || j < 0)
                        throw std::runtime_error("negative coords from constraint relationships!");
                    is.push_back(i);
                    js.push_back(j);
                }
            }
#else
            if (rel->isGlueConstraint()) {
                if (i < 0 || j < 0)
                    throw std::runtime_error("negative coords from constraint relationships!");
                is.push_back(i);
                js.push_back(j);
            }
#endif
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
        fmt::print("{} rigid bodies, {} groups (incl compounds). {} active constraints left to be processed.\n", nrbs,
                   ncompounds, numPreservedCons);

        ///
        ///
        /// @brief compounds
        ///
        ///
        /// @brief construct compounds
        // mass
        std::vector<float> cpdMasses(ncompounds);
        std::vector<float> cpdLinearDampings(ncompounds);
        std::vector<float> cpdAngularDampings(ncompounds);
        std::vector<float> cpdFrictions(ncompounds);
        std::vector<float> cpdRestitutions(ncompounds);
        pol(enumerate(rbs), [&cpdMasses, &cpdLinearDampings, &cpdAngularDampings, &cpdFrictions, &cpdRestitutions, &fas,
                             tab = proxy<space>(tab)](int rbi, const auto &rb) {
            auto fa = fas[rbi];
            auto compId = tab.query(fa);
            auto &body = rb->body;
            auto m = body->getMass();
            atomic_add(exec_omp, &cpdMasses[compId], m);
            atomic_add(exec_omp, &cpdLinearDampings[compId], m * body->getLinearDamping());
            atomic_add(exec_omp, &cpdAngularDampings[compId], m * body->getAngularDamping());
            atomic_add(exec_omp, &cpdFrictions[compId], m * body->getFriction());
            atomic_add(exec_omp, &cpdRestitutions[compId], m * body->getRestitution());
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
                btTransform trans = rbs[rbi]->getWorldTransform();
                cpdPtr->addChildShape(trans, bodyPtr->getCollisionShape());
                cpdChildMasses[compId].push_back(bodyPtr->getMass());
                // cpdOrigins[compId] += (bodyPtr->getMass() * bodyPtr->getCenterOfMassPosition());

                primList->push_back(rbs[rbi]->userData().get("prim"));
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
        std::vector<btVector3> accumCpdVs(ncompounds);    // linear momentum
        std::vector<btVector3> accumCpdWs(ncompounds);    // angular momentum (accepted one)
        std::vector<btVector3> accumCpdWRefs(ncompounds); // angular momentum (direct)

        pol(zip(accumCpdVs, accumCpdWs, accumCpdWRefs), [](auto &a, auto &b, auto &c) {
            a.setZero();
            b.setZero();
            c.setZero();
        });
        pol(range(nrbs), [&](int rbi) {
            if (isRbCompound[rbi]) {
                auto &rb = rbs[rbi];
                auto &body = rb->body;
                auto compId = rbDstCompId[rbi];

                const auto rbTrans = rb->getWorldTransform();
                const auto ci = rbTrans.getOrigin();
                const auto &cpdTrans = cpdTransforms[compId];
                const auto cc = cpdTrans.getOrigin();

                auto mi = body->getMass();
                // add linear momentum
                auto vi = body->getLinearVelocity();
                for (int d = 0; d != 3; ++d)
                    atomic_add(exec_omp, &accumCpdVs[compId][d], mi * vi[d]);

                // add angular momentum (ref, proven wrong)
                auto wi = body->getAngularVelocity();
                auto Ii_cpd = getMoI(rbTrans.getBasis(), rb->getInertia(), mi, ci, cc);
                auto tmp = Ii_cpd * wi;
                for (int d = 0; d != 3; ++d)
                    atomic_add(exec_omp, &accumCpdWRefs[compId][d], tmp[d]);
            }
        });
        pol(range(ncompounds), [&](int cpdi) {
            if (isCompound[cpdi]) {
                auto mass = cpdMasses[cpdi];
                auto cpdv = accumCpdVs[cpdi] / mass;
                accumCpdVs[cpdi] = cpdv;
            }
        });
        pol(range(nrbs), [&](int rbi) {
            if (isRbCompound[rbi]) {
                auto &rb = rbs[rbi];
                auto &body = rb->body;
                auto compId = rbDstCompId[rbi];

                const auto rbTrans = rb->getWorldTransform();
                const auto ci = rbTrans.getOrigin();
                const auto &cpdTrans = cpdTransforms[compId];
                const auto cc = cpdTrans.getOrigin();

                auto mi = body->getMass();
                auto vi = body->getLinearVelocity();
                auto vc = accumCpdVs[compId];

                // add angular momentum
                auto wi = body->getAngularVelocity();
                auto Ii = getMoI(rbTrans.getBasis(), rb->getInertia());
                auto tmp = (Ii * wi + mi * (ci - cc).cross(vi - vc));
                for (int d = 0; d != 3; ++d)
                    atomic_add(exec_omp, &accumCpdWs[compId][d], tmp[d]);
            }
        });
        pol(range(ncompounds), [&](int cpdi) {
            if (isCompound[cpdi]) {
                auto ICpdInv = getMoI(cpdTransforms[cpdi].getBasis(), vecInv(cpdInertia[cpdi]));
                auto cpdw = ICpdInv * accumCpdWs[cpdi];
                accumCpdWs[cpdi] = cpdw;
                auto cpdwref = ICpdInv * accumCpdWRefs[cpdi];
                accumCpdWRefs[cpdi] = cpdwref;
#if DEBUG_CPD
                fmt::print(fg(fmt::color::red), "cpd [{}] v<{}, {}, {}>, w<{}, {}, {}>, wref<{}, {}, {}>.\n", cpdi,
                           accumCpdVs[cpdi][0], accumCpdVs[cpdi][1], accumCpdVs[cpdi][2], cpdw[0], cpdw[1], cpdw[2],
                           cpdwref[0], cpdwref[1], cpdwref[2]);
#endif
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
        pol(zip(isCompound, cpdMasses, accumCpdVs, accumCpdWs, cpdLinearDampings, cpdAngularDampings, cpdFrictions,
                cpdRestitutions, cpdTransforms, cpdInertia, btCpdShapes, primLists, rblist->arr),
            [](bool isCpd, float cpdMass, const btVector3 &v, const btVector3 &w, float cpdLinearDamping,
               float cpdAngularDamping, float friction, float restitution, const auto &cpdTransform,
               const auto &inertia, auto &btShape, auto primList, auto &cpdBody) {
                if (isCpd) {
                    auto tmp = std::make_shared<BulletCollisionShape>(std::move(btShape));
                    // list of PrimitiveObject, corresponding with each CompoundShape children
                    tmp->userData().set("prim", primList);
                    // cpdBody = std::make_shared<BulletObject>(cpdMass, cpdTransform, tmp);
                    auto rb = std::make_shared<BulletObject>(cpdMass, cpdTransform, inertia, tmp);
                    cpdBody = rb;
                    if (cpdMass) {
                        rb->body->setDamping(cpdLinearDamping / cpdMass, cpdAngularDamping / cpdMass);
                        rb->body->setRestitution(restitution / cpdMass);
                        rb->body->setFriction(friction / cpdMass);
                        rb->body->setRestitution(restitution / cpdMass);
                    }
                    rb->body->setLinearVelocity(v);
                    rb->body->setAngularVelocity(w);
                }
            });

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
        set_output("constraintList", conslist);

#if DEBUG_CPD
        iters++;
#endif
    }
};

ZENDEFNODE(BulletMaintainCompoundsAndConstraints, {
                                                      {
                                                          "rbList",
                                                          "constraint_relationships",
                                                      },
                                                      {
                                                          "compoundList",
                                                          "constraintList",
                                                      },
                                                      {},
                                                      {"Bullet"},
                                                  });

/// ultimate solution
struct BulletMaintainRigidBodiesAndConstraints : zeno::INode {
    virtual void apply() override {
        using namespace zs;
        constexpr auto space = execspace_e::openmp;
        auto pol = omp_exec();

        /// @note simple (non-compound) rigid bodies
        auto rblist = get_input<ListObject>("rbList");
        auto rbs = rblist->get<BulletObject>();
        const auto nrbs = rbs.size();

#if DEBUG_CPD
        fmt::print(fg(fmt::color::green), "read rblist at [{}]\n", (void *)rblist.get());
#endif

        /// @note constraint relationships
        auto relationships = get_input<ListObject>("constraint_relationships")->get<BulletConstraintRelationship>();
        const auto ncons = relationships.size();

        ///
        ///
        /// @brief filter constraints (glue + non-glue)
        ///
        ///

        ///
        ///
        /// @brief filter constraints (glue + non-glue)
        ///
        ///
        /// @note does not allow duplicate entries
        // auto comp = [](const std::shared_ptr<BulletObject> &a, const std::shared_ptr<BulletObject> &b) {
        //     return (std::uintptr_t)a.get() < (std::uintptr_t)b.get();
        // };
        // std::sort(std::begin(rbs), std::end(rbs), comp);
        std::vector<int> rbIndices(nrbs);
        std::vector<std::pair<BulletObject *, int>> kvs(nrbs);
        {
            pol(enumerate(rbs, kvs),
                [](int id, auto key, std::pair<BulletObject *, int> &kv) { kv = std::make_pair(key.get(), id); });
            struct {
                constexpr bool operator()(const std::pair<BulletObject *, int> &a,
                                          const std::pair<BulletObject *, int> &b) const {
                    return a.first < b.first;
                }
            } lessOp;
            std::sort(kvs.begin(), kvs.end(), lessOp);
            // sort rbs (ptr) in ascending order
            pol(enumerate(kvs), [&rbIndices /*, &rbs*/](int no, auto kv) {
                rbIndices[no] = kv.second;
                // rbs[no] = kv.first;
            });
        }

        auto find_id = [&rbs = kvs, nrbs](const BulletObject *target) {
            int st = 0, ed = nrbs - 1, mid;
            while (ed >= st) {
                mid = st + (ed - st) / 2;
                if (target == rbs[mid].first)
                    return mid;
                if (target < rbs[mid].first)
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
            auto i = rbIndices[find_id(rel->rb0)];
            auto j = rbIndices[find_id(rel->rb1)];
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
        pol(range(nrbs), [&fas, tab = proxy<space>(tab)](int vi) mutable {
            auto fa = fas[vi];
            while (fa != fas[fa])
                fa = fas[fa];
            fas[vi] = fa;
            tab.insert(fa);
        });
        auto ncompounds = tab.size();
        std::vector<int> cpdSizes(ncompounds);

        ///
        /// sort compounds upon minimum index within
        ///
        std::vector<int> fwdMap(ncompounds);
        {
            std::vector<std::pair<int, int>> kvs(ncompounds);
            auto keys = tab._activeKeys;
            pol(enumerate(keys, kvs),
                [](int id, auto key, std::pair<int, int> &kv) { kv = std::make_pair(key[0], id); });
            struct {
                constexpr bool operator()(const std::pair<int, int> &a, const std::pair<int, int> &b) const {
                    return a.first < b.first;
                }
            } lessOp;
            std::sort(kvs.begin(), kvs.end(), lessOp);
            pol(enumerate(kvs), [&fwdMap](int no, auto kv) { fwdMap[kv.second] = no; });
        }

        // map rigid body indices to target compound indices
        std::vector<int> rbDstCompId(nrbs);
        pol(range(nrbs), [&fas, &rbDstCompId, &cpdSizes, &fwdMap, tab = proxy<space>(tab)](int rbi) mutable {
            auto fa = fas[rbi];
            auto compId = fwdMap[tab.query(fa)];
            rbDstCompId[rbi] = compId;
            atomic_add(exec_omp, &cpdSizes[compId], 1);
        });
        pol(range(nrbs), [&rbDstCompId, &cpdSizes, &isRbCompound](int rbi) {
            auto compId = rbDstCompId[rbi];
            // if (rbi < 10)
            //     printf("rb[%d] belongs to cpd [%d]\n", rbi, (int)compId);
            isRbCompound[rbi] = cpdSizes[compId] > 1;
        });

        ///
        ///
        /// @brief maintain info
        ///
        ///

        std::vector<btTransform> rbTransforms(nrbs);
        pol(zip(rbs, rbTransforms),
            [](std::shared_ptr<BulletObject> rb, btTransform &trans) { trans = rb->getWorldTransform(); });
        ///
        /// @brief update compound transform to rbs
        std::shared_ptr<ListObject> groupList;
        if (rblist->userData().has("compounds")) {
            // std::vector<std::shared_ptr<BulletObject>>
            groupList = rblist->userData().get<ListObject>("compounds");
            auto prevGroups = groupList->get<BulletObject>();
#if DEBUG_CPD
            fmt::print(fg(fmt::color::gold), "read (cpd)grouplist at [{}]\n",
                       (void *)rblist->userData().get<ListObject>("compounds").get());

#if DEBUG_DISPLAY
            puts("previous rbs\' and compounds\' states");

            pol(prevGroups, [](auto cpd) {
                auto v = cpd->body->getLinearVelocity();
                auto w = cpd->body->getAngularVelocity();
                fmt::print(fg(fmt::color::orange), "cpd to-be-dismembered [{}] v<{}, {}, {}>, w<{}, {}, {}>.\n",
                           (void *)cpd.get(), v[0], v[1], v[2], w[0], w[1], w[2]);
            });
#endif
#endif
            /// TBD: update rigid body transforms if exit the compounds
            pol(range(nrbs), [&](int rbi) {
                auto rb = rbs[rbi];
                if (rb->userData().has("cpd_no")) {
                    auto cpdNo = get_value<int>(rb, "cpd_no");
                    const auto &cpd = prevGroups[cpdNo];
                    if (cpd->isCompound()) {
                        /// @note trans
                        auto chTrans = cpd->getChildTransform(rb->body->getCollisionShape());
                        // rb trans = cpd trans * rb trans (either got from cpd child trans or directly from rb)
                        rb->setWorldTransform(cpd->getWorldTransform() * chTrans);
                        rbTransforms[rbi] = cpd->getWorldTransform() * chTrans;

                        /// @note linear vel
                        auto rel = rbTransforms[rbi].getOrigin() - cpd->getWorldTransform().getOrigin();
                        rb->body->setLinearVelocity(cpd->body->getVelocityInLocalPoint(rel));
                        /// @note angular vel
                        rb->body->setAngularVelocity(cpd->body->getAngularVelocity());

#if DEBUG_CPD
                        int rbi = cpd->locateChild(rb->body->getCollisionShape());
                        if (rbi < 2 && rbi >= 0) {
                            // fmt::print(" chk [{}] world pos: <{}, {}, {}>\n", rbi, rbTransforms[rbi].getOrigin()[0],
                            //           rbTransforms[rbi].getOrigin()[1], rbTransforms[rbi].getOrigin()[2]);
                            cpd->printChildDynamics(rbi, fmt::color::pink);
                        }
#endif
                    }
                }
            });

#if DEBUG_CPD
            pol(range(prevGroups), [&](auto &cpd) {
                if (cpd->isCompound()) {
                    cpd->printDynamics(fmt::color::light_green);
                }
            });
#endif
        }
        groupList = std::make_shared<ListObject>();
        /// @note the output BulletObject list
        groupList->arr.resize(ncompounds);
        set_attrib<ListObject>(rblist, "compounds", groupList);

#if DEBUG_CPD
        fmt::print(fg(fmt::color::yellow), "write (cpd)grouplist at [{}]\n", (void *)groupList.get());
#endif

        ///
        ///
        /// @brief maintain info
        ///
        ///
        pol(zip(rbs, rbDstCompId),
            [](std::shared_ptr<BulletObject> rb, int cpdNo) { set_value<int>(rb, "cpd_no", cpdNo); });

        /// @note isolated rigid bodies are delegated to this BulletObject list here!
        // determine compound or not pass on rbs that are does not belong in any compound
        std::vector<int> isCompound(ncompounds);
        pol(range(nrbs),
            [&isCompound, &isRbCompound, &rbDstCompId, &fas, &rbs, &groupList = groupList->arr](int rbi) mutable {
                auto isRbCpd = isRbCompound[rbi];
                auto compId = rbDstCompId[rbi];
                if (isRbCpd)
                    isCompound[compId] = 1;
                else
                    groupList[compId] = rbs[rbi];
            });

        std::vector<int> consMarks(ncons + 1); // 0: discard, 1: preserve
        pol(range(ncons), [&consMarks, &relationships, &rbDstCompId, &consIs, &consJs, &fas](int k) mutable {
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
        fmt::print("{} rigid bodies, {} groups (incl compounds). {} active constraints left to be processed.\n", nrbs,
                   ncompounds, numPreservedCons);

        ///
        ///
        /// @brief compounds
        ///
        ///

        ///
        /// @brief construct compounds
        ///

        // mass
        std::vector<float> cpdMasses(ncompounds);
        std::vector<float> cpdLinearDampings(ncompounds);
        std::vector<float> cpdAngularDampings(ncompounds);
        std::vector<float> cpdFrictions(ncompounds);
        std::vector<float> cpdRestitutions(ncompounds);
        pol(enumerate(rbs), [&cpdMasses, &cpdLinearDampings, &cpdAngularDampings, &cpdFrictions, &cpdRestitutions, &fas,
                             &rbDstCompId](int rbi, const auto &rb) {
            auto compId = rbDstCompId[rbi];
            auto &body = rb->body;
            auto m = body->getMass();
            atomic_add(exec_omp, &cpdMasses[compId], m);
            atomic_add(exec_omp, &cpdLinearDampings[compId], m * body->getLinearDamping());
            atomic_add(exec_omp, &cpdAngularDampings[compId], m * body->getAngularDamping());
            atomic_add(exec_omp, &cpdFrictions[compId], m * body->getFriction());
            atomic_add(exec_omp, &cpdRestitutions[compId], m * body->getRestitution());
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
        pol(range(nrbs), [&](int rbi) mutable {
            std::unique_ptr<btRigidBody> &bodyPtr = rbs[rbi]->body;
            auto compId = rbDstCompId[rbi];
            if (isCompound[compId]) {
                std::lock_guard<std::mutex> guard(comLocks[compId]);
                auto &cpdPtr = btCpdShapes[compId];
                auto &primList = primLists[compId];
#if 1
                btTransform trans = rbs[rbi]->getWorldTransform();
                cpdPtr->addChildShape(trans, bodyPtr->getCollisionShape());
#else
                cpdPtr->addChildShape(rbTransforms[rbi], bodyPtr->getCollisionShape());
#endif
                cpdChildMasses[compId].push_back(bodyPtr->getMass());
                // cpdOrigins[compId] += (bodyPtr->getMass() * bodyPtr->getCenterOfMassPosition());

                primList->push_back(rbs[rbi]->userData().get("prim"));
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

        std::vector<btVector3> accumCpdVs(ncompounds);    // linear momentum
        std::vector<btVector3> accumCpdWs(ncompounds);    // angular momentum (accepted one)
        std::vector<btVector3> accumCpdWRefs(ncompounds); // angular momentum (direct)
        pol(zip(accumCpdVs, accumCpdWs, accumCpdWRefs), [](auto &a, auto &b, auto &c) {
            a.setZero();
            b.setZero();
            c.setZero();
        });
#if 0
        fmt::print(fg(fmt::color::red), "tab status {}\n", tab._buildSuccess.getVal());
        for (int i = 0; i != 2; ++i)
            fmt::print(fg(fmt::color::red), "rb[{}] is cpd: {}, dst cpd no: {}\n", i, isRbCompound[i], rbDstCompId[i]);
#endif
        pol(range(nrbs), [&](int rbi) {
            if (isRbCompound[rbi]) {
                auto &rb = rbs[rbi];
                auto &body = rb->body;
                auto compId = rbDstCompId[rbi];

                auto mi = body->getMass();
                // add linear momentum
                auto vi = body->getLinearVelocity();
                for (int d = 0; d != 3; ++d)
                    atomic_add(exec_omp, &accumCpdVs[compId][d], mi * vi[d]);
#if 0
                fmt::print(fg(fmt::color::red), "rb[{}] mass {} lv <{}, {}, {}>\n", rbi, mi, vi[0], vi[1], vi[2]);
#endif

                {
                    const auto rbTrans = rb->getWorldTransform();
                    const auto ci = rbTrans.getOrigin();
                    const auto &cpdTrans = cpdTransforms[compId];
                    const auto cc = cpdTrans.getOrigin();

                    // add angular momentum (ref, proven wrong)
                    auto wi = body->getAngularVelocity();
                    auto Ii_cpd = getMoI(rbTrans.getBasis(), rb->getInertia(), mi, ci, cc);
                    auto tmp = Ii_cpd * wi;
                    for (int d = 0; d != 3; ++d)
                        atomic_add(exec_omp, &accumCpdWRefs[compId][d], tmp[d]);
                }
            }
        });
        pol(range(ncompounds), [&](int cpdi) {
            if (isCompound[cpdi]) {
                auto mass = cpdMasses[cpdi];
                auto cpdv = accumCpdVs[cpdi] / mass;
                accumCpdVs[cpdi] = cpdv;
            }
        });
        pol(range(nrbs), [&](int rbi) {
            if (isRbCompound[rbi]) {
                auto &rb = rbs[rbi];
                auto &body = rb->body;
                auto compId = rbDstCompId[rbi];

#if 1
                const auto rbTrans = rb->getWorldTransform();
#else
                const auto rbTrans = rbTransforms[rbi];
#endif
                const auto ci = rbTrans.getOrigin();
                const auto &cpdTrans = cpdTransforms[compId];
                const auto cc = cpdTrans.getOrigin();

                auto mi = body->getMass();
                auto vi = body->getLinearVelocity();
                auto vc = accumCpdVs[compId];

                // add angular momentum
                auto wi = body->getAngularVelocity();
                auto Ii = getMoI(rbTrans.getBasis(), rb->getInertia());
                auto tmp = (Ii * wi + mi * (ci - cc).cross(vi - vc));
                for (int d = 0; d != 3; ++d)
                    atomic_add(exec_omp, &accumCpdWs[compId][d], tmp[d]);
            }
        });
        pol(range(ncompounds), [&](int cpdi) {
            if (isCompound[cpdi]) {
                auto ICpdInv = getMoI(cpdTransforms[cpdi].getBasis(), vecInv(cpdInertia[cpdi]));
                auto cpdw = ICpdInv * accumCpdWs[cpdi];
                accumCpdWs[cpdi] = cpdw;

                auto cpdwref = ICpdInv * accumCpdWRefs[cpdi];
                accumCpdWRefs[cpdi] = cpdwref;
#if DEBUG_CPD
                fmt::print(fg(fmt::color::red),
                           "reassembled cpd [{}] v<{}, {}, {}>, w<{}, {}, {}>, wRef<{}, {}, {}>.\n", cpdi,
                           accumCpdVs[cpdi][0], accumCpdVs[cpdi][1], accumCpdVs[cpdi][2], cpdw[0], cpdw[1], cpdw[2],
                           cpdwref[0], cpdwref[1], cpdwref[2]);
#endif
            }
        });

        /// @note adjust compound children transforms according to the principal compound transforms
        pol(zip(isCompound, cpdTransforms, btCpdShapes), [](bool isCpd, const auto &principalTrans, auto &cpdShape) {
            if (isCpd) {
                for (int rbi = 0; rbi != cpdShape->getNumChildShapes(); ++rbi) {
                    cpdShape->updateChildTransform(rbi, principalTrans.inverse() * cpdShape->getChildTransform(rbi));
                }
            }
        });

        // assemble true compound shapes/rigidbodies
        pol(zip(isCompound, cpdMasses, accumCpdVs, accumCpdWs, cpdLinearDampings, cpdAngularDampings, cpdFrictions,
                cpdRestitutions, cpdTransforms, cpdInertia, btCpdShapes, primLists, groupList->arr),
            [](bool isCpd, float cpdMass, const btVector3 &v, const btVector3 &w, float cpdLinearDamping,
               float cpdAngularDamping, float friction, float restitution, const auto &cpdTransform,
               const auto &inertia, auto &btShape, auto primList, auto &cpdBody) {
                if (isCpd) {
                    auto tmp = std::make_shared<BulletCollisionShape>(std::move(btShape));
                    // list of PrimitiveObject, corresponding with each CompoundShape children
                    tmp->userData().set("prim", primList);
                    // cpdBody = std::make_shared<BulletObject>(cpdMass, cpdTransform, tmp);
                    auto rb = std::make_shared<BulletObject>(cpdMass, cpdTransform, inertia, tmp);
                    cpdBody = rb;
                    if (cpdMass) {
                        rb->body->setDamping(cpdLinearDamping / cpdMass, cpdAngularDamping / cpdMass);
                        rb->body->setRestitution(restitution / cpdMass);
                        rb->body->setFriction(friction / cpdMass);
                        rb->body->setRestitution(restitution / cpdMass);
                    }
                    rb->body->setLinearVelocity(v);
                    rb->body->setAngularVelocity(w);
                }
            });

#if DEBUG_CPD
        puts("newly prepared compounds\' states");
        {
            auto cpds = groupList->get<BulletObject>();
            pol(range(cpds), [&](auto &cpd) {
                if (cpd->isCompound()) {
                    cpd->printDynamics(fmt::color::blue);

                    for (int rbi = 0; rbi != 2; ++rbi)
                        cpd->printChildDynamics(rbi, fmt::color::light_pink);
                }
            });
        }
#endif

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

        set_output("compoundList", groupList);
        set_output("constraintList", conslist);
    }
};

ZENDEFNODE(BulletMaintainRigidBodiesAndConstraints, {
                                                        {
                                                            "rbList",
                                                            "constraint_relationships",
                                                        },
                                                        {
                                                            "compoundList",
                                                            "constraintList",
                                                        },
                                                        {},
                                                        {"Bullet"},
                                                    });

} // namespace zeno