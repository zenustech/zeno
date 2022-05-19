#include <memory>
#include <vector>

// zeno basics
#include <zeno/ListObject.h>
#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/logger.h>
#include <zeno/utils/UserData.h>
#include <zeno/zeno.h>
#include <zeno/utils/fileio.h>

#include "RigidTest.h"

// convex decomposition
#include <VHACD/inc/VHACD.h>
#include <hacdHACD.h>
#include <hacdICHull.h>
#include <hacdVector.h>

// bullet basics
#include <BulletCollision/CollisionDispatch/btCollisionDispatcherMt.h>
#include <BulletCollision/CollisionShapes/btShapeHull.h>
#include <BulletDynamics/ConstraintSolver/btSequentialImpulseConstraintSolverMt.h>
#include <BulletDynamics/Dynamics/btDiscreteDynamicsWorldMt.h>
#include <LinearMath/btConvexHullComputer.h>
#include <btBulletDynamicsCommon.h>

// multibody dynamcis
#include <BulletDynamics/Featherstone/btMultiBody.h>
#include <BulletDynamics/Featherstone/btMultiBodyJointFeedback.h>


// multibody inverse kinematics/dynamics
#include "BussIK/IKTrajectoryHelper.h"
#include <Bullet3Common/b3HashMap.h>
#include "BulletInverseDynamics/MultiBodyTree.hpp"
#include "BulletInverseDynamics/btMultiBodyTreeCreator.hpp"

namespace {
using namespace zeno;

/*
 *  Bullet Position & Rotation
 */


struct BulletMakeTransform : zeno::INode {
    virtual void apply() override {
        auto trans = std::make_unique<BulletTransform>();
        trans->trans.setIdentity();
        if (has_input("translate")) {
            auto origin = get_input<zeno::NumericObject>("translate")->get<zeno::vec3f>();
            trans->trans.setOrigin(zeno::vec_to_other<btVector3>(origin));
        }
        if (has_input("rotation")) {
            if (get_input<zeno::NumericObject>("rotation")->is<zeno::vec3f>()) {
                auto rotation = get_input<zeno::NumericObject>("rotation")->get<zeno::vec3f>();
                trans->trans.setRotation(zeno::vec_to_other<btQuaternion>(rotation)); // ypr
            } else {
                auto rotation = get_input<zeno::NumericObject>("rotation")->get<zeno::vec4f>();
                trans->trans.setRotation(zeno::vec_to_other<btQuaternion>(rotation));
            }
        }
        set_output("trans", std::move(trans));
    }
};

ZENDEFNODE(BulletMakeTransform, {
    {{"vec3f", "translate"},  "rotation"},
    {"trans"},
    {},
    {"Bullet"},
});

struct BulletTransformSetBasisEuler : zeno::INode {
    virtual void apply() override {
        auto trans = get_input<BulletTransform>("trans")->trans;
        auto euler = get_input<zeno::NumericObject>("eulerZYX")->get<zeno::vec3f>();
        trans.getBasis().setEulerZYX(euler[0], euler[1], euler[2]);
    }
};

ZENDEFNODE(BulletTransformSetBasisEuler, {
    {"trans", "eulerZYX"},
    {},
    {},
    {"Bullet"}
});

struct BulletMakeFrameFromPivotAxis : zeno::INode {
	virtual void apply() override {
		auto pivot = zeno::vec_to_other<btVector3>(get_input<zeno::NumericObject>("pivot")->get<zeno::vec3f>());
		auto axis = zeno::vec_to_other<btVector3>(get_input<zeno::NumericObject>("axis")->get<zeno::vec3f>());

        auto trans = std::make_shared<BulletTransform>();

        trans->trans.setOrigin(pivot);
        trans->trans.getBasis().setValue(axis.getX(),axis.getX(),axis.getX(),axis.getY(),axis.getY(),axis.getY(),axis.getZ(),axis.getZ(),axis.getZ());

		set_output("frame", std::move(trans));
	}
};

ZENDEFNODE(BulletMakeFrameFromPivotAxis, {
	{"pivot", "axis"},
	{"frame"},
	{},
	{"Bullet"}
});

struct BulletQuatRotate : zeno::INode {
    virtual void apply() override {
        auto quat = zeno::vec_to_other<btQuaternion>(get_input<zeno::NumericObject>("quat")->get<zeno::vec4f>());
        auto v = zeno::vec_to_other<btVector3>(get_input<zeno::NumericObject>("vec3")->get<zeno::vec3f>());

        auto res = quatRotate(quat, v);

        auto res2 = std::make_shared<zeno::NumericObject>();
        res2->set(vec3f(other_to_vec<3>(res)));
        set_output("vec3", std::move(res2));
    }
};

ZENDEFNODE(BulletQuatRotate, {
    {"quat", "vec3"},
    {"vec3"},
    {},
    {"Bullet"}
});

struct BulletComposeTransform : zeno::INode {
    virtual void apply() override {
        auto transFirst = get_input<BulletTransform>("transFirst")->trans;
        auto transSecond = get_input<BulletTransform>("transSecond")->trans;
        auto trans = std::make_unique<BulletTransform>();
        trans->trans = transFirst * transSecond;
        set_output("trans", std::move(trans));
    }
};

ZENDEFNODE(BulletComposeTransform, {
    {"transFirst", "transSecond"},
    {"trans"},
    {},
    {"Bullet"},
});


/*
 * Bullet Geometry
 */
struct PrimitiveToBulletMesh : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto mesh = std::make_unique<BulletTriangleMesh>();
        auto pos = prim->attr<zeno::vec3f>("pos");
        for (int i = 0; i < prim->tris.size(); i++) {
            auto f = prim->tris[i];
            mesh->mesh.addTriangle(
                    zeno::vec_to_other<btVector3>(pos[f[0]]),
                    zeno::vec_to_other<btVector3>(pos[f[1]]),
                    zeno::vec_to_other<btVector3>(pos[f[2]]), true);
        }
        set_output("mesh", std::move(mesh));
    }
};

ZENDEFNODE(PrimitiveToBulletMesh, {
    {"prim"},
    {"mesh"},
    {},
    {"Bullet"},
});


struct VHACDParameters
{
    bool m_run;
    VHACD::IVHACD::Parameters m_paramsVHACD;
    VHACDParameters(void)
    {
        m_run = true;
    }
};

struct PrimitiveConvexDecompositionV : zeno::INode {
    /*
    *  Use VHACD to do convex decomposition
    */
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto &pos = prim->attr<zeno::vec3f>("pos");

        //auto resolution = get_input2<int>("resolution");


        std::vector<float> points;
        std::vector<int> triangles;

        for (size_t i = 0; i < pos.size(); i++){
            //std::cout << "pos: " << pos[i][0] << "," << pos[i][1] << "," << pos[i][2] <<std::endl;
            points.push_back(pos[i][0]);
            points.push_back(pos[i][1]);
            points.push_back(pos[i][2]);
        }

        for (size_t i = 0; i < prim->tris.size(); i++){
            //std::cout << "tri: " << prim->tris[i][0] << "," << prim->tris[i][1] << "," << prim->tris[i][2] <<std::endl;
            triangles.push_back(prim->tris[i][0]);
            triangles.push_back(prim->tris[i][1]);
            triangles.push_back(prim->tris[i][2]);
        }

        VHACDParameters params;
        // TODO: get more parameters from INode, currently it is only for testing.
        params.m_paramsVHACD.m_resolution = 100000; // Maximum number of voxels generated during the voxelization stage (default=100,000, range=10,000-16,000,000)
        params.m_paramsVHACD.m_depth = 20; // Maximum number of clipping stages. During each split stage, parts with a concavity higher than the user defined threshold are clipped according the "best" clipping plane (default=20, range=1-32)
        params.m_paramsVHACD.m_concavity = 0.001; // Maximum allowed concavity (default=0.0025, range=0.0-1.0)
        params.m_paramsVHACD.m_planeDownsampling = 4; // Controls the granularity of the search for the "best" clipping plane (default=4, range=1-16)
        params.m_paramsVHACD.m_convexhullDownsampling  = 4; // Controls the precision of the convex-hull generation process during the clipping plane selection stage (default=4, range=1-16)
        params.m_paramsVHACD.m_alpha = 0.05; // Controls the bias toward clipping along symmetry planes (default=0.05, range=0.0-1.0)
        params.m_paramsVHACD.m_beta = 0.05; // Controls the bias toward clipping along revolution axes (default=0.05, range=0.0-1.0)
        params.m_paramsVHACD.m_gamma = 0.0005; // Controls the maximum allowed concavity during the merge stage (default=0.00125, range=0.0-1.0)
        params.m_paramsVHACD.m_pca = 0; // Enable/disable normalizing the mesh before applying the convex decomposition (default=0, range={0,1})
        params.m_paramsVHACD.m_mode = 0; // 0: voxel-based approximate convex decomposition, 1: tetrahedron-based approximate convex decomposition (default=0, range={0,1})
        params.m_paramsVHACD.m_maxNumVerticesPerCH = 64; // Controls the maximum number of triangles per convex-hull (default=64, range=4-1024)
        params.m_paramsVHACD.m_minVolumePerCH = 0.0001; // Controls the adaptive sampling of the generated convex-hulls (default=0.0001, range=0.0-0.01)
        params.m_paramsVHACD.m_convexhullApproximation = true; // Enable/disable approximation when computing convex-hulls (default=1, range={0,1})
        params.m_paramsVHACD.m_oclAcceleration = true; // Enable/disable OpenCL acceleration (default=0, range={0,1})


        VHACD::IVHACD* interfaceVHACD = VHACD::CreateVHACD();
        bool res = interfaceVHACD->Compute(&points[0], 3, (unsigned int)points.size() / 3,
                                           &triangles[0], 3, (unsigned int)triangles.size() / 3, params.m_paramsVHACD);


        // save output
        auto listPrim = std::make_shared<zeno::ListObject>();
        listPrim->arr.clear();


        unsigned int nConvexHulls = interfaceVHACD->GetNConvexHulls();
        //std::cout<< "Generate output:" << nConvexHulls << " convex-hulls" << std::endl;
        log_debugf("Generate output: %d convex-hulls \n", nConvexHulls);

        bool good_ch_flag = true;
        VHACD::IVHACD::ConvexHull ch;
        size_t vertexOffset = 0; // triangle index start from 1
        for (size_t c = 0; c < nConvexHulls; c++) {
            interfaceVHACD->GetConvexHull(c, ch);
            size_t nPoints = ch.m_nPoints;
            size_t nTriangles = ch.m_nTriangles;

            auto outprim = std::make_shared<zeno::PrimitiveObject>();
            outprim->resize(nPoints);
            outprim->tris.resize(nTriangles);

            auto &outpos = outprim->add_attr<zeno::vec3f>("pos");

            if (nPoints > 0) {
                for (size_t i = 0; i < nPoints; i ++) {
                    size_t ind = i * 3;
                    outpos[i] = zeno::vec3f(ch.m_points[ind], ch.m_points[ind + 1], ch.m_points[ind + 2]);
                }
            }
            else{
                good_ch_flag = false;
            }
            if (nTriangles > 0)
            {
                for (size_t i = 0; i < nTriangles; i++) {
                    size_t ind = i * 3;
                    outprim->tris[i] = zeno::vec3i(ch.m_triangles[ind], ch.m_triangles[ind + 1],ch.m_triangles[ind + 2]);
                }
            }
            else{
                good_ch_flag = false;
            }

            if(good_ch_flag) {
                listPrim->arr.push_back(std::move(outprim));
            }
        }

        interfaceVHACD->Clean();
        interfaceVHACD->Release();

        set_output("listPrim", std::move(listPrim));
    }
};

ZENDEFNODE(PrimitiveConvexDecompositionV, {
    {"prim"},
    {"listPrim"},
    {},
    {"Bullet"},
});

struct PrimitiveConvexDecomposition : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto &pos = prim->attr<zeno::vec3f>("pos");

        std::vector<HACD::Vec3<HACD::Real>> points;
        std::vector<HACD::Vec3<long>> triangles;

        for (int i = 0; i < pos.size(); i++) {
            points.push_back(
                    zeno::vec_to_other<HACD::Vec3<HACD::Real>>(pos[i]));
        }

        for (int i = 0; i < prim->tris.size(); i++) {
            triangles.push_back(
                    zeno::vec_to_other<HACD::Vec3<long>>(prim->tris[i]));
        }

        HACD::HACD hacd;
        hacd.SetPoints(points.data());
        hacd.SetNPoints(points.size());
        hacd.SetTriangles(triangles.data());
        hacd.SetNTriangles(triangles.size());

        hacd.SetCompacityWeight(0.1);
        hacd.SetVolumeWeight(0.0);
        hacd.SetNClusters(2);
        hacd.SetNVerticesPerCH(100);
        hacd.SetConcavity(100.0);
        hacd.SetAddExtraDistPoints(false);
        hacd.SetAddNeighboursDistPoints(false);
        hacd.SetAddFacesPoints(false);

        hacd.Compute();
        size_t nClusters = hacd.GetNClusters();

        auto listPrim = std::make_shared<zeno::ListObject>();
        listPrim->arr.clear();

        log_debugf("hacd got %d clusters\n", nClusters);
        for (size_t c = 0; c < nClusters; c++) {
            size_t nPoints = hacd.GetNPointsCH(c);
            size_t nTriangles = hacd.GetNTrianglesCH(c);
            log_debugf("hacd cluster %d have %d points, %d triangles\n",
                       c, nPoints, nTriangles);

            points.clear();
            points.resize(nPoints);
            triangles.clear();
            triangles.resize(nTriangles);
            hacd.GetCH(c, points.data(), triangles.data());

            auto outprim = std::make_shared<zeno::PrimitiveObject>();
            outprim->resize(nPoints);
            outprim->tris.resize(nTriangles);

            auto &outpos = outprim->add_attr<zeno::vec3f>("pos");
            for (size_t i = 0; i < nPoints; i++) {
                auto p = points[i];
                //printf("point %d: %f %f %f\n", i, p.X(), p.Y(), p.Z());
                outpos[i] = zeno::vec3f(p.X(), p.Y(), p.Z());
            }

            for (size_t i = 0; i < nTriangles; i++) {
                auto p = triangles[i];
                //printf("triangle %d: %d %d %d\n", i, p.X(), p.Y(), p.Z());
                outprim->tris[i] = zeno::vec3i(p.X(), p.Y(), p.Z());
            }

            listPrim->arr.push_back(std::move(outprim));
        }

        set_output("listPrim", std::move(listPrim));
    }
};

ZENDEFNODE(PrimitiveConvexDecomposition, {
    {"prim"},
    {"listPrim"},
    {},
    {"Bullet"},
});


/*
 *  Bullet Collision
 */

struct BulletMakeBoxShape : zeno::INode {
    virtual void apply() override {
        auto size = get_input<zeno::NumericObject>("semiSize")->get<zeno::vec3f>();
        auto shape = std::make_shared<BulletCollisionShape>(
            std::make_unique<btBoxShape>(zeno::vec_to_other<btVector3>(size)));
        set_output("shape", std::move(shape));
    }
};

ZENDEFNODE(BulletMakeBoxShape, {
    {{"vec3f", "semiSize", "1,1,1"}},
    {"shape"},
    {},
    {"Bullet"},
});

struct BulletMakeSphereShape : zeno::INode {
    virtual void apply() override {
        auto radius = get_input<zeno::NumericObject>("radius")->get<float>();
        auto shape = std::make_shared<BulletCollisionShape>(
            std::make_unique<btSphereShape>(btScalar(radius)));
        set_output("shape", std::move(shape));
    }
};

ZENDEFNODE(BulletMakeSphereShape, {
    {{"float", "radius", "1"}},
    {"shape"},
    {},
    {"Bullet"},
});

struct BulletMakeStaticPlaneShape : zeno::INode {
	virtual void apply() override {
		auto planeNormal = zeno::vec_to_other<btVector3>(get_input<zeno::NumericObject>("planeNormal")->get<zeno::vec3f>());
		auto planeConstant = btScalar(get_input2<float>("planeConstant"));

		auto shape = std::make_shared<BulletCollisionShape>(std::make_unique<btStaticPlaneShape>(planeNormal, planeConstant));
		set_output("shape", std::move(shape));
	}
};

ZENDEFNODE(BulletMakeStaticPlaneShape, {
	{"planeNormal", {"float", "planeConstant", "40"}},
	{"shape"},
	{},
	{"Bullet"},
});

struct BulletMakeCapsuleShape : zeno::INode {
	virtual void apply() override {
		auto radius = get_input2<float>("radius");
		auto height = get_input2<float>("height");

		auto shape = std::make_shared<BulletCollisionShape>(std::make_unique<btCapsuleShape>(btScalar(radius), btScalar(height)));
		set_output("shape", std::move(shape));
	}
};

ZENDEFNODE(BulletMakeCapsuleShape, {
	{{"float", "radius", "1"}, {"float", "height", "1"}},
	{"shape"},
	{},
	{"Bullet"},
});

struct BulletMakeCylinderShape : zeno::INode {
	virtual void apply() override {
		auto halfExtents = zeno::vec_to_other<btVector3>(get_input<zeno::NumericObject>("halfExtents")->get<zeno::vec3f>());

		auto shape = std::make_shared<BulletCollisionShape>(std::make_unique<btCylinderShape>(halfExtents));
		set_output("shape", std::move(shape));
	}
};

ZENDEFNODE(BulletMakeCylinderShape, {
	{"halfExtents"},
	{"shape"},
	{},
	{"Bullet"},
});

struct BulletCompoundShape : BulletCollisionShape {
	std::vector<std::shared_ptr<BulletCollisionShape>> children;

	using BulletCollisionShape::BulletCollisionShape;

	void addChild(btTransform const &trans,
	              std::shared_ptr<BulletCollisionShape> child) {
		auto comShape = static_cast<btCompoundShape *>(shape.get());
		comShape->addChildShape(trans, child->shape.get());
		children.push_back(std::move(child));
	}
};

struct BulletMakeCompoundShape : zeno::INode {
    virtual void apply() override {
        auto compound = std::make_unique<btCompoundShape>();
        auto shape = std::make_shared<BulletCompoundShape>(std::move(compound));
        set_output("compound", std::move(shape));
    }
};

ZENDEFNODE(BulletMakeCompoundShape, {
    {},
    {"compound"},
    {},
    {"Bullet"},
});

struct BulletCompoundAddChild : zeno::INode {
    virtual void apply() override {
        auto compound = get_input<BulletCompoundShape>("compound");
        auto childShape = get_input<BulletCollisionShape>("childShape");
        auto trans = get_input<BulletTransform>("trans")->trans;

        compound->addChild(trans, std::move(childShape));
        set_output("compound", get_input("compound"));
    }
};

ZENDEFNODE(BulletCompoundAddChild, {
    {"compound", "childShape", "trans"},
    {"compound"},
    {},
    {"Bullet"},
});

struct BulletColShapeCalcLocalInertia : zeno::INode {
	virtual void apply() override {
		auto isCompound = (std::get<std::string>(get_param("isCompound")) == "true");
		auto mass = get_input2<float>("mass");
		auto localInertia = zeno::IObject::make<zeno::NumericObject>();
		btVector3 lInertia;
		if (isCompound){
			auto colObject = get_input<BulletCompoundShape>("colObject");
			colObject->shape->calculateLocalInertia(btScalar(mass), lInertia);
		}
		else {
			auto colObject = get_input<BulletCollisionShape>("colObject");
			colObject->shape->calculateLocalInertia(btScalar(mass), lInertia);
		}

		localInertia->set<zeno::vec3f>(zeno::vec3f(lInertia[0], lInertia[1], lInertia[2]));
		set_output("localInertia", std::move(localInertia));
	}
};

ZENDEFNODE(BulletColShapeCalcLocalInertia, {
	{"colObject", {"float", "mass", "1"}},
	{"localInertia"},
	{{"enum true false", "isCompound", "false"}},
	{"Bullet"}
});

// it moves mesh to CollisionShape
struct BulletMakeConvexMeshShape : zeno::INode {
	virtual void apply() override {
		auto triMesh = &get_input<BulletTriangleMesh>("triMesh")->mesh;
		auto inShape = std::make_unique<btConvexTriangleMeshShape>(triMesh);

		auto shape = std::make_shared<BulletCollisionShape>(std::move(inShape));
		set_output("shape", std::move(shape));
	}
};

ZENDEFNODE(BulletMakeConvexMeshShape, {
	{"triMesh"},
	{"shape"},
	{},
	{"Bullet"},
});

// it moves mesh to CollisionShape
struct BulletMakeConvexHullShape : zeno::INode {
	virtual void apply() override {
#if 1
		auto triMesh = &get_input<BulletTriangleMesh>("triMesh")->mesh;
		auto inShape = std::make_unique<btConvexTriangleMeshShape>(triMesh);
		auto hull = std::make_unique<btShapeHull>(inShape.get());
		auto margin = get_input2<float>("margin");
		hull->buildHull(margin, 0);
		auto convex = std::make_unique<btConvexHullShape>(
		(const btScalar *)hull->getVertexPointer(), hull->numVertices());
		convex->setMargin(btScalar(margin));
#else
		auto prim = get_input<PrimitiveObject>("prim");
auto convexHC = std::make_unique<btConvexHullComputer>();
std::vector<float> vertices;
vertices.reserve(prim->size() * 3);
for (int i = 0; i < prim->size(); i++) {
btVector3 coor = vec_to_other<btVector3>(prim->verts[i]);
vertices.push_back(coor[0]);
vertices.push_back(coor[1]);
vertices.push_back(coor[2]);
}
auto margin = get_input2<float>("margin");
convexHC->compute(vertices.data(), sizeof(float) * 3, vertices.size() / 3, 0.0f, 0.0f);
auto convex = std::make_unique<btConvexHullShape>(
    &(convexHC->vertices[0].getX()), convexHC->vertices.size());
convex->setMargin(btScalar(margin));
#endif

		// auto convex = std::make_unique<btConvexPointCloudShape>();
		// btVector3* points = new btVector3[inShape->getNumVertices()];
		// for(int i=0;i<inShape->getNumVertices(); i++)
		// {
		//     btVector3 v;
		//     inShape->getVertex(i, v);
		//     points[i]=v;
		// }

		auto shape = std::make_shared<BulletCollisionShape>(std::move(convex));
		set_output("shape", std::move(shape));
	}
};

ZENDEFNODE(BulletMakeConvexHullShape, {
	{"triMesh", {"float", "margin", "0"}},
	{"shape"},
	{},
	{"Bullet"},
});

/*
 * Bullet Object
 */
struct BulletMakeObject : zeno::INode {
    virtual void apply() override {
        auto shape = get_input<BulletCollisionShape>("shape");
        auto mass = get_input<zeno::NumericObject>("mass")->get<float>();
        auto trans = get_input<BulletTransform>("trans");
        auto object = std::make_shared<BulletObject>(
            mass, trans->trans, shape);
        object->body->setDamping(0, 0);
        set_output("object", std::move(object));
    }
};

ZENDEFNODE(BulletMakeObject, {
    {"shape", "trans", {"float", "mass", "0"}},
    {"object"},
    {},
    {"Bullet"},
});

struct BulletObjectSetDamping : zeno::INode {
    virtual void apply() override {
        auto object = get_input<BulletObject>("object");
        auto dampLin = get_input2<float>("dampLin");
        auto dampAug = get_input2<float>("dampAug");
        log_debug("set object {} with dampLin={}, dampAug={}", (void*)object.get(), dampLin, dampAug);
        object->body->setDamping(dampLin, dampAug);
        set_output("object", std::move(object));
    }
};

ZENDEFNODE(BulletObjectSetDamping, {
    {"object", {"float", "dampLin", "0"}, {"float", "dampAug", "0"}},
    {"object"},
    {},
    {"Bullet"},
});

struct BulletObjectSetFriction : zeno::INode {
    virtual void apply() override {
        auto object = get_input<BulletObject>("object");
        auto friction = get_input2<float>("friction");
        log_debug("set object {} with friction={}", (void*)object.get(), friction);
        object->body->setFriction(friction);
        set_output("object", std::move(object));
    }
};

ZENDEFNODE(BulletObjectSetFriction, {
    {"object", {"float", "friction", "0"}},
    {"object"},
    {},
    {"Bullet"},
});

struct BulletObjectSetRestitution : zeno::INode {
    virtual void apply() override {
        auto object = get_input<BulletObject>("object");
        auto restitution = get_input2<float>("restitution");
        log_debug("set object {} with restituion={}", (void*)object.get(), restitution);
        object->body->setRestitution(restitution);
        set_output("object", std::move(object));
    }
};

ZENDEFNODE(BulletObjectSetRestitution, {
    {"object", {"float", "restitution", "0"}},
    {"object"},
    {},
    {"Bullet"},
});

struct BulletObjectGetTransform : zeno::INode {
    virtual void apply() override {
        auto obj = get_input<BulletObject>("object");
        auto body = obj->body.get();
        auto trans = std::make_shared<BulletTransform>();

        if (body && body->getMotionState()) {
                body->getMotionState()->getWorldTransform(trans->trans);
        } else {
                trans->trans = static_cast<btCollisionObject *>(body)->getWorldTransform();
        }
        set_output("trans", std::move(trans));
    }
};

ZENDEFNODE(BulletObjectGetTransform, {
    {"object"},
    {"trans"},
    {},
    {"Bullet"},
});

struct BulletInverseTransform : zeno::INode {
	virtual void apply() override {
		auto trans = get_input<BulletTransform>("trans");
		btTransform t = trans->trans.inverse();
		trans->trans.setOrigin(t.getOrigin());
		trans->trans.setRotation(t.getRotation());
		set_output("trans_inv", std::move(trans));
	}
};

ZENDEFNODE(BulletInverseTransform, {
	{"trans"},
	{"trans_inv"},
	{},
	{"Bullet"}
});

struct BulletObjectGetVel : zeno::INode {
    virtual void apply() override {
        auto obj = get_input<BulletObject>("object");
        auto body = obj->body.get();
        auto linearVel = zeno::IObject::make<zeno::NumericObject>();
        auto angularVel = zeno::IObject::make<zeno::NumericObject>();
        linearVel->set<zeno::vec3f>(zeno::vec3f(0));
        angularVel->set<zeno::vec3f>(zeno::vec3f(0));

        if (body && body->getLinearVelocity() ) {
            auto v = body->getLinearVelocity();
            linearVel->set<zeno::vec3f>(zeno::vec3f(v.x(), v.y(), v.z()));
        }
        if (body && body->getAngularVelocity() ){
            auto w = body->getAngularVelocity();
            angularVel->set<zeno::vec3f>(zeno::vec3f(w.x(), w.y(), w.z()));
        }
        set_output("linearVel", linearVel);
        set_output("angularVel", angularVel);
    }
};

ZENDEFNODE(BulletObjectGetVel, {
    {"object"},
    {"linearVel", "angularVel"},
    {},
    {"Bullet"},
});

struct RigidVelToPrimitive : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto com = get_input<zeno::NumericObject>("centroid")->get<zeno::vec3f>();
        auto lin = get_input<zeno::NumericObject>("linearVel")->get<zeno::vec3f>();
        auto ang = get_input<zeno::NumericObject>("angularVel")->get<zeno::vec3f>();

        auto &pos = prim->attr<zeno::vec3f>("pos");
        auto &vel = prim->add_attr<zeno::vec3f>("vel");
        #pragma omp parallel for
        for (size_t i = 0; i < prim->size(); i++) {
            vel[i] = lin + zeno::cross(ang, pos[i] - com);
        }

        set_output("prim", get_input("prim"));
    }
};

ZENDEFNODE(RigidVelToPrimitive, {
    {"prim", "centroid", "linearVel", "angularVel"},
    {"prim"},
    {},
    {"Bullet"},
});

struct BulletExtractTransform : zeno::INode {
    virtual void apply() override {
        auto trans = &get_input<BulletTransform>("trans")->trans;
        auto origin = std::make_shared<zeno::NumericObject>();
        auto rotation = std::make_shared<zeno::NumericObject>();
        origin->set(vec3f(other_to_vec<3>(trans->getOrigin())));
        rotation->set(vec4f(other_to_vec<4>(trans->getRotation())));
        set_output("origin", std::move(origin));
        set_output("rotation", std::move(rotation));
    }
};

ZENDEFNODE(BulletExtractTransform, {
    {"trans"},
    {{"vec3f","origin"}, {"vec4f", "rotation"}},
    {},
    {"Bullet"},
});


/*static class btTaskSchedulerManager {
	btAlignedObjectArray<btITaskScheduler*> m_taskSchedulers;
	btAlignedObjectArray<btITaskScheduler*> m_allocatedTaskSchedulers;

public:
	btTaskSchedulerManager() {}
	void init()
	{
		addTaskScheduler(btGetSequentialTaskScheduler());
#if BT_THREADSAFE
		if (btITaskScheduler* ts = btCreateDefaultTaskScheduler())
		{
			m_allocatedTaskSchedulers.push_back(ts);
			addTaskScheduler(ts);
		}
		addTaskScheduler(btGetOpenMPTaskScheduler());
		addTaskScheduler(btGetTBBTaskScheduler());
		addTaskScheduler(btGetPPLTaskScheduler());
		if (getNumTaskSchedulers() > 1)
		{
			// prefer a non-sequential scheduler if available
			btSetTaskScheduler(m_taskSchedulers[1]);
		}
		else
		{
			btSetTaskScheduler(m_taskSchedulers[0]);
		}
#endif  // #if BT_THREADSAFE
	}
	void shutdown()
	{
		for (int i = 0; i < m_allocatedTaskSchedulers.size(); ++i)
		{
			delete m_allocatedTaskSchedulers[i];
		}
		m_allocatedTaskSchedulers.clear();
	}

	void addTaskScheduler(btITaskScheduler* ts)
	{
		if (ts)
		{
#if BT_THREADSAFE
			// if initial number of threads is 0 or 1,
			if (ts->getNumThreads() <= 1)
			{
				// for OpenMP, TBB, PPL set num threads to number of logical cores
				ts->setNumThreads(ts->getMaxNumThreads());
			}
#endif  // #if BT_THREADSAFE
			m_taskSchedulers.push_back(ts);
		}
	}
	int getNumTaskSchedulers() const { return m_taskSchedulers.size(); }
	btITaskScheduler* getTaskScheduler(int i) { return m_taskSchedulers[i]; }
} gTaskSchedulerMgr; */

struct BulletMakeConstraint : zeno::INode {
    virtual void apply() override {
        auto constraintType = std::get<std::string>(get_param("constraintType"));
        auto obj1 = get_input<BulletObject>("obj1");
		if (has_input("obj2")) {
			auto obj2 = get_input<BulletObject>("obj2");
			auto cons = std::make_shared<BulletConstraint>(obj1->body.get(), obj2->body.get(), constraintType);
			set_output("constraint", std::move(cons));
		}
		else{
			auto cons = std::make_shared<BulletConstraint>(obj1->body.get(), constraintType);
			set_output("constraint", std::move(cons));
		}
    }
};

ZENDEFNODE(BulletMakeConstraint, {
    {"obj1", "obj2"},
    {"constraint"},
    {{"enum ConeTwist Fixed Gear Generic6Dof Generic6DofSpring Generic6DofSpring2 Hinge Hinge2 Point2Point Slider Universal", "constraintType", "Fixed"}},
    {"Bullet"},
});


struct BulletConstraintSetBreakThres : zeno::INode {
    virtual void apply() override {
        auto cons = get_input<BulletConstraint>("constraint");
        cons->setBreakingThreshold(get_input2<float>("threshold"));
        set_output("constraint", std::move(cons));
    }
};

ZENDEFNODE(BulletConstraintSetBreakThres, {
    {"constraint", {"float", "threshold", "3.0"}},
    {"constraint"},
    {},
    {"Bullet"},
});

struct BulletConstraintSetFrames : zeno::INode {
    virtual void apply() override {
        auto cons = get_input<BulletConstraint>("constraint");
	    auto frame1 = get_input<BulletTransform>("frame1");
        auto frame2 = get_input<BulletTransform>("frame2");
        auto constraintType = std::get<std::string>(get_param("constraintType"));

        if (constraintType == "ConeTwist") {
            dynamic_cast<btConeTwistConstraint *>(cons->constraint.get())->setFrames(frame1->trans, frame2->trans);
        }
        else if (constraintType == "Fixed") {
            dynamic_cast<btFixedConstraint *>(cons->constraint.get())->setFrames(frame1->trans, frame2->trans);
        }
        else if (constraintType == "Generic6Dof") {
            dynamic_cast<btGeneric6DofConstraint *>(cons->constraint.get())->setFrames(frame1->trans, frame2->trans);
        }
        else if (constraintType == "Generic6DofSpring") {
            dynamic_cast<btGeneric6DofSpringConstraint *>(cons->constraint.get())->setFrames(frame1->trans, frame2->trans);
        }
        else if (constraintType == "Generic6DofSpring2") {
            dynamic_cast<btGeneric6DofSpring2Constraint *>(cons->constraint.get())->setFrames(frame1->trans, frame2->trans);
        }
        else if (constraintType == "Hinge") {
            dynamic_cast<btHingeConstraint *>(cons->constraint.get())->setFrames(frame1->trans, frame2->trans);
        }
        else if (constraintType == "Hinge2") {
            dynamic_cast<btHinge2Constraint *>(cons->constraint.get())->setFrames(frame1->trans, frame2->trans);
        }
        else if (constraintType == "Slider") {
            dynamic_cast<btSliderConstraint *>(cons->constraint.get())->setFrames(frame1->trans, frame2->trans);
        }
        else if (constraintType == "Universal") {
            dynamic_cast<btUniversalConstraint *>(cons->constraint.get())->setFrames(frame1->trans, frame2->trans);
        }
        set_output("constraint", std::move(cons));
    }
};

ZENDEFNODE(BulletConstraintSetFrames, {
    {"constraint", "frame1", "frame2"},
    {"constraint"},
    {{"enum ConeTwist Fixed Generic6Dof Generic6DofSpring Generic6DofSpring2 Hinge Hinge2 Slider Universal", "constraintType", "Universal"}},
    {"Bullet"},
});

struct BulletConstraintGetFrames : zeno::INode {
	virtual void apply() override {
		auto cons = get_input<BulletConstraint>("constraint");
		auto constraintType = std::get<std::string>(get_param("constraintType"));

		auto frame1 = std::make_shared<BulletTransform>();
		auto frame2 = std::make_shared<BulletTransform>();

		if (constraintType == "ConeTwist") {
			frame1->trans = dynamic_cast<btConeTwistConstraint *>(cons->constraint.get())->getFrameOffsetA();
			frame2->trans = dynamic_cast<btConeTwistConstraint *>(cons->constraint.get())->getFrameOffsetB();
		}
		else if (constraintType == "Fixed") {
			frame1->trans = dynamic_cast<btFixedConstraint *>(cons->constraint.get())->getFrameOffsetA();
			frame2->trans = dynamic_cast<btFixedConstraint *>(cons->constraint.get())->getFrameOffsetB();
		}
		else if (constraintType == "Generic6Dof") {
			frame1->trans = dynamic_cast<btGeneric6DofConstraint *>(cons->constraint.get())->getFrameOffsetA();
			frame2->trans = dynamic_cast<btGeneric6DofConstraint *>(cons->constraint.get())->getFrameOffsetB();
		}
		else if (constraintType == "Generic6DofSpring") {
			frame1->trans = dynamic_cast<btGeneric6DofSpringConstraint *>(cons->constraint.get())->getFrameOffsetA();
			frame2->trans = dynamic_cast<btGeneric6DofSpringConstraint *>(cons->constraint.get())->getFrameOffsetB();
		}
		else if (constraintType == "Generic6DofSpring2") {
			frame1->trans = dynamic_cast<btGeneric6DofSpring2Constraint *>(cons->constraint.get())->getFrameOffsetA();
			frame2->trans = dynamic_cast<btGeneric6DofSpring2Constraint *>(cons->constraint.get())->getFrameOffsetB();
		}
		else if (constraintType == "Hinge") {
			std::cout<< "get constraint " << (void *)cons.get() << std::endl;
			frame1->trans = dynamic_cast<btHingeConstraint *>(cons->constraint.get())->getFrameOffsetA();
			frame2->trans = dynamic_cast<btHingeConstraint *>(cons->constraint.get())->getFrameOffsetB();
			std::cout << "frame1:" << frame1->trans.getOrigin()[0] << " " <<  frame1->trans.getOrigin()[1] << " " << frame1->trans.getOrigin()[2] << std::endl;
			std::cout << "frame2:" << frame2->trans.getOrigin()[0] << " " <<  frame2->trans.getOrigin()[1] << " " << frame2->trans.getOrigin()[2] << std::endl;
		}
		else if (constraintType == "Hinge2") {
			frame1->trans = dynamic_cast<btHinge2Constraint *>(cons->constraint.get())->getFrameOffsetA();
			frame2->trans = dynamic_cast<btHinge2Constraint *>(cons->constraint.get())->getFrameOffsetB();
		}
		else if (constraintType == "Slider") {
			frame1->trans = dynamic_cast<btSliderConstraint *>(cons->constraint.get())->getFrameOffsetA();
			frame2->trans = dynamic_cast<btSliderConstraint *>(cons->constraint.get())->getFrameOffsetB();
		}
		else if (constraintType == "Universal") {
			frame1->trans = dynamic_cast<btUniversalConstraint *>(cons->constraint.get())->getFrameOffsetA();
			frame2->trans = dynamic_cast<btUniversalConstraint *>(cons->constraint.get())->getFrameOffsetB();
		}

		set_output("frame1", std::move(frame1));
		set_output("frame2", std::move(frame2));
	}
};

ZENDEFNODE(BulletConstraintGetFrames, {
	{"constraint"},
	{"frame1", "frame2"},
	{{"enum ConeTwist Fixed Generic6Dof Generic6DofSpring Generic6DofSpring2 Hinge Hinge2 Slider Universal", "constraintType", "Universal"}},
	{"Bullet"}
});

struct BulletConstraintSetLimitByAxis : zeno::INode {
    virtual void apply() override {
        auto cons = get_input<BulletConstraint>("constraint");
        auto constraintType = std::get<std::string>(get_param("constraintType"));
        auto axisId = std::get<std::string>(get_param("axisId"));
        int axis;
        if (axisId == "linearX"){axis = 0;}
        else if (axisId == "linearY"){axis = 1;}
        else if (axisId == "linearZ"){axis = 2;}
        else if (axisId == "angularX"){axis = 3;}
        else if (axisId == "angularY"){axis = 4;}
        else {axis=5;} // "angularZ"

        auto low = btScalar(get_input2<float>("lowLimit"));
        auto high = btScalar(get_input2<float>("highLimit"));

		if (has_input("axisId")){
			if (constraintType == "ConeTwist") {
				if ((high > low) && (high - low) < 1e-5) {
					dynamic_cast<btConeTwistConstraint *>(cons->constraint.get())->setLimit(axis, low); // axis >= 3
				}
			}
			else if (constraintType == "Fixed") {
				dynamic_cast<btFixedConstraint *>(cons->constraint.get())->setLimit(axis, low, high);
			}
			else if (constraintType == "Generic6Dof") {
				dynamic_cast<btGeneric6DofConstraint *>(cons->constraint.get())->setLimit(axis, low, high);
			}
			else if (constraintType == "Generic6DofSpring") {
				dynamic_cast<btGeneric6DofSpringConstraint *>(cons->constraint.get())->setLimit(axis, low, high);
			}
			else if (constraintType == "Generic6DofSpring2") {
				dynamic_cast<btGeneric6DofSpring2Constraint *>(cons->constraint.get())->setLimit(axis, low, high);
			}
			else if (constraintType == "Hinge2") {
				dynamic_cast<btHinge2Constraint *>(cons->constraint.get())->setLimit(axis, low, high);
			}
			else if (constraintType == "Slider") {
				if (axis < 3) {
					dynamic_cast<btSliderConstraint *>(cons->constraint.get())->setLowerLinLimit(low);
					dynamic_cast<btSliderConstraint *>(cons->constraint.get())->setUpperLinLimit(high);
				}
				else{
					dynamic_cast<btSliderConstraint *>(cons->constraint.get())->setLowerAngLimit(low);
					dynamic_cast<btSliderConstraint *>(cons->constraint.get())->setUpperAngLimit(high);
				}

			}
			else if (constraintType == "Universal") {
				dynamic_cast<btUniversalConstraint *>(cons->constraint.get())->setLimit(axis, low, high);
			}
		}
		else {
			if (constraintType == "Hinge") {
				dynamic_cast<btHingeConstraint *>(cons->constraint.get())->setLimit(low, high);
			}
		}

        set_output("constraint", std::move(cons));
    }
};

ZENDEFNODE(BulletConstraintSetLimitByAxis, {
    {"constraint", "lowLimit", "highLimit"},
    {"constraint"},
    {{"enum ConeTwist Fixed Generic6Dof Generic6DofSpring Generic6DofSpring2 Hinge Hinge2 Slider Universal", "constraintType", "Universal"}, {"enum linearX linearY linearZ angularX angularY angularZ", "axisId", "linearX"}},
    {"Bullet"},
});

struct BulletConstraintSetRefFrameA : zeno::INode {
    virtual void apply() override {
        auto cons = get_input<BulletConstraint>("constraint");
        auto constraintType = std::get<std::string>(get_param("constraintType"));
        auto useLinearReferenceFrameA = std::get<std::string>(get_param("useReferenceFrameA"));

		bool flag = (useLinearReferenceFrameA == "true");

		// std::cout << "check useLinearReferenceFrameA bool: " << flag << std::endl;

        if (constraintType == "Generic6Dof") {
			dynamic_cast<btGeneric6DofConstraint *>(cons->constraint.get())->setUseLinearReferenceFrameA(flag);
        }
        else if (constraintType == "Generic6DofSpring") {
            dynamic_cast<btGeneric6DofSpringConstraint *>(cons->constraint.get())->setUseLinearReferenceFrameA(flag);
        }
        else if (constraintType == "Hinge") {
            dynamic_cast<btHingeConstraint *>(cons->constraint.get())->setUseReferenceFrameA(flag); // NAME DIFFERENCE
        }
        else if (constraintType == "Universal") {
            dynamic_cast<btUniversalConstraint *>(cons->constraint.get())->setUseLinearReferenceFrameA(flag);
        }
        set_output("constraint", std::move(cons));
    }
};

ZENDEFNODE(BulletConstraintSetRefFrameA, {
    {"constraint"},
    {"constraint"},
    {{"enum Generic6Dof Generic6DofSpring Hinge Universal", "constraintType", "Universal"}, {"enum true false", "useReferenceFrameA", "true"}},
    {"Bullet"},
});

struct BulletConstraintSetAxis : zeno::INode {
    virtual void apply() override {
        auto cons = get_input<BulletConstraint>("constraint");
        auto axis1 = zeno::vec_to_other<btVector3>(get_input<zeno::NumericObject>("axis1")->get<zeno::vec3f>());
        auto axis2 = zeno::vec_to_other<btVector3>(get_input<zeno::NumericObject>("axis2")->get<zeno::vec3f>());
        auto constraintType = std::get<std::string>(get_param("constraintType"));


        if (constraintType == "Fixed") {
            dynamic_cast<btFixedConstraint *>(cons->constraint.get())->setAxis(axis1, axis2);
        }
        else if (constraintType == "Gear") {
            dynamic_cast<btGearConstraint *>(cons->constraint.get())->setAxisA(axis1);
            dynamic_cast<btGearConstraint *>(cons->constraint.get())->setAxisB(axis2);
        }
        else if (constraintType == "Generic6Dof") {
            dynamic_cast<btGeneric6DofConstraint *>(cons->constraint.get())->setAxis(axis1, axis2);
        }
        else if (constraintType == "Generic6DofSpring") {
            dynamic_cast<btGeneric6DofSpringConstraint *>(cons->constraint.get())->setAxis(axis1, axis2);
        }
        else if (constraintType == "Generic6DofSpring2") {
            dynamic_cast<btGeneric6DofSpring2Constraint *>(cons->constraint.get())->setAxis(axis1, axis2);
        }
        else if (constraintType == "Hinge") {
            dynamic_cast<btHingeConstraint *>(cons->constraint.get())->setAxis(axis1);
        }
        else if (constraintType == "Hinge2") {
            dynamic_cast<btHinge2Constraint *>(cons->constraint.get())->setAxis(axis1, axis2);
        }
        else if (constraintType == "Point2Point") {
            dynamic_cast<btPoint2PointConstraint *>(cons->constraint.get())->setPivotA(axis1);
            dynamic_cast<btPoint2PointConstraint *>(cons->constraint.get())->setPivotB(axis2);
        }
        else if (constraintType == "Universal") {
            dynamic_cast<btUniversalConstraint *>(cons->constraint.get())->setAxis(axis1, axis2);
        }
        set_output("constraint", std::move(cons));
    }
};

ZENDEFNODE(BulletConstraintSetAxis, {
    {"constraint", "axis1", "axis2"},
    {"constraint"},
    {{"enum Fixed Gear Generic6Dof Generic6DofSpring Generic6DofSpring2 Hinge Hinge2 Point2Point Universal", "constraintType", "Universal"}},
    {"Bullet"},
});

struct BulletConstraintSetSpring : zeno::INode {
    virtual void apply() override {
        auto cons = get_input<BulletConstraint>("constraint");
        auto constraintType = std::get<std::string>(get_param("constraintType"));
        auto axisId = std::get<std::string>(get_param("axisId"));
        int axis;
        if (axisId == "linearX"){axis = 0;}
        else if (axisId == "linearY"){axis = 1;}
        else if (axisId == "linearZ"){axis = 2;}
        else if (axisId == "angularX"){axis = 3;}
        else if (axisId == "angularY"){axis = 4;}
        else {axis=5;} // "angularZ"
        auto enabled = get_input2<bool>("enable");
        auto stiffness = btScalar(get_input2<float>("stiffness"));
        auto damping = btScalar(get_input2<float>("damping"));
        auto epVal = btScalar(get_input2<float>("equilibriumPointVal"));

        if (constraintType == "ConeTwist") {
            dynamic_cast<btConeTwistConstraint *>(cons->constraint.get())->setDamping(damping);

        }
        if (constraintType == "Fixed") {
            dynamic_cast<btFixedConstraint *>(cons->constraint.get())->enableSpring(axis, enabled);
            dynamic_cast<btFixedConstraint *>(cons->constraint.get())->setStiffness(axis, stiffness);
            dynamic_cast<btFixedConstraint *>(cons->constraint.get())->setDamping(axis, damping);
            dynamic_cast<btFixedConstraint *>(cons->constraint.get())->setEquilibriumPoint(axis, epVal);
        }
        else if (constraintType == "Generic6DofSpring") {
            dynamic_cast<btGeneric6DofSpringConstraint *>(cons->constraint.get())->enableSpring(axis, enabled);
            dynamic_cast<btGeneric6DofSpringConstraint *>(cons->constraint.get())->setStiffness(axis, stiffness);
            dynamic_cast<btGeneric6DofSpringConstraint *>(cons->constraint.get())->setDamping(axis, damping);
            dynamic_cast<btGeneric6DofSpringConstraint *>(cons->constraint.get())->setEquilibriumPoint(axis, epVal);
        }
        else if (constraintType == "Generic6DofSpring2") {
            dynamic_cast<btGeneric6DofSpring2Constraint *>(cons->constraint.get())->enableSpring(axis, enabled);
            dynamic_cast<btGeneric6DofSpring2Constraint *>(cons->constraint.get())->setEquilibriumPoint(axis, epVal);
            dynamic_cast<btGeneric6DofSpring2Constraint *>(cons->constraint.get())->setDamping(axis, damping);
            dynamic_cast<btGeneric6DofSpring2Constraint *>(cons->constraint.get())->setStiffness(axis, stiffness);
        }
        else if (constraintType == "Hinge2") {
            dynamic_cast<btHinge2Constraint *>(cons->constraint.get())->enableSpring(axis, enabled);
            dynamic_cast<btHinge2Constraint *>(cons->constraint.get())->setEquilibriumPoint(axis, epVal);
            dynamic_cast<btHinge2Constraint *>(cons->constraint.get())->setDamping(axis, damping);
            dynamic_cast<btHinge2Constraint *>(cons->constraint.get())->setStiffness(axis, stiffness);
        }
        set_output("constraint", std::move(cons));
    }
};

ZENDEFNODE(BulletConstraintSetSpring , {
    {"constraint", {"bool", "enable", "true"}, "stiffness", "damping", "equilibriumPointVal"},
    {"constraint"},
    {{"enum Fixed Generic6DofSpring Generic6DofSpring2 Hinge2", "constraintType", "Fixed"}, {"enum linearX linearY linearZ angularX angularY angularZ", "axisId", "linearX"}},
    {"Bullet"},
});

struct BulletConstraintSetMotor : zeno::INode {
    virtual void apply() override {
        auto cons = get_input<BulletConstraint>("constraint");
        auto constraintType = std::get<std::string>(get_param("constraintType"));
        auto axisId = std::get<std::string>(get_param("axisId"));
        int axis;
        if (axisId == "linearX"){axis = 0;}
        else if (axisId == "linearY"){axis = 1;}
        else if (axisId == "linearZ"){axis = 2;}
        else if (axisId == "angularX"){axis = 3;}
        else if (axisId == "angularY"){axis = 4;}
        else {axis=5;} // "angularZ"

        auto bounce = btScalar(get_input2<float>("bounce"));
        auto enableMotor = get_input2<bool>("enableMotor");
        auto enableServo = get_input2<bool>("enableServo");
        auto maxMotorForce = btScalar(get_input2<float>("maxMotorForce"));
        auto servoTarget = btScalar(get_input2<float>("servoTarget"));
        auto targetVelocity = btScalar(get_input2<float>("targetVelocity"));

        auto maxMotorImpulse = btScalar(get_input2<float>("maxMotorImpulse"));
        auto maxMotorImpulseNormalized = btScalar(get_input2<float>("maxMotorImpulseNormalized"));
        auto motorTarget = zeno::vec_to_other<btQuaternion>(get_input<zeno::NumericObject>("motorTarget")->get<zeno::vec4f>());
        auto motorTargetConstraint = zeno::vec_to_other<btQuaternion>(get_input<zeno::NumericObject>("motorTargetConstraint")->get<zeno::vec4f>());
        auto angularOnly = get_input2<bool>("angularOnly");
        auto fixThresh = btScalar(get_input2<float>("fixThresh"));

        auto dt = btScalar(get_input2<float>("dt"));
        if (constraintType == "ConeTwist") {
            dynamic_cast<btConeTwistConstraint *>(cons->constraint.get())->enableMotor(enableMotor);
            dynamic_cast<btConeTwistConstraint *>(cons->constraint.get())->setMaxMotorImpulse(maxMotorImpulse);
            dynamic_cast<btConeTwistConstraint *>(cons->constraint.get())->setMaxMotorImpulseNormalized(maxMotorImpulseNormalized);
            dynamic_cast<btConeTwistConstraint *>(cons->constraint.get())->setMotorTarget(motorTarget);
            dynamic_cast<btConeTwistConstraint *>(cons->constraint.get())->setMotorTargetInConstraintSpace(motorTargetConstraint);
            dynamic_cast<btConeTwistConstraint *>(cons->constraint.get())->setAngularOnly(angularOnly);

            dynamic_cast<btConeTwistConstraint *>(cons->constraint.get())->setFixThresh(fixThresh);
        }
		else if (constraintType == "Generic6Dof") {
	        dynamic_cast<btGeneric6DofConstraint *>(cons->constraint.get())->getTranslationalLimitMotor()->m_enableMotor[axis] = enableMotor;
	        dynamic_cast<btGeneric6DofConstraint *>(cons->constraint.get())->getTranslationalLimitMotor()->m_targetVelocity[axis] = targetVelocity;
	        dynamic_cast<btGeneric6DofConstraint *>(cons->constraint.get())->getTranslationalLimitMotor()->m_maxMotorForce[axis] = maxMotorForce;
		}
        else if (constraintType == "Generic6DofSpring2") {
            dynamic_cast<btGeneric6DofSpring2Constraint *>(cons->constraint.get())->setBounce(axis, bounce);
            dynamic_cast<btGeneric6DofSpring2Constraint *>(cons->constraint.get())->enableMotor(axis, enableMotor);
            dynamic_cast<btGeneric6DofSpring2Constraint *>(cons->constraint.get())->setServo(axis, enableServo);
            dynamic_cast<btGeneric6DofSpring2Constraint *>(cons->constraint.get())->setMaxMotorForce(axis, maxMotorForce);
            dynamic_cast<btGeneric6DofSpring2Constraint *>(cons->constraint.get())->setServoTarget(axis, servoTarget);
            dynamic_cast<btGeneric6DofSpring2Constraint *>(cons->constraint.get())->setTargetVelocity(axis, targetVelocity);
        }
        else if (constraintType == "Hinge")
        {
            dynamic_cast<btHingeConstraint *>(cons->constraint.get())->setAngularOnly(angularOnly);
            dynamic_cast<btHingeConstraint *>(cons->constraint.get())->setMaxMotorImpulse(maxMotorImpulse);
            dynamic_cast<btHingeConstraint *>(cons->constraint.get())->setMotorTarget(motorTarget, dt);
            dynamic_cast<btHingeConstraint *>(cons->constraint.get())->setMotorTargetVelocity(targetVelocity);
        }
        else if (constraintType == "Hinge2") {
            dynamic_cast<btHinge2Constraint *>(cons->constraint.get())->setBounce(axis, bounce);
            dynamic_cast<btHinge2Constraint *>(cons->constraint.get())->enableMotor(axis, enableMotor);
            dynamic_cast<btHinge2Constraint *>(cons->constraint.get())->setServo(axis, enableServo);
            dynamic_cast<btHinge2Constraint *>(cons->constraint.get())->setMaxMotorForce(axis, maxMotorForce);
            dynamic_cast<btHinge2Constraint *>(cons->constraint.get())->setServoTarget(axis, servoTarget);
            dynamic_cast<btHinge2Constraint *>(cons->constraint.get())->setTargetVelocity(axis, targetVelocity);
        }
        set_output("constraint", std::move(cons));
    }
};

ZENDEFNODE(BulletConstraintSetMotor , {
    {"constraint", {"float","bounce","0"}, {"bool", "enableMotor", "1"}, {"bool","enableServo","1"}, {"float", "maxMotorForce", "0"}, {"float","servoTarget","0"}, {"float", "targetVelocity", "0"}, {"float","maxMotorImpulse","0"}, {"float","maxMotorImpulseNormalized","0"}, {"vec4f","motorTarget","0,0,0,1"}, {"vec4f","motorTargetConstraint","0,0,0,1"}, {"float","angularOnly","1"}, {"float","fixThresh","0"}, {"float","dt","0"}},
    {"constraint"},
    {{"enum ConeTwist Generic6Dof Generic6DofSpring2 Hinge Hinge2", "constraintType", "ConeTwist"}, {"enum linearX linearY linearZ angularX angularY angularZ", "axisId", "linearX"}},
    {"Bullet"},
});

struct BulletConstraintSetRotOrder : zeno::INode {
    virtual void apply() override {
        auto cons = get_input<BulletConstraint>("constraint");
        auto constraintType = std::get<std::string>(get_param("constraintType"));
        auto rotateOrder = std::get<std::string>(get_param("rotateOrder"));

        RotateOrder rotOrder;
        if (rotateOrder == "XYZ"){
            rotOrder = RO_XYZ;
        }
        else if (rotateOrder == "XZY"){
            rotOrder = RO_XZY;
        }
        else if (rotateOrder == "YXZ"){
            rotOrder = RO_YXZ;
        }
        else if (rotateOrder == "YZX") {
            rotOrder = RO_YZX;
        }
        else if (rotateOrder == "ZXY") {
            rotOrder = RO_ZXY;
        }
        else {
            rotOrder = RO_ZYX;
        }

        if (constraintType == "Generic6DofSpring2") {
            dynamic_cast<btGeneric6DofSpring2Constraint *>(cons->constraint.get())->setRotationOrder(rotOrder);
        }
        else if (constraintType == "Hinge2") {
            dynamic_cast<btHinge2Constraint *>(cons->constraint.get())->setRotationOrder(rotOrder);
        }
        set_output("constraint", std::move(cons));
    }
};

ZENDEFNODE(BulletConstraintSetRotOrder, {
    {"constraint"},
    {"constraint"},
    {{"enum Generic6DofSpring2 Hinge2", "constraintType", "Hinge2"}, {"enum XYZ XZY YXZ YZX ZXY ZYX", "rotateOrder", "XYZ"}},
    {"Bullet"},
});


struct BulletGearConstraintSetRatio : zeno::INode {
    virtual void apply() override {
        auto cons = get_input<BulletConstraint>("constraint");
        auto ratio = btScalar(get_input2<float>("ratio"));
        dynamic_cast<btGearConstraint *>(cons->constraint.get())->setRatio(ratio);
        set_output("constraint", std::move(cons));
    }
};

ZENDEFNODE(BulletGearConstraintSetRatio, {
    {"constraint", {"float", "ratio", "1"}},
    {"constraint"},
    {},
    {"Bullet"},
});


struct BulletSliderConstraintSetSpring : zeno::INode {
    virtual void apply() override {
        auto cons = get_input<BulletConstraint>("constraint");
        auto dampingDirAng = btScalar(get_input2<float>("dampingDirAng"));
        auto dampingDirLin = btScalar(get_input2<float>("dampingDirLin"));
        auto dampingLimAng = btScalar(get_input2<float>("dampingLimAng"));
        auto dampingLimLin = btScalar(get_input2<float>("dampingLimLin"));
        auto dampingOrthoAng = btScalar(get_input2<float>("dampingOrthoAng"));
        auto dampingOrthoLin = btScalar(get_input2<float>("dampingOrthoLin"));
        auto maxAngMotorForce = btScalar(get_input2<float>("maxAngMotorForce"));
        auto maxLinMotorForce = btScalar(get_input2<float>("maxLinMotorForce"));
        auto poweredAngMotor = get_input2<bool>("poweredAngMotor");
        auto poweredLinMotor = get_input2<bool>("poweredLinMotor");
        auto restitutionDirAng = btScalar(get_input2<float>("restitutionDirAng"));
        auto restitutionDirLin = btScalar(get_input2<float>("restitutionDirLin"));
        auto restitutionLimAng = btScalar(get_input2<float>("restitutionLimAng"));
        auto restitutionLimLin = btScalar(get_input2<float>("restitutionLimLin"));
        auto restitutionOrthoAng = btScalar(get_input2<float>("restitutionOrthoAng"));
        auto restitutionOrthoLin = btScalar(get_input2<float>("restitutionOrthoLin"));
        auto softnessDirAng = btScalar(get_input2<float>("softnessDirAng"));
        auto softnessDirLin = btScalar(get_input2<float>("softnessDirLin"));
        auto softnessLimAng = btScalar(get_input2<float>("softnessLimAng"));
        auto softnessLimLin = btScalar(get_input2<float>("softnessLimLin"));
        auto softnessOrthoAng = btScalar(get_input2<float>("softnessOrthoAng"));
        auto softnessOrthoLin = btScalar(get_input2<float>("softnessOrthoLin"));
        auto targetAngMotorVelocity = btScalar(get_input2<float>("targetAngMotorVelocity"));
        auto targetLinMotorVelocity = btScalar(get_input2<float>("targetLinMotorVelocity"));

        dynamic_cast<btSliderConstraint *>(cons->constraint.get())->setDampingDirAng(dampingDirAng);
        dynamic_cast<btSliderConstraint *>(cons->constraint.get())->setDampingDirLin(dampingDirLin);

        dynamic_cast<btSliderConstraint *>(cons->constraint.get())->setDampingLimAng(dampingLimAng);
        dynamic_cast<btSliderConstraint *>(cons->constraint.get())->setDampingLimLin(dampingLimLin);
        dynamic_cast<btSliderConstraint *>(cons->constraint.get())->setDampingOrthoAng(dampingOrthoAng);
        dynamic_cast<btSliderConstraint *>(cons->constraint.get())->setDampingOrthoLin(dampingOrthoLin);

        dynamic_cast<btSliderConstraint *>(cons->constraint.get())->setMaxAngMotorForce(maxAngMotorForce);
        dynamic_cast<btSliderConstraint *>(cons->constraint.get())->setMaxLinMotorForce(maxLinMotorForce);
        dynamic_cast<btSliderConstraint *>(cons->constraint.get())->setPoweredAngMotor(poweredAngMotor);
        dynamic_cast<btSliderConstraint *>(cons->constraint.get())->setPoweredLinMotor(poweredLinMotor);
        dynamic_cast<btSliderConstraint *>(cons->constraint.get())->setRestitutionDirAng(restitutionDirAng);
        dynamic_cast<btSliderConstraint *>(cons->constraint.get())->setRestitutionDirLin(restitutionDirLin);
        dynamic_cast<btSliderConstraint *>(cons->constraint.get())->setRestitutionLimAng(restitutionLimAng);
        dynamic_cast<btSliderConstraint *>(cons->constraint.get())->setRestitutionLimLin(restitutionLimLin);
        dynamic_cast<btSliderConstraint *>(cons->constraint.get())->setRestitutionOrthoAng(restitutionOrthoAng);
        dynamic_cast<btSliderConstraint *>(cons->constraint.get())->setRestitutionOrthoLin(restitutionOrthoLin);
        dynamic_cast<btSliderConstraint *>(cons->constraint.get())->setSoftnessDirAng(softnessDirAng);
        dynamic_cast<btSliderConstraint *>(cons->constraint.get())->setSoftnessDirLin(softnessDirLin);
        dynamic_cast<btSliderConstraint *>(cons->constraint.get())->setSoftnessLimAng(softnessLimAng);
        dynamic_cast<btSliderConstraint *>(cons->constraint.get())->setSoftnessLimLin(softnessLimLin);
        dynamic_cast<btSliderConstraint *>(cons->constraint.get())->setSoftnessOrthoAng(softnessOrthoAng);
        dynamic_cast<btSliderConstraint *>(cons->constraint.get())->setSoftnessOrthoLin(softnessOrthoLin);
        dynamic_cast<btSliderConstraint *>(cons->constraint.get())->setTargetAngMotorVelocity(targetAngMotorVelocity);
        dynamic_cast<btSliderConstraint *>(cons->constraint.get())->setTargetLinMotorVelocity(targetLinMotorVelocity);

        set_output("constraint", std::move(cons));
    }
};

ZENDEFNODE(BulletSliderConstraintSetSpring, {
    {"constraint", "dampingDirAng", "dampingDirLin", "dampingLimAng", "dampingLimLin", "dampingOrthoAng", "dampingOrthoLin", "maxAngMotorForce", "maxLinMotorForce", "poweredAngMotor", "poweredLinMotor", "restitutionDirAng", "poweredLinMotor", "restitutionDirAng", "restitutionDirLin", "restitutionLimAng", "restitutionLimLin", "restitutionOrthoAng", "restitutionOrthoLin", "softnessDirAng", "softnessDirLin", "softnessLimAng", "softnessLimLin", "softnessOrthoAng", "softnessOrthoLin", "targetAngMotorVelocity", "targetLinMotorVelocity"},
    {"constraint"},
    {},
    {"Bullet"},
});

/*
 *  Bullet World
 */

struct BulletMakeWorld : zeno::INode {
    virtual void apply() override {
        auto world = std::make_shared<BulletWorld>();
        set_output("world", std::move(world));
    }
};

ZENDEFNODE(BulletMakeWorld, {
                                {},
                                {"world"},
                                {},
                                {"Bullet"},
                            });

struct BulletWorldSetGravity : zeno::INode {
    virtual void apply() override {
        auto world = get_input<BulletWorld>("world");
        auto gravity = get_input<zeno::NumericObject>("gravity")->get<zeno::vec3f>();
        world->dynamicsWorld->setGravity(zeno::vec_to_other<btVector3>(gravity));
        set_output("world", std::move(world));
    }
};

ZENDEFNODE(BulletWorldSetGravity, {
                                      {"world", {"vec3f", "gravity", "0,0,-9.8"}},
                                      {"world"},
                                      {},
                                      {"Bullet"},
                                  });

struct BulletStepWorld : zeno::INode {
    virtual void apply() override {
        auto world = get_input<BulletWorld>("world");
        auto dt = get_input<zeno::NumericObject>("dt")->get<float>();
        auto steps = get_input<zeno::NumericObject>("steps")->get<int>();
        world->step(dt, steps);
        set_output("world", std::move(world));
    }
};

ZENDEFNODE(BulletStepWorld, {
                                {"world", {"float", "dt", "0.04"}, {"int", "steps", "1"}},
                                {"world"},
                                {},
                                {"Bullet"},
                            });

struct BulletWorldAddObject : zeno::INode {
    virtual void apply() override {
        auto world = get_input<BulletWorld>("world");
        auto object = get_input<BulletObject>("object");
        world->addObject(std::move(object));
        set_output("world", get_input("world"));
    }
};

ZENDEFNODE(BulletWorldAddObject, {
                                     {"world", "object"},
                                     {"world"},
                                     {},
                                     {"Bullet"},
                                 });

struct BulletWorldRemoveObject : zeno::INode {
    virtual void apply() override {
        auto world = get_input<BulletWorld>("world");
        auto object = get_input<BulletObject>("object");
        world->removeObject(std::move(object));
        set_output("world", get_input("world"));
    }
};

ZENDEFNODE(BulletWorldRemoveObject, {
                                        {"world", "object"},
                                        {"world"},
                                        {},
                                        {"Bullet"},
                                    });

struct BulletWorldSetObjList : zeno::INode {
    virtual void apply() override {
        auto world = get_input<BulletWorld>("world");
        auto objList = get_input<ListObject>("objList")->get<std::shared_ptr<BulletObject>>();
        world->setObjectList(std::move(objList));
        set_output("world", get_input("world"));
    }
};

ZENDEFNODE(BulletWorldSetObjList, {
                                      {"world", "objList"},
                                      {"world"},
                                      {},
                                      {"Bullet"},
                                  });

struct BulletWorldAddConstraint : zeno::INode {
    virtual void apply() override {
        auto world = get_input<BulletWorld>("world");
        auto constraint = get_input<BulletConstraint>("constraint");
        world->addConstraint(constraint);
        set_output("world", get_input("world"));
    }
};

ZENDEFNODE(BulletWorldAddConstraint, {
                                         {"world", "constraint"},
                                         {"world"},
                                         {},
                                         {"Bullet"},
                                     });

struct BulletWorldRemoveConstraint : zeno::INode {
    virtual void apply() override {
        auto world = get_input<BulletWorld>("world");
        auto constraint = get_input<BulletConstraint>("constraint");
        world->removeConstraint(std::move(constraint));
        set_output("world", get_input("world"));
    }
};

ZENDEFNODE(BulletWorldRemoveConstraint, {
                                            {"world", "constraint"},
                                            {"world"},
                                            {},
                                            {"Bullet"},
                                        });

struct BulletWorldSetConsList : zeno::INode {
    virtual void apply() override {
        auto world = get_input<BulletWorld>("world");
        auto consList = get_input<ListObject>("consList")->get<std::shared_ptr<BulletConstraint>>();
        world->setConstraintList(std::move(consList));
        set_output("world", get_input("world"));
    }
};

ZENDEFNODE(BulletWorldSetConsList, {
                                       {"world", "consList"},
                                       {"world"},
                                       {},
                                       {"Bullet"},
                                   });


struct BulletObjectApplyForce:zeno::INode {
    virtual void apply() override {
        auto object = get_input<BulletObject>("object");
        auto forceImpulse = get_input<zeno::NumericObject>("ForceImpulse")->get<zeno::vec3f>();
        auto torqueImpulse = get_input<zeno::NumericObject>("TorqueImpulse")->get<zeno::vec3f>();
        object->body->applyCentralImpulse(zeno::vec_to_other<btVector3>(forceImpulse));
        object->body->applyTorqueImpulse(zeno::vec_to_other<btVector3>(torqueImpulse));
    }
};

ZENDEFNODE(BulletObjectApplyForce, {
                                       {"object", {"vec3f", "ForceImpulse", "0,0,0"}, {"vec3f", "TorqueImpulse", "0,0,0"}},
                                       {},
                                       {},
                                       {"Bullet"},
                                   });


/*
 * Bullet MultiBody
 */



struct BulletMultiBodyObjectMakeStart : zeno::INode {
    virtual void apply() override {
	auto n_links = get_input2<int>("nLinks");
	auto mass = get_input<zeno::NumericObject>("mass")->get<float>();
	btVector3 inertia(0.f, 0.f, 0.f);
	if (has_input("inertia"))
		inertia = zeno::vec_to_other<btVector3>(get_input<zeno::NumericObject>("inertia")->get<zeno::vec3f>());
	auto fixedBase = (std::get<std::string>(get_param("fixedBase")) == "true");
	auto canSleep = (std::get<std::string>(get_param("canSleep")) == "true");
	auto object = std::make_shared<BulletMultiBodyObject>(n_links, mass, inertia, fixedBase, canSleep);

	set_output("object", std::move(object));
    }
};

ZENDEFNODE(BulletMultiBodyObjectMakeStart, {
	{"nLinks", {"float", "mass", "0"}, "inertia"},
	{"object"},
	{{"enum true false", "fixedBase", "true"}, {"enum true false", "canSleep", "true"}},
	{"Bullet"}
});

struct BulletMultiBodyObjectMakeEnd : zeno::INode {
    virtual void apply() override {
        auto object = get_input<BulletMultiBodyObject>("object");
        object->multibody->finalizeMultiDof();
        set_output("object", std::move(object));
    }
};

ZENDEFNODE(BulletMultiBodyObjectMakeEnd, {
	{"object"},
	{"object"},
	{},
	{"Bullet"}
});

struct BulletMultiBodySetCollider : zeno::INode {
    virtual void apply() override {
        auto object = get_input<BulletMultiBodyObject>("object");
        auto link_id = get_input2<int>("linkIndex");
        auto collider = get_input<BulletMultiBodyLinkCollider>("collider");
        if (link_id < 0) {
                object->multibody->setBaseCollider(collider->linkCollider.get());
        }
        else{
                object->multibody->getLink(link_id).m_collider = collider->linkCollider.get();
        }
        set_output("object", std::move(object));
    }
};

ZENDEFNODE(BulletMultiBodySetCollider, {
	{"object", "linkIndex", "collider"},
	{"object"},
	{},
	{"Bullet"},
});

struct BulletMultiBodySetupJoint : zeno::INode {
    virtual void apply() override {
        auto object = get_input<BulletMultiBodyObject>("object");
        auto jointType = std::get<std::string>(get_param("jointType"));
        auto i = get_input2<int>("linkIndex");
        auto parent = get_input2<int>("parentIndex");
        auto mass = get_input2<float>("mass");
        auto inertia = zeno::vec_to_other<btVector3>(get_input<zeno::NumericObject>("inertia")->get<zeno::vec3f>());
        auto jointAxis = zeno::vec_to_other<btVector3>(get_input<zeno::NumericObject>("jointAxis")->get<zeno::vec3f>());
        auto rotParentToThis = zeno::vec_to_other<btQuaternion>(get_input<zeno::NumericObject>("rotParentToThis")->get<zeno::vec4f>());
        auto parentComToThisPivotOffset = zeno::vec_to_other<btVector3>(get_input<zeno::NumericObject>("parentComToThisPivotOffset")->get<zeno::vec3f>());
        auto thisPivotToThisComOffset = zeno::vec_to_other<btVector3>(get_input<zeno::NumericObject>("thisPivotToThisComOffset")->get<zeno::vec3f>());
        auto disableParentCollision= (std::get<std::string>(get_param("disableParentCollision")) == "true");

        if (jointType == "Fixed") {
                object->multibody->setupFixed(i, mass, inertia, parent, rotParentToThis, parentComToThisPivotOffset, thisPivotToThisComOffset, disableParentCollision);
        }
        else if (jointType == "Prismatic") {
                object->multibody->setupPrismatic(i, mass, inertia, parent, rotParentToThis, jointAxis, parentComToThisPivotOffset, thisPivotToThisComOffset, disableParentCollision);
        }
        else if (jointType == "Revolute") {
                object->multibody->setupRevolute(i, mass, inertia, parent, rotParentToThis, jointAxis, parentComToThisPivotOffset, thisPivotToThisComOffset, disableParentCollision);
        }
        else if (jointType == "Spherical") {
                object->multibody->setupSpherical(i, mass, inertia, parent, rotParentToThis, parentComToThisPivotOffset, thisPivotToThisComOffset, disableParentCollision);
        }
        else { // planar
                object->multibody->setupPlanar(i, mass, inertia, parent, rotParentToThis, jointAxis, parentComToThisPivotOffset, disableParentCollision);
        }

        set_output("object", std::move(object));
    }
};

ZENDEFNODE(BulletMultiBodySetupJoint, {
	{"object", "linkIndex", "parentIndex", "mass", "inertia", "jointAxis", "rotParentToThis", "parentComToThisPivotOffset", "thisPivotToThisComOffset"},
	{"object"},
	{{"enum Fixed Prismatic Revolute Spherical Planar", "jointType", "Revolute"}, {"enum true false", "disableParentCollision", "false"}},
	{"Bullet"}
});

struct BulletMultiBodySetJointProperty : zeno::INode{
	virtual void apply() override {
		auto object = get_input<BulletMultiBodyObject>("object");
		auto link_id = get_input2<int>("linkIndex");
		if(has_input("damping")){
			auto damping = get_input2<float>("damping");
			object->multibody->getLink(link_id).m_jointDamping = damping;
		}
		if(has_input("friction")){
			auto friction = get_input2<float>("friction");
			object->multibody->getLink(link_id).m_jointFriction = friction;
		}
		if(has_input("lowerLimit")){
			auto lowerLimit = get_input2<float>("lowerLimit");
			object->multibody->getLink(link_id).m_jointLowerLimit = lowerLimit;
		}
		if(has_input("upperLimit")){
			auto upperLimit = get_input2<float>("upperLimit");
			object->multibody->getLink(link_id).m_jointUpperLimit = upperLimit;
		}
		if(has_input("maxForce")){
			auto maxForce = get_input2<float>("maxForce");
			object->multibody->getLink(link_id).m_jointMaxForce = maxForce;
		}
		if(has_input("maxVelocity")){
			auto maxVelocity = get_input2<float>("maxVelocity");
			object->multibody->getLink(link_id).m_jointMaxVelocity = maxVelocity;
		}
		set_output("object", std::move(object));
	}
};

ZENDEFNODE(BulletMultiBodySetJointProperty, {
	{"object", {"int", "linkIndex", "0"}, "damping", "friction", "lowerLimit", "upperLimit", "maxForce", "maxVelocity"},
	{"object"},
	{},
	{"Bullet"}
});

struct BulletMultiBodySetBaseTransform : zeno::INode {
	virtual void apply() override {
		auto object = get_input<BulletMultiBodyObject>("object");
		auto trans = get_input<BulletTransform>("baseTrans");

		object->multibody->setBaseWorldTransform(trans->trans);
		set_output("object", std::move(object));
	}
};

ZENDEFNODE(BulletMultiBodySetBaseTransform, {
	{"object", "baseTrans"},
	{"object"},
	{},
	{"Bullet"}
});

struct BulletMultiBodyLinkGetTransform : zeno::INode {
	virtual void apply() override {
		auto object = get_input<BulletMultiBodyObject>("object");
		auto link_id = get_input2<int>("linkIndex");
        auto trans = std::make_shared<BulletTransform>();
        trans->trans = object->multibody->getLink(link_id).m_collider->getWorldTransform();
		set_output("trans", std::move(trans));
	}
};

ZENDEFNODE(BulletMultiBodyLinkGetTransform, {
	{"object", "linkIndex"},
	{"trans"},
	{},
	{"Bullet"}
});

struct BulletMultiBodyGetTransform : zeno::INode {
    virtual void apply() override {
        auto object = get_input<BulletMultiBodyObject>("object");
        auto transList = std::make_shared<zeno::ListObject>();
        transList->arr.clear();

        for (size_t i = 0; i < object->multibody->getNumLinks(); i++) {
            auto trans = std::make_shared<BulletTransform>();
            trans->trans = object->multibody->getLink(i).m_collider->getWorldTransform();
            std::cout<< "\nlink #" << i << ": " << trans->trans.getOrigin()[0] << "," << trans->trans.getOrigin()[1] << "," << trans->trans.getOrigin()[2];
            transList->arr.push_back(trans);
        }

//        std::cout<<"current Transform: ";
//		for (size_t i = 0; i< object->multibody->getNumLinks(); i++) {
//			btVector3 tmpTrans;
//			btVector3 tmpOri;
//
//			object->multibody->localPosToWorld(i, tmpTrans);
//			object->multibody->localDirToWorld(i, tmpOri);
//            std::cout<< "link #" << i << ": " << tmpTrans[0] << "," << tmpTrans[1] << "," << tmpTrans[2];
//			auto tmpFrame = std::make_shared<BulletTransform>();
//            tmpFrame->trans.setOrigin(tmpTrans);
//            btQuaternion q(tmpOri[0], tmpOri[1], tmpOri[2]);
//            tmpFrame->trans.setRotation(q);
//            transList->arr.push_back(tmpFrame);
//		}
//        std::cout<<std::endl;
        set_output("transList", std::move(transList));
    }
};

ZENDEFNODE(BulletMultiBodyGetTransform, {
    {"object"},
    {"transList"},
    {},
    {"Bullet"}
});

struct BulletMultiBodyWorldGetTransform : zeno::INode {
    // TODO: this part should be more adaptive later!!
	virtual void apply() override {
		auto world = get_input<BulletMultiBodyWorld>("world");
        auto transList = std::make_shared<zeno::ListObject>();
        transList->arr.clear();

        int numCollisionObjects = world->dynamicsWorld->getNumCollisionObjects();
        for (size_t i = 0; i < numCollisionObjects; i++) {
            auto linkTrans = std::make_shared<BulletTransform>();
            btCollisionObject* colObj = world->dynamicsWorld->getCollisionObjectArray()[i];
            btCollisionShape* collisionShape = colObj->getCollisionShape();

            linkTrans->trans = colObj->getWorldTransform();
            int graphicsIndex = colObj->getUserIndex();
            std::cout << "graphicsId #" << graphicsIndex << ":" << linkTrans->trans.getOrigin()[0] << "," << linkTrans->trans.getOrigin()[1] << "," << linkTrans->trans.getOrigin()[2] << "\n";
            if (graphicsIndex >= 0) {
                transList->arr.push_back(linkTrans);
            }
        }

        set_output("transList", std::move(transList));
	}
};

ZENDEFNODE(BulletMultiBodyWorldGetTransform, {
	{"world"},
	{"transList"},
	{},
	{"Bullet"}
});

struct BulletMultiBodySetProperty : zeno::INode {
	virtual void apply() override {
		auto object = get_input<BulletMultiBodyObject>("object");
		auto canSleep = (std::get<std::string>(get_param("canSleep")) == "true");
		auto selfCollide = (std::get<std::string>(get_param("selfCollide")) == "true");
		auto useGyro = (std::get<std::string>(get_param("useGyro")) == "true");
		auto linearDamping = get_input2<float>("linearDamp");
		auto angularDamping = get_input2<float>("angularDamp");

		object->multibody->setCanSleep(canSleep);
		object->multibody->setHasSelfCollision(selfCollide);
		object->multibody->setUseGyroTerm(useGyro);
		object->multibody->setLinearDamping(linearDamping);
		object->multibody->setAngularDamping(angularDamping);

		set_output("object", std::move(object));
	}
};

ZENDEFNODE(BulletMultiBodySetProperty, {
	{"object", {"float", "linearDamp", "0"}, {"float", "angularDamp", "0"}},
	{"object"},
	{{"enum true false", "canSleep", "false"},
		{"enum true false", "selfCollide", "false"},
		{"enum true false", "useGyro", "false"}},
	{"Bullet"}
});


struct BulletMultiBodySetCollisionShapeForLink : zeno::INode {
	virtual void apply() override {
		auto object = get_input<BulletMultiBodyObject>("object");
		auto link_id = get_input2<int>("linkIndex");
		auto collisionShape = get_input<BulletCollisionShape>("colShape");
		auto col = std::make_shared<BulletMultiBodyLinkCollider>(object->multibody.get(), link_id);
		col->linkCollider->setCollisionShape(collisionShape->shape.get());
        if (link_id < 0) {
            object->multibody->setBaseCollider(col->linkCollider.get());
        }
        else {
            object->multibody->getLink(
                link_id).m_collider = col->linkCollider.get();
        }
        set_output("object", std::move(object));
		set_output("collider", std::move(col));
	}
};

ZENDEFNODE(BulletMultiBodySetCollisionShapeForLink, {
	{"object", "linkIndex", "colShape"},
	{"object", "collider"},
	{},
	{"Bullet"},
});

struct BulletMultiBodyLinkColliderSetTransform : zeno::INode {
	virtual void apply() override {
		auto col = get_input<BulletMultiBodyLinkCollider>("collider");
		auto trans = get_input<BulletTransform>("trans");

		col->linkCollider->setWorldTransform(trans->trans);
		set_output("collider", std::move(col));
	}
};

ZENDEFNODE(BulletMultiBodyLinkColliderSetTransform, {
	{"collider", "trans"},
	{"collider"},
	{},
	{"Bullet"},
});

struct BulletMultiBodyForwardKinematics : zeno::INode {
	virtual void apply() override {
		auto object = get_input<BulletMultiBodyObject>("object");
		btAlignedObjectArray<btQuaternion> scratch_q;
		btAlignedObjectArray<btVector3> scratch_m;
		object->multibody->forwardKinematics(scratch_q, scratch_m);
		set_output("object", std::move(object));
	}
};

ZENDEFNODE(BulletMultiBodyForwardKinematics, {
	{"object"},
	{"object"},
	{},
	{"Bullet"},
});

struct BulletMultiBodyUpdateColObjectTransform : zeno::INode {
	virtual void apply() override {
		auto object = get_input<BulletMultiBodyObject>("object");
		btAlignedObjectArray<btQuaternion> world_to_local;
		btAlignedObjectArray<btVector3> local_origin;
		object->multibody->updateCollisionObjectWorldTransforms(world_to_local, local_origin);
		set_output("object", std::move(object));
	}
};

ZENDEFNODE(BulletMultiBodyUpdateColObjectTransform, {
	{"object"},
	{"object"},
	{},
	{"Bullet"},
});

struct BulletMultiBodyAddJointTorque : zeno::INode {
	virtual void apply() override {
		auto object = get_input<BulletMultiBodyObject>("object");
		auto link_id = get_input2<int>("linkIndex");
		auto torque = get_input2<float>("torque");

		object->multibody->addJointTorque(link_id, torque);
		set_output("object", std::move(object));
	}
};

ZENDEFNODE(BulletMultiBodyAddJointTorque, {
	{"object", "linkIndex", "torque"},
	{"object"},
	{},
	{"Bullet"},
});

struct BulletMultiBodySetJointPosMultiDof : zeno::INode {
	virtual void apply() override {
		auto object = get_input<BulletMultiBodyObject>("object");
		auto link_id = get_input2<int>("startIndex");
		auto isSpherical = (std::get<std::string>(get_param("isSpherical")) == "true");
		if (isSpherical){
			auto pos = zeno::vec_to_other<btQuaternion>(get_input<zeno::NumericObject>("pos")->get<zeno::vec4f>());
			pos.normalize();
			object->multibody->setJointPosMultiDof(link_id, pos);
		}
		else{
			auto pos = get_input2<float>("pos");
			object->multibody->setJointPosMultiDof(link_id, &pos);
		}

		set_output("object", std::move(object));
	}
};

ZENDEFNODE(BulletMultiBodySetJointPosMultiDof, {
	{"object", {"int", "startIndex", "0"}, "pos"},
	{"object"},
	{{"enum true false", "isSpherical", "false"}},
	{"Bullet"},
});

struct BulletMultiBodySetJointVelMultiDof : zeno::INode {
	virtual void apply() override {
		auto object = get_input<BulletMultiBodyObject>("object");
		auto link_id = get_input2<int>("startIndex");
		auto isSpherical = (std::get<std::string>(get_param("isSpherical")) == "true");
		if (isSpherical){
			auto pos = zeno::vec_to_other<btQuaternion>(get_input<zeno::NumericObject>("pos")->get<zeno::vec4f>());
			pos.normalize();
			object->multibody->setJointVelMultiDof(link_id, pos);
		}
		else{
			auto pos = get_input2<float>("pos");
			object->multibody->setJointVelMultiDof(link_id, &pos);
		}

		set_output("object", std::move(object));
	}
};

ZENDEFNODE(BulletMultiBodySetJointVelMultiDof, {
	{"object", {"int", "startIndex", "0"}, "pos"},
	{"object"},
	{{"enum true false", "isSpherical", "false"}},
	{"Bullet"},
});


struct BulletMultiBodySetJointFeedback : zeno::INode {
	virtual void apply() override {
		auto object = get_input<BulletMultiBodyObject>("object");

		auto fb = std::make_shared<MultiBodyJointFeedback>();

		for (int i = 0; i < object->multibody->getNumLinks(); i++)
		{
			object->multibody->getLink(i).m_jointFeedback = &fb->jointFeedback;
			object->jointFeedbacks.push_back(&fb->jointFeedback);
		}

		set_output("object", std::move(object));
	}
};

ZENDEFNODE(BulletMultiBodySetJointFeedback, {
	{"object"},
	{"object"},
	{},
	{"Bullet"},
});

struct BulletMultiBodyPDControl : zeno::INode {
	virtual void apply() override {
		auto object = get_input<BulletMultiBodyObject>("object");
		auto kp = get_input2<float>("kp");
		auto kd = get_input2<float>("kd");
		auto maxForce = get_input2<float>("maxForce");

		btAlignedObjectArray<btScalar> qDesiredArray;
		btAlignedObjectArray<btScalar> qdDesiredArray;

        if (has_input("qDesiredList")){
            {
                auto numericObjs = get_input<zeno::ListObject>(
                    "qDesiredList")->get<std::shared_ptr<NumericObject>>();
                for (auto &&no: numericObjs)
                    qDesiredArray.push_back(no->get<float>());
            }
		}
        else{
            qDesiredArray.resize(object->multibody->getNumLinks(), 0);
        }
        std::cout << "check target pose: ";
        for (size_t i = 0; i< qDesiredArray.size(); i++) {
            std::cout << qDesiredArray[i] << "," ;
        }
        std::cout << std::endl;
		if (has_input("dqDesiredList")) {
            {
                auto numericObjs = get_input<zeno::ListObject>(
                    "dqDesiredList")->get<std::shared_ptr<NumericObject>>();
                for (auto &&no: numericObjs)
                    qdDesiredArray.push_back(no->get<float>());
            }
		}
        else {
            qdDesiredArray.resize(object->multibody->getNumLinks(), 0);
        }

        std::cout << std::endl;
        for (size_t i = 0; i< qDesiredArray.size(); i++){
            std::cout << qDesiredArray[i] << ", ";
        }
        std::cout << std::endl;

		for (int joint = 0; joint < object->multibody->getNumLinks(); joint++)
		{
			int dof1 = 0;
			btScalar qActual = object->multibody->getJointPosMultiDof(joint)[dof1];
			btScalar qdActual = object->multibody->getJointVelMultiDof(joint)[dof1];
			btScalar positionError = (qDesiredArray[joint] - qActual);
			btScalar velocityError = (qdDesiredArray[joint] - qdActual);
			btScalar force = kp * positionError + kd * velocityError;
			btClamp(force, -maxForce, maxForce);
            std::cout << "current force for link #" << joint << " is " << force << std::endl;
			object->multibody->addJointTorque(joint, force);
		}
	}
};

ZENDEFNODE(BulletMultiBodyPDControl, {
	{"object", {"float", "kp", "100"}, {"float", "kd", "20"}, {"float", "maxForce", "100"}, "qDesiredList", "dqDesiredList"},
	{},
	{},
	{"Bullet"},
});
/*
 * Bullet MultiBody World
 */

struct BulletMakeMultiBodyWorld : zeno::INode {
	virtual void apply() override {
		auto solverType = std::get<std::string>(get_param("solverType"));
		auto world = std::make_unique<BulletMultiBodyWorld>(solverType);
		set_output("world", std::move(world));
	}
};

ZENDEFNODE(BulletMakeMultiBodyWorld, {
	{},
	{"world"},
    {{"enum SequentialImpulse ProjectedGaussSeidel Dantzig", "solverType", "SequentialImpulse"}},
	{"Bullet"},
});

struct BulletMultiBodyWorldSetGravity : zeno::INode {
	virtual void apply() override {
		auto world = get_input<BulletMultiBodyWorld>("world");
		auto gravity = get_input<zeno::NumericObject>("gravity")->get<zeno::vec3f>();
		world->dynamicsWorld->setGravity(zeno::vec_to_other<btVector3>(gravity));
		set_output("world", std::move(world));
	}
};

ZENDEFNODE(BulletMultiBodyWorldSetGravity, {
	{"world", {"vec3f", "gravity", "0,0,-9.8"}},
	{"world"},
	{},
	{"Bullet"},
});

struct BulletStepMultiBodyWorld : zeno::INode {
	virtual void apply() override {
		auto world = get_input<BulletMultiBodyWorld>("world");
		auto dt = get_input<zeno::NumericObject>("dt")->get<float>();
		auto steps = get_input<zeno::NumericObject>("maxSubSteps")->get<int>();
        auto fixedTimeStep = get_input<zeno::NumericObject>("fixedTimeStep")->get<float>();

        world->dynamicsWorld->stepSimulation(dt, steps, fixedTimeStep);
		set_output("world", std::move(world));
	}
};

ZENDEFNODE(BulletStepMultiBodyWorld, {
	{"world", {"float", "dt", "0.04"}, {"int", "maxSubSteps", "1"}, {"float", "fixedTimeStep", "0.0042"}},
	{"world"},
	{},
	{"Bullet"},
});

struct BulletMultiBodyWorldAddObject : zeno::INode {
	virtual void apply() override {
		auto world = get_input<BulletMultiBodyWorld>("world");
        auto objType = std::get<std::string>(get_param("objType"));
        if (objType == "multi") {
            auto object = get_input<BulletMultiBodyObject>("object");
            world->dynamicsWorld->addMultiBody(object->multibody.get());
        }
        else {
            auto object = get_input<BulletObject>("object");
            world->dynamicsWorld->addRigidBody(object->body.get());
        }
		set_output("world", std::move(world));
	}
};


ZENDEFNODE(BulletMultiBodyWorldAddObject, {
	{"world", "object"},
	{"world"},
	{{"enum rigid multi", "objType", "multi"}},
	{"Bullet"},
});

struct BulletMultiBodyWorldRemoveObject : zeno::INode {
	virtual void apply() override {
		auto world = get_input<BulletMultiBodyWorld>("world");
        auto objType = std::get<std::string>(get_param("objType"));

        if (objType == "multi") {
            auto object = get_input<BulletMultiBodyObject>("object");
            world->dynamicsWorld->removeMultiBody(object->multibody.get());
        }
        else {
            auto object = get_input<BulletObject>("object");
            world->dynamicsWorld->removeRigidBody(object->body.get());
        }

		set_output("world", get_input("world"));
	}
};

ZENDEFNODE(BulletMultiBodyWorldRemoveObject, {
	{"world", "object"},
	{"world"},
	{{"enum rigid multi", "objType", "multi"}},
	{"Bullet"},
});


struct BulletMultiBodyWorldAddCollisionObject : zeno::INode {
	virtual void apply() override {
		auto world = get_input<BulletMultiBodyWorld>("world");
		auto isDynamic = (std::get<std::string>(get_param("isDynamic")) == "true");
		auto col = get_input<BulletMultiBodyLinkCollider>("collider");
		int collisionFilterGroup = isDynamic ? int(btBroadphaseProxy::DefaultFilter) : int(btBroadphaseProxy::StaticFilter);
		int collisionFilterMask = isDynamic ? int(btBroadphaseProxy::AllFilter) : int(btBroadphaseProxy::AllFilter ^ btBroadphaseProxy::StaticFilter);

        std::cout<< "add collider here: " << (void *)col.get() << std::endl;
		world->dynamicsWorld->addCollisionObject(col->linkCollider.get(), collisionFilterGroup, collisionFilterMask);

		set_output("world", world);
	}
};

ZENDEFNODE(BulletMultiBodyWorldAddCollisionObject, {
	{"world", "collider"},
	{"world"},
	{{"enum true false", "isDynamic", "true"}},
	{"Bullet"},
});


struct BulletMultiBodyMakeConstraint : zeno::INode {
	virtual void apply() override {
		auto constraintType = std::get<std::string>(get_param("constraintType"));
		auto bodyA = get_input<BulletMultiBodyObject>("bodyA");
		auto linkA = get_input2<int>("linkA");

		std::map<std::string, btScalar> config;

		if (has_input("bodyB")) {
			auto bodyB = get_input<BulletMultiBodyObject>("bodyB");
			auto linkB = get_input2<int>("linkB");
			auto cons = std::make_shared<BulletMultiBodyConstraint>(bodyA->multibody.get(), bodyB->multibody.get(), linkA, linkB, constraintType);
			set_output("constraint", cons);
		}
		else{
			if (has_input("lowerLimit")){
				auto lowerLimit = btScalar(get_input2<float>("lowerLimit"));
				config["jointLowerLimit"] = lowerLimit;
			}
			if(has_input("upperLimit")) {
				auto upperLimit = btScalar(get_input2<float>("upperLimit"));
				config["jointUpperLimit"] = upperLimit;
			}
			if(has_input("twistLimit")) {
				auto twistLimit = btScalar(get_input2<float>("twistLimit"));
				config["jointTwistLimit"] = twistLimit;
			}
			if(has_input("jointMaxForce")) {
				auto jointMaxForce = btScalar(get_input2<float>("jointMaxForce"));
				config["jointMaxForce"] = jointMaxForce;
			}
			if(has_input("desiredVelocity")) {
				auto desiredVelocity = btScalar(get_input2<float>("desiredVelocity"));
				config["desiredVelocity"] = desiredVelocity;
			}
			auto cons = std::make_shared<BulletMultiBodyConstraint>(bodyA->multibody.get(), linkA, constraintType, config);
			set_output("constraint", cons);
		}
	}
};

ZENDEFNODE(BulletMultiBodyMakeConstraint, {
	{"bodyA", "linkA", "bodyB", "linkB", "lowerLimit", "upperLimit", "twistLimit", "jointMaxForce", "desiredVelocity"},
	{"constraint"},
	{{"enum Default DefaultMotor Spherical SphericalMotor Fixed Gear Point2Point Slider", "constraintType", "Default"}},
	{"Bullet"},
});

struct BulletMultiBodyWorldAddConstraint : zeno::INode {
	virtual void apply() override {
		auto world = get_input<BulletMultiBodyWorld>("world");
		auto constraint = get_input<BulletMultiBodyConstraint>("constraint");
		world->dynamicsWorld->addMultiBodyConstraint(constraint->constraint.get());
		set_output("world", get_input("world"));
	}
};

ZENDEFNODE(BulletMultiBodyWorldAddConstraint, {
	{"world", "constraint"},
	{"world"},
	{},
	{"Bullet"},
});

struct BulletMultiBodyWorldRemoveConstraint : zeno::INode {
	virtual void apply() override {
		auto world = get_input<BulletMultiBodyWorld>("world");
		auto constraint = get_input<BulletMultiBodyConstraint>("constraint");
		world->dynamicsWorld->removeMultiBodyConstraint(constraint->constraint.get());
		set_output("world", get_input("world"));
	}
};

ZENDEFNODE(BulletMultiBodyWorldRemoveConstraint, {
	{"world", "constraint"},
	{"world"},
	{},
	{"Bullet"},
});

struct BulletMultiBodyWorldAddConstraintEnd : zeno::INode {
    virtual void apply() override {
        auto world = get_input<BulletMultiBodyWorld>("world");
        for (int i = 0; i < world->dynamicsWorld->getNumMultiBodyConstraints(); i++)
        {
            world->dynamicsWorld->getMultiBodyConstraint(i)->finalizeMultiDof();
        }
        set_output("world", std::move(world));
    }
};

ZENDEFNODE(BulletMultiBodyWorldAddConstraintEnd, {
    {"world"},
    {"world"},
    {},
    {"Bullet"}
});
/*
 * Bullet Kinematics
 */

struct BulletCalcInverseKinematics : zeno::INode {
    virtual void apply() override {
        auto object = get_input<BulletMultiBodyObject>("object");
        btVector3 gravity(0, -9.81, 0);
        if (has_input("gravity")) {
            gravity = zeno::vec_to_other<btVector3>(
                get_input<zeno::NumericObject>("gravity")->get<zeno::vec3f>());
        }

        std::vector<int> endEffectorLinkIndices;
        {
            auto numericObjs = get_input<zeno::ListObject>(
                "endEffectorLinkIndices")->get<std::shared_ptr<NumericObject>>();
            for (auto &&no: numericObjs)
                endEffectorLinkIndices.push_back(no->get<int>());
        }

        auto numEndEffectorLinkIndices = endEffectorLinkIndices.size();

        std::vector<vec3f> targetPositions;
        {
            auto numericObjs = get_input<zeno::ListObject>(
                "targetPositions")->get<std::shared_ptr<NumericObject>>();
            for (auto &&no: numericObjs)
                targetPositions.push_back(no->get<vec3f>());
        }

        std::vector<vec4f> targetOrientations;
        {
            auto numericObjs = get_input<zeno::ListObject>("targetOrientations")->get<std::shared_ptr<NumericObject>>();
            for (auto &&no : numericObjs)
                targetOrientations.push_back(no->get<vec4f>());
        }

		auto numIterations = get_input2<int>("numIterations"); // 20
		auto residualThreshold = get_input2<float>("residualThreshold"); // 1e-4

		auto ikMethod = std::get<std::string>(get_param("IKMethod"));

		auto ikHelperPtr = std::make_shared<IKTrajectoryHelper>();

	    btAlignedObjectArray<double> startingPositions;
	    startingPositions.reserve(object->multibody->getNumLinks());

        {
		    for (int i = 0; i < object->multibody->getNumLinks(); ++i)
		    {
			    if (object->multibody->getLink(i).m_jointType >= 0 && object->multibody->getLink(i).m_jointType <= 2)
			    {
				    // 0, 1, 2 represent revolute, prismatic, and spherical joint types respectively. Skip the fixed joints.
				    double curPos = 0;
				    curPos = object->multibody->getJointPos(i);

				    startingPositions.push_back(curPos);
			    }
		    }
	    }

	    btScalar currentDiff = 1e30f;
	    b3AlignedObjectArray<double> endEffectorTargetWorldPositions;
	    b3AlignedObjectArray<double> endEffectorTargetWorldOrientations;
	    b3AlignedObjectArray<double> endEffectorCurrentWorldPositions;
	    b3AlignedObjectArray<double> jacobian_linear;
	    b3AlignedObjectArray<double> jacobian_angular;
	    btAlignedObjectArray<double> q_current;
	    btAlignedObjectArray<double> q_new;
	    btAlignedObjectArray<double> lower_limit;
	    btAlignedObjectArray<double> upper_limit;
	    btAlignedObjectArray<double> joint_range;
	    btAlignedObjectArray<double> rest_pose;
	    const int numDofs = object->multibody->getNumDofs();
	    int baseDofs = object->multibody->hasFixedBase() ? 0 : 6;
	    btInverseDynamics::vecx nu(numDofs + baseDofs), qdot(numDofs + baseDofs), q(numDofs + baseDofs), joint_force(numDofs + baseDofs);

	    endEffectorTargetWorldPositions.resize(0);
	    endEffectorTargetWorldPositions.reserve(numEndEffectorLinkIndices * 3);
	    endEffectorTargetWorldOrientations.resize(0);
	    endEffectorTargetWorldOrientations.reserve(numEndEffectorLinkIndices * 4);

	    bool validEndEffectorLinkIndices = true;

		// setarget position
	    for (int ne = 0; ne < numEndEffectorLinkIndices; ne++)
	    {
		    int endEffectorLinkIndex = endEffectorLinkIndices[ne];
		    validEndEffectorLinkIndices = validEndEffectorLinkIndices && (endEffectorLinkIndex < object->multibody->getNumLinks());

		    btVector3 targetPosWorld(targetPositions[ne][0],
		                             targetPositions[ne][1],
		                             targetPositions[ne][2]);

		    btQuaternion targetOrnWorld(targetOrientations[ne][0],
		                                targetOrientations[ne][1],
		                                targetOrientations[ne][2],
		                                targetOrientations[ne][3]);

		    btTransform targetBaseCoord;
		    btTransform targetWorld;
		    targetWorld.setOrigin(targetPosWorld);
		    targetWorld.setRotation(targetOrnWorld);
		    btTransform tr = object->multibody->getBaseWorldTransform();
		    targetBaseCoord = tr.inverse() * targetWorld;


		    btVector3DoubleData targetPosBaseCoord;
		    btQuaternionDoubleData targetOrnBaseCoord;
		    targetBaseCoord.getOrigin().serializeDouble(targetPosBaseCoord);
		    targetBaseCoord.getRotation().serializeDouble(targetOrnBaseCoord);

		    endEffectorTargetWorldPositions.push_back(targetPosBaseCoord.m_floats[0]);
		    endEffectorTargetWorldPositions.push_back(targetPosBaseCoord.m_floats[1]);
		    endEffectorTargetWorldPositions.push_back(targetPosBaseCoord.m_floats[2]);

		    endEffectorTargetWorldOrientations.push_back(targetOrnBaseCoord.m_floats[0]);
		    endEffectorTargetWorldOrientations.push_back(targetOrnBaseCoord.m_floats[1]);
		    endEffectorTargetWorldOrientations.push_back(targetOrnBaseCoord.m_floats[2]);
		    endEffectorTargetWorldOrientations.push_back(targetOrnBaseCoord.m_floats[3]);
	    }
		// IK iteration
	    for (int i = 0; i < numIterations && currentDiff > residualThreshold; i++)
	    {
            std::cout << "currentDiff: " << currentDiff << std::endl;
		    if (ikHelperPtr && validEndEffectorLinkIndices)
		    {
			    jacobian_linear.resize(numEndEffectorLinkIndices * 3 * numDofs);
			    jacobian_angular.resize(numEndEffectorLinkIndices * 3 * numDofs);
			    int jacSize = 0;

                btInverseDynamics::MultiBodyTree* tree = 0;
                btInverseDynamics::btMultiBodyTreeCreator id_creator;
                if (-1 == id_creator.createFromBtMultiBody(object->multibody.get(), false))
                {
                }
                else
                {
                    tree = btInverseDynamics::CreateMultiBodyTree(id_creator);
                }

			    q_current.resize(numDofs);

			    if (tree && ((numDofs + baseDofs) == tree->numDoFs()))
			    {
				    btInverseDynamics::vec3 world_origin;
				    btInverseDynamics::mat33 world_rot;

				    jacSize = jacobian_linear.size();
				    // Set jacobian value

				    int DofIndex = 0;
				    for (int i = 0; i < object->multibody->getNumLinks(); ++i)
				    {
					    if (object->multibody->getLink(i).m_jointType >= 0 && object->multibody->getLink(i).m_jointType <= 2)
					    {
						    // 0, 1, 2 represent revolute, prismatic, and spherical joint types respectively. Skip the fixed joints.
						    double curPos = startingPositions[DofIndex];
						    q_current[DofIndex] = curPos;
						    q[DofIndex + baseDofs] = curPos;
						    qdot[DofIndex + baseDofs] = 0;
						    nu[DofIndex + baseDofs] = 0;
						    DofIndex++;
					    }
				    }  // Set the gravity to correspond to the world gravity
				    btInverseDynamics::vec3 id_grav(gravity);

				    {
					    if (-1 != tree->setGravityInWorldFrame(id_grav) &&
					        -1 != tree->calculateInverseDynamics(q, qdot, nu, &joint_force))
					    {
						    tree->calculateJacobians(q);
						    btInverseDynamics::mat3x jac_t(3, numDofs + baseDofs);
						    btInverseDynamics::mat3x jac_r(3, numDofs + baseDofs);
						    currentDiff = 0;

						    endEffectorCurrentWorldPositions.resize(0);
						    endEffectorCurrentWorldPositions.reserve(numEndEffectorLinkIndices * 3);

						    for (int ne = 0; ne < numEndEffectorLinkIndices; ne++)
						    {
							    int endEffectorLinkIndex2 = endEffectorLinkIndices[ne];

							    // Note that inverse dynamics uses zero-based indexing of bodies, not starting from -1 for the base link.
							    tree->getBodyJacobianTrans(endEffectorLinkIndex2 + 1, &jac_t);
							    tree->getBodyJacobianRot(endEffectorLinkIndex2 + 1, &jac_r);

							    //calculatePositionKinematics is already done inside calculateInverseDynamics

							    tree->getBodyOrigin(endEffectorLinkIndex2 + 1, &world_origin);
							    tree->getBodyTransform(endEffectorLinkIndex2 + 1, &world_rot);

							    for (int i = 0; i < 3; ++i)
							    {
								    for (int j = 0; j < numDofs; ++j)
								    {
									    jacobian_linear[(ne * 3 + i) * numDofs + j] = jac_t(i, (baseDofs + j));
									    jacobian_angular[(ne * 3 + i) * numDofs + j] = jac_r(i, (baseDofs + j));
								    }
							    }

							    endEffectorCurrentWorldPositions.push_back(world_origin[0]);
							    endEffectorCurrentWorldPositions.push_back(world_origin[1]);
							    endEffectorCurrentWorldPositions.push_back(world_origin[2]);

							    btInverseDynamics::vec3 targetPos(btVector3(endEffectorTargetWorldPositions[ne * 3 + 0],
							                                                endEffectorTargetWorldPositions[ne * 3 + 1],
							                                                endEffectorTargetWorldPositions[ne * 3 + 2]));
							    //diff
							    currentDiff = btMax(currentDiff, (world_origin - targetPos).length());
						    }
					    }
				    }

				    q_new.resize(numDofs);
					int IKMethod;
                    if (ikMethod == "JACOB_TRANS") {
                        IKMethod = IK2_JACOB_TRANS;
                    }
				    else if (ikMethod == "VEL_DLS_ORI_NULL")
				    {
					    //Nullspace task only works with DLS now. TODO: add nullspace task to SDLS.
					    IKMethod = IK2_VEL_DLS_WITH_ORIENTATION_NULLSPACE;
				    }
				    else if (ikMethod == "VEL_SDLS_ORI")
				    {
					    IKMethod = IK2_VEL_SDLS_WITH_ORIENTATION;
					}
					else if (ikMethod == "VEL_DLS_ORI")
					{
						IKMethod = IK2_VEL_DLS_WITH_ORIENTATION;
					}
				    else if (ikMethod == "VEL_DLS_NULL")
				    {
					    //Nullspace task only works with DLS now. TODO: add nullspace task to SDLS.
					    IKMethod = IK2_VEL_DLS_WITH_NULLSPACE;
				    }
				    else if (ikMethod == "VEL_SDLS")
				    {
					    IKMethod = IK2_VEL_SDLS;
					}
					else // VEL_DLS
					{
						IKMethod = IK2_VEL_DLS;
				    }

				    if (ikMethod == "VEL_DLS_ORI_NULL" || ikMethod == "VEL_DLS_NULL")
				    {
					    lower_limit.resize(numDofs);
					    upper_limit.resize(numDofs);
					    joint_range.resize(numDofs);
					    rest_pose.resize(numDofs);
					    for (int i = 0; i < numDofs; ++i) // TODO: use default data from multibody!
					    {
						    lower_limit[i] = object->multibody->getLink(i).m_jointLowerLimit;
						    upper_limit[i] = object->multibody->getLink(i).m_jointUpperLimit;
						    joint_range[i] = upper_limit[i] - lower_limit[i];
						    rest_pose[i] = startingPositions[i];
					    }
					    {
						    ikHelperPtr->computeNullspaceVel(numDofs, &q_current[0], &lower_limit[0], &upper_limit[0], &joint_range[0], &rest_pose[0]);
					    }
				    }

				    //btTransform endEffectorTransformWorld = bodyHandle->m_multiBody->getLink(endEffectorLinkIndex).m_cachedWorldTransform * bodyHandle->m_linkLocalInertialFrames[endEffectorLinkIndex].inverse();

				    btVector3DoubleData endEffectorWorldPosition;
				    btQuaternionDoubleData endEffectorWorldOrientation;

				    //get the position from the inverse dynamics (based on q) instead of endEffectorTransformWorld
				    btVector3 endEffectorPosWorldOrg = world_origin;
				    btQuaternion endEffectorOriWorldOrg;
				    world_rot.getRotation(endEffectorOriWorldOrg);

				    btTransform endEffectorBaseCoord;
				    endEffectorBaseCoord.setOrigin(endEffectorPosWorldOrg);
				    endEffectorBaseCoord.setRotation(endEffectorOriWorldOrg);

				    btQuaternion endEffectorOriBaseCoord = endEffectorBaseCoord.getRotation();

				    endEffectorBaseCoord.getOrigin().serializeDouble(endEffectorWorldPosition);
				    endEffectorBaseCoord.getRotation().serializeDouble(endEffectorWorldOrientation);

				    // Set joint damping coefficents. A small default
				    // damping constant is added to prevent singularity
				    // with pseudo inverse. The user can set joint damping
				    // coefficients differently for each joint. The larger
				    // the damping coefficient is, the less we rely on
				    // this joint to achieve the IK target.
				    btAlignedObjectArray<double> joint_damping;
				    joint_damping.resize(numDofs, 0.5);

				    for (int i = 0; i < numDofs; ++i)
				    {
					    joint_damping[i] = object->multibody->getLink(i).m_jointDamping;
				    }
				    ikHelperPtr->setDampingCoeff(numDofs, &joint_damping[0]);

				    double targetDampCoeff[6] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
				    bool performedIK = false;
				    if (numEndEffectorLinkIndices == 1)
				    {
					    ikHelperPtr->computeIK(&endEffectorTargetWorldPositions[0],
					                           &endEffectorTargetWorldOrientations[0],
					                           endEffectorWorldPosition.m_floats, endEffectorWorldOrientation.m_floats,
					                           &q_current[0],
					                           numDofs, endEffectorLinkIndices[0],
					                           &q_new[0], IKMethod, &jacobian_linear[0], &jacobian_angular[0], jacSize * 2, targetDampCoeff);
					    performedIK = true;
				    }
				    else
				    {
					    if (numEndEffectorLinkIndices > 1)
					    {
						    ikHelperPtr->computeIK2(&endEffectorTargetWorldPositions[0],
						                            &endEffectorCurrentWorldPositions[0],
						                            numEndEffectorLinkIndices,
						    //endEffectorWorldOrientation.m_floats,
						                            &q_current[0],
						                            numDofs,
						                            &q_new[0], IKMethod, &jacobian_linear[0], targetDampCoeff);
						    performedIK = true;
					    }
				    }
				    if (performedIK)
				    {
					    for (int i = 0; i < numDofs; i++)
					    {
						    startingPositions[i] = q_new[i];
					    }
				    }
			    }
		    }
	    }
        auto outputPoses = std::make_shared<ListObject>();
        outputPoses->arr.clear();
        for (size_t i = 0; i < startingPositions.size(); i++){
            auto p = std::make_shared<zeno::NumericObject>(float(startingPositions[i]));
            outputPoses->arr.push_back(p);
        }
	    set_output("poses", std::move(outputPoses));
    }
};

ZENDEFNODE(BulletCalcInverseKinematics, {
	{"object", "gravity", "endEffectorLinkIndices", "targetPositions", "targetOrientations", {"int", "numIterations", "20"}, {"float", "residualThreshold", "0.0001"}},
	{"poses"},
	{{"enum VEL_DLS_ORI_NULL VEL_SDLS_ORI VEL_DLS_ORI VEL_DLS_NULL VEL_SDLS VEL_DLS JACOB_TRANS", "IKMethod", "VEL_DLS_ORI_NULL"}},
	{"Bullet"},
});


struct BulletMultiBodyMakeJointMotor : zeno::INode {
    virtual void apply() override {
        auto object = get_input<BulletMultiBodyObject>("object");
        auto world = get_input<BulletMultiBodyWorld>("world");
        auto linkIndex = get_input2<int>("linkIndex");
        int linkDof = 0;
        if (has_input("linkDof")){
            linkDof = get_input2<int>("linkDof");
        }
        auto desiredVelocity = btScalar(get_input2<float>("desiredVelocity"));
        auto maxMotorImpulse = btScalar(get_input2<float>("maxMotorImpulse"));

        auto jointMotor = std::make_shared<BulletMultiBodyJointMotor>(object->multibody.get(), linkIndex, linkDof, desiredVelocity, maxMotorImpulse);
        world->dynamicsWorld->addMultiBodyConstraint(jointMotor->jointMotor.get());
        jointMotor->jointMotor->finalizeMultiDof();
        set_output("object", std::move(object));
        set_output("world", std::move(world));
    }
};

ZENDEFNODE(BulletMultiBodyMakeJointMotor, {
    {"object", "linkIndex", {"int", "linkDof", "0"}, {"float", "desiredVelocity", "0"}, {"float", "maxMotorImpulse", "10.0"}},
    {"object", "world"},
    {},
    {"Bullet"}
});


struct BulletMultiBodyGetJointTorque : zeno::INode {
    virtual void apply() override {
        auto object = get_input<BulletMultiBodyObject>("object");
        auto link_id = get_input2<int>("jointIndex");
        btScalar torque;

        torque = object->multibody->getJointTorque(link_id);
        // out_torque = vec1f(other_to_vec<1>(torque));

        auto out_torque = std::make_shared<zeno::NumericObject>(torque);
        set_output("joint_torque", std::move(out_torque));
    }
};

ZENDEFNODE(BulletMultiBodyGetJointTorque, {
    {"object", "jointIndex"},
    {"torque"},
    {},
    {"Bullet"}
});

struct BulletMultiBodyGetLinkForce : zeno::INode {
    virtual void apply() override {
        auto object = get_input<BulletMultiBodyObject>("object");
        auto link_id = get_input2<int>("linkIndex");
        btVector3 force;
        auto force_out = zeno::IObject::make<zeno::NumericObject>();
        force = object->multibody->getLinkForce(link_id);
        force_out->set<zeno::vec3f>(zeno::vec3f(force.x(), force.y(), force.z()));
        set_output("force", std::move(force_out));
    }
};

ZENDEFNODE(BulletMultiBodyGetLinkForce, {
                                              {"object", "linkIndex"},
                                              {"force"},
                                              {},
                                              {"Bullet"}
                                          });

struct BulletMultiBodyGetLinkTorque : zeno::INode {
    virtual void apply() override {
        auto object = get_input<BulletMultiBodyObject>("object");
        auto link_id = get_input2<int>("linkIndex");
        btVector3 torque;
        auto torque_out = zeno::IObject::make<zeno::NumericObject>();
        torque = object->multibody->getLinkTorque(link_id);
        torque_out->set<zeno::vec3f>(zeno::vec3f(torque.x(), torque.y(), torque.z()));
        set_output("torque", std::move(torque_out));
    }
};

ZENDEFNODE(BulletMultiBodyGetLinkTorque, {
                                            {"object", "linkIndex"},
                                            {"torque"},
                                            {},
                                            {"Bullet"}
                                        });

struct BulletMultiBodyGetJointVelPos : zeno::INode {
    virtual void apply() override {
        auto object = get_input<BulletMultiBodyObject>("object");
        auto link_id = get_input2<int>("linkIndex");
        btScalar vel;
        btScalar pos;

        vel = object->multibody->getJointVel(link_id);
        pos = object->multibody->getJointPos(link_id);
        // out_torque = vec1f(other_to_vec<1>(torque));

        auto vel_ = std::make_shared<zeno::NumericObject>(vel);
        auto pos_ = std::make_shared<zeno::NumericObject>(pos);
        set_output("vel", std::move(vel_));
        set_output("pos", std::move(pos_));
    }
};

ZENDEFNODE(BulletMultiBodyGetJointVelPos, {
                                              {"object", "linkIndex"},
                                              {"vel", "pos"},
                                              {},
                                              {"Bullet"}
                                          });

struct BulletMultiBodyGetBaseTransform : zeno::INode {
    virtual void apply() {
        auto object = get_input<BulletMultiBodyObject>("object");
        auto trans = std::make_unique<BulletTransform>();
        trans->trans = object->multibody->getBaseWorldTransform();
        set_output("trans", std::move(trans));
    }
};

 ZENDEFNODE(BulletMultiBodyGetBaseTransform, {
     {"object"},
     {"trans"},
     {},
     {"Bullet"},
 });

 struct BulletMultiBodyGetBaseVelocity : zeno::INode {
     virtual void apply() {
         auto object = get_input<BulletMultiBodyObject>("object");
         auto vel = zeno::IObject::make<zeno::NumericObject>();
         btVector3 vel_;
         vel_ = object->multibody->getBaseVel();
         vel->set<zeno::vec3f>(zeno::vec3f(vel_.x(), vel_.y(), vel_.z()));
         set_output("vel", std::move(vel));
     }
 };

 ZENDEFNODE(BulletMultiBodyGetBaseVelocity, {
                                                 {"object"},
                                                 {"vel"},
                                                 {},
                                                 {"Bullet"},
                                             });

struct BulletMultiBodyClearJointStates : zeno::INode {
    virtual void apply() {
        auto object = get_input<BulletMultiBodyObject>("object");
        object->multibody->clearVelocities();
        object->multibody->clearForcesAndTorques();
//        object->multibody->clearConstraintForces();
//        Not sure if clear constraint forces is necessary
    }
};

ZENDEFNODE(BulletMultiBodyClearJointStates, {
                                               {"object"},
                                               {},
                                               {},
                                               {"Bullet"},
                                           });


};