
#include "VHACD/inc/VHACD.h"
#include <BulletCollision/CollisionDispatch/btCollisionDispatcherMt.h>
#include <BulletCollision/CollisionShapes/btShapeHull.h>
#include <BulletDynamics/ConstraintSolver/btSequentialImpulseConstraintSolverMt.h>
#include <BulletDynamics/Dynamics/btDiscreteDynamicsWorldMt.h>
#include <BulletDynamics/Dynamics/btSimulationIslandManagerMt.h>
#include <LinearMath/btConvexHullComputer.h>
#include <btBulletDynamicsCommon.h>
#include <hacdCircularList.h>
#include <hacdGraph.h>
#include <hacdHACD.h>
#include <hacdICHull.h>
#include <hacdVector.h>
#include <memory>
#include <vector>
#include <zeno/ListObject.h>
#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/logger.h>
#include <zeno/utils/UserData.h>
#include <zeno/zeno.h>
#include <zeno/utils/fileio.h>

// multibody dynamcis
#include <BulletDynamics/Featherstone/btMultiBody.h>
#include <BulletDynamics/Featherstone/btMultiBodyDynamicsWorld.h>
#include <BulletDynamics/Featherstone/btMultiBodyLinkCollider.h>
#include <BulletDynamics/Featherstone/btMultiBodyJointFeedback.h>
#include <BulletDynamics/Featherstone/btMultiBodyConstraint.h>
#include <BulletDynamics/Featherstone/btMultiBodyJointLimitConstraint.h>
#include <BulletDynamics/Featherstone/btMultiBodySphericalJointLimit.h>
#include <BulletDynamics/Featherstone/btMultiBodySphericalJointMotor.h>
#include <BulletDynamics/Featherstone/btMultiBodyFixedConstraint.h>
#include <BulletDynamics/Featherstone/btMultiBodyGearConstraint.h>
#include <BulletDynamics/Featherstone/btMultiBodyJointMotor.h>
#include <BulletDynamics/Featherstone/btMultiBodyPoint2Point.h>
#include <BulletDynamics/Featherstone/btMultiBodySliderConstraint.h>
#include <BulletDynamics/Featherstone/btMultiBodyMLCPConstraintSolver.h>
#include "BulletDynamics/MLCPSolvers/btSolveProjectedGaussSeidel.h"
#include "BulletDynamics/MLCPSolvers/btDantzigSolver.h"

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
struct BulletTransform : zeno::IObject {
    btTransform trans;
};

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

struct BulletSetTransformBasisEuler : zeno::INode {
	virtual void apply() override {
		auto trans = get_input<BulletTransform>("trans")->trans;
		auto euler = get_input<zeno::NumericObject>("eulerZYX")->get<zeno::vec3f>();
		trans.getBasis().setEulerZYX(euler[0], euler[1], euler[2]);
	}
};

ZENDEFNODE(BulletSetTransformBasisEuler, {
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
struct BulletTriangleMesh : zeno::IObject {
    btTriangleMesh mesh;
};

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
struct BulletCollisionShape : zeno::IObject {
    std::unique_ptr<btCollisionShape> shape;

    BulletCollisionShape(std::unique_ptr<btCollisionShape> &&shape)
        : shape(std::move(shape)){

    }
};

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
        auto shape = std::make_unique<BulletCollisionShape>(
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

		auto shape = std::make_unique<BulletCollisionShape>(std::make_unique<btStaticPlaneShape>(planeNormal, planeConstant));
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

		auto shape = std::make_unique<BulletCollisionShape>(std::make_unique<btCapsuleShape>(btScalar(radius), btScalar(height)));
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

		auto shape = std::make_unique<BulletCollisionShape>(std::make_unique<btCylinderShape>(halfExtents));
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

struct BulletCalculateLocalInertia : zeno::INode {
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

ZENDEFNODE(BulletCalculateLocalInertia, {
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
struct BulletObject : zeno::IObject {
	// TODO: when btRigidBody get destroyed, should remove ref constraints first.
    std::unique_ptr<btDefaultMotionState> myMotionState;
    std::unique_ptr<btRigidBody> body;
    std::shared_ptr<BulletCollisionShape> colShape;
    btScalar mass = 0.f;
    btTransform trans;

    BulletObject(btScalar mass_,
        btTransform const &trans,
        std::shared_ptr<BulletCollisionShape> colShape_)
        : mass(mass_), colShape(std::move(colShape_))
    {
        btVector3 localInertia(0, 0, 0);
        if (mass != 0)
            colShape->shape->calculateLocalInertia(mass, localInertia);
        
        myMotionState = std::make_unique<btDefaultMotionState>(trans);
        btRigidBody::btRigidBodyConstructionInfo rbInfo(mass, myMotionState.get(), colShape->shape.get(), localInertia);
        body = std::make_unique<btRigidBody>(rbInfo);
    }
};

struct BulletMakeObject : zeno::INode {
    virtual void apply() override {
        auto shape = get_input<BulletCollisionShape>("shape");
        auto mass = get_input<zeno::NumericObject>("mass")->get<float>();
        auto trans = get_input<BulletTransform>("trans");
        auto object = std::make_unique<BulletObject>(
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

struct BulletSetObjectDamping : zeno::INode {
    virtual void apply() override {
        auto object = get_input<BulletObject>("object");
        auto dampLin = get_input2<float>("dampLin");
        auto dampAug = get_input2<float>("dampAug");
        log_debug("set object {} with dampLin={}, dampAug={}", (void*)object.get(), dampLin, dampAug);
        object->body->setDamping(dampLin, dampAug);
        set_output("object", std::move(object));
    }
};

ZENDEFNODE(BulletSetObjectDamping, {
    {"object", {"float", "dampLin", "0"}, {"float", "dampAug", "0"}},
    {"object"},
    {},
    {"Bullet"},
});

struct BulletSetObjectFriction : zeno::INode {
    virtual void apply() override {
        auto object = get_input<BulletObject>("object");
        auto friction = get_input2<float>("friction");
        log_debug("set object {} with friction={}", (void*)object.get(), friction);
        object->body->setFriction(friction);
        set_output("object", std::move(object));
    }
};

ZENDEFNODE(BulletSetObjectFriction, {
    {"object", {"float", "friction", "0"}},
    {"object"},
    {},
    {"Bullet"},
});

struct BulletSetObjectRestitution : zeno::INode {
    virtual void apply() override {
        auto object = get_input<BulletObject>("object");
        auto restitution = get_input2<float>("restitution");
        log_debug("set object {} with restituion={}", (void*)object.get(), restitution);
        object->body->setRestitution(restitution);
        set_output("object", std::move(object));
    }
};

ZENDEFNODE(BulletSetObjectRestitution, {
    {"object", {"float", "restitution", "0"}},
    {"object"},
    {},
    {"Bullet"},
});

struct BulletGetObjTransform : zeno::INode {
    virtual void apply() override {
        auto obj = get_input<BulletObject>("object");
        auto body = obj->body.get();
        auto trans = std::make_unique<BulletTransform>();

		if (body && body->getMotionState()) {
			body->getMotionState()->getWorldTransform(trans->trans);
		} else {
			trans->trans = static_cast<btCollisionObject *>(body)->getWorldTransform();
		}
		set_output("trans", std::move(trans));
    }
};

ZENDEFNODE(BulletGetObjTransform, {
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

struct BulletGetObjVel : zeno::INode {
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

ZENDEFNODE(BulletGetObjVel, {
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
        auto origin = std::make_unique<zeno::NumericObject>();
        auto rotation = std::make_unique<zeno::NumericObject>();
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

/*
 * Bullet Constraints
 */
//struct BulletConstraint : zeno::IObject {
//    std::unique_ptr<btTypedConstraint> constraint;
//
//    BulletObject *obj1;
//    BulletObject *obj2;
//
//    BulletConstraint(BulletObject *obj1, BulletObject *obj2)
//            : obj1(obj1), obj2(obj2)
//    {
//        //btTransform gf;
//        //gf.setIdentity();
//        //gf.setOrigin(cposw);
//        auto trA = obj1->body->getWorldTransform().inverse();// * gf;
//        auto trB = obj2->body->getWorldTransform().inverse();// * gf;
//#if 1
//        constraint = std::make_unique<btFixedConstraint>(
//                *obj1->body, *obj2->body, trA, trB);
//#else
//        constraint = std::make_unique<btGeneric6DofConstraint>(
//            *obj1->body, *obj2->body, trA, trB, true);
//    for (int i = 0; i < 6; i++)
//        static_cast<btGeneric6DofConstraint *>(constraint.get())->setLimit(i, 0, 0);
//#endif
//    }
//
//    void setBreakingThreshold(float breakingThreshold) {
//        auto totalMass = obj1->body->getMass() + obj2->body->getMass();
//        constraint->setBreakingImpulseThreshold(breakingThreshold * totalMass);
//    }
//};
//
//struct BulletMakeConstraint : zeno::INode {
//    virtual void apply() override {
//        auto obj1 = get_input<BulletObject>("obj1");
//        auto obj2 = get_input<BulletObject>("obj2");
//        auto cons = std::make_shared<BulletConstraint>(obj1.get(), obj2.get());
//        //cons->constraint->setOverrideNumSolverIterations(400);
//        set_output("constraint", std::move(cons));
//    }
//};
//
//ZENDEFNODE(BulletMakeConstraint, {
//    {"obj1", "obj2"},
//    {"constraint"},
//    {},
//    {"Bullet"},
//});

struct BulletConstraint : zeno::IObject {
    std::unique_ptr<btTypedConstraint> constraint;

	btRigidBody *obj1;
	btRigidBody *obj2;
    std::string constraintType;
	btTransform frame1;
	btTransform frame2;
	btVector3 axis1;
    btVector3 axis2;
    btVector3 pivot1;
    btVector3 pivot2;

    BulletConstraint(btRigidBody *obj1, btRigidBody *obj2, std::string constraintType)
            : obj1(obj1), obj2(obj2), constraintType(constraintType)
    {

        // a bad example
        //constraint = std::make_unique<btTypedConstraint>(D6_CONSTRAINT_TYPE, *obj1->body, *obj2->body);
        std::cout << "current constraintType: " << constraintType << std::endl;
        if (constraintType == "ConeTwist") {
            //frame1.setIdentity();
            //frame2.setIdentity();
			frame1 = obj1->getWorldTransform().inverse(); // local identity
	        frame2 = obj2->getWorldTransform().inverse();
			obj1->getWorldTransform().getBasis().setEulerZYX(0,0,0);
	        obj2->getWorldTransform().getBasis().setEulerZYX(0,0,0);
            frame1.getBasis().setEulerZYX(0,0,SIMD_PI/2);
	        frame2.getBasis().setEulerZYX(0,0,SIMD_PI/2);
			constraint = std::make_unique<btConeTwistConstraint>(*obj1, *obj2, frame1, frame2);
        }
        else if (constraintType == "Fixed") {
	        frame1 = obj1->getWorldTransform().inverse();
	        frame2 = obj2->getWorldTransform().inverse();
            constraint = std::make_unique<btFixedConstraint>(*obj1, *obj2, frame1, frame2);
        }
        else if (constraintType == "Gear") {
            axis1.setValue(0, 1, 0);
            axis2.setValue(0, 1, 0);
            btScalar ratio;
            ratio = (2-std::tan(SIMD_PI / 4.f)) / std::cos(SIMD_PI /4.f);
            constraint = std::make_unique<btGearConstraint>(*obj1, *obj2, axis1, axis2, ratio);
        }
        else if (constraintType == "Generic6Dof") {
	        frame1 = obj1->getWorldTransform(); // attach to the middle point
	        frame2 = obj2->getWorldTransform();

			auto mid_origin = (frame1.getOrigin() + frame2.getOrigin())/2;
			auto diff_origin1 = mid_origin - frame1.getOrigin();
			auto diff_origin2 = mid_origin - frame2.getOrigin();
			frame1.setOrigin(diff_origin1);
			frame2.setOrigin(diff_origin2);
            constraint = std::make_unique<btGeneric6DofConstraint>(*obj1, *obj2, frame1, frame2, false);
        }
        else if (constraintType == "Generic6DofSpring") {
	        frame1 = obj1->getWorldTransform(); // attached to the child object
			frame2 = obj2->getWorldTransform();
	        auto diff_origin = frame2.getOrigin() - frame1.getOrigin();
	        frame1.setOrigin(diff_origin);
            constraint = std::make_unique<btGeneric6DofSpringConstraint>(*obj1, *obj2, frame1, frame2, false);
        }
        else if (constraintType == "Generic6DofSpring2") {
	        frame1 = obj1->getWorldTransform();
	        frame2 = obj2->getWorldTransform();
	        auto diff_origin = frame2.getOrigin() - frame1.getOrigin();
	        frame1.setOrigin(diff_origin);
			frame2 = frame2.inverse();
            constraint = std::make_unique<btGeneric6DofSpring2Constraint>(*obj1, *obj2, frame1, frame2);
        }
        else if (constraintType == "Hinge") {
//            axis1.setValue(0, 1, 0);
//            axis2.setValue(0, 1, 0);
//            pivot1.setValue(-5, 0, 0);
//            pivot2.setValue(5, 0, 0);
	        frame1 = obj1->getWorldTransform();
	        frame2 = obj2->getWorldTransform();
	        auto mid_origin = (frame1.getOrigin() + frame2.getOrigin())/2;
	        auto diff_origin1 = mid_origin - frame1.getOrigin();
	        auto diff_origin2 = mid_origin - frame2.getOrigin();
	        frame1.setOrigin(diff_origin1);
	        frame2.setOrigin(diff_origin2);
            constraint = std::make_unique<btHingeConstraint>(*obj1, *obj2, frame1, frame2, false);
        }
        else if (constraintType == "Hinge2") {
            axis1.setValue(0, 1, 0);
            axis2.setValue(1, 0, 0);
			btVector3 anchor = obj2->getWorldTransform().getOrigin(); // attach to child
            constraint = std::make_unique<btHinge2Constraint>(*obj1, *obj2, anchor, axis1, axis2);
        }
        else if (constraintType == "Point2Point") {
	        frame1 = obj1->getCenterOfMassTransform();
	        frame2 = obj2->getCenterOfMassTransform();
	        auto mid_origin = (frame1.getOrigin() + frame2.getOrigin())/2;
	        pivot1 = mid_origin - frame1.getOrigin();
	        pivot2 = mid_origin - frame2.getOrigin();

			std::cout<< "mid_origin:" << mid_origin[0] << " " << mid_origin[1] << " " << mid_origin[2] << std::endl;
			std::cout<< "pivot1" << pivot1[0] << " " << pivot1[1] << " " << pivot1[2] << " " <<std::endl;
	        std::cout<< "pivot2" << pivot2[0] << " " << pivot2[1] << " " << pivot2[2] << " " <<std::endl;
            constraint = std::make_unique<btPoint2PointConstraint>(*obj1, *obj2, pivot1, pivot2);
        }
        else if (constraintType == "Slider") {
	        frame1 = obj1->getWorldTransform().inverse();
	        frame2 = obj2->getWorldTransform().inverse();
            constraint = std::make_unique<btSliderConstraint>(*obj1, *obj2, frame1, frame2, true);
        }
        else if (constraintType == "Universal") {
            axis1.setValue(1, 0, 0);
            axis2.setValue(0, 0, 1);
	        frame1 = obj1->getWorldTransform();
	        frame2 = obj2->getWorldTransform();
	        btVector3 anchor = (frame1.getOrigin() + frame2.getOrigin())/2;
            constraint = std::make_unique<btUniversalConstraint>(*obj1, *obj2, anchor, axis1, axis2);
        }
    }

	BulletConstraint(btRigidBody *obj1, std::string constraintType): obj1(obj1), constraintType(constraintType){
		if (constraintType == "ConeTwist") {
			//frame1.setIdentity();
			//frame2.setIdentity();
			frame1 = obj1->getWorldTransform().inverse(); // local identity
			constraint = std::make_unique<btConeTwistConstraint>(*obj1, frame1);
		}
		else if (constraintType == "Generic6Dof") {
			frame1 = obj1->getWorldTransform().inverse();
			constraint = std::make_unique<btGeneric6DofConstraint>(*obj1, frame1, false);
		}
		else if (constraintType == "Generic6DofSpring") {
			frame1 = obj1->getWorldTransform().inverse();
			constraint = std::make_unique<btGeneric6DofSpringConstraint>(*obj1,  frame1, false);
		}
		else if (constraintType == "Generic6DofSpring2") {
			frame1 = obj1->getWorldTransform().inverse();
			constraint = std::make_unique<btGeneric6DofSpring2Constraint>(*obj1, frame1);
		}
		else if (constraintType == "Hinge") {
			//axis1.setValue(0, 1, 0);
			//pivot1.setValue(-2, 0, 0);
			frame1 = obj1->getCenterOfMassTransform().inverse();
			constraint = std::make_unique<btHingeConstraint>(*obj1,  frame1, false);
		}
		else if (constraintType == "Point2Point") {
			pivot1.setValue(1, -1, -1);
			constraint = std::make_unique<btPoint2PointConstraint>(*obj1, pivot1);
		}
		else if (constraintType == "Slider") {
			frame1 = obj1->getWorldTransform().inverse();
			constraint = std::make_unique<btSliderConstraint>(*obj1, frame1, true);
		}
	}
    void setBreakingThreshold(float breakingThreshold) {
        auto totalMass = obj1->getMass() + obj2->getMass();
        constraint->setBreakingImpulseThreshold(breakingThreshold * totalMass);
    }
};

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


        //cons->constraint->setOverrideNumSolverIterations(400);

    }
};

ZENDEFNODE(BulletMakeConstraint, {
    {"obj1", "obj2"},
    {"constraint"},
    {{"enum ConeTwist Fixed Gear Generic6Dof Generic6DofSpring Generic6DofSpring2 Hinge Hinge2 Point2Point Slider Universal", "constraintType", "Fixed"}},
    {"Bullet"},
});


struct BulletSetConstraintBreakThres : zeno::INode {
    virtual void apply() override {
        auto cons = get_input<BulletConstraint>("constraint");
        cons->setBreakingThreshold(get_input2<float>("threshold"));
        set_output("constraint", std::move(cons));
    }
};

ZENDEFNODE(BulletSetConstraintBreakThres, {
    {"constraint", {"float", "threshold", "3.0"}},
    {"constraint"},
    {},
    {"Bullet"},
});

struct BulletSetConstraintFrames : zeno::INode {
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

ZENDEFNODE(BulletSetConstraintFrames, {
    {"constraint", "frame1", "frame2"},
    {"constraint"},
    {{"enum ConeTwist Fixed Generic6Dof Generic6DofSpring Generic6DofSpring2 Hinge Hinge2 Slider Universal", "constraintType", "Universal"}},
    {"Bullet"},
});

struct BulletGetConstraintFrames : zeno::INode {
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

ZENDEFNODE(BulletGetConstraintFrames, {
	{"constraint"},
	{"frame1", "frame2"},
	{{"enum ConeTwist Fixed Generic6Dof Generic6DofSpring Generic6DofSpring2 Hinge Hinge2 Slider Universal", "constraintType", "Universal"}},
	{"Bullet"}
});

struct BulletSetConstraintLimitByAxis : zeno::INode {
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

ZENDEFNODE(BulletSetConstraintLimitByAxis, {
    {"constraint", "lowLimit", "highLimit"},
    {"constraint"},
    {{"enum ConeTwist Fixed Generic6Dof Generic6DofSpring Generic6DofSpring2 Hinge Hinge2 Slider Universal", "constraintType", "Universal"}, {"enum linearX linearY linearZ angularX angularY angularZ", "axisId", "linearX"}},
    {"Bullet"},
});

struct BulletSetConstraintRefFrameA : zeno::INode {
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

ZENDEFNODE(BulletSetConstraintRefFrameA, {
    {"constraint"},
    {"constraint"},
    {{"enum Generic6Dof Generic6DofSpring Hinge Universal", "constraintType", "Universal"}, {"enum true false", "useReferenceFrameA", "true"}},
    {"Bullet"},
});

struct BulletSetConstraintAxis : zeno::INode {
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

ZENDEFNODE(BulletSetConstraintAxis, {
    {"constraint", "axis1", "axis2"},
    {"constraint"},
    {{"enum Fixed Gear Generic6Dof Generic6DofSpring Generic6DofSpring2 Hinge Hinge2 Point2Point Universal", "constraintType", "Universal"}},
    {"Bullet"},
});

struct BulletSetConstraintSpring : zeno::INode {
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

ZENDEFNODE(BulletSetConstraintSpring , {
    {"constraint", {"bool", "enable", "true"}, "stiffness", "damping", "equilibriumPointVal"},
    {"constraint"},
    {{"enum Fixed Generic6DofSpring Generic6DofSpring2 Hinge2", "constraintType", "Fixed"}, {"enum linearX linearY linearZ angularX angularY angularZ", "axisId", "linearX"}},
    {"Bullet"},
});

struct BulletSetConstraintMotor : zeno::INode {
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

ZENDEFNODE(BulletSetConstraintMotor , {
    {"constraint", {"float","bounce","0"}, {"bool", "enableMotor", "1"}, {"bool","enableServo","1"}, {"float", "maxMotorForce", "0"}, {"float","servoTarget","0"}, {"float", "targetVelocity", "0"}, {"float","maxMotorImpulse","0"}, {"float","maxMotorImpulseNormalized","0"}, {"vec4f","motorTarget","0,0,0,1"}, {"vec4f","motorTargetConstraint","0,0,0,1"}, {"float","angularOnly","1"}, {"float","fixThresh","0"}, {"float","dt","0"}},
    {"constraint"},
    {{"enum ConeTwist Generic6Dof Generic6DofSpring2 Hinge Hinge2", "constraintType", "ConeTwist"}, {"enum linearX linearY linearZ angularX angularY angularZ", "axisId", "linearX"}},
    {"Bullet"},
});

struct BulletSetConstraintRotOrder : zeno::INode {
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

ZENDEFNODE(BulletSetConstraintRotOrder, {
    {"constraint"},
    {"constraint"},
    {{"enum Generic6DofSpring2 Hinge2", "constraintType", "Hinge2"}, {"enum XYZ XZY YXZ YZX ZXY ZYX", "rotateOrder", "XYZ"}},
    {"Bullet"},
});


struct BulletSetGearConstraintRatio : zeno::INode {
    virtual void apply() override {
        auto cons = get_input<BulletConstraint>("constraint");
        auto ratio = btScalar(get_input2<float>("ratio"));
        dynamic_cast<btGearConstraint *>(cons->constraint.get())->setRatio(ratio);
        set_output("constraint", std::move(cons));
    }
};

ZENDEFNODE(BulletSetGearConstraintRatio, {
    {"constraint", {"float", "ratio", "1"}},
    {"constraint"},
    {},
    {"Bullet"},
});


struct BulletSetSliderConstraintSpring : zeno::INode {
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

ZENDEFNODE(BulletSetSliderConstraintSpring, {
    {"constraint", "dampingDirAng", "dampingDirLin", "dampingLimAng", "dampingLimLin", "dampingOrthoAng", "dampingOrthoLin", "maxAngMotorForce", "maxLinMotorForce", "poweredAngMotor", "poweredLinMotor", "restitutionDirAng", "poweredLinMotor", "restitutionDirAng", "restitutionDirLin", "restitutionLimAng", "restitutionLimLin", "restitutionOrthoAng", "restitutionOrthoLin", "softnessDirAng", "softnessDirLin", "softnessLimAng", "softnessLimLin", "softnessOrthoAng", "softnessOrthoLin", "targetAngMotorVelocity", "targetLinMotorVelocity"},
    {"constraint"},
    {},
    {"Bullet"},
});

/*
 *  Bullet World
 */
struct BulletWorld : zeno::IObject {
#ifdef ZENO_RIGID_MULTITHREADING
    // mt bullet not working for now
    std::unique_ptr<btDefaultCollisionConfiguration> collisionConfiguration;
    std::unique_ptr<btCollisionDispatcherMt> dispatcher;
    std::unique_ptr<btBroadphaseInterface> broadphase;
    std::unique_ptr<btSequentialImpulseConstraintSolverMt> solver;
    std::vector<std::unique_ptr<btSequentialImpulseConstraintSolver>> solvers;
    std::unique_ptr<btConstraintSolverPoolMt> solverPool;

    std::unique_ptr<btDiscreteDynamicsWorldMt> dynamicsWorld;

    std::set<std::shared_ptr<BulletObject>> objects;
    std::set<std::shared_ptr<BulletConstraint>> constraints;

    BulletWorld() {
        /*if (NULL != btGetTaskScheduler() && gTaskSchedulerMgr.getNumTaskSchedulers() > 1) {
            log_critical("bullet multithreading enabled!");
        } else {
            log_critical("bullet multithreading disabled...");
        }*/
        collisionConfiguration = std::make_unique<btDefaultCollisionConfiguration>();
        dispatcher = std::make_unique<btCollisionDispatcherMt>(collisionConfiguration.get());
        broadphase = std::make_unique<btDbvtBroadphase>();
        solver = std::make_unique<btSequentialImpulseConstraintSolverMt>();
        std::vector<btConstraintSolver *> solversPtr;
        for (int i = 0; i < BT_MAX_THREAD_COUNT; i++) {
            auto sol = std::make_unique<btSequentialImpulseConstraintSolver>();
            solversPtr.push_back(sol.get());
            solvers.push_back(std::move(sol));
        }
        solverPool = std::make_unique<btConstraintSolverPoolMt>(solversPtr.data(), solversPtr.size());
        dynamicsWorld = std::make_unique<btDiscreteDynamicsWorldMt>(
            dispatcher.get(), broadphase.get(), solverPool.get(), solver.get(),
            collisionConfiguration.get());
        dynamicsWorld->setGravity(btVector3(0, -10, 0));
    }
#else
    std::unique_ptr<btDefaultCollisionConfiguration> collisionConfiguration;
    std::unique_ptr<btCollisionDispatcher> dispatcher;
    std::unique_ptr<btBroadphaseInterface> broadphase;
    std::unique_ptr<btSequentialImpulseConstraintSolver> solver;

    std::unique_ptr<btDiscreteDynamicsWorld> dynamicsWorld;

    std::set<std::shared_ptr<BulletObject>> objects;
    std::set<std::shared_ptr<BulletConstraint>> constraints;

    BulletWorld() {
        collisionConfiguration = std::make_unique<btDefaultCollisionConfiguration>();
        /*btDefaultCollisionConstructionInfo cci;
		cci.m_defaultMaxPersistentManifoldPoolSize = 80000;
		cci.m_defaultMaxCollisionAlgorithmPoolSize = 80000;
        collisionConfiguration = std::make_unique<btDefaultCollisionConfiguration>(cci);*/

        dispatcher = std::make_unique<btCollisionDispatcher>(collisionConfiguration.get());
        broadphase = std::make_unique<btDbvtBroadphase>();
        solver = std::make_unique<btSequentialImpulseConstraintSolver>();
        dynamicsWorld = std::make_unique<btDiscreteDynamicsWorld>(
            dispatcher.get(), broadphase.get(), solver.get(),
            collisionConfiguration.get());
        dynamicsWorld->setGravity(btVector3(0, -10, 0));

        log_debug("creating bullet world {}", (void *)this);
    }
#endif

    void addObject(std::shared_ptr<BulletObject> obj) {
        log_debug("adding object {}", (void *)obj.get());
        dynamicsWorld->addRigidBody(obj->body.get());
        objects.insert(std::move(obj));
    }

    void removeObject(std::shared_ptr<BulletObject> const &obj) {
        log_debug("removing object {}", (void *)obj.get());
        dynamicsWorld->removeRigidBody(obj->body.get());
        objects.erase(obj);
    }

    void setObjectList(std::vector<std::shared_ptr<BulletObject>> objList) {
        std::set<std::shared_ptr<BulletObject>> objSet;
        log_debug("setting object list len={}", objList.size());
        log_debug("existing object list len={}", objects.size());
        for (auto const &object: objList) {
            objSet.insert(object);
            if (objects.find(object) == objects.end()) {
                addObject(std::move(object));
            }
        }
        for (auto const &object: std::set(objects)) {
            if (objSet.find(object) == objSet.end()) {
                removeObject(object);
            }
        }
    }

    void addConstraint(std::shared_ptr<BulletConstraint> cons) {
        log_debug("adding constraint {}", (void *)cons.get());
        dynamicsWorld->addConstraint(cons->constraint.get(), true);
        constraints.insert(std::move(cons));
    }

    void removeConstraint(std::shared_ptr<BulletConstraint> const &cons) {
        log_debug("removing constraint {}", (void *)cons.get());
        dynamicsWorld->removeConstraint(cons->constraint.get());
        constraints.erase(cons);
    }

    void setConstraintList(std::vector<std::shared_ptr<BulletConstraint>> consList) {
        std::set<std::shared_ptr<BulletConstraint>> consSet;
        log_debug("setting constraint list len={}", consList.size());
        log_debug("existing constraint list len={}", constraints.size());
        for (auto const &constraint: consList) {
            if (!constraint->constraint->isEnabled())
                continue;
            consSet.insert(constraint);
            if (constraints.find(constraint) == constraints.end()) {
                addConstraint(std::move(constraint));
            }
        }
        for (auto const &constraint: std::set(constraints)) {
            if (consSet.find(constraint) == consSet.end()) {
                removeConstraint(constraint);
            }
        }
    }

    /*
    void addGround() {
        auto groundShape = std::make_unique<btBoxShape>(btVector3(btScalar(50.), btScalar(50.), btScalar(50.)));

        btTransform groundTransform;
        groundTransform.setIdentity();
        groundTransform.setOrigin(btVector3(0, -56, 0));

        btScalar mass(0.);

        addObject(std::make_unique<BulletObject>(mass, groundTransform, std::move(groundShape)));
    }

    void addBall() {
        auto colShape = std::make_unique<btSphereShape>(btScalar(1.));

        btTransform startTransform;
        startTransform.setIdentity();

        btScalar mass(1.f);

        addObject(std::make_unique<BulletObject>(mass, startTransform, std::move(colShape)));
    }*/

    void step(float dt = 1.f / 60.f, int steps = 1) {
        log_debug("stepping with dt={}, steps={}, len(objects)={}", dt, steps, objects.size());
        //dt /= steps;
        for(int i=0;i<steps;i++)
            dynamicsWorld->stepSimulation(dt/(float)steps, 1, dt / (float)steps);

        /*for (int j = dynamicsWorld->getNumCollisionObjects() - 1; j >= 0; j--)
        {
            btCollisionObject* obj = dynamicsWorld->getCollisionObjectArray()[j];
            btRigidBody* body = btRigidBody::upcast(obj);
            btTransform trans;
            if (body && body->getMotionState())
            {
                body->getMotionState()->getWorldTransform(trans);
            }
            else
            {
                trans = obj->getWorldTransform();
            }
            printf("world pos object %d = %f,%f,%f\n", j, float(trans.getOrigin().getX()), float(trans.getOrigin().getY()), float(trans.getOrigin().getZ()));
        }*/
    }
};

struct BulletMakeWorld : zeno::INode {
    virtual void apply() override {
        auto world = std::make_unique<BulletWorld>();
        set_output("world", std::move(world));
    }
};

ZENDEFNODE(BulletMakeWorld, {
                                {},
                                {"world"},
                                {},
                                {"Bullet"},
                            });

struct BulletSetWorldGravity : zeno::INode {
    virtual void apply() override {
        auto world = get_input<BulletWorld>("world");
        auto gravity = get_input<zeno::NumericObject>("gravity")->get<zeno::vec3f>();
        world->dynamicsWorld->setGravity(zeno::vec_to_other<btVector3>(gravity));
        set_output("world", std::move(world));
    }
};

ZENDEFNODE(BulletSetWorldGravity, {
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
        auto consList = get_input<ListObject>("consList")
                            ->get<std::shared_ptr<BulletConstraint>>();
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

struct MultiBodyJointFeedback : zeno::IObject {
	btMultiBodyJointFeedback jointFeedback;
};

struct BulletMultiBodyObject : zeno::IObject {

    int n_links;
    btScalar mass;
    btVector3 inertia;
    bool fixedBase;
    bool canSleep;
    std::unique_ptr<btMultiBody> multibody;
	btAlignedObjectArray<btMultiBodyJointFeedback*> jointFeedbacks;

    BulletMultiBodyObject(int n_links, btScalar mass, btVector3 inertia, bool fixedBase, bool canSleep) : n_links(n_links), mass(mass), inertia(inertia), fixedBase(fixedBase), canSleep(canSleep)
    {
        multibody = std::make_unique<btMultiBody>(n_links, mass, inertia, fixedBase, canSleep);
		multibody->setBaseWorldTransform(btTransform::getIdentity());
    }
};

struct BulletMultiBodyLinkCollider : zeno::IObject{
	// it is a child class of btCollisionObject.
	std::unique_ptr<btMultiBodyLinkCollider> linkCollider;

	BulletMultiBodyLinkCollider(btMultiBody *multiBody, int link){
		linkCollider = std::make_unique<btMultiBodyLinkCollider>(multiBody, link);
	}
};

struct BulletStartMultiBodyObject : zeno::INode {
    virtual void apply() override {
		auto n_links = get_input2<int>("nLinks");
		auto mass = get_input<zeno::NumericObject>("mass")->get<float>();
		btVector3 inertia(0.f, 0.f, 0.f);
		if (has_input("inertia"))
			inertia = zeno::vec_to_other<btVector3>(get_input<zeno::NumericObject>("inertia")->get<zeno::vec3f>());
	    auto fixedBase = (std::get<std::string>(get_param("fixedBase")) == "true");
		auto canSleep = (std::get<std::string>(get_param("canSleep")) == "true");
		auto object = std::make_unique<BulletMultiBodyObject>(n_links, mass, inertia, fixedBase, canSleep);
	    set_output("object", std::move(object));
    }
};

ZENDEFNODE(BulletStartMultiBodyObject, {
	{"nLinks", {"float", "mass", "0"}, "inertia"},
	{"object"},
	{{"enum true false", "fixedBase", "true"}, {"enum true false", "canSleep", "true"}},
	{"Bullet"}
});

struct BulletEndMultiBodyObject : zeno::INode {
	virtual void apply() override {
		auto object = get_input<BulletMultiBodyObject>("object");
		object->multibody->finalizeMultiDof();
		set_output("object", std::move(object));
	}
};

ZENDEFNODE(BulletEndMultiBodyObject, {
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

struct BulletSetMultiBodyJointProperty : zeno::INode{
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

ZENDEFNODE(BulletSetMultiBodyJointProperty, {
	{"object", {"int", "linkIndex", "0"}, "damping", "friction", "lowerLimit", "upperLimit", "maxForce", "maxVelocity"},
	{"object"},
	{},
	{"Bullet"}
});

struct BulletSetMultiBodyBaseTransform : zeno::INode {
	virtual void apply() override {
		auto object = get_input<BulletMultiBodyObject>("object");
		auto trans = get_input<BulletTransform>("baseTrans");

		object->multibody->setBaseWorldTransform(trans->trans);
		set_output("object", std::move(object));
	}
};

ZENDEFNODE(BulletSetMultiBodyBaseTransform, {
	{"object", "baseTrans"},
	{"object"},
	{},
	{"Bullet"}
});

struct BulletExtractMultiBodyLinkTransform : zeno::INode {
	virtual void apply() override {
		auto object = get_input<BulletMultiBodyObject>("object");
		auto link_id = get_input2<int>("linkIndex");

		auto trans = std::make_unique<zeno::NumericObject>();
		auto rot = std::make_unique<zeno::NumericObject>();

		btVector3 tmpTrans;
		btVector3 tmpOri;
		object->multibody->localPosToWorld(link_id, tmpTrans);
		object->multibody->localDirToWorld(link_id, tmpOri);
		trans->set(vec3f(other_to_vec<3>(tmpTrans)));
		rot->set(vec3f(other_to_vec<3>(tmpOri)));

		std::cout<<"trans:" << tmpTrans[0] << " " << tmpTrans[1] << " " << tmpTrans[2] << std::endl;
		std::cout<<"rot:" << tmpOri[0] << " " << tmpOri[1] << " " << tmpOri[2] << std::endl;
		set_output("trans", std::move(trans));
		set_output("rot", std::move(rot));
	}
};

ZENDEFNODE(BulletExtractMultiBodyLinkTransform, {
	{"object", "linkIndex"},
	{"trans", "rot"},
	{},
	{"Bullet"}
});

struct BulletExtractMultiBodyTransform : zeno::INode {
	virtual void apply() override {
		auto object = get_input<BulletMultiBodyObject>("object");
		auto listTrans = std::make_shared<zeno::ListObject>();
		auto listOri = std::make_shared<zeno::ListObject>();

		for (size_t i = 0; i< object->multibody->getNumLinks(); i++) {
			btVector3 tmpTrans;
			btVector3 tmpOri;
			object->multibody->localPosToWorld(i, tmpTrans);
			object->multibody->localDirToWorld(i, tmpOri);
			listTrans->arr.push_back(tmpTrans);
			listOri->arr.push_back(tmpOri);
		}

		set_output("listTrans", std::move(listTrans));
		set_output("listOri", std::move(listOri));
	}
};

ZENDEFNODE(BulletExtractMultiBodyTransform, {
	{"object"},
	{"listTrans", "listOri"},
	{},
	{"Bullet"}
});

struct BulletSetMultiBodyProperty : zeno::INode {
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

ZENDEFNODE(BulletSetMultiBodyProperty, {
	{"object", {"float", "linearDamp", "0"}, {"float", "angularDamp", "0"}},
	{"object"},
	{{"enum true false", "canSleep", "false"},
		{"enum true false", "selfCollide", "false"},
		{"enum true false", "useGyro", "false"}},
	{"Bullet"}
});


struct BulletSetCollisionShapeForLink : zeno::INode {
	virtual void apply() override {
		auto object = get_input<BulletMultiBodyObject>("object");
		auto link_id = get_input2<int>("linkIndex");
		auto collisionShape = get_input<BulletCollisionShape>("colShape");

		auto col = std::make_shared<BulletMultiBodyLinkCollider>(object->multibody.get(), link_id);
		col->linkCollider->setCollisionShape(collisionShape->shape.get());
		object->multibody->getLink(link_id).m_collider = col->linkCollider.get();

		set_output("object", std::move(object));
		set_output("collider", std::move(col));
	}
};

ZENDEFNODE(BulletSetCollisionShapeForLink, {
	{"object", "linkIndex", "colShape"},
	{"object", "collider"},
	{},
	{"Bullet"},
});

struct BulletSetLinkColliderTransform : zeno::INode {
	virtual void apply() override {
		auto col = get_input<BulletMultiBodyLinkCollider>("collider");
		auto trans = get_input<BulletTransform>("trans");

		col->linkCollider->setWorldTransform(trans->trans);
		set_output("collider", std::move(col));
	}
};

ZENDEFNODE(BulletSetLinkColliderTransform, {
	{"collider", "trans"},
	{"collider"},
	{},
	{"Bullet"},
});

struct BulletForwardKinematics : zeno::INode {
	virtual void apply() override {
		auto object = get_input<BulletMultiBodyObject>("object");
		btAlignedObjectArray<btQuaternion> scratch_q;
		btAlignedObjectArray<btVector3> scratch_m;
		object->multibody->forwardKinematics(scratch_q, scratch_m);
		set_output("object", std::move(object));
	}
};

ZENDEFNODE(BulletForwardKinematics, {
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
	{"object", "startIndex", "pos"},
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
	{"object", "startIndex", "pos"},
	{"object"},
	{{"enum true false", "isSpherical", "false"}},
	{"Bullet"},
});


struct BulletSetMultiBodyJointFeedback : zeno::INode {
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

ZENDEFNODE(BulletSetMultiBodyJointFeedback, {
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
		qDesiredArray.resize(object->multibody->getNumLinks(), 0);
		btAlignedObjectArray<btScalar> qdDesiredArray;
		qdDesiredArray.resize(object->multibody->getNumLinks(), 0);
		if (has_input("qDesiredList")){
			auto temp =  get_input<ListObject>("qDesiredList")->get<float>();
			for (size_t i; i<qDesiredArray.size();i++)
				qDesiredArray[i] = temp[i];
		}
		if (has_input("dqDesiredList")) {
			auto temp =  get_input<ListObject>("qDesiredList")->get<float>();
			for (size_t i; i<qdDesiredArray.size();i++)
				qdDesiredArray[i] = temp[i];
		}

		for (int joint = 0; joint < object->multibody->getNumLinks(); joint++)
		{
			int dof1 = 0;
			btScalar qActual = object->multibody->getJointPosMultiDof(joint)[dof1];
			btScalar qdActual = object->multibody->getJointVelMultiDof(joint)[dof1];
			btScalar positionError = (qDesiredArray[joint] - qActual);
			btScalar velocityError = (qdDesiredArray[joint] - qdActual);
			btScalar force = kp * positionError + kd * velocityError;
			btClamp(force, -maxForce, maxForce);
			object->multibody->addJointTorque(joint, force);
		}
	}
};

ZENDEFNODE(BulletMultiBodyPDControl, {
	{"object", {"float", "kp", "100"}, {"float", "kd", "20"}, {"float", "maxForce", "100"}},
	{},
	{},
	{"Bullet"},
});
/*
 * Bullet MultiBody World
 */
struct BulletMultiBodyWorld : zeno::IObject {
    std::unique_ptr<btDefaultCollisionConfiguration> collisionConfiguration;
    std::unique_ptr<btCollisionDispatcher> dispatcher;
    std::unique_ptr<btBroadphaseInterface> broadphase;
    std::unique_ptr<btMultiBodyConstraintSolver> solver;

    std::unique_ptr<btMultiBodyDynamicsWorld> dynamicsWorld;

    std::string solverType;

    BulletMultiBodyWorld(std::string solverType) : solverType(solverType) {
        collisionConfiguration = std::make_unique<btDefaultCollisionConfiguration>();

        dispatcher = std::make_unique<btCollisionDispatcher>(collisionConfiguration.get());
        broadphase = std::make_unique<btDbvtBroadphase>();

        if (solverType == "SequentialImpulse") {
            solver = std::make_unique<btMultiBodyConstraintSolver>();
        }
        else if (solverType == "ProjectedGaussSeidel") {
            auto mlcp = std::make_unique<btSolveProjectedGaussSeidel>();
            solver = std::make_unique<btMultiBodyMLCPConstraintSolver>(mlcp.get());
        }
        else { // solverType == "Dantzig"
            auto mlcp = std::make_unique<btDantzigSolver>();
            solver = std::make_unique<btMultiBodyMLCPConstraintSolver>(mlcp.get());
        }
        dynamicsWorld = std::make_unique<btMultiBodyDynamicsWorld>(
            dispatcher.get(), broadphase.get(), solver.get(),
            collisionConfiguration.get());
        dynamicsWorld->setGravity(btVector3(0, -10, 0));

        log_debug("creating bullet multibody dynamics world {}", (void *)this);
    }

};

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
	{{"enum SequentialImpulse ProjectedGaussSeidel Dantzig", "solverType", "ProjectedGaussSeidel"}},
	{"Bullet"},
});

struct BulletSetMultiBodyWorldGravity : zeno::INode {
	virtual void apply() override {
		auto world = get_input<BulletMultiBodyWorld>("world");
		auto gravity = get_input<zeno::NumericObject>("gravity")->get<zeno::vec3f>();
		world->dynamicsWorld->setGravity(zeno::vec_to_other<btVector3>(gravity));
		set_output("world", std::move(world));
	}
};

ZENDEFNODE(BulletSetMultiBodyWorldGravity, {
	{"world", {"vec3f", "gravity", "0,0,-9.8"}},
	{"world"},
	{},
	{"Bullet"},
});

struct BulletStepMultiBodyWorld : zeno::INode {
	virtual void apply() override {
		auto world = get_input<BulletMultiBodyWorld>("world");
		auto dt = get_input<zeno::NumericObject>("dt")->get<float>();
		auto steps = get_input<zeno::NumericObject>("steps")->get<int>();
		for (size_t i = 0; i< steps;i++)
			world->dynamicsWorld->stepSimulation(dt/(float)steps, 1, dt/(float)steps);
		set_output("world", std::move(world));
	}
};

ZENDEFNODE(BulletStepMultiBodyWorld, {
	{"world", {"float", "dt", "0.04"}, {"int", "steps", "1"}},
	{"world"},
	{},
	{"Bullet"},
});

struct BulletWorldAddMultiBodyObject : zeno::INode {
	virtual void apply() override {
		auto world = get_input<BulletMultiBodyWorld>("world");
		auto object = get_input<BulletMultiBodyObject>("object");
		world->dynamicsWorld->addMultiBody(object->multibody.get());

		set_output("world", std::move(world));
	}
};


ZENDEFNODE(BulletWorldAddMultiBodyObject, {
	{"world", "object"},
	{"world"},
	{},
	{"Bullet"},
});

struct BulletWorldRemoveMultiBodyObject : zeno::INode {
	virtual void apply() override {
		auto world = get_input<BulletMultiBodyWorld>("world");
		auto object = get_input<BulletMultiBodyObject>("object");
		world->dynamicsWorld->removeMultiBody(object->multibody.get());
		set_output("world", get_input("world"));
	}
};

ZENDEFNODE(BulletWorldRemoveMultiBodyObject, {
	{"world", "object"},
	{"world"},
	{},
	{"Bullet"},
});


struct BulletMultiBodyWorldAddCollisionObject : zeno::INode {
	virtual void apply() override {
		auto world = get_input<BulletMultiBodyWorld>("world");
		auto isDynamic = (std::get<std::string>(get_param("isDynamic")) == "true");
		auto col = get_input<BulletMultiBodyLinkCollider>("collider");
		int collisionFilterGroup = isDynamic ? int(btBroadphaseProxy::DefaultFilter) : int(btBroadphaseProxy::StaticFilter);
		int collisionFilterMask = isDynamic ? int(btBroadphaseProxy::AllFilter) : int(btBroadphaseProxy::AllFilter ^ btBroadphaseProxy::StaticFilter);
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

struct BulletMultiBodyConstraint : zeno::IObject {
	std::unique_ptr<btMultiBodyConstraint> constraint;
	btMultiBody *bodyA;
	btMultiBody *bodyB;
	int linkA;
	int linkB;
	std::map<std::string, btScalar> config;
	std::string constraintType;
	btVector3 pivotA;
	btVector3 pivotB;
	btVector3 jointAxis;
	btMatrix3x3 frameA;
	btMatrix3x3 frameB;

	BulletMultiBodyConstraint(btMultiBody * bodyA, btMultiBody * bodyB, int linkA, int linkB, std::string constraintType):
	bodyA(bodyA), bodyB(bodyB), linkA(linkA), linkB(linkB), constraintType(constraintType){
		if (constraintType == "Slider"){
			constraint = std::make_unique<btMultiBodySliderConstraint>(bodyA, linkA, bodyB, linkB, pivotA, pivotB, frameA, frameB, jointAxis);
		}
		else if (constraintType == "Point2Point"){
			constraint = std::make_unique<btMultiBodyPoint2Point>(bodyA, linkA, bodyB, linkB, pivotA, pivotB);
		}
		else if (constraintType == "Gear") {
			constraint = std::make_unique<btMultiBodyGearConstraint>(bodyA, linkA, bodyB, linkB, pivotA, pivotB, frameA, frameB);
		}
		else if (constraintType == "Fixed") {
			constraint = std::make_unique<btMultiBodyFixedConstraint>(bodyA, linkA, bodyB, linkB, pivotA, pivotB, frameA, frameB);
		}
	}

	BulletMultiBodyConstraint(btMultiBody * bodyA, int linkA, std::string constraintType, std::map<std::string, btScalar> config):
	bodyA(bodyA), linkA(linkA), constraintType(constraintType), config(config){
		if (constraintType == "Spherical") {
			btScalar swingxRange = config["jointLowerLimit"];
			btScalar swingyRange = config["jointUpperLimit"];
			btScalar twistRange = config["twistLimit"];
			btScalar maxAppliedImpulse = config["jointMaxForce"];
			constraint = std::make_unique<btMultiBodySphericalJointLimit>(bodyA, linkA, swingxRange, swingyRange, twistRange, maxAppliedImpulse);
		}
		else if (constraintType == "Default") {
			btScalar lower = config["jointLowerLimit"];
			btScalar upper = config["jointUpperLimit"];
			constraint = std::make_unique<btMultiBodyJointLimitConstraint>(bodyA, linkA, lower, upper);
		}
		else if (constraintType == "DefaultMotor") {
			int linkDof = (int)config["linkDof"];
			btScalar desiredVelocity = config["desiredVelocity"];
			btScalar maxMotorImpulse = config["jointMaxForce"];
			constraint = std::make_unique<btMultiBodyJointMotor>(bodyA, linkA, linkDof, desiredVelocity, maxMotorImpulse);
		}
		else if (constraintType == "SphericalMotor") {
			btScalar maxMotorImpulse = config["jointMaxForce"];
			constraint = std::make_unique<btMultiBodySphericalJointMotor>(bodyA, linkA, maxMotorImpulse);
		}
	}
};

struct BulletMakeMultiBodyConstraint : zeno::INode {
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

ZENDEFNODE(BulletMakeMultiBodyConstraint, {
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
/*
 * Bullet Kinematics
 */

struct BulletCalcInverseKinematics : zeno::INode {
    virtual void apply() override {
		auto object = get_input<BulletMultiBodyObject>("object");
		auto gravity = zeno::vec_to_other<btVector3>(get_input<zeno::NumericObject>("gravity")->get<zeno::vec3f>());

		auto endEffectorLinkIndices = get_input<zeno::ListObject>("endEffectorLinkIndices")->get<int>();
		auto numEndEffectorLinkIndices = endEffectorLinkIndices.size();

		auto targetPositions = get_input<zeno::ListObject>("targetPositions")->get<zeno::vec3f>();
		auto targetOrientations = get_input<zeno::ListObject>("targetOrientations")->get<zeno::vec4f>();

		auto numIterations = get_input2<int>("numIterations"); // 20
		auto residualThreshold = get_input2<float>("residualThreshold"); // 1e-4

		auto ikMethod = std::get<std::string>(get_param("ikMethod"));

		auto ikHelperPtr = std::make_shared<IKTrajectoryHelper>();

	    btAlignedObjectArray<double> startingPositions;
	    startingPositions.reserve(object->multibody->getNumLinks());

		auto outputPoses = std::make_shared<ListObject>();
	    {
		    int DofIndex = 0;
		    for (int i = 0; i < object->multibody->getNumLinks(); ++i)
		    {
			    if (object->multibody->getLink(i).m_jointType >= 0 && object->multibody->getLink(i).m_jointType <= 2)
			    {
				    // 0, 1, 2 represent revolute, prismatic, and spherical joint types respectively. Skip the fixed joints.
				    double curPos = 0;
				    curPos = object->multibody->getJointPos(i);

				    startingPositions.push_back(curPos);
				    DofIndex++;
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
		    if (ikHelperPtr && validEndEffectorLinkIndices)
		    {
			    jacobian_linear.resize(numEndEffectorLinkIndices * 3 * numDofs);
			    jacobian_angular.resize(numEndEffectorLinkIndices * 3 * numDofs);
			    int jacSize = 0;

			    btInverseDynamics::btMultiBodyTreeCreator id_creator;
			    id_creator.createFromBtMultiBody(object->multibody.get(), false);
			    btInverseDynamics::MultiBodyTree* tree = tree = btInverseDynamics::CreateMultiBodyTree(id_creator);

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
				    if (ikMethod == "VEL_DLS_ORI_NULL")
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
						    outputPoses->arr.push_back(q_new[i]);
					    }
				    }
			    }
		    }
	    }

	    set_output("poses", std::move(outputPoses));
    }
};

ZENDEFNODE(BulletCalcInverseKinematics, {
	{"object", "gravity", "endEffectorLinkIndices", "targetPositions", "targetOrientations", {"int", "numIterations", "20"}, {"float", "residualThreshold", "0.0001"}},
	{"poses"},
	{{"enum VEL_DLS_ORI_NULL VEL_SDLS_ORI VEL_DLS_ORI VEL_DLS_NULL VEL_SDLS VEL_DLS", "IKMethod", "VEL_DLS_ORI_NULL"}},
	{"Bullet"},
});


/*
 * Bullet URDF Helper
 */

// URDFLinkContactInfo to BulletLinkCollider
struct BulletSetContactParameters : zeno::INode {
	virtual void apply() override {
		auto col = get_input<BulletMultiBodyLinkCollider>("collider");
		if(has_input("lateralFriction")){
			auto lateralFriction = get_input2<float>("lateralFriction");
			col->linkCollider->setFriction(lateralFriction);
		}
		if(has_input("restitution")){
			auto restitution = get_input2<float>("restitution");
			col->linkCollider->setRestitution(restitution);
		}
		if(has_input("rollingFriction")){
			auto rollingFriction = get_input2<float>("rollingFriction");
			col->linkCollider->setRollingFriction(rollingFriction);
		}
		if(has_input("spinningFriction")){
			auto spinningFriction = get_input2<float>("spinningFriction");
			col->linkCollider->setSpinningFriction(spinningFriction);
		}
		if(has_input("stiffness")){
			auto stiffness = get_input2<float>("stiffness");
			if(has_input("damping")) {
				auto damping = get_input2<float>("damping");
				col->linkCollider->setContactStiffnessAndDamping(stiffness, damping);
			}
			else{
				col->linkCollider->setContactStiffnessAndDamping(stiffness,0);
			}
		}else{
			if(has_input("damping")) {
				auto damping = get_input2<float>("damping");
				col->linkCollider->setContactStiffnessAndDamping(0, damping);
			}
		}
		if(std::get<std::string>(get_param("frictionAnchor"))=="true"){
			col->linkCollider->setCollisionFlags(col->linkCollider->getCollisionFlags() | btCollisionObject::CF_HAS_FRICTION_ANCHOR);
		}
		set_output("collider", std::move(col));
	}
};

ZENDEFNODE(BulletSetContactParameters, {
	{"collider", "lateralFriction", "restitution", "rollingFriction", "spinningFriction", "stiffness", "damping"},
	{"collider"},
	{{"enum true false", "frictionAnchor", "false"}},
	{"Bullet"}
});


struct BulletMultiBodyGetJointTorque : zeno::INode {
    virtual void apply() override {
        auto object = get_input<BulletMultiBodyObject>("object");
        auto link_id = get_input2<int>("linkIndex");
        btScalar torque;

        torque = object->multibody->getJointTorque(link_id);
        // out_torque = vec1f(other_to_vec<1>(torque));
        
        auto out_torque = std::make_shared<zeno::NumericObject>(torque);
        set_output("joint_torque", std::move(out_torque));
    }
};

ZENDEFNODE(BulletMultiBodyGetJointTorque, {
    {"object", "linkIndex"},
    {"torque"},
    {},
    {"Bullet"}
});

struct BulletMultiBodyGetJointState : zeno::INode {
    virtual void apply() override {
        auto object = get_input<BulletMultiBodyObject>("object");
        auto link_id = get_input2<int>("linkIndex");
        btScalar vel;
        btScalar pos;

        vel = object->multibody->getJointVel(link_id);
        pos = object -> multibody ->getJointPos(link_id);
        // out_torque = vec1f(other_to_vec<1>(torque));

        auto vel_ = std::make_shared<zeno::NumericObject>(vel);
        auto pos_ = std::make_shared<zeno::NumericObject>(pos);
        set_output("vel", std::move(vel_));
        set_output("pos", std::move(pos_));
    }
};

ZENDEFNODE(BulletMultiBodyGetJointState, {
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

// struct BulletCalculateEEForce : zeno::INode {
//     virtual void apply() {
//         auto object = get_input<BulletMultiBodyObject>("object");
//         auto endEffectorLinkIndices = get_input<zeno::ListObject>("endEffectorLinkIndices")->get<int>();
//         auto numEndEffectorLinkIndices = endEffectorLinkIndices.size();
//
//     }
// };
// ZENDEFNODE(BulletCalculateEEForce, {
//                                                {"object"},
//                                                {"EEForce"},
//                                                {},
//                                                {"Bullet"},
//                                            });
};