#include <zeno/zeno.h>
#include <zeno/ListObject.h>
#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>
#include <btBulletDynamicsCommon.h>
#include <BulletCollision/CollisionShapes/btShapeHull.h>
#include <BulletCollision/CollisionShapes/btConvexPointCloudShape.h>
#include <hacdCircularList.h>
#include <hacdVector.h>
#include <hacdICHull.h>
#include <hacdGraph.h>
#include <hacdHACD.h>
#include <memory>
#include <vector>


struct BulletTransform : zen::IObject {
    btTransform trans;
};

struct BulletCollisionShape : zen::IObject {
    std::unique_ptr<btCollisionShape> shape;

    BulletCollisionShape(std::unique_ptr<btCollisionShape> &&shape)
        : shape(std::move(shape)) {
    }
};

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

struct BulletMakeBoxShape : zen::INode {
    virtual void apply() override {
        auto v3size = get_input<zen::NumericObject>("v3size")->get<zen::vec3f>();
        auto shape = std::make_shared<BulletCollisionShape>(
            std::make_unique<btBoxShape>(zen::vec_to_other<btVector3>(v3size)));
        set_output("shape", std::move(shape));
    }
};

ZENDEFNODE(BulletMakeBoxShape, {
    {"v3size"},
    {"shape"},
    {},
    {"Rigid"},
});

struct BulletMakeSphereShape : zen::INode {
    virtual void apply() override {
        auto radius = get_input<zen::NumericObject>("radius")->get<float>();
        auto shape = std::make_unique<BulletCollisionShape>(
            std::make_unique<btSphereShape>(btScalar(radius)));
        set_output("shape", std::move(shape));
    }
};

ZENDEFNODE(BulletMakeSphereShape, {
    {"radius"},
    {"shape"},
    {},
    {"Rigid"},
});


struct BulletTriangleMesh : zen::IObject {
    btTriangleMesh mesh;
};

struct PrimitiveToBulletMesh : zen::INode {
    virtual void apply() override {
        auto prim = get_input<zen::PrimitiveObject>("prim");
        auto mesh = std::make_unique<BulletTriangleMesh>();
        auto pos = prim->attr<zen::vec3f>("pos");
        for (int i = 0; i < prim->tris.size(); i++) {
            auto f = prim->tris[i];
            mesh->mesh.addTriangle(
                zen::vec_to_other<btVector3>(pos[f[0]]),
                zen::vec_to_other<btVector3>(pos[f[1]]),
                zen::vec_to_other<btVector3>(pos[f[2]]), true);
        }
        set_output("mesh", std::move(mesh));
    }
};

ZENDEFNODE(PrimitiveToBulletMesh, {
    {"prim"},
    {"mesh"},
    {},
    {"Rigid"},
});

struct PrimitiveConvexDecomposition : zen::INode {
    virtual void apply() override {
        auto prim = get_input<zen::PrimitiveObject>("prim");
        auto &pos = prim->attr<zen::vec3f>("pos");

        std::vector<HACD::Vec3<HACD::Real>> points;
        std::vector<HACD::Vec3<long>> triangles;

        for (int i = 0; i < pos.size(); i++) {
            points.push_back(
                zen::vec_to_other<HACD::Vec3<HACD::Real>>(pos[i]));
        }

        for (int i = 0; i < prim->tris.size(); i++) {
            triangles.push_back(
                zen::vec_to_other<HACD::Vec3<long>>(prim->tris[i]));
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

        auto listPrim = std::make_shared<zen::ListObject>();
        listPrim->arr.clear();

        printf("hacd got %d clusters\n", nClusters);
        for (size_t c = 0; c < nClusters; c++) {
            size_t nPoints = hacd.GetNPointsCH(c);
            size_t nTriangles = hacd.GetNTrianglesCH(c);
            printf("hacd cluster %d have %d points, %d triangles\n",
                c, nPoints, nTriangles);

            points.clear();
            points.resize(nPoints);
            triangles.clear();
            triangles.resize(nTriangles);
            hacd.GetCH(c, points.data(), triangles.data());

            auto outprim = std::make_shared<zen::PrimitiveObject>();
            outprim->resize(nPoints);
            outprim->tris.resize(nTriangles);

            auto &outpos = outprim->add_attr<zen::vec3f>("pos");
            for (size_t i = 0; i < nPoints; i++) {
                auto p = points[i];
                //printf("point %d: %f %f %f\n", i, p.X(), p.Y(), p.Z());
                outpos[i] = zen::vec3f(p.X(), p.Y(), p.Z());
            }

            for (size_t i = 0; i < nTriangles; i++) {
                auto p = triangles[i];
                //printf("triangle %d: %d %d %d\n", i, p.X(), p.Y(), p.Z());
                outprim->tris[i] = zen::vec3i(p.X(), p.Y(), p.Z());
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
    {"Rigid"},
});


struct BulletMakeConvexHullShape : zen::INode {
    virtual void apply() override {
        auto triMesh = &get_input<BulletTriangleMesh>("triMesh")->mesh;
        auto inShape = std::make_unique<btConvexTriangleMeshShape>(triMesh);
        auto hull = std::make_unique<btShapeHull>(inShape.get());
        hull->buildHull(0.01, 256);
        auto convex = std::make_unique<btConvexHullShape>(
             (const float *)hull->getVertexPointer(), hull->numVertices());
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
    {"triMesh"},
    {"shape"},
    {},
    {"Rigid"},
});

struct BulletMakeCompoundShape : zen::INode {
    virtual void apply() override {
        auto compound = std::make_unique<btCompoundShape>();
        auto shape = std::make_shared<BulletCompoundShape>(std::move(compound));
        set_output("compound", std::move(shape));
    }
};

ZENDEFNODE(BulletMakeCompoundShape, {
    {""},
    {"compound"},
    {},
    {"Rigid"},
});

struct BulletCompoundAddChild : zen::INode {
    virtual void apply() override {
        auto compound = get_input<BulletCompoundShape>("compound");
        auto childShape = get_input<BulletCollisionShape>("childShape");
        auto trans = get_input<BulletTransform>("trans")->trans;

        compound->addChild(trans, std::move(childShape));
        set_output_ref("compound", get_input_ref("compound"));
    }
};

ZENDEFNODE(BulletCompoundAddChild, {
    {"compound", "childShape", "trans"},
    {"compound"},
    {},
    {"Rigid"},
});


struct BulletMakeTransform : zen::INode {
    virtual void apply() override {
        auto trans = std::make_unique<BulletTransform>();
        trans->trans.setIdentity();
        if (has_input("origin")) {
            auto origin = get_input<zen::NumericObject>("origin")->get<zen::vec3f>();
            trans->trans.setOrigin(zen::vec_to_other<btVector3>(origin));
        }
        if (has_input("rotation")) {
            auto rotation = get_input<zen::NumericObject>("rotation")->get<zen::vec3f>();
            trans->trans.setRotation(zen::vec_to_other<btQuaternion>(rotation));
        }
        set_output("trans", std::move(trans));
    }
};

ZENDEFNODE(BulletMakeTransform, {
    {"origin", "rotation"},
    {"trans"},
    {},
    {"Rigid"},
});

struct BulletComposeTransform : zen::INode {
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
    {"Rigid"},
});


struct BulletObject : zen::IObject {
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

struct BulletMakeObject : zen::INode {
    virtual void apply() override {
        auto shape = get_input<BulletCollisionShape>("shape");
        auto mass = get_input<zen::NumericObject>("mass")->get<float>();
        auto trans = get_input<BulletTransform>("trans");
        auto object = std::make_unique<BulletObject>(
            mass, trans->trans, shape);
        set_output("object", std::move(object));
    }
};

ZENDEFNODE(BulletMakeObject, {
    {"shape", "trans", "mass"},
    {"object"},
    {},
    {"Rigid"},
});

struct BulletGetObjTransform : zen::INode {
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
    {"Rigid"},
});

struct BulletGetObjMotion : zen::INode {
    virtual void apply() override {
        auto obj = get_input<BulletObject>("object");
        auto body = obj->body.get();
        auto linearVel = zen::IObject::make<zen::NumericObject>();
        auto angularVel = zen::IObject::make<zen::NumericObject>();
        linearVel->set<zen::vec3f>(zen::vec3f(0));
        angularVel->set<zen::vec3f>(zen::vec3f(0));

        if (body && body->getLinearVelocity() ) {
            auto v = body->getLinearVelocity();
            linearVel->set<zen::vec3f>(zen::vec3f(v.x(), v.y(), v.z()));
        }
        if (body && body->getAngularVelocity() ){
            auto w = body->getAngularVelocity();
            angularVel->set<zen::vec3f>(zen::vec3f(w.x(), w.y(), w.z()));
        }
        set_output("linearVel", linearVel);
        set_output("angularVel", angularVel);
    }
};

ZENDEFNODE(BulletGetObjMotion, {
    {"object"},
    {"linearVel", "angularVel"},
    {},
    {"Rigid"},
});

struct RigidVelToPrimitive : zen::INode {
    virtual void apply() override {
        auto prim = get_input<zen::PrimitiveObject>("prim");
        auto com = get_input<zen::NumericObject>("centroid")->get<zen::vec3f>();
        auto lin = get_input<zen::NumericObject>("linearVel")->get<zen::vec3f>();
        auto ang = get_input<zen::NumericObject>("angularVel")->get<zen::vec3f>();

        auto &pos = prim->attr<zen::vec3f>("pos");
        auto &vel = prim->add_attr<zen::vec3f>("vel");
        for (size_t i = 0; i < prim->size(); i++) {
            vel[i] = lin + zen::cross(ang, pos[i] - com);
        }

        set_output_ref("prim", get_input_ref("prim"));
    }
};

ZENDEFNODE(RigidVelToPrimitive, {
    {"prim", "centroid", "linearVel", "angularVel"},
    {"prim"},
    {},
    {"Rigid"},
});

struct BulletExtractTransform : zen::INode {
    virtual void apply() override {
        auto trans = &get_input<BulletTransform>("trans")->trans;
        auto origin = std::make_unique<zen::NumericObject>();
        auto rotation = std::make_unique<zen::NumericObject>();
        origin->set(zen::other_to_vec<3>(trans->getOrigin()));
        rotation->set(zen::other_to_vec<4>(trans->getRotation()));
        set_output("origin", std::move(origin));
        set_output("rotation", std::move(rotation));
    }
};

ZENDEFNODE(BulletExtractTransform, {
    {"trans"},
    {"origin", "rotation"},
    {},
    {"Rigid"},
});


struct BulletWorld : zen::IObject {
    std::unique_ptr<btDefaultCollisionConfiguration> collisionConfiguration = std::make_unique<btDefaultCollisionConfiguration>();
    std::unique_ptr<btCollisionDispatcher> dispatcher = std::make_unique<btCollisionDispatcher>(collisionConfiguration.get());
    std::unique_ptr<btBroadphaseInterface> overlappingPairCache = std::make_unique<btDbvtBroadphase>();
    std::unique_ptr<btSequentialImpulseConstraintSolver> solver = std::make_unique<btSequentialImpulseConstraintSolver>();

    std::unique_ptr<btDiscreteDynamicsWorld> dynamicsWorld = std::make_unique<btDiscreteDynamicsWorld>(dispatcher.get(), overlappingPairCache.get(), solver.get(), collisionConfiguration.get());

    std::vector<std::shared_ptr<BulletObject>> objects;

    BulletWorld() {
        dynamicsWorld->setGravity(btVector3(0, -10, 0));
    }

    void addObject(std::shared_ptr<BulletObject> obj) {
        dynamicsWorld->addRigidBody(obj->body.get());
        objects.push_back(std::move(obj));
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

    void step(float dt = 1.f / 60.f) {
        std::cout<<dt<<std::endl;
        for(int i=0;i<10;i++)
            dynamicsWorld->stepSimulation(0.1*dt, 1, 0.1*dt);

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

struct BulletMakeWorld : zen::INode {
    virtual void apply() override {
        auto world = std::make_unique<BulletWorld>();
        set_output("world", std::move(world));
    }
};

ZENDEFNODE(BulletMakeWorld, {
    {},
    {"world"},
    {},
    {"Rigid"},
});

struct BulletSetWorldGravity : zen::INode {
    virtual void apply() override {
        auto world = get_input<BulletWorld>("world");
        auto gravity = get_input<zen::NumericObject>("gravity")->get<zen::vec3f>();
        world->dynamicsWorld->setGravity(zen::vec_to_other<btVector3>(gravity));
    }
};

ZENDEFNODE(BulletSetWorldGravity, {
    {"world", "gravity"},
    {},
    {},
    {"Rigid"},
});

struct BulletStepWorld : zen::INode {
    virtual void apply() override {
        auto world = get_input<BulletWorld>("world");
        auto dt = get_input<zen::NumericObject>("dt")->get<float>();
        world->step(dt);
    }
};

ZENDEFNODE(BulletStepWorld, {
    {"world", "dt"},
    {},
    {},
    {"Rigid"},
});

struct BulletWorldAddObject : zen::INode {
    virtual void apply() override {
        auto world = get_input<BulletWorld>("world");
        auto object = get_input<BulletObject>("object");
        object->body->setDamping(0,0);
        world->addObject(std::move(object));
        set_output_ref("world", get_input_ref("world"));
    }
};

ZENDEFNODE(BulletWorldAddObject, {
    {"world", "object"},
    {"world"},
    {},
    {"Rigid"},
});

struct BulletObjectApplyForce:zen::INode {
    virtual void apply() override {
        auto object = get_input<BulletObject>("object");
        auto forceImpulse = get_input<zen::NumericObject>("ForceImpulse")->get<zen::vec3f>();
        auto torqueImpulse = get_input<zen::NumericObject>("TorqueImpulse")->get<zen::vec3f>();
        object->body->applyCentralImpulse(zen::vec_to_other<btVector3>(forceImpulse));
        object->body->applyTorqueImpulse(zen::vec_to_other<btVector3>(torqueImpulse));
    }
};

ZENDEFNODE(BulletObjectApplyForce, {
    {"object", "ForceImpulse", "TorqueImpulse"},
    {},
    {},
    {"Rigid"},
});