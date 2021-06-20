#include <zen/zen.h>
#include <zen/NumericObject.h>
#include <zen/PrimitiveObject.h>
#include <btBulletDynamicsCommon.h>
#include <BulletCollision/CollisionShapes/btShapeHull.h>
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
                zen::vec_to_other<btVector3>(pos[f[2]]));
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

struct BulletMakeConvexHullShape : zen::INode {
    virtual void apply() override {

        auto triMesh = &get_input<BulletTriangleMesh>("triMesh")->mesh;
        auto inShape = std::make_unique<btConvexTriangleMeshShape>(triMesh);
        auto hull = std::make_unique<btShapeHull>(inShape.get());
        hull->buildHull(inShape->getMargin());
        auto convex = std::make_unique<btConvexHullShape>(
            reinterpret_cast<const float *>(hull->getVertexPointer()),
            hull->numVertices());

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
        dynamicsWorld->stepSimulation(dt, 10);

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


#if 0
/// This is a Hello World program for running a basic Bullet physics simulation

int main(int argc, char** argv)
{
    ///-----initialization_start-----

    ///collision configuration contains default setup for memory, collision setup. Advanced users can create their own configuration.
    BulletWorld w;

    ///-----initialization_end-----

    //keep track of the shapes, we release memory at exit.
    //make sure to re-use collision shapes among rigid bodies whenever possible!
    w.addGround();
    w.addBall();

    ///create a few basic rigid bodies

    /// Do some simulation

    ///-----stepsimulation_start-----
    for (int i = 0; i < 120; i++) {
        w.step();
    }

    ///-----stepsimulation_end-----

    //cleanup in the reverse order of creation/initialization

    ///-----cleanup_start-----

    //remove the rigidbodies from the dynamics world and delete them


    return 0;
}
#endif
