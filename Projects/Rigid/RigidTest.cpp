#include <zen/zen.h>
#include <stdio.h>
#include <btBulletDynamicsCommon.h>
#include <memory>
#include <vector>


struct BulletCollisionShape : zen::IObject {
    std::unique_ptr<btCollisionShape> shape;

    BulletCollisionShape(std::unique_ptr<btCollisionShape> &&shape)
        : shape(shape) {
    }
};

struct BulletMakeBoxShape : zen::INode {
    virtual void apply() override {
        auto v3size = get_input<zen::NumericObject>("v3size").get<zen::vec3f>();
        auto shape = std::make_unique<BulletCollisionShape>(
            std::make_unique<btBoxShape>(v3size));
        set_output("shape", std::move(shape));
    }
};

ZENDEFNODE(BulletMakeBoxShape, {
    {"v3size"},
    {"shape"},
    {},
    {"rigidbody"},
});

struct BulletMakeSphereShape : zen::INode {
    virtual void apply() override {
        auto radius = get_input<zen::NumericObject>("radius").get<float>();
        auto shape = std::make_unique<BulletCollisionShape>(
            std::make_unique<btBoxShape>(v3size));
        set_output("shape", std::move(shape));
    }
};

ZENDEFNODE(BulletMakeSphereShape, {
    {"radius"},
    {"shape"},
    {},
    {"rigidbody"},
});


struct BulletTransform : zen::IObject {
    btTransform trans;
};

struct BulletMakeTransform : zen::INode {
    virtual void apply() override {
        auto origin = get_input<zen::NumericObject>("origin").get<zen::vec3f>();
        auto trans = std::make_unique<BulletTransform>();
        trans->trans.setIdentity();
        trans->trans.setOrigin(btVector3(origin[0], origin[1], origin[2]));
        set_output("trans", std::move(trans));
    }
};

ZENDEFNODE(BulletMakeTransform, {
    {"origin"},
    {"trans"},
    {},
    {"rigidbody"},
});


struct BulletObject : zen::IObject {
    std::unique_ptr<btCollisionShape> colShape;
    std::unique_ptr<btDefaultMotionState> myMotionState;
    std::unique_ptr<btRigidBody> body;
    btScalar mass = 0.f;
    btTransform trans;

    BulletObject(btScalar mass_,
        btTransform const &trans,
        std::unique_ptr<btCollisionShape> &&colShape_)
        : mass(mass_), colShape(std::move(colShape_))
    {
        btVector3 localInertia(0, 0, 0);
        if (mass != 0)
            colShape->calculateLocalInertia(mass, localInertia);

        //using motionstate is optional, it provides interpolation capabilities, and only synchronizes 'active' objects
        myMotionState = std::make_unique<btDefaultMotionState>(trans);
        btRigidBody::btRigidBodyConstructionInfo rbInfo(mass, myMotionState.get(), colShape.get(), localInertia);
        body = std::make_unique<btRigidBody>(rbInfo);
    }
};

struct BulletMakeObject : zen::INode {
    virtual void apply() override {
        auto shape = get_input<BulletCollisionShape>("shape");
        auto mass = get_input<zen::NumericObject>("mass").get<float>();
        auto trans = get_input<BulletTransform>();
        auto object = std::make_unique<BulletObject>(
            mass, trans->trans, std::move(shape));
        set_output("object", std::move(object));
    }
};

ZENDEFNODE(BulletMakeObject, {
    {"shape", "trans", "mass"},
    {"object"},
    {},
    {"rigidbody"},
});


struct BulletWorld {

    std::unique_ptr<btDefaultCollisionConfiguration> collisionConfiguration = std::make_unique<btDefaultCollisionConfiguration>();
    std::unique_ptr<btCollisionDispatcher> dispatcher = std::make_unique<btCollisionDispatcher>(collisionConfiguration.get());
    std::unique_ptr<btBroadphaseInterface> overlappingPairCache = std::make_unique<btDbvtBroadphase>();
    std::unique_ptr<btSequentialImpulseConstraintSolver> solver = std::make_unique<btSequentialImpulseConstraintSolver>();

    std::unique_ptr<btDiscreteDynamicsWorld> dynamicsWorld = std::make_unique<btDiscreteDynamicsWorld>(dispatcher.get(), overlappingPairCache.get(), solver.get(), collisionConfiguration.get());

    std::vector<std::unique_ptr<BulletObject>> objects;

    BulletWorld() {
        dynamicsWorld->setGravity(btVector3(0, -10, 0));
    }

    void addObject(std::unique_ptr<BulletObject> &&obj) {
        dynamicsWorld->addRigidBody(obj->body.get());
        objects.push_back(std::move(obj));
    }

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
    }

    void step() {
        dynamicsWorld->stepSimulation(1.f / 60.f, 10);

        for (int j = dynamicsWorld->getNumCollisionObjects() - 1; j >= 0; j--)
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
        }
    }
};

struct BulletMakeWorld : zen::INode {
    virtual void apply() override {
        auto world = std::make_unique<BulletWorld>();
        set_output("world", std::move(world));
    }
};

ZENDEFNODE(BulletMakeWorld, {
    {"world"},
    {},
    {"rigidbody"},
});

struct BulletSetWorldGravity : zen::INode {
    virtual void apply() override {
        auto world = get_input<BulletWorld>("world");
        auto gravity = get_input<zen::NumericObject>("gravity").get<zen::vec3f>();
        world->dynamicsWorld->setGravity(
            btVector3(gravity[0], gravity[1], gravity[2]));
    }
};

ZENDEFNODE(BulletSetWorldGravity, {
    {"world", "gravity"},
    {},
    {},
    {"rigidbody"},
});

struct BulletWorldAddObject : zen::INode {
    virtual void apply() override {
        auto world = get_input<BulletWorld>("world");
        auto object = get_input<BulletObject>("object");
        world->addObject(object);
    }
};

ZENDEFNODE(BulletWorldAddObject, {
    {"world", "object"},
    {},
    {},
    {"rigidbody"},
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
