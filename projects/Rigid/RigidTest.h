#include <zeno/utils/logger.h>

// bullet basics
#include <BulletCollision/CollisionDispatch/btCollisionDispatcherMt.h>
#include <BulletCollision/CollisionShapes/btShapeHull.h>
#include <BulletDynamics/ConstraintSolver/btSequentialImpulseConstraintSolverMt.h>
#include <BulletDynamics/Dynamics/btDiscreteDynamicsWorldMt.h>
#include <BulletDynamics/Dynamics/btSimulationIslandManagerMt.h>
#include <LinearMath/btConvexHullComputer.h>
#include <btBulletDynamicsCommon.h>

// multibody dynamcis
#include <BulletDynamics/Featherstone/btMultiBody.h>
#include <BulletDynamics/Featherstone/btMultiBodyDynamicsWorld.h>
#include <BulletDynamics/Featherstone/btMultiBodyLinkCollider.h>
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
#include <BulletDynamics/Featherstone/btMultiBodyDynamicsWorld.h>
#include <BulletDynamics/Featherstone/btMultiBodyJointFeedback.h>

#ifndef ZENO_RIGIDTEST_H
#define ZENO_RIGIDTEST_H

using namespace zeno;

struct BulletTransform : zeno::IObject {
    btTransform trans;
};

struct BulletTriangleMesh : zeno::IObject {
    btTriangleMesh mesh;
};

struct BulletMultiBodyLinkCollider : zeno::IObject{
    // it is a child class of btCollisionObject.
    std::unique_ptr<btMultiBodyLinkCollider> linkCollider;

    BulletMultiBodyLinkCollider(btMultiBody *multiBody, int link){
        linkCollider = std::make_unique<btMultiBodyLinkCollider>(multiBody, link);
    }
};

struct BulletCollisionShape : zeno::IObject {
    std::unique_ptr<btCollisionShape> shape;

    BulletCollisionShape(std::unique_ptr<btCollisionShape> &&shape)
        : shape(std::move(shape)){

    }
};

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
    std::unique_ptr<btCollisionWorld> collisionWorld;

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
        zeno::log_debug("creating bullet world {}", (void *)this);
    }
#endif

    void addObject(std::shared_ptr<BulletObject> obj) {
        zeno::log_debug("adding object {}", (void *)obj.get());
        dynamicsWorld->addRigidBody(obj->body.get());
        objects.insert(std::move(obj));
    }

    void removeObject(std::shared_ptr<BulletObject> const &obj) {
        zeno::log_debug("removing object {}", (void *)obj.get());
        dynamicsWorld->removeRigidBody(obj->body.get());
        objects.erase(obj);
    }

    void setObjectList(std::vector<std::shared_ptr<BulletObject>> objList) {
        std::set<std::shared_ptr<BulletObject>> objSet;
        zeno::log_debug("setting object list len={}", objList.size());
        zeno::log_debug("existing object list len={}", objects.size());
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
        zeno::log_debug("adding constraint {}", (void *)cons.get());
        dynamicsWorld->addConstraint(cons->constraint.get(), true);
        constraints.insert(std::move(cons));
    }

    void removeConstraint(std::shared_ptr<BulletConstraint> const &cons) {
        zeno::log_debug("removing constraint {}", (void *)cons.get());
        dynamicsWorld->removeConstraint(cons->constraint.get());
        constraints.erase(cons);
    }

    void setConstraintList(std::vector<std::shared_ptr<BulletConstraint>> consList) {
        std::set<std::shared_ptr<BulletConstraint>> consSet;
        zeno::log_debug("setting constraint list len={}", consList.size());
        zeno::log_debug("existing constraint list len={}", constraints.size());
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
        zeno::log_debug("stepping with dt={}, steps={}, len(objects)={}", dt, steps, objects.size());
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

    BulletMultiBodyObject(btMultiBody * mb){
        multibody = std::unique_ptr<btMultiBody>(mb);
    }

    BulletMultiBodyObject(int n_links, btScalar mass, btVector3 inertia, bool fixedBase, bool canSleep) : n_links(n_links), mass(mass), inertia(inertia), fixedBase(fixedBase), canSleep(canSleep)
    {
        multibody = std::make_unique<btMultiBody>(n_links, mass, inertia, fixedBase, canSleep);
        multibody->setBaseWorldTransform(btTransform::getIdentity());
    }
};

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
        dynamicsWorld->setGravity(btVector3(0, -9.81, 0));

        zeno::log_debug("creating bullet multibody dynamics world {}", (void *)this);
    }

};

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

struct BulletMultiBodyJointMotor : zeno::IObject {
    std::unique_ptr<btMultiBodyJointMotor> jointMotor;

    BulletMultiBodyJointMotor(btMultiBody* mb, int linkIndex, int linkDof, btScalar desiredVelocity, btScalar maxMotorImpulse){
        jointMotor = std::make_unique<btMultiBodyJointMotor>(mb, linkIndex, linkDof, desiredVelocity, maxMotorImpulse);
    }
};

#endif //ZENO_RIGIDTEST_H
