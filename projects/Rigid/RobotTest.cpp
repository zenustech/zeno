#include <memory>
#include <vector>
#include <iostream>

// zeno basics
#include <zeno/ListObject.h>
#include <zeno/DictObject.h>
#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/logger.h>
#include <zeno/zeno.h>
#include <zeno/utils/fileio.h>

#include "RigidTest.h"
#include "zeno/types/StringObject.h"
#include "zeno/types/DictObject.h"

#include <BulletCollision/CollisionDispatch/btCollisionDispatcherMt.h>
#include <BulletCollision/CollisionShapes/btShapeHull.h>


// Bullet URDFImporter
#include <URDFImporter/BulletUrdfImporter.h>
#include <URDFImporter/MyMultiBodyCreator.h>
#include <URDFImporter/MultiBodyCreationInterface.h>
#include <URDFImporter/URDF2Bullet.h>

namespace {
using namespace zeno;

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
{"Robot"}
});

bool supportsJointMotor (btMultiBody* mb, int mbLinkIndex){
    bool canHaveMotor = (mb->getLink(mbLinkIndex).m_jointType == btMultibodyLink::eRevolute || mb->getLink(mbLinkIndex).m_jointType == btMultibodyLink::ePrismatic);
    return canHaveMotor;
}

struct RobotLoadURDF : zeno::INode {
    virtual void apply() override {
        auto path = get_input<zeno::StringObject>("path")->get();
        auto globalScaling = get_input2<float>("globalScaling");
        auto world = get_input<BulletMultiBodyWorld>("world");
        auto fixedBase = (std::get<std::string>(get_param("fixedBase")) == "true");

        // load URDF by BulletURDFImporter
        int flags = 0; // TODO: make it configurable later

        BulletURDFImporter u2b(globalScaling, flags); // create internal m_data
        bool loadOk = u2b.loadURDF(path.c_str(), fixedBase);

        if (loadOk) {
            btTransform identityTrans;
            identityTrans.setIdentity();

            // convert the URDF model to MultiBody
            auto mb = ConvertURDF2Bullet(u2b, identityTrans, world->dynamicsWorld.get(), 1);
            auto object = std::make_shared<BulletMultiBodyObject>(mb);

            for (size_t i = 0; i < object->multibody->getNumLinks(); i++){
                btMultiBodyJointFeedback* fb = new btMultiBodyJointFeedback();
                object->multibody->getLink(i).m_jointFeedback = fb;
            }
            std::cout<< "\nload URDF done! We have " << object->multibody->getNumLinks() << " links in this Robot!\n";

            auto visualMap = u2b.getVisualMap();
            std::cout<< "\nload URDF done! We have " << visualMap.size() << " links in this Robot!\n";
            auto finalVisualMap = std::make_shared<zeno::DictObject>();

            for (auto &kv : visualMap){
                finalVisualMap->lut[std::to_string(kv.first)] = kv.second;
                std::cout << "graphicsId: "<< kv.first << std::endl;
            }

            int numLinks = mb->getNumLinks();
            for (int i = 0; i < numLinks; i++)
            {
                int mbLinkIndex = i;
                float maxMotorImpulse = 1.f;

                if (supportsJointMotor(mb, mbLinkIndex))
                {
                    int dof = 0;
                    btScalar desiredVelocity = 0.f;
                    btMultiBodyJointMotor* motor = new btMultiBodyJointMotor(mb, mbLinkIndex, dof, desiredVelocity, maxMotorImpulse);
                    motor->setPositionTarget(0, 0);
                    motor->setVelocityTarget(0, 1);
                    //motor->setRhsClamp(gRhsClamp);
                    //motor->setMaxAppliedImpulse(0);
                    mb->getLink(mbLinkIndex).m_userPtr = motor;
#ifndef USE_DISCRETE_DYNAMICS_WORLD
                    world->dynamicsWorld->addMultiBodyConstraint(motor);
#endif
                    motor->finalizeMultiDof();
                }
                if (mb->getLink(mbLinkIndex).m_jointType == btMultibodyLink::eSpherical)
                {
                    btMultiBodySphericalJointMotor* motor = new btMultiBodySphericalJointMotor(mb, mbLinkIndex, 1000 * maxMotorImpulse);
                    mb->getLink(mbLinkIndex).m_userPtr = motor;
#ifndef USE_DISCRETE_DYNAMICS_WORLD
                    world->dynamicsWorld->addMultiBodyConstraint(motor);
#endif
                    motor->finalizeMultiDof();
                }
            }

            int numCollisionObjects = world->dynamicsWorld->getNumCollisionObjects();
            std::cout << "now the transform for colobject:\n";
            for (size_t i = 0; i < numCollisionObjects; i++) {
                btCollisionObject* colObj = world->dynamicsWorld->getCollisionObjectArray()[i];
                btCollisionShape* collisionShape = colObj->getCollisionShape();

                btTransform trans = colObj->getWorldTransform();
                int graphicsIndex = colObj->getUserIndex();
                std::cout << "graphicsId: " << graphicsIndex << " -- " << trans.getOrigin()[0] << "," << trans.getOrigin()[1] << "," << trans.getOrigin()[2] << "\n";
                std::cout << trans.getRotation()[0] << "," << trans.getRotation()[1] << "," << trans.getRotation()[2] << "," << trans.getRotation()[3] << "\n";

            }

            set_output("world", std::move(world));
            set_output("object", std::move(object));
            set_output("visualMap", std::move(finalVisualMap));
        }
    }
};

ZENDEFNODE(RobotLoadURDF, {
    {"path", {"float", "globalScaling", "1"}, "world"},
    {"world", "object", "visualMap"},
    {{"enum true false", "fixedBase", "true"}},
    {"Robot"},
});

struct RobotGetLinkName : zeno::INode{
    virtual void apply() override {
        auto index = get_input2<int>("linkIndex");
        auto mb = get_input<BulletMultiBodyObject>("object");
        std::cout<<"Name for Link # " << index << " is " << mb->multibody->getLink(index).m_linkName << std::endl;
    }
};

ZENDEFNODE(RobotGetLinkName, {
    {"object", "index"},
    {},
    {},
    {"Robot"},
});

struct RobotGetJointName : zeno::INode{
    virtual void apply() override {
        auto index = get_input2<int>("linkIndex");
        auto mb = get_input<BulletMultiBodyObject>("object");
        std::cout<<"Name for Joint # " << index << " is " << mb->multibody->getLink(index).m_jointName << std::endl;
    }
};

ZENDEFNODE(RobotGetJointName, {
    {"object", "index"},
    {},
    {},
    {"Robot"},
});

struct RobotSetJointPoses : zeno::INode {
    virtual void apply() override {
        auto object = get_input<BulletMultiBodyObject>("object");
        btAlignedObjectArray<btScalar> qDesiredArray;

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

        for (size_t i = 0; i< object->multibody->getNumLinks(); i++){
            object->multibody->setJointPos(i, qDesiredArray[i]);
        }

        set_output("object", object);
    }
};

ZENDEFNODE(RobotSetJointPoses, {
    {"object", "qDesiredList"},
    {"object"},
    {},
    {"Robot"}
});
};