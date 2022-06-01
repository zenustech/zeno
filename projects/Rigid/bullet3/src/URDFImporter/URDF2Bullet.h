#ifndef _URDF2BULLET_H
#define _URDF2BULLET_H
#include "../LinearMath/btAlignedObjectArray.h"
#include "../LinearMath/btTransform.h"
#include <string>
#include "URDFJointTypes.h"  //for UrdfMaterialColor cache
#include "../BulletDynamics/Featherstone/btMultiBody.h"
#include "zeno/types/PrimitiveObject.h"
#include <vector>
#include <memory>

class btVector3;
class btTransform;
class btMultiBodyDynamicsWorld;
class btDiscreteDynamicsWorld;
class btTransform;

class URDFImporterInterface;

struct UrdfVisualShapeCache
{
	btAlignedObjectArray<UrdfMaterialColor> m_cachedUrdfLinkColors;
	btAlignedObjectArray<int> m_cachedUrdfLinkVisualShapeIndices;
};
//#define USE_DISCRETE_DYNAMICS_WORLD
#ifdef USE_DISCRETE_DYNAMICS_WORLD
	void ConvertURDF2Bullet(const URDFImporterInterface& u2b,
						MultiBodyCreationInterface& creationCallback,
						const btTransform& rootTransformInWorldSpace,
						btDiscreteDynamicsWorld* world,
						bool createMultiBody,
						const char* pathPrefix,
						int flags = 0,
						UrdfVisualShapeCache* cachedLinkGraphicsShapes = 0);

#else
btMultiBody* ConvertURDF2Bullet(const URDFImporterInterface& u2b,
						const btTransform& rootTransformInWorldSpace,
						btMultiBodyDynamicsWorld* world,
						bool createMultiBody,
						int flags = 0,
						UrdfVisualShapeCache* cachedLinkGraphicsShapes = 0);

#endif
#endif  //_URDF2BULLET_H
