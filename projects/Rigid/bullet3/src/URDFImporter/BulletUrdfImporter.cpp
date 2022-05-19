/* Copyright (C) 2015 Google

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it freely,
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/
#include <string>
#include <iostream>
#include <fstream>
#include <list>

#include "BulletUrdfImporter.h"
//#include "URDFImporterInterface.h"
#include "CommonFileIOInterface.h"
#include "b3ResourcePath.h"
#include "b3BulletDefaultFileIO.h"
#include "UrdfParser.h"
#include "ShapeData.h"
#include "URDF2Bullet.h"  //for flags
#include "ReadObjPrim.h"

#include "../btBulletCollisionCommon.h"
#include "../BulletCollision/CollisionShapes/btShapeHull.h"  //to create a tesselation of a generic btConvexShape
#include "../Bullet3Common/b3FileUtils.h"

#include "zeno/zeno.h"
#include "zeno/utils/vec.h"
#include "zeno/utils/fileio.h"
#include "zeno/types/ListObject.h"
#include "zeno/types/PrimitiveObject.h"
#include "b3ImportMeshUtility.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/quaternion.hpp>

static btScalar gUrdfDefaultCollisionMargin = 0.001;
using namespace zeno;

ATTRIBUTE_ALIGNED16(struct)
BulletURDFInternalData
{
	BT_DECLARE_ALIGNED_ALLOCATOR();
	b3BulletDefaultFileIO m_defaultFileIO;
	UrdfParser m_urdfParser;
	struct CommonFileIOInterface* m_fileIO;
	std::string m_sourceFile;
	char m_pathPrefix[1024];
	int m_bodyId;
	btHashMap<btHashInt, UrdfMaterialColor> m_linkColors; //link to primitive
	btAlignedObjectArray<btCollisionShape*> m_allocatedCollisionShapes;
	btAlignedObjectArray<int> m_allocatedTextures; //link to primitive
	mutable btAlignedObjectArray<btTriangleMesh*> m_allocatedMeshInterfaces;
	btHashMap<btHashPtr, UrdfCollision> m_bulletCollisionShape2UrdfCollision;
	int m_flags;
    std::vector<std::shared_ptr<zeno::PrimitiveObject>> visualList;
    std::vector<std::shared_ptr<zeno::PrimitiveObject>> mbVisualList;

	void setSourceFile(const std::string& fileName)
	{
		m_sourceFile = fileName;
		m_urdfParser.setSourceFile(fileName);
	}

	BulletURDFInternalData()
		:m_urdfParser(&m_defaultFileIO),
		m_fileIO(&m_defaultFileIO)
	{
		m_pathPrefix[0] = 0;
		m_flags = 0;
	}

	void setGlobalScaling(btScalar scaling)
	{
		m_urdfParser.setGlobalScaling(scaling);
	}


};

void BulletURDFImporter::printTree()
{
	//	btAssert(0);
}

BulletURDFImporter::BulletURDFImporter(double globalScaling, int flags)
{
	m_data = new BulletURDFInternalData();
	m_data->setGlobalScaling(globalScaling);
	m_data->m_flags = flags;
}

struct BulletErrorLogger : public ErrorLogger
{
	int m_numErrors;
	int m_numWarnings;

	BulletErrorLogger()
		: m_numErrors(0),
		  m_numWarnings(0)
	{
	}
	virtual void reportError(const char* error)
	{
		m_numErrors++;
		b3Error(error);
	}
	virtual void reportWarning(const char* warning)
	{
		m_numWarnings++;
		b3Warning(warning);
	}

	virtual void printMessage(const char* msg)
	{
		b3Printf(msg);
	}
};

std::vector<std::shared_ptr<zeno::PrimitiveObject>> BulletURDFImporter::getVisualShapes() const
{
    return m_data->mbVisualList;
}


void BulletURDFImporter::setMultiBodyInfo(std::vector<int> mb2graphics, int numLinks) const
{
    for (size_t i=0;i<numLinks;i++){
        m_data->mbVisualList.push_back(m_data->visualList[mb2graphics[i]]);
    }
    std::cout << "sync visual 2 mb DONE!\n";
}

bool BulletURDFImporter::loadURDF(const char* fileName, bool forceFixedBase)
{
	if (strlen(fileName) == 0)
		return false;

	b3FileUtils fu;

    char findFileName[1024];
	//bool fileFound = fu.findFile(fileName, relativeFileName, 1024);
	bool fileFound = m_data->m_fileIO->findResourcePath(fileName, findFileName, 1024);

	std::string xml_string;

	if (!fileFound)
	{
		b3Warning("URDF file '%s' not found\n", fileName);
		return false;
	}
	else
	{
		char path[1024];
		m_data->setSourceFile(findFileName);
		//read file
		int fileId = m_data->m_fileIO->fileOpen(findFileName,"r");

		char destBuffer[8192];
		char* line = 0;
		do
		{
			line = m_data->m_fileIO->readLine(fileId, destBuffer, 8192);
			if (line)
			{
				xml_string += (std::string(destBuffer) + "\n");
			}
		}
		while (line);
		m_data->m_fileIO->fileClose(fileId);

	}

	BulletErrorLogger loggie;
	bool result = false;
	if (xml_string.length())
	{
			result = m_data->m_urdfParser.loadUrdf(xml_string.c_str(), &loggie, forceFixedBase, (m_data->m_flags & CUF_PARSE_SENSORS));

			if (m_data->m_flags & CUF_IGNORE_VISUAL_SHAPES)
			{
				for (int i=0; i < m_data->m_urdfParser.getModel().m_links.size(); i++)
				{
					UrdfLink* linkPtr = *m_data->m_urdfParser.getModel().m_links.getAtIndex(i);
					linkPtr->m_visualArray.clear();
				}
			}
			if (m_data->m_flags & CUF_IGNORE_COLLISION_SHAPES)
			{
				for (int i=0; i < m_data->m_urdfParser.getModel().m_links.size(); i++)
				{
					UrdfLink* linkPtr = *m_data->m_urdfParser.getModel().m_links.getAtIndex(i);
					linkPtr->m_collisionArray.clear();
				}
			}
			if (m_data->m_urdfParser.getModel().m_rootLinks.size())
			{
				if (m_data->m_flags & CUF_MERGE_FIXED_LINKS)
				{
					m_data->m_urdfParser.mergeFixedLinks(m_data->m_urdfParser.getModel(), m_data->m_urdfParser.getModel().m_rootLinks[0], &loggie, forceFixedBase, 0);
					m_data->m_urdfParser.getModel().m_links.clear();
					m_data->m_urdfParser.getModel().m_joints.clear();
					m_data->m_urdfParser.recreateModel(m_data->m_urdfParser.getModel(), m_data->m_urdfParser.getModel().m_rootLinks[0], &loggie);
				}
				if (m_data->m_flags & CUF_PRINT_URDF_INFO)
				{
					m_data->m_urdfParser.printTree(m_data->m_urdfParser.getModel().m_rootLinks[0], &loggie, 0);
				}
			}
	}
	return result;
}

void BulletURDFImporter::setBodyUniqueId(int bodyId)
{
	m_data->m_bodyId = bodyId;
}

int BulletURDFImporter::getBodyUniqueId() const
{
	return m_data->m_bodyId;
}

BulletURDFImporter::~BulletURDFImporter()
{
	delete m_data;
}

int BulletURDFImporter::getRootLinkIndex() const
{
	if (m_data->m_urdfParser.getModel().m_rootLinks.size() == 1)
	{
		return m_data->m_urdfParser.getModel().m_rootLinks[0]->m_linkIndex;
	}
	return -1;
};

void BulletURDFImporter::getLinkChildIndices(int linkIndex, btAlignedObjectArray<int>& childLinkIndices) const
{
	childLinkIndices.resize(0);
	UrdfLink* const* linkPtr = m_data->m_urdfParser.getModel().m_links.getAtIndex(linkIndex);
	if (linkPtr)
	{
		const UrdfLink* link = *linkPtr;
		//int numChildren = m_data->m_urdfParser->getModel().m_links.getAtIndex(linkIndex)->

		for (int i = 0; i < link->m_childLinks.size(); i++)
		{
			int childIndex = link->m_childLinks[i]->m_linkIndex;
			childLinkIndices.push_back(childIndex);
		}
	}
}

std::string BulletURDFImporter::getLinkName(int linkIndex) const
{
	UrdfLink* const* linkPtr = m_data->m_urdfParser.getModel().m_links.getAtIndex(linkIndex);
	btAssert(linkPtr);
	if (linkPtr)
	{
		UrdfLink* link = *linkPtr;
		return link->m_name;
	}
	return "";
}

std::string BulletURDFImporter::getBodyName() const
{
	return m_data->m_urdfParser.getModel().m_name;
}

std::string BulletURDFImporter::getJointName(int linkIndex) const
{
	UrdfLink* const* linkPtr = m_data->m_urdfParser.getModel().m_links.getAtIndex(linkIndex);
	btAssert(linkPtr);
	if (linkPtr)
	{
		UrdfLink* link = *linkPtr;
		if (link->m_parentJoint)
		{
			return link->m_parentJoint->m_name;
		}
	}
	return "";
}

void BulletURDFImporter::getMassAndInertia2(int urdfLinkIndex, btScalar& mass, btVector3& localInertiaDiagonal, btTransform& inertialFrame, int flags) const
{
	if (flags & CUF_USE_URDF_INERTIA)
	{
		getMassAndInertia(urdfLinkIndex, mass, localInertiaDiagonal, inertialFrame);
	}
	else
	{
		//the link->m_inertia is NOT necessarily aligned with the inertial frame
		//so an additional transform might need to be computed
		UrdfLink* const* linkPtr = m_data->m_urdfParser.getModel().m_links.getAtIndex(urdfLinkIndex);

		btAssert(linkPtr);
		if (linkPtr)
		{
			UrdfLink* link = *linkPtr;
			btScalar linkMass;
			if (link->m_parentJoint == 0 && m_data->m_urdfParser.getModel().m_overrideFixedBase)
			{
				linkMass = 0.f;
			}
			else
			{
				linkMass = link->m_inertia.m_mass;
			}
			mass = linkMass;
			localInertiaDiagonal.setValue(0, 0, 0);
			inertialFrame.setOrigin(link->m_inertia.m_linkLocalFrame.getOrigin());
			inertialFrame.setBasis(link->m_inertia.m_linkLocalFrame.getBasis());
		}
		else
		{
			mass = 1.f;
			localInertiaDiagonal.setValue(1, 1, 1);
			inertialFrame.setIdentity();
		}
	}
}

void BulletURDFImporter::getMassAndInertia(int linkIndex, btScalar& mass, btVector3& localInertiaDiagonal, btTransform& inertialFrame) const
{
	//the link->m_inertia is NOT necessarily aligned with the inertial frame
	//so an additional transform might need to be computed
	UrdfLink* const* linkPtr = m_data->m_urdfParser.getModel().m_links.getAtIndex(linkIndex);

	btAssert(linkPtr);
	if (linkPtr)
	{
		UrdfLink* link = *linkPtr;
		btMatrix3x3 linkInertiaBasis;
		btScalar linkMass, principalInertiaX, principalInertiaY, principalInertiaZ;
		if (link->m_parentJoint == 0 && m_data->m_urdfParser.getModel().m_overrideFixedBase)
		{
			linkMass = 0.f;
			principalInertiaX = 0.f;
			principalInertiaY = 0.f;
			principalInertiaZ = 0.f;
			linkInertiaBasis.setIdentity();
		}
		else
		{
			linkMass = link->m_inertia.m_mass;
			if (link->m_inertia.m_ixy == 0.0 &&
				link->m_inertia.m_ixz == 0.0 &&
				link->m_inertia.m_iyz == 0.0)
			{
				principalInertiaX = link->m_inertia.m_ixx;
				principalInertiaY = link->m_inertia.m_iyy;
				principalInertiaZ = link->m_inertia.m_izz;
				linkInertiaBasis.setIdentity();
			}
			else
			{
				principalInertiaX = link->m_inertia.m_ixx;
				btMatrix3x3 inertiaTensor(link->m_inertia.m_ixx, link->m_inertia.m_ixy, link->m_inertia.m_ixz,
										  link->m_inertia.m_ixy, link->m_inertia.m_iyy, link->m_inertia.m_iyz,
										  link->m_inertia.m_ixz, link->m_inertia.m_iyz, link->m_inertia.m_izz);
				btScalar threshold = 1.0e-6;
				int numIterations = 30;
				inertiaTensor.diagonalize(linkInertiaBasis, threshold, numIterations);
				principalInertiaX = inertiaTensor[0][0];
				principalInertiaY = inertiaTensor[1][1];
				principalInertiaZ = inertiaTensor[2][2];
			}
		}
		mass = linkMass;
		if (principalInertiaX < 0 ||
			principalInertiaX > (principalInertiaY + principalInertiaZ) ||
			principalInertiaY < 0 ||
			principalInertiaY > (principalInertiaX + principalInertiaZ) ||
			principalInertiaZ < 0 ||
			principalInertiaZ > (principalInertiaX + principalInertiaY))
		{
			b3Warning("Bad inertia tensor properties, setting inertia to zero for link: %s\n", link->m_name.c_str());
			principalInertiaX = 0.f;
			principalInertiaY = 0.f;
			principalInertiaZ = 0.f;
			linkInertiaBasis.setIdentity();
		}
		localInertiaDiagonal.setValue(principalInertiaX, principalInertiaY, principalInertiaZ);
		inertialFrame.setOrigin(link->m_inertia.m_linkLocalFrame.getOrigin());
		inertialFrame.setBasis(link->m_inertia.m_linkLocalFrame.getBasis() * linkInertiaBasis);
	}
	else
	{
		mass = 1.f;
		localInertiaDiagonal.setValue(1, 1, 1);
		inertialFrame.setIdentity();
	}
}

bool BulletURDFImporter::getJointInfo2(int urdfLinkIndex, btTransform& parent2joint, btTransform& linkTransformInWorld, btVector3& jointAxisInJointSpace, int& jointType, btScalar& jointLowerLimit, btScalar& jointUpperLimit, btScalar& jointDamping, btScalar& jointFriction, btScalar& jointMaxForce, btScalar& jointMaxVelocity) const
{
	btScalar twistLimit;
	return getJointInfo3(urdfLinkIndex, parent2joint, linkTransformInWorld, jointAxisInJointSpace, jointType, jointLowerLimit, jointUpperLimit, jointDamping, jointFriction, jointMaxForce, jointMaxVelocity, twistLimit);
}

bool BulletURDFImporter::getJointInfo3(int urdfLinkIndex, btTransform& parent2joint, btTransform& linkTransformInWorld, btVector3& jointAxisInJointSpace, int& jointType, btScalar& jointLowerLimit, btScalar& jointUpperLimit, btScalar& jointDamping, btScalar& jointFriction, btScalar& jointMaxForce, btScalar& jointMaxVelocity, btScalar& twistLimit) const
{
	jointLowerLimit = 0.f;
	jointUpperLimit = 0.f;
	jointDamping = 0.f;
	jointFriction = 0.f;
	jointMaxForce = 0.f;
	jointMaxVelocity = 0.f;

	UrdfLink* const* linkPtr = m_data->m_urdfParser.getModel().m_links.getAtIndex(urdfLinkIndex);
	btAssert(linkPtr);
	if (linkPtr)
	{
		UrdfLink* link = *linkPtr;
		linkTransformInWorld = link->m_linkTransformInWorld;

		if (link->m_parentJoint)
		{
			UrdfJoint* pj = link->m_parentJoint;
			parent2joint = pj->m_parentLinkToJointTransform;
			jointType = pj->m_type;
			jointAxisInJointSpace = pj->m_localJointAxis;
			jointLowerLimit = pj->m_lowerLimit;
			jointUpperLimit = pj->m_upperLimit;
			jointDamping = pj->m_jointDamping;
			jointFriction = pj->m_jointFriction;
			jointMaxForce = pj->m_effortLimit;
			jointMaxVelocity = pj->m_velocityLimit;
			twistLimit = pj->m_twistLimit;
			return true;
		}
		else
		{
			parent2joint.setIdentity();
			return false;
		}
	}

	return false;
};

bool BulletURDFImporter::getJointInfo(int urdfLinkIndex, btTransform& parent2joint, btTransform& linkTransformInWorld, btVector3& jointAxisInJointSpace, int& jointType, btScalar& jointLowerLimit, btScalar& jointUpperLimit, btScalar& jointDamping, btScalar& jointFriction) const
{
	btScalar jointMaxForce;
	btScalar jointMaxVelocity;
	return getJointInfo2(urdfLinkIndex, parent2joint, linkTransformInWorld, jointAxisInJointSpace, jointType, jointLowerLimit, jointUpperLimit, jointDamping, jointFriction, jointMaxForce, jointMaxVelocity);
}

void BulletURDFImporter::setRootTransformInWorld(const btTransform& rootTransformInWorld)
{
	m_data->m_urdfParser.getModel().m_rootTransformInWorld = rootTransformInWorld;
}

bool BulletURDFImporter::getRootTransformInWorld(btTransform& rootTransformInWorld) const
{
	rootTransformInWorld = m_data->m_urdfParser.getModel().m_rootTransformInWorld;
	return true;
}

static btCollisionShape* createConvexHullFromShapes(std::vector<std::shared_ptr<zeno::PrimitiveObject>> shapes, const btVector3& geomScale, int flags)
{
	B3_PROFILE("createConvexHullFromShapes");
	btCompoundShape* compound = new btCompoundShape();
	compound->setMargin(gUrdfDefaultCollisionMargin);

	btTransform identity;
	identity.setIdentity();

	for (int s = 0; s < shapes.size(); s++)
	{
		btConvexHullShape* convexHull = new btConvexHullShape();
		convexHull->setMargin(gUrdfDefaultCollisionMargin);
        std::shared_ptr<zeno::PrimitiveObject> shape = shapes[s];
		int faceCount = shape->tris.size();

		for (int f = 0; f < faceCount; f += 3)
		{
			btVector3 pt;
			pt.setValue(shape->verts[shape->tris[f][0]][0],
                        shape->verts[shape->tris[f][0]][1],
                        shape->verts[shape->tris[f][0]][2]);

			convexHull->addPoint(pt * geomScale, false);

			pt.setValue(shape->verts[shape->tris[f][1]][0],
                        shape->verts[shape->tris[f][1]][1],
                        shape->verts[shape->tris[f][1]][2]);
			convexHull->addPoint(pt * geomScale, false);

			pt.setValue(shape->verts[shape->tris[f][2]][0],
                        shape->verts[shape->tris[f][2]][1],
                        shape->verts[shape->tris[f][2]][2]);
			convexHull->addPoint(pt * geomScale, false);
		}
		convexHull->recalcLocalAabb();
		convexHull->optimizeConvexHull();
		if (flags & CUF_INITIALIZE_SAT_FEATURES)
		{
			convexHull->initializePolyhedralFeatures();
		}

		compound->addChildShape(identity, convexHull);
	}

	return compound;
}

int BulletURDFImporter::getUrdfFromCollisionShape(const btCollisionShape* collisionShape, UrdfCollision& collision) const
{
	UrdfCollision* col = m_data->m_bulletCollisionShape2UrdfCollision.find(collisionShape);
	if (col)
	{
		collision = *col;
		return 1;
	}
	return 0;
}

btCollisionShape* BulletURDFImporter::convertURDFToCollisionShape(const UrdfCollision* collision) const
{
	BT_PROFILE("convertURDFToCollisionShape");

	btCollisionShape* shape = 0;

	switch (collision->m_geometry.m_type)
	{
		case URDF_GEOM_PLANE:
		{
			btVector3 planeNormal = collision->m_geometry.m_planeNormal;
			btScalar planeConstant = 0;  //not available?
			btStaticPlaneShape* plane = new btStaticPlaneShape(planeNormal, planeConstant);
			shape = plane;
			shape->setMargin(gUrdfDefaultCollisionMargin);
			break;
		}
		case URDF_GEOM_CAPSULE:
		{
			btScalar radius = collision->m_geometry.m_capsuleRadius;
			btScalar height = collision->m_geometry.m_capsuleHeight;
			btCapsuleShapeZ* capsuleShape = new btCapsuleShapeZ(radius, height);
			shape = capsuleShape;
			shape->setMargin(gUrdfDefaultCollisionMargin);
			break;
		}

		case URDF_GEOM_CYLINDER:
		{
			btScalar cylRadius = collision->m_geometry.m_capsuleRadius;
			btScalar cylHalfLength = 0.5 * collision->m_geometry.m_capsuleHeight;
			if (m_data->m_flags & CUF_USE_IMPLICIT_CYLINDER)
			{
				btVector3 halfExtents(cylRadius, cylRadius, cylHalfLength);
				btCylinderShapeZ* cylZShape = new btCylinderShapeZ(halfExtents);
				shape = cylZShape;
				shape->setMargin(gUrdfDefaultCollisionMargin);
			}
			else
			{
				btAlignedObjectArray<btVector3> vertices;
				//int numVerts = sizeof(barrel_vertices)/(9*sizeof(float));
				int numSteps = 32;
				for (int i = 0; i < numSteps; i++)
				{
					btVector3 vert(cylRadius * btSin(SIMD_2_PI * (float(i) / numSteps)), cylRadius * btCos(SIMD_2_PI * (float(i) / numSteps)), cylHalfLength);
					vertices.push_back(vert);
					vert[2] = -cylHalfLength;
					vertices.push_back(vert);
				}
				btConvexHullShape* cylZShape = new btConvexHullShape(&vertices[0].x(), vertices.size(), sizeof(btVector3));
				cylZShape->setMargin(gUrdfDefaultCollisionMargin);
				cylZShape->recalcLocalAabb();
				if (m_data->m_flags & CUF_INITIALIZE_SAT_FEATURES)
				{
					cylZShape->initializePolyhedralFeatures();
				}
				cylZShape->optimizeConvexHull();
				shape = cylZShape;
			}

			break;
		}
		case URDF_GEOM_BOX:
		{
			btVector3 extents = collision->m_geometry.m_boxSize;
			btBoxShape* boxShape = new btBoxShape(extents * 0.5f);
			//btConvexShape* boxShape = new btConeShapeX(extents[2]*0.5,extents[0]*0.5);
			if (m_data->m_flags & CUF_INITIALIZE_SAT_FEATURES)
			{
				boxShape->initializePolyhedralFeatures();
			}
			shape = boxShape;
			shape->setMargin(gUrdfDefaultCollisionMargin);
			break;
		}
		case URDF_GEOM_SPHERE:
		{
			btScalar radius = collision->m_geometry.m_sphereRadius;
			btSphereShape* sphereShape = new btSphereShape(radius);
			shape = sphereShape;
			shape->setMargin(gUrdfDefaultCollisionMargin);
			break;
		}
		case URDF_GEOM_MESH:
		{
			std::shared_ptr<zeno::PrimitiveObject> zenoMesh;

			switch (collision->m_geometry.m_meshFileType)
			{
				case UrdfGeometry::FILE_OBJ: //currently only obj is supported
					if (collision->m_flags & URDF_FORCE_CONCAVE_TRIMESH)
                    {
						char relativeFileName[1024];
						char pathPrefix[1024];
						pathPrefix[0] = 0;
						m_data->m_fileIO->findResourcePath(collision->m_geometry.m_meshFileName.c_str(), relativeFileName, 1024);

                        std::string path = pathPrefix;
                        path += collision->m_geometry.m_meshFileName;
                        auto binary = file_get_binary(path);
                        zenoMesh = parse_obj(std::move(binary));
					}
					else
					{
						std::vector<std::shared_ptr<zeno::PrimitiveObject>> shapes;
						auto singleObj = parse_obj(file_get_binary(collision->m_geometry.m_meshFileName)); // TODO: fake multi-group here, implement later.
                        shapes.push_back(std::move(singleObj));
                        //create a convex hull for each shape, and store it in a btCompoundShape
						shape = createConvexHullFromShapes(shapes, collision->m_geometry.m_meshScale, m_data->m_flags);
						m_data->m_bulletCollisionShape2UrdfCollision.insert(shape, *collision);
						return shape;
					}
					break;

			}

			if (!zenoMesh || zenoMesh->verts.size() <= 0)
			{
				b3Warning("%s: cannot extract mesh from '%s'\n", collision->m_geometry.m_meshFileName.c_str());
				break;
			}

			btAlignedObjectArray<btVector3> convertedVerts;
			convertedVerts.reserve(zenoMesh->verts.size());
			for (int i = 0; i < zenoMesh->verts.size(); i++)
			{
				convertedVerts.push_back(btVector3(
                    zenoMesh->verts[i][0] * collision->m_geometry.m_meshScale[0],
                    zenoMesh->verts[i][1] * collision->m_geometry.m_meshScale[1],
                    zenoMesh->verts[i][2] * collision->m_geometry.m_meshScale[2]));
			}

			if (collision->m_flags & URDF_FORCE_CONCAVE_TRIMESH)
			{
				BT_PROFILE("convert trimesh");
				btTriangleMesh* meshInterface = new btTriangleMesh();
				m_data->m_allocatedMeshInterfaces.push_back(meshInterface);
				{
					BT_PROFILE("convert vertices");

					for (int i = 0; i < zenoMesh->tris.size(); i++)
					{
						const btVector3& v0 = convertedVerts[zenoMesh->tris[i][0]];
						const btVector3& v1 = convertedVerts[zenoMesh->tris[i][0]];
						const btVector3& v2 = convertedVerts[zenoMesh->tris[i][0]];
						meshInterface->addTriangle(v0, v1, v2);
					}
				}
				{
					BT_PROFILE("create btBvhTriangleMeshShape");
					btBvhTriangleMeshShape* trimesh = new btBvhTriangleMeshShape(meshInterface, true, true);
					//trimesh->setLocalScaling(collision->m_geometry.m_meshScale);
					shape = trimesh;
				}
			}
			else
			{
				BT_PROFILE("convert btConvexHullShape");
				btConvexHullShape* convexHull = new btConvexHullShape(&convertedVerts[0].getX(), convertedVerts.size(), sizeof(btVector3));
				convexHull->optimizeConvexHull();
				if (m_data->m_flags & CUF_INITIALIZE_SAT_FEATURES)
				{
					convexHull->initializePolyhedralFeatures();
				}
				convexHull->setMargin(gUrdfDefaultCollisionMargin);
				convexHull->recalcLocalAabb();
				//convexHull->setLocalScaling(collision->m_geometry.m_meshScale);
				shape = convexHull;
			}

			break;
		}  // mesh case

		default:
			b3Warning("Error: unknown collision geometry type %i\n", collision->m_geometry.m_type);
	}
	if (shape && collision->m_geometry.m_type == URDF_GEOM_MESH)
	{
		m_data->m_bulletCollisionShape2UrdfCollision.insert(shape, *collision);
	}
	return shape;
}

static std::shared_ptr<zeno::PrimitiveObject> transformShapeInternal(btTransform trans, std::shared_ptr<zeno::PrimitiveObject> shape) {

    auto &pos = shape->attr<zeno::vec3f>("pos");
    #pragma omp parallel for
    for (int i = 0; i < pos.size(); i++) {
        auto p = zeno::vec_to_other<btVector3>(pos[i]);
        auto vt = trans * p;
        pos[i] = zeno::other_to_vec<3>(vt);
    }

    return shape;
}

bool BulletURDFImporter::convertURDFToVisualShapeInternal(const UrdfVisual* visual, const btTransform& visualTransform, struct b3ImportMeshData& meshData) const
{
    bool graphicsFlag = false;
	BT_PROFILE("convertURDFToVisualShapeInternal");

    std::shared_ptr<zeno::PrimitiveObject> zenoMesh;

	btConvexShape* convexColShape = 0;

	switch (visual->m_geometry.m_type)
	{
        case URDF_GEOM_CAPSULE:
        {
           btScalar radius = visual->m_geometry.m_capsuleRadius;
			btScalar height = visual->m_geometry.m_capsuleHeight;
			btCapsuleShapeZ* capsuleShape = new btCapsuleShapeZ(radius, height);
			convexColShape = capsuleShape;
			convexColShape->setMargin(gUrdfDefaultCollisionMargin);
            break;
        }
		case URDF_GEOM_CYLINDER:
		{
			btAlignedObjectArray<btVector3> vertices;

			//int numVerts = sizeof(barrel_vertices)/(9*sizeof(float));
			int numSteps = 32;
			for (int i = 0; i < numSteps; i++)
			{
				btScalar cylRadius = visual->m_geometry.m_capsuleRadius;
				btScalar cylLength = visual->m_geometry.m_capsuleHeight;

				btVector3 vert(cylRadius * btSin(SIMD_2_PI * (float(i) / numSteps)), cylRadius * btCos(SIMD_2_PI * (float(i) / numSteps)), cylLength / 2.);
				vertices.push_back(vert);
				vert[2] = -cylLength / 2.;
				vertices.push_back(vert);
			}

			btConvexHullShape* cylZShape = new btConvexHullShape(&vertices[0].x(), vertices.size(), sizeof(btVector3));
			cylZShape->setMargin(gUrdfDefaultCollisionMargin);
			cylZShape->recalcLocalAabb();
			convexColShape = cylZShape;
			break;
		}

		case URDF_GEOM_BOX:
		{
			btVector3 extents = visual->m_geometry.m_boxSize;
			int strideInBytes = 9 * sizeof(float);
			int numVertices = sizeof(cube_vertices_textured) / strideInBytes;
			int numTriangles = sizeof(cube_indices) / sizeof(int) / 3;
			zenoMesh = std::make_shared<zeno::PrimitiveObject>();
            zenoMesh->resize(numVertices);
            zenoMesh->tris.resize(numTriangles);
            auto &nrm = zenoMesh->add_attr<zeno::vec3f>("nrm");
			for (int k = 0; k < numTriangles; k++)
			{
				zenoMesh->tris[k][0] = cube_indices[k * 3];
                zenoMesh->tris[k][1] = cube_indices[k * 3 + 1];
                zenoMesh->tris[k][2] = cube_indices[k * 3 + 2];
			}

			btScalar halfExtentsX = extents[0] * 0.5;
			btScalar halfExtentsY = extents[1] * 0.5;
			btScalar halfExtentsZ = extents[2] * 0.5;
			btScalar textureScaling = 1;

			for (int i = 0; i < numVertices; i++)
			{
				zenoMesh->verts[i][0] = halfExtentsX * cube_vertices_textured[i * 9];
                zenoMesh->verts[i][1] = halfExtentsY * cube_vertices_textured[i * 9 + 1];
                zenoMesh->verts[i][2] = halfExtentsZ * cube_vertices_textured[i * 9 + 2];
				//unused w, verts[i].xyzw[3] = cube_vertices_textured[i * 9 + 3];


                nrm[i] = zeno::vec3f(cube_vertices_textured[i * 9 + 4], cube_vertices_textured[i * 9 + 5], cube_vertices_textured[i * 9 + 6]);

                // TODO: add uv, currently uv is more complex to set
				//verts[i].uv[0] = cube_vertices_textured[i * 9 + 7] * textureScaling;
				//verts[i].uv[1] = cube_vertices_textured[i * 9 + 8] * textureScaling;
			}
			break;
		}

		case URDF_GEOM_SPHERE:
		{
			btScalar radius = visual->m_geometry.m_sphereRadius;
			btSphereShape* sphereShape = new btSphereShape(radius);
			convexColShape = sphereShape;
			convexColShape->setMargin(gUrdfDefaultCollisionMargin);
			break;
		}

		case URDF_GEOM_MESH:
		{
			switch (visual->m_geometry.m_meshFileType)
			{
				case UrdfGeometry::MEMORY_VERTICES:
				{
                    zenoMesh = std::make_shared<zeno::PrimitiveObject>();
					//		int index = 0;
					zenoMesh->resize(visual->m_geometry.m_vertices.size());
					zenoMesh->tris.resize(visual->m_geometry.m_indices.size()/3);
                    auto &nrm = zenoMesh->add_attr<zeno::vec3f>("nrm");
					for (int i = 0; i < visual->m_geometry.m_vertices.size(); i++)
					{
                        zenoMesh->verts[i][0] = visual->m_geometry.m_vertices[i].x();
                        zenoMesh->verts[i][1]= visual->m_geometry.m_vertices[i].y();
                        zenoMesh->verts[i][2]= visual->m_geometry.m_vertices[i].z();

						btVector3 normal(visual->m_geometry.m_vertices[i]);
						if (visual->m_geometry.m_normals.size() == visual->m_geometry.m_vertices.size())
						{
							normal = visual->m_geometry.m_normals[i];
						}
						else
						{
							normal.safeNormalize();
						}

						btVector3 uv(0.5, 0.5, 0);
						if (visual->m_geometry.m_uvs.size() == visual->m_geometry.m_vertices.size())
						{
							uv = visual->m_geometry.m_uvs[i];
						}
                        nrm[i] = zeno::vec3f(normal[0], normal[1], normal[2]);
                        //TODO: add uv later
						//glmesh->m_vertices->at(i).uv[0] = uv[0];
						//glmesh->m_vertices->at(i).uv[1] = uv[1];

					}
					for (int i = 0; i < visual->m_geometry.m_indices.size(); i++)
					{
						zenoMesh->tris[i/3][i%3] = visual->m_geometry.m_indices[i];
					}

					break;
				}
				case UrdfGeometry::FILE_OBJ:
				{
					if (b3ImportMeshUtility::loadAndRegisterMeshFromFileInternal(visual->m_geometry.m_meshFileName, meshData, m_data->m_fileIO))
					{
						if (meshData.m_textureImage1)
						{
							BulletURDFTexture texData;
							texData.m_width = meshData.m_textureWidth;
							texData.m_height = meshData.m_textureHeight;
							texData.textureData1 = meshData.m_textureImage1;
							texData.m_isCached = meshData.m_isCached;
							//texturesOut.push_back(texData); // TODO: find another way to attach texture to primitive object
						}
						zenoMesh = meshData.m_gfxShape;
					}
					break;
				}
			}  // switch file type

			if (!zenoMesh  || zenoMesh->verts.size() <= 0)
			{
				b3Warning("%s: cannot extract anything useful from mesh '%s'\n", visual->m_geometry.m_meshFileName.c_str());
				break;
			}

			//apply the geometry scaling
			for (int i = 0; i < zenoMesh->verts.size(); i++)
			{
                zenoMesh->verts[i][0] *= visual->m_geometry.m_meshScale[0];
                zenoMesh->verts[i][1] *= visual->m_geometry.m_meshScale[1];
                zenoMesh->verts[i][2] *= visual->m_geometry.m_meshScale[2];
			}
			break;
		}
		case URDF_GEOM_PLANE:
		{
			b3Warning("No default visual for URDF_GEOM_PLANE");
			break;
		}
		default:
		{
			b3Warning("Error: unknown visual geometry type %i\n", visual->m_geometry.m_type);
		}
	}

	//if we have a convex, tesselate into localVertices/localIndices
	if (convexColShape)
	{
		BT_PROFILE("convexColShape");

		btShapeHull* hull = new btShapeHull(convexColShape);
		hull->buildHull(0.0);
		{
			//	int strideInBytes = 9*sizeof(float);
			int numVertices = hull->numVertices();
			int numIndices = hull->numIndices();

			zenoMesh = std::make_shared<zeno::PrimitiveObject>();
			//	int index = 0;
			zenoMesh->resize(numVertices);
            zenoMesh->tris.resize(numIndices/3);
            auto &nrm = zenoMesh->add_attr<zeno::vec3f>("nrm");
			for (int i = 0; i < numVertices; i++)
			{
				btVector3 pos = hull->getVertexPointer()[i];
				zenoMesh->verts[i][0] = pos.x();
                zenoMesh->verts[i][1] = pos.y();
                zenoMesh->verts[i][2] = pos.z();
				pos.normalize();
                //TODO: attach uv later
                nrm[i] = zeno::vec3f(pos.x(), pos.y(), pos.z());
				//vtx.normal[0] = pos.x();
				//vtx.normal[1] = pos.y();
				//vtx.normal[2] = pos.z();
				//btScalar u = btAtan2(vtx.normal[0], vtx.normal[2]) / (2 * SIMD_PI) + 0.5;
				//btScalar v = vtx.normal[1] * 0.5 + 0.5;
				//vtx.uv[0] = u;
				//vtx.uv[1] = v;
			}

			btAlignedObjectArray<int> indices;
			for (int i = 0; i < numIndices; i++)
			{
				zenoMesh->tris[i/3][i%3] = hull->getIndexPointer()[i];
			}
		}
		delete hull;
		delete convexColShape;
		convexColShape = 0;
	}

	if (zenoMesh && zenoMesh->verts.size() > 0)
	{
		BT_PROFILE("zenoMesh");
        //zenoMesh = mapplypos(visualTransform, zenoMesh);
        //zenoMesh = transformShapeInternal(visualTransform, zenoMesh);
        m_data->visualList.push_back(zenoMesh);
        graphicsFlag = 1;
	}
    return graphicsFlag;
}

int BulletURDFImporter::convertLinkVisualShapes(int linkIndex, const btTransform& localInertiaFrame) const
{
	int graphicsIndex = -1;
    bool graphicsFlag;
	btAlignedObjectArray<vec3f> vertices;
	btAlignedObjectArray<int> indices;
	btTransform startTrans;
	startTrans.setIdentity();
	btAlignedObjectArray<BulletURDFTexture> textures;

	const UrdfModel& model = m_data->m_urdfParser.getModel();
	UrdfLink* const* linkPtr = model.m_links.getAtIndex(linkIndex);
	if (linkPtr)
	{
		const UrdfLink* link = *linkPtr;

		for (int v = 0; v < link->m_visualArray.size(); v++)
		{
			const UrdfVisual& vis = link->m_visualArray[v];
			btTransform childTrans = vis.m_linkLocalFrame;
			btHashString matName(vis.m_materialName.c_str());
			UrdfMaterial* const* matPtr = model.m_materials[matName];
			b3ImportMeshData meshData;

			graphicsFlag = convertURDFToVisualShapeInternal(&vis, localInertiaFrame.inverse() * childTrans, meshData);

			bool mtlOverridesUrdfColor = false;
			if ((meshData.m_flags & B3_IMPORT_MESH_HAS_RGBA_COLOR) &&
					(meshData.m_flags & B3_IMPORT_MESH_HAS_SPECULAR_COLOR))
			{
				mtlOverridesUrdfColor = (m_data->m_flags & CUF_USE_MATERIAL_COLORS_FROM_MTL) != 0;
				UrdfMaterialColor matCol;
				if (m_data->m_flags&CUF_USE_MATERIAL_TRANSPARANCY_FROM_MTL)
				{
					matCol.m_rgbaColor.setValue(meshData.m_rgbaColor[0],
								meshData.m_rgbaColor[1],
								meshData.m_rgbaColor[2],
								meshData.m_rgbaColor[3]);
				} else
				{
					matCol.m_rgbaColor.setValue(meshData.m_rgbaColor[0],
								meshData.m_rgbaColor[1],
								meshData.m_rgbaColor[2],
								1);
				}

				matCol.m_specularColor.setValue(meshData.m_specularColor[0],
					meshData.m_specularColor[1],
					meshData.m_specularColor[2]);
				m_data->m_linkColors.insert(linkIndex, matCol);
			}
			if (matPtr && !mtlOverridesUrdfColor)
			{
				UrdfMaterial* const mat = *matPtr;
				//printf("UrdfMaterial %s, rgba = %f,%f,%f,%f\n",mat->m_name.c_str(),mat->m_rgbaColor[0],mat->m_rgbaColor[1],mat->m_rgbaColor[2],mat->m_rgbaColor[3]);
				UrdfMaterialColor matCol;
				matCol.m_rgbaColor = mat->m_matColor.m_rgbaColor;
				matCol.m_specularColor = mat->m_matColor.m_specularColor;
				m_data->m_linkColors.insert(linkIndex, matCol);
			}
		}
	}
	if (graphicsFlag)
	{
		//		graphicsIndex  = m_data->m_guiHelper->registerGraphicsShape(&vertices[0].xyzw[0], vertices.size(), &indices[0], indices.size());
		//graphicsIndex  = m_data->m_guiHelper->registerGraphicsShape(&vertices[0].xyzw[0], vertices.size(), &indices[0], indices.size());

		//CommonRenderInterface* renderer = m_data->m_guiHelper->getRenderInterface();

		if (1)
		{
			int textureIndex = -1;
			if (textures.size())
			{
				//textureIndex = m_data->m_guiHelper->registerTexture(textures[0].textureData1, textures[0].m_width, textures[0].m_height);
				//if (textureIndex >= 0)
				//{
				//	m_data->m_allocatedTextures.push_back(textureIndex);
				//}
			}
			{
				B3_PROFILE("registerGraphicsShape");
                graphicsIndex = m_data->visualList.size();
			}
		}
	}

	//delete textures
	for (int i = 0; i < textures.size(); i++)
	{
		B3_PROFILE("free textureData");
		if (!textures[i].m_isCached)
		{
			free(textures[i].textureData1);
		}
	}
	return graphicsIndex;
}

static std::shared_ptr<zeno::PrimitiveObject> mapplypos(btTransform trans, std::shared_ptr<zeno::PrimitiveObject> shape) {
    glm::mat4 matTrans = glm::translate(glm::vec3(trans.getOrigin()[0], trans.getOrigin()[1], trans.getOrigin()[2]));
    glm::quat myQuat(trans.getRotation()[3], trans.getRotation()[0], trans.getRotation()[1], trans.getRotation()[2]);
    glm::mat4 matQuat  = glm::toMat4(myQuat);
    auto matrix = matTrans*matQuat;

    auto &pos = shape->attr<zeno::vec3f>("pos");
#pragma omp parallel for
    for (int i = 0; i < pos.size(); i++) {
        auto p = zeno::vec_to_other<glm::vec3>(pos[i]);
        auto vector4 =  matrix * glm::vec4(p, 1.0f);
        p = glm::vec3(vector4) / vector4.w;
        pos[i] = zeno::other_to_vec<3>(p);
    }

    return shape;
}

void BulletURDFImporter::convertLinkVisualShapes2(int urdfIndex, const btTransform& localInertiaFrame, class btCollisionObject* colObj) const
{
    if (colObj->getUserIndex() < 0)
    {
        btCollisionShape* shape = colObj->getCollisionShape();
        btTransform startTransform = colObj->getWorldTransform();
        int graphicsShapeId = shape->getUserIndex();
        if (graphicsShapeId >= 0)
        {
            //	btAssert(graphicsShapeId >= 0);
            //the graphics shape is already scaled
            btVector3 localScaling(1, 1, 1);
            //auto shape = m_data->visualList[graphicsShapeId-1];
            //shape = mapplypos(localInertiaFrame, shape);
            //shape = mapplypos(startTransform, shape);
            colObj->setUserIndex(graphicsShapeId-1);
        }
    }
}


void BulletURDFImporter::createCollisionObjectGraphicsInstance(int linkIndex, class btCollisionObject* colObj, const btVector3& colorRgba) const
{
    if (colObj->getUserIndex() < 0)
    {
        btCollisionShape* shape = colObj->getCollisionShape();
        btTransform startTransform = colObj->getWorldTransform();
        int graphicsShapeId = shape->getUserIndex();
        if (graphicsShapeId >= 0)
        {
            //	btAssert(graphicsShapeId >= 0);
            //the graphics shape is already scaled
            btVector3 localScaling(1, 1, 1);
            //auto shape = m_data->visualList[graphicsShapeId-1];
            //transform it with startTransform.getOrigin(), startTransform.getRotation()
            //shape = mapplypos(startTransform, shape);
            colObj->setUserIndex(graphicsShapeId-1);
        }
    }
}

bool BulletURDFImporter::getLinkColor(int linkIndex, btVector4& colorRGBA) const
{
	const UrdfMaterialColor* matColPtr = m_data->m_linkColors[linkIndex];
	if (matColPtr)
	{
		colorRGBA = matColPtr->m_rgbaColor;
		return true;
	}
	return false;
}

bool BulletURDFImporter::getLinkColor2(int linkIndex, UrdfMaterialColor& matCol) const
{
	UrdfMaterialColor* matColPtr = m_data->m_linkColors[linkIndex];
	if (matColPtr)
	{
		matCol = *matColPtr;
		return true;
	}
	return false;
}

void BulletURDFImporter::setLinkColor2(int linkIndex, struct UrdfMaterialColor& matCol) const
{
	m_data->m_linkColors.insert(linkIndex, matCol);
}

bool BulletURDFImporter::getLinkContactInfo(int urdflinkIndex, URDFLinkContactInfo& contactInfo) const
{
	UrdfLink* const* linkPtr = m_data->m_urdfParser.getModel().m_links.getAtIndex(urdflinkIndex);
	if (linkPtr)
	{
		const UrdfLink* link = *linkPtr;
		contactInfo = link->m_contactInfo;
		return true;
	}
	return false;
}

int BulletURDFImporter::getNumAllocatedCollisionShapes() const
{
	return m_data->m_allocatedCollisionShapes.size();
}

btCollisionShape* BulletURDFImporter::getAllocatedCollisionShape(int index)
{
	return m_data->m_allocatedCollisionShapes[index];
}

int BulletURDFImporter::getNumAllocatedMeshInterfaces() const
{
	return m_data->m_allocatedMeshInterfaces.size();
}

btStridingMeshInterface* BulletURDFImporter::getAllocatedMeshInterface(int index)
{
	return m_data->m_allocatedMeshInterfaces[index];
}

int BulletURDFImporter::getNumAllocatedTextures() const
{
	return m_data->m_allocatedTextures.size();
}

int BulletURDFImporter::getAllocatedTexture(int index) const
{
	return m_data->m_allocatedTextures[index];
}

int BulletURDFImporter::getCollisionGroupAndMask(int linkIndex, int& colGroup, int& colMask) const
{
	int result = 0;
	UrdfLink* const* linkPtr = m_data->m_urdfParser.getModel().m_links.getAtIndex(linkIndex);
	btAssert(linkPtr);
	if (linkPtr)
	{
		UrdfLink* link = *linkPtr;
		for (int v = 0; v < link->m_collisionArray.size(); v++)
		{
			const UrdfCollision& col = link->m_collisionArray[v];
			if (col.m_flags & URDF_HAS_COLLISION_GROUP)
			{
				colGroup = col.m_collisionGroup;
				result |= URDF_HAS_COLLISION_GROUP;
			}
			if (col.m_flags & URDF_HAS_COLLISION_MASK)
			{
				colMask = col.m_collisionMask;
				result |= URDF_HAS_COLLISION_MASK;
			}
		}
	}
	return result;
}

class btCompoundShape* BulletURDFImporter::convertLinkCollisionShapes(int linkIndex, const btTransform& localInertiaFrame) const
{
	btCompoundShape* compoundShape = new btCompoundShape();
	m_data->m_allocatedCollisionShapes.push_back(compoundShape);

	compoundShape->setMargin(gUrdfDefaultCollisionMargin);
	UrdfLink* const* linkPtr = m_data->m_urdfParser.getModel().m_links.getAtIndex(linkIndex);
	btAssert(linkPtr);

	if (linkPtr)
	{
		UrdfLink* link = *linkPtr;
		for (int v = 0; v < link->m_collisionArray.size(); v++)
		{
			const UrdfCollision& col = link->m_collisionArray[v];
			btCollisionShape* childShape = convertURDFToCollisionShape(&col); // TODO: seems fine for mesh, other geom should also check
			if (childShape)
			{
				m_data->m_allocatedCollisionShapes.push_back(childShape);
				if (childShape->getShapeType() == COMPOUND_SHAPE_PROXYTYPE)
				{
					btCompoundShape* compound = (btCompoundShape*)childShape;
					for (int i = 0; i < compound->getNumChildShapes(); i++)
					{
						m_data->m_allocatedCollisionShapes.push_back(compound->getChildShape(i));
					}
				}

				btTransform childTrans = col.m_linkLocalFrame;

				compoundShape->addChildShape(localInertiaFrame.inverse() * childTrans, childShape);
			}
		}
	}
	return compoundShape;
}

const struct UrdfModel* BulletURDFImporter::getUrdfModel() const {
	return &m_data->m_urdfParser.getModel();
};
