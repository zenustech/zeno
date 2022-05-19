#ifndef BULLET_URDF_IMPORTER_H
#define BULLET_URDF_IMPORTER_H

#include <vector>
#include <memory>
#include "URDFImporterInterface.h"
#include "UrdfParser.h"
#include "../LinearMath/btAlignedObjectArray.h"
#include "../LinearMath/btVector3.h"
#include "../LinearMath/btTransform.h"
#include "zeno/utils/vec.h"
#include "zeno/types/PrimitiveObject.h"

struct BulletURDFTexture
{
	int m_width;
	int m_height;
	unsigned char* textureData1;
	bool m_isCached;
};

struct b3VisualShapeData
{
    int m_objectUniqueId;
    int m_linkIndex;
    int m_visualGeometryType;  //box primitive, sphere primitive, triangle mesh
    double m_dimensions[3];    //meaning depends on m_visualGeometryType
    char m_meshAssetFileName[1024];
    double m_localVisualFrame[7];  //pos[3], orn[4]
    //todo: add more data if necessary (material color etc, although material can be in asset file .obj file)
    double m_rgbaColor[4];
    int m_tinyRendererTextureId;
    int m_textureUniqueId;
    int m_openglTextureId;
};

class BulletURDFImporter : public URDFImporterInterface
{
	struct BulletURDFInternalData* m_data;

public:
	BulletURDFImporter(double globalScaling=1, int flags=0);

	virtual ~BulletURDFImporter();

	virtual bool loadURDF(const char* fileName, bool forceFixedBase = false);

	virtual void setBodyUniqueId(int bodyId);
	virtual int getBodyUniqueId() const;

	void printTree();  //for debugging

	virtual int getRootLinkIndex() const;

	virtual void getLinkChildIndices(int linkIndex, btAlignedObjectArray<int>& childLinkIndices) const;

	virtual std::string getBodyName() const;

	virtual std::string getLinkName(int linkIndex) const;

	virtual bool getLinkColor(int linkIndex, btVector4& colorRGBA) const;

	virtual bool getLinkColor2(int linkIndex, UrdfMaterialColor& matCol) const;

	virtual void setLinkColor2(int linkIndex, struct UrdfMaterialColor& matCol) const;

	virtual bool getLinkContactInfo(int urdflinkIndex, URDFLinkContactInfo& contactInfo) const;

	virtual std::string getJointName(int linkIndex) const;

	virtual void getMassAndInertia(int linkIndex, btScalar& mass, btVector3& localInertiaDiagonal, btTransform& inertialFrame) const;
	virtual void getMassAndInertia2(int urdfLinkIndex, btScalar& mass, btVector3& localInertiaDiagonal, btTransform& inertialFrame, int flags) const;

	virtual bool getJointInfo(int urdfLinkIndex, btTransform& parent2joint, btTransform& linkTransformInWorld, btVector3& jointAxisInJointSpace, int& jointType, btScalar& jointLowerLimit, btScalar& jointUpperLimit, btScalar& jointDamping, btScalar& jointFriction) const;
	virtual bool getJointInfo2(int urdfLinkIndex, btTransform& parent2joint, btTransform& linkTransformInWorld, btVector3& jointAxisInJointSpace, int& jointType, btScalar& jointLowerLimit, btScalar& jointUpperLimit, btScalar& jointDamping, btScalar& jointFriction, btScalar& jointMaxForce, btScalar& jointMaxVelocity) const;
	virtual bool getJointInfo3(int urdfLinkIndex, btTransform& parent2joint, btTransform& linkTransformInWorld, btVector3& jointAxisInJointSpace, int& jointType, btScalar& jointLowerLimit, btScalar& jointUpperLimit, btScalar& jointDamping, btScalar& jointFriction, btScalar& jointMaxForce, btScalar& jointMaxVelocity, btScalar& twistLimit) const;

	virtual bool getRootTransformInWorld(btTransform& rootTransformInWorld) const;
	virtual void setRootTransformInWorld(const btTransform& rootTransformInWorld);

	virtual int convertLinkVisualShapes(int linkIndex, const btTransform& inertialFrame) const;
    virtual void convertLinkVisualShapes2(int urdfIndex, const btTransform& inertialFrame, class btCollisionObject* colObj) const;
	class btCollisionShape* convertURDFToCollisionShape(const struct UrdfCollision* collision) const;

	virtual int getUrdfFromCollisionShape(const btCollisionShape* collisionShape, UrdfCollision& collision) const;

	///todo(erwincoumans) refactor this convertLinkCollisionShapes/memory allocation

	virtual const struct UrdfModel* getUrdfModel() const;

	virtual class btCompoundShape* convertLinkCollisionShapes(int linkIndex, const btTransform& localInertiaFrame) const;

	virtual int getCollisionGroupAndMask(int linkIndex, int& colGroup, int& colMask) const;

	virtual int getNumAllocatedCollisionShapes() const;
	virtual class btCollisionShape* getAllocatedCollisionShape(int index);

	virtual int getNumAllocatedMeshInterfaces() const;
	virtual class btStridingMeshInterface* getAllocatedMeshInterface(int index);

	virtual int getNumAllocatedTextures() const;
	virtual int getAllocatedTexture(int index) const;

	bool convertURDFToVisualShapeInternal(const struct UrdfVisual* visual, const class btTransform& visualTransform, struct b3ImportMeshData& meshData) const;

    ///optionally create some graphical representation from a collision object, usually for visual debugging purposes.
    virtual void createCollisionObjectGraphicsInstance(int linkIndex, class btCollisionObject* colObj, const btVector3& colorRgba) const;

    virtual std::vector<std::shared_ptr<zeno::PrimitiveObject>> getVisualShapes() const;
    virtual void setMultiBodyInfo(std::vector<int> mb2graphics, int numLinks) const;


};

#endif  //BULLET_URDF_IMPORTER_H
