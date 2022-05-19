#ifndef ZENO_B3IMPORTMESHUTILITY_H
#define ZENO_B3IMPORTMESHUTILITY_H
#include <string>
#include <memory>
#include "zeno/types/PrimitiveObject.h"

using namespace std;
enum b3ImportMeshDataFlags
{
    B3_IMPORT_MESH_HAS_RGBA_COLOR=1,
    B3_IMPORT_MESH_HAS_SPECULAR_COLOR=2,
};

struct b3ImportMeshData
{
    std::shared_ptr<zeno::PrimitiveObject> m_gfxShape;

    unsigned char* m_textureImage1;  //in 3 component 8-bit RGB data
    bool m_isCached;
    int m_textureWidth;
    int m_textureHeight;
    double m_rgbaColor[4];
    double m_specularColor[4];
    int m_flags;

    b3ImportMeshData()
        :m_gfxShape(0),
         m_textureImage1(0),
         m_isCached(false),
         m_textureWidth(0),
         m_textureHeight(0),
         m_flags(0)
    {
    }

};

class b3ImportMeshUtility
{
public:
    static bool loadAndRegisterMeshFromFileInternal(const std::string& fileName, b3ImportMeshData& meshData, struct CommonFileIOInterface* fileIO);
};
#endif //ZENO_B3IMPORTMESHUTILITY_H
