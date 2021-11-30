#include "utest.h"
#include <mcut/mcut.h>
#include <string>
#include <vector>

#include "off.h"

#ifdef _WIN32
#pragma warning(disable : 26812) // Unscoped enums from mcut.h
#endif // _WIN32

struct PolygonsWithHoles { 
    McContext context_ {};

    float* pSrcMeshVertices {};
    uint32_t* pSrcMeshFaceIndices {};
    uint32_t* pSrcMeshFaceSizes {};
    uint32_t numSrcMeshVertices{};
    uint32_t numSrcMeshFaces {};

    float* pCutMeshVertices {};
    uint32_t* pCutMeshFaceIndices {};
    uint32_t* pCutMeshFaceSizes {};
    uint32_t numCutMeshVertices {};
    uint32_t numCutMeshFaces {};
};

UTEST_F_SETUP(PolygonsWithHoles)
{
    McResult err = mcCreateContext(&utest_fixture->context_, 0);
    EXPECT_TRUE(utest_fixture->context_ != nullptr);
    EXPECT_EQ(err, MC_NO_ERROR);

    utest_fixture->pSrcMeshVertices = nullptr;
    utest_fixture->pSrcMeshFaceIndices = nullptr;
    utest_fixture->pSrcMeshFaceSizes = nullptr;
    utest_fixture->numSrcMeshVertices = 0;
    utest_fixture->numSrcMeshFaces = 0;

    utest_fixture->pCutMeshVertices = nullptr;
    utest_fixture->pCutMeshFaceIndices = nullptr;
    utest_fixture->pCutMeshFaceSizes = nullptr;
    utest_fixture->numCutMeshVertices = 0;
    utest_fixture->numCutMeshFaces = 0;
}

UTEST_F_TEARDOWN(PolygonsWithHoles)
{
    EXPECT_EQ(mcReleaseContext(utest_fixture->context_), MC_NO_ERROR);

    if (utest_fixture->pSrcMeshVertices)
        free(utest_fixture->pSrcMeshVertices);

    if (utest_fixture->pSrcMeshFaceIndices)
        free(utest_fixture->pSrcMeshFaceIndices);

    if (utest_fixture->pSrcMeshFaceSizes)
        free(utest_fixture->pSrcMeshFaceSizes);

    if (utest_fixture->pCutMeshVertices)
        free(utest_fixture->pCutMeshVertices);

    if (utest_fixture->pCutMeshFaceIndices)
        free(utest_fixture->pCutMeshFaceIndices);

    if (utest_fixture->pCutMeshFaceSizes)
        free(utest_fixture->pCutMeshFaceSizes);
}

UTEST_F(PolygonsWithHoles, outputWillHaveHoles)
{
    // partial cut intersection between a cube and a quad

    const std::string srcMeshPath = std::string(MESHES_DIR) + "/cube-flattened.off";

    readOFF(srcMeshPath.c_str(), &utest_fixture->pSrcMeshVertices, &utest_fixture->pSrcMeshFaceIndices, &utest_fixture->pSrcMeshFaceSizes, &utest_fixture->numSrcMeshVertices, &utest_fixture->numSrcMeshFaces);

    ASSERT_TRUE(utest_fixture->pSrcMeshVertices != nullptr);
    ASSERT_TRUE(utest_fixture->pSrcMeshFaceIndices != nullptr);
    ASSERT_TRUE(utest_fixture->pSrcMeshVertices != nullptr);
    ASSERT_GT((int)utest_fixture->numSrcMeshVertices, 2);
    ASSERT_GT((int)utest_fixture->numSrcMeshFaces, 0);

    const std::string cutMeshPath = std::string(MESHES_DIR) + "/cube-flattened-with-holes.off";

    readOFF(cutMeshPath.c_str(), &utest_fixture->pCutMeshVertices, &utest_fixture->pCutMeshFaceIndices, &utest_fixture->pCutMeshFaceSizes, &utest_fixture->numCutMeshVertices, &utest_fixture->numCutMeshFaces);
    ASSERT_TRUE(utest_fixture->pCutMeshVertices != nullptr);
    ASSERT_TRUE(utest_fixture->pCutMeshFaceIndices != nullptr);
    ASSERT_TRUE(utest_fixture->pCutMeshFaceSizes != nullptr);
    ASSERT_GT((int)utest_fixture->numCutMeshVertices, 2);
    ASSERT_GT((int)utest_fixture->numCutMeshFaces, 0);

    ASSERT_EQ(mcDispatch(
                  utest_fixture->context_,
                  MC_DISPATCH_VERTEX_ARRAY_FLOAT,
                  utest_fixture->pSrcMeshVertices,
                  utest_fixture->pSrcMeshFaceIndices,
                  utest_fixture->pSrcMeshFaceSizes,
                  utest_fixture->numSrcMeshVertices,
                  utest_fixture->numSrcMeshFaces,
                  utest_fixture->pCutMeshVertices,
                  utest_fixture->pCutMeshFaceIndices,
                  utest_fixture->pCutMeshFaceSizes,
                  utest_fixture->numCutMeshVertices,
                  utest_fixture->numCutMeshFaces),
        MC_INVALID_OPERATION);
}
