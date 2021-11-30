#include "utest.h"
#include <mcut/mcut.h>
#include <string>
#include <vector>

#include "off.h"

#ifdef _WIN32
#pragma warning(disable : 26812) // Unscoped enums from mcut.h
#endif // _WIN32

struct SeamConnectedComponent {
    std::vector<McConnectedComponent> connComps_ = {};
    McContext context_ = MC_NULL_HANDLE;

    float* pSrcMeshVertices = NULL;
    uint32_t* pSrcMeshFaceIndices = NULL;
    uint32_t* pSrcMeshFaceSizes = NULL;
    uint32_t numSrcMeshVertices = 0;
    uint32_t numSrcMeshFaces = 0;

    float* pCutMeshVertices = NULL;
    uint32_t* pCutMeshFaceIndices = NULL;
    uint32_t* pCutMeshFaceSizes = NULL;
    uint32_t numCutMeshVertices = 0;
    uint32_t numCutMeshFaces = 0;
};

UTEST_F_SETUP(SeamConnectedComponent)
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

UTEST_F_TEARDOWN(SeamConnectedComponent)
{
    if (utest_fixture->connComps_.size() > 0) {
        EXPECT_EQ(mcReleaseConnectedComponents(
                      utest_fixture->context_,
                      (uint32_t)utest_fixture->connComps_.size(),
                      utest_fixture->connComps_.data()),
            MC_NO_ERROR);
    }

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

UTEST_F(SeamConnectedComponent, queryVertices)
{
    // partial cut intersection between a cube and a quad

    const std::string srcMeshPath = std::string(MESHES_DIR) + "/benchmarks/src-mesh013.off";

    readOFF(srcMeshPath.c_str(), &utest_fixture->pSrcMeshVertices, &utest_fixture->pSrcMeshFaceIndices, &utest_fixture->pSrcMeshFaceSizes, &utest_fixture->numSrcMeshVertices, &utest_fixture->numSrcMeshFaces);

    ASSERT_TRUE(utest_fixture->pSrcMeshVertices != nullptr);
    ASSERT_TRUE(utest_fixture->pSrcMeshFaceIndices != nullptr);
    ASSERT_TRUE(utest_fixture->pSrcMeshVertices != nullptr);
    ASSERT_GT((int)utest_fixture->numSrcMeshVertices, 2);
    ASSERT_GT((int)utest_fixture->numSrcMeshFaces, 0);

    const std::string cutMeshPath = std::string(MESHES_DIR) + "/benchmarks/cut-mesh013.off";

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
        MC_NO_ERROR);

    uint32_t numConnComps = 0;

    ASSERT_EQ(mcGetConnectedComponents(utest_fixture->context_, MC_CONNECTED_COMPONENT_TYPE_SEAM, 0, NULL, &numConnComps), MC_NO_ERROR);
    // NOTE: there can only be a seamed mesh whose origin/parent is the cut-mesh in this test
    // a seam conn-comp whose origin is the cut-mesh is guarranteed to exist if the src-mesh is water-tight.
    // More generally, a seamed mesh is guarranteed to exist if and only if discovered seams/cut-paths are either 1) "circular" (loop) or 2) "linear"
    // which means that they sever/partition the respective origin (src-mesh or cut-mesh)
    ASSERT_EQ(numConnComps, uint32_t(1));

    utest_fixture->connComps_.resize(numConnComps);

    ASSERT_EQ(mcGetConnectedComponents(utest_fixture->context_, MC_CONNECTED_COMPONENT_TYPE_SEAM, (uint32_t)utest_fixture->connComps_.size(), utest_fixture->connComps_.data(), NULL), MC_NO_ERROR);

    for (int c = 0; c < (int)utest_fixture->connComps_.size(); ++c) {
        McConnectedComponent cc = utest_fixture->connComps_[c]; // connected compoenent id

        // indices of the vertices which define the seam
        uint64_t connCompSeamVertexIndicesBytes = 0;
        ASSERT_EQ(mcGetConnectedComponentData(utest_fixture->context_, cc, MC_CONNECTED_COMPONENT_DATA_SEAM_VERTEX, 0, NULL, &connCompSeamVertexIndicesBytes), MC_NO_ERROR);

        std::vector<uint32_t> seamVertexIndices;
        seamVertexIndices.resize(connCompSeamVertexIndicesBytes / sizeof(uint32_t));
        ASSERT_EQ(mcGetConnectedComponentData(utest_fixture->context_, cc, MC_CONNECTED_COMPONENT_DATA_SEAM_VERTEX, connCompSeamVertexIndicesBytes, seamVertexIndices.data(), NULL), MC_NO_ERROR);

        for (int i = 0; i < (int)seamVertexIndices.size(); ++i) {
            ASSERT_GE((uint32_t)seamVertexIndices[i], (uint32_t)0);
        }

        ASSERT_EQ((uint32_t)seamVertexIndices.size(), 4u); // specifc to benchmark meshes used (see setup function).
    }
}

UTEST_F(SeamConnectedComponent, queryOriginPartialCut)
{
    const std::string srcMeshPath = std::string(MESHES_DIR) + "/benchmarks/src-mesh013.off";

    readOFF(srcMeshPath.c_str(), &utest_fixture->pSrcMeshVertices, &utest_fixture->pSrcMeshFaceIndices, &utest_fixture->pSrcMeshFaceSizes, &utest_fixture->numSrcMeshVertices, &utest_fixture->numSrcMeshFaces);

    ASSERT_TRUE(utest_fixture->pSrcMeshVertices != nullptr);
    ASSERT_TRUE(utest_fixture->pSrcMeshFaceIndices != nullptr);
    ASSERT_TRUE(utest_fixture->pSrcMeshVertices != nullptr);
    ASSERT_GT((int)utest_fixture->numSrcMeshVertices, 2);
    ASSERT_GT((int)utest_fixture->numSrcMeshFaces, 0);

    const std::string cutMeshPath = std::string(MESHES_DIR) + "/benchmarks/cut-mesh013.off";

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
        MC_NO_ERROR);

    uint32_t numConnComps = 0;

    ASSERT_EQ(mcGetConnectedComponents(utest_fixture->context_, MC_CONNECTED_COMPONENT_TYPE_SEAM, 0, NULL, &numConnComps), MC_NO_ERROR);
    ASSERT_EQ(uint32_t(1), numConnComps);

    utest_fixture->connComps_.resize(numConnComps);

    ASSERT_EQ(mcGetConnectedComponents(utest_fixture->context_, MC_CONNECTED_COMPONENT_TYPE_SEAM, (uint32_t)utest_fixture->connComps_.size(), utest_fixture->connComps_.data(), NULL), MC_NO_ERROR);

    McConnectedComponent cc = utest_fixture->connComps_[0]; // connected compoenent id

    McSeamOrigin orig = (McSeamOrigin)(0);
    ASSERT_EQ(mcGetConnectedComponentData(utest_fixture->context_, cc, MC_CONNECTED_COMPONENT_DATA_ORIGIN, sizeof(McSeamOrigin), &orig, NULL), MC_NO_ERROR);

    ASSERT_TRUE(orig == MC_SEAM_ORIGIN_CUTMESH);
}

UTEST_F(SeamConnectedComponent, queryConnectedComponentType_CompleteCut)
{
    // complete cut: cube and quad with two polygons
    //mySetup("src-mesh014.off", "cut-mesh014.off");

    const std::string srcMeshPath = std::string(MESHES_DIR) + "/benchmarks/src-mesh014.off";

    readOFF(srcMeshPath.c_str(), &utest_fixture->pSrcMeshVertices, &utest_fixture->pSrcMeshFaceIndices, &utest_fixture->pSrcMeshFaceSizes, &utest_fixture->numSrcMeshVertices, &utest_fixture->numSrcMeshFaces);

    ASSERT_TRUE(utest_fixture->pSrcMeshVertices != nullptr);
    ASSERT_TRUE(utest_fixture->pSrcMeshFaceIndices != nullptr);
    ASSERT_TRUE(utest_fixture->pSrcMeshVertices != nullptr);
    ASSERT_GT((int)utest_fixture->numSrcMeshVertices, 2);
    ASSERT_GT((int)utest_fixture->numSrcMeshFaces, 0);

    const std::string cutMeshPath = std::string(MESHES_DIR) + "/benchmarks/cut-mesh014.off";

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
        MC_NO_ERROR);

    uint32_t numConnComps = 0;

    ASSERT_EQ(mcGetConnectedComponents(utest_fixture->context_, MC_CONNECTED_COMPONENT_TYPE_SEAM, 0, NULL, &numConnComps), MC_NO_ERROR);
    ASSERT_EQ(numConnComps, 2u);

    utest_fixture->connComps_.resize(numConnComps);

    ASSERT_EQ(mcGetConnectedComponents(utest_fixture->context_, MC_CONNECTED_COMPONENT_TYPE_SEAM, (uint32_t)utest_fixture->connComps_.size(), utest_fixture->connComps_.data(), NULL), MC_NO_ERROR);

    bool foundSeamedMeshFromSrcMesh = false;
    bool foundSeamedMeshFromCutMesh = false;
    for (int i = 0; i < (int)utest_fixture->connComps_.size(); ++i) {
        McConnectedComponent cc = utest_fixture->connComps_[i]; // connected compoenent id

        McSeamOrigin orig = McSeamOrigin::MC_SEAM_ORIGIN_ALL;
        ASSERT_EQ(mcGetConnectedComponentData(utest_fixture->context_, cc, MC_CONNECTED_COMPONENT_DATA_ORIGIN, sizeof(McSeamOrigin), &orig, NULL), MC_NO_ERROR);

        ASSERT_TRUE(orig == MC_SEAM_ORIGIN_SRCMESH || orig == MC_SEAM_ORIGIN_CUTMESH);

        if (orig == MC_SEAM_ORIGIN_SRCMESH) {
            foundSeamedMeshFromSrcMesh = true;
        } else {
            foundSeamedMeshFromCutMesh = true;
        }
    }

    ASSERT_TRUE(foundSeamedMeshFromSrcMesh || foundSeamedMeshFromCutMesh);
}

UTEST_F(SeamConnectedComponent, dispatchRequireThroughCuts_CompleteCut)
{
    // complete cut: cube and quad with two polygons
    //mySetup("src-mesh014.off", "cut-mesh014.off");

    const std::string srcMeshPath = std::string(MESHES_DIR) + "/benchmarks/src-mesh014.off";

    readOFF(srcMeshPath.c_str(), &utest_fixture->pSrcMeshVertices, &utest_fixture->pSrcMeshFaceIndices, &utest_fixture->pSrcMeshFaceSizes, &utest_fixture->numSrcMeshVertices, &utest_fixture->numSrcMeshFaces);

    ASSERT_TRUE(utest_fixture->pSrcMeshVertices != nullptr);
    ASSERT_TRUE(utest_fixture->pSrcMeshFaceIndices != nullptr);
    ASSERT_TRUE(utest_fixture->pSrcMeshVertices != nullptr);
    ASSERT_GT((int)utest_fixture->numSrcMeshVertices, 2);
    ASSERT_GT((int)utest_fixture->numSrcMeshFaces, 0);

    const std::string cutMeshPath = std::string(MESHES_DIR) + "/benchmarks/cut-mesh014.off";

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
        MC_NO_ERROR);

    uint32_t numConnComps = 0;

    ASSERT_EQ(mcGetConnectedComponents(utest_fixture->context_, MC_CONNECTED_COMPONENT_TYPE_SEAM, 0, NULL, &numConnComps), MC_NO_ERROR);
    ASSERT_EQ(numConnComps, 2u);
}

UTEST_F(SeamConnectedComponent, dispatchRequireThroughCuts_PartialCut)
{
    //mySetup("src-mesh013.off", "cut-mesh013.off");

    const std::string srcMeshPath = std::string(MESHES_DIR) + "/benchmarks/src-mesh013.off";

    readOFF(srcMeshPath.c_str(), &utest_fixture->pSrcMeshVertices, &utest_fixture->pSrcMeshFaceIndices, &utest_fixture->pSrcMeshFaceSizes, &utest_fixture->numSrcMeshVertices, &utest_fixture->numSrcMeshFaces);

    ASSERT_TRUE(utest_fixture->pSrcMeshVertices != nullptr);
    ASSERT_TRUE(utest_fixture->pSrcMeshFaceIndices != nullptr);
    ASSERT_TRUE(utest_fixture->pSrcMeshVertices != nullptr);
    ASSERT_GT((int)utest_fixture->numSrcMeshVertices, 2);
    ASSERT_GT((int)utest_fixture->numSrcMeshFaces, 0);

    const std::string cutMeshPath = std::string(MESHES_DIR) + "/benchmarks/cut-mesh013.off";

    readOFF(cutMeshPath.c_str(), &utest_fixture->pCutMeshVertices, &utest_fixture->pCutMeshFaceIndices, &utest_fixture->pCutMeshFaceSizes, &utest_fixture->numCutMeshVertices, &utest_fixture->numCutMeshFaces);

    ASSERT_TRUE(utest_fixture->pCutMeshVertices != nullptr);
    ASSERT_TRUE(utest_fixture->pCutMeshFaceIndices != nullptr);
    ASSERT_TRUE(utest_fixture->pCutMeshFaceSizes != nullptr);
    ASSERT_GT((int)utest_fixture->numCutMeshVertices, 2);
    ASSERT_GT((int)utest_fixture->numCutMeshFaces, 0);

    ASSERT_EQ(mcDispatch(
                  utest_fixture->context_,
                  MC_DISPATCH_VERTEX_ARRAY_FLOAT | MC_DISPATCH_REQUIRE_THROUGH_CUTS,
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
        MC_NO_ERROR);

    uint32_t numConnComps = 0;

    ASSERT_EQ(mcGetConnectedComponents(utest_fixture->context_, MC_CONNECTED_COMPONENT_TYPE_ALL, 0, NULL, &numConnComps), MC_NO_ERROR);
    ASSERT_EQ(numConnComps, 2u); // there should be no connected components besides inputs
}
