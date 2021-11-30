#if defined(_WIN32)
#define _CRT_SECURE_NO_WARNINGS 1
#endif

#include "utest.h"
#include <mcut/mcut.h>

#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "off.h"

#define NUMBER_OF_BENCHMARKS 59 //

struct Benchmark {
    McContext myContext = MC_NULL_HANDLE;
    int benchmarkIndex = 0;

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

UTEST_I_SETUP(Benchmark)
{
    if (utest_index < NUMBER_OF_BENCHMARKS) {
        // create with no flags (default)
        EXPECT_EQ(mcCreateContext(&utest_fixture->myContext, MC_NULL_HANDLE), MC_NO_ERROR);
        EXPECT_TRUE(utest_fixture->myContext != nullptr);
        utest_fixture->benchmarkIndex = (int)utest_index;

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
}

UTEST_I_TEARDOWN(Benchmark)
{
    if (utest_index < NUMBER_OF_BENCHMARKS) {
        EXPECT_EQ(mcReleaseContext(utest_fixture->myContext), MC_NO_ERROR);
    }
}

UTEST_I(Benchmark, inputID, NUMBER_OF_BENCHMARKS)
{
    std::vector<std::pair<std::string, std::string>> benchmarkMeshPairs;

    std::stringstream ss;
    ss << std::setfill('0') << std::setw(3) << utest_fixture->benchmarkIndex;
    std::string s = ss.str();
    benchmarkMeshPairs.emplace_back("src-mesh" + s + ".off", "cut-mesh" + s + ".off");

    for (auto& i : benchmarkMeshPairs) {
        const std::string srcMeshName = i.first;
        const std::string cutMeshName = i.second;

        // do the cutting
        // --------------
        const std::string srcMeshPath = std::string(MESHES_DIR) + "/benchmarks/" + srcMeshName;

        readOFF(srcMeshPath.c_str(), &utest_fixture->pSrcMeshVertices, &utest_fixture->pSrcMeshFaceIndices, &utest_fixture->pSrcMeshFaceSizes, &utest_fixture->numSrcMeshVertices, &utest_fixture->numSrcMeshFaces);

        ASSERT_TRUE(utest_fixture->pSrcMeshVertices != nullptr);
        ASSERT_TRUE(utest_fixture->pSrcMeshFaceIndices != nullptr);
        ASSERT_TRUE(utest_fixture->pSrcMeshVertices != nullptr);
        ASSERT_GT((int)utest_fixture->numSrcMeshVertices, 2);
        ASSERT_GT((int)utest_fixture->numSrcMeshFaces, 0);

        const std::string cutMeshPath = std::string(MESHES_DIR) + "/benchmarks/" + cutMeshName;

        readOFF(cutMeshPath.c_str(), &utest_fixture->pCutMeshVertices, &utest_fixture->pCutMeshFaceIndices, &utest_fixture->pCutMeshFaceSizes, &utest_fixture->numCutMeshVertices, &utest_fixture->numCutMeshFaces);
        ASSERT_TRUE(utest_fixture->pCutMeshVertices != nullptr);
        ASSERT_TRUE(utest_fixture->pCutMeshFaceIndices != nullptr);
        ASSERT_TRUE(utest_fixture->pCutMeshFaceSizes != nullptr);
        ASSERT_GT((int)utest_fixture->numCutMeshVertices, 2);
        ASSERT_GT((int)utest_fixture->numCutMeshFaces, 0);

        ASSERT_EQ(mcDispatch(
                      utest_fixture->myContext,
                      MC_DISPATCH_VERTEX_ARRAY_FLOAT | MC_DISPATCH_ENFORCE_GENERAL_POSITION,
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

        free(utest_fixture->pSrcMeshVertices); 
        free(utest_fixture->pSrcMeshFaceIndices); 
        free(utest_fixture->pSrcMeshFaceSizes); 

        free(utest_fixture->pCutMeshVertices); 
        free(utest_fixture->pCutMeshFaceIndices); 
        free(utest_fixture->pCutMeshFaceSizes); 
        
        uint32_t numConnComps = 0;

        ASSERT_EQ(mcGetConnectedComponents(utest_fixture->myContext, MC_CONNECTED_COMPONENT_TYPE_ALL, 0, NULL, &numConnComps), MC_NO_ERROR);

        if (numConnComps == 0) {
            printf("no connected component found\n");
        }
        std::vector<McConnectedComponent> connComps;
        connComps.resize(numConnComps);

        ASSERT_EQ(mcGetConnectedComponents(utest_fixture->myContext, MC_CONNECTED_COMPONENT_TYPE_ALL, (uint32_t)connComps.size(), connComps.data(), NULL), MC_NO_ERROR);

        //
        // query connected component data
        //
        for (int c = 0; c < (int)connComps.size(); ++c) {
            McConnectedComponent cc = connComps[c]; // connected compoenent id

            // vertex array
            uint64_t connCompVerticesBytes = 0;
            ASSERT_EQ(mcGetConnectedComponentData(utest_fixture->myContext, cc, MC_CONNECTED_COMPONENT_DATA_VERTEX_FLOAT, 0, NULL, &connCompVerticesBytes), MC_NO_ERROR);
            ASSERT_GT(connCompVerticesBytes, uint64_t(0));
            ASSERT_GE(connCompVerticesBytes, uint64_t(sizeof(float) * 9)); // triangle
            const uint32_t numberOfVertices = (uint32_t)(connCompVerticesBytes / (sizeof(float) * 3));

            std::vector<float> vertices(numberOfVertices*3);
            ASSERT_EQ(mcGetConnectedComponentData(utest_fixture->myContext, cc, MC_CONNECTED_COMPONENT_DATA_VERTEX_FLOAT, connCompVerticesBytes, (void*)vertices.data(), NULL), MC_NO_ERROR);

            // face indices
            uint64_t connCompFaceIndicesBytes = 0;
            ASSERT_EQ(mcGetConnectedComponentData(utest_fixture->myContext, cc, MC_CONNECTED_COMPONENT_DATA_FACE, 0, NULL, &connCompFaceIndicesBytes), MC_NO_ERROR);
            ASSERT_GT(connCompFaceIndicesBytes, uint64_t(0));
            ASSERT_GE(connCompFaceIndicesBytes, uint64_t(sizeof(uint32_t) * 3)); // triangle
            std::vector<uint32_t> faceIndices;
            faceIndices.resize(connCompFaceIndicesBytes / sizeof(uint32_t));
            ASSERT_EQ(mcGetConnectedComponentData(utest_fixture->myContext, cc, MC_CONNECTED_COMPONENT_DATA_FACE, connCompFaceIndicesBytes, faceIndices.data(), NULL), MC_NO_ERROR);

            for (int v = 0; v < (int)faceIndices.size(); ++v) {
                ASSERT_GE((uint32_t)faceIndices[v], (uint32_t)0);
                ASSERT_LT((uint32_t)faceIndices[v], numberOfVertices); //  "out of bounds vertex index"
            }

            // face sizes
            uint64_t connCompFaceSizesBytes = 0;
            ASSERT_EQ(mcGetConnectedComponentData(utest_fixture->myContext, cc, MC_CONNECTED_COMPONENT_DATA_FACE_SIZE, 0, NULL, &connCompFaceSizesBytes), MC_NO_ERROR);
            ASSERT_GE(connCompFaceIndicesBytes, sizeof(uint32_t));
            std::vector<uint32_t> faceSizes;
            faceSizes.resize(connCompFaceSizesBytes / sizeof(uint32_t));

            ASSERT_EQ(mcGetConnectedComponentData(utest_fixture->myContext, cc, MC_CONNECTED_COMPONENT_DATA_FACE_SIZE, connCompFaceSizesBytes, faceSizes.data(), NULL), MC_NO_ERROR);
            ASSERT_GT(faceSizes.size(), std::size_t(0)); //  "there has to be at least one face in a connected component"

            for (int v = 0; v < (int)faceSizes.size(); ++v) {
                ASSERT_GE(faceSizes[v], (uint32_t)3); // "3 is the minimum possible number of vertices in a polygon, which is a triangle"
            }

            // edge indices
            uint64_t connCompEdgesBytes = 0;
            ASSERT_EQ(mcGetConnectedComponentData(utest_fixture->myContext, cc, MC_CONNECTED_COMPONENT_DATA_EDGE, 0, NULL, &connCompEdgesBytes), MC_NO_ERROR);
            ASSERT_GE(connCompEdgesBytes, uint64_t(sizeof(uint32_t) * 6)); // triangle

            std::vector<uint32_t> edgeIndices;
            edgeIndices.resize(connCompEdgesBytes / sizeof(uint32_t));
            ASSERT_EQ(mcGetConnectedComponentData(utest_fixture->myContext, cc, MC_CONNECTED_COMPONENT_DATA_EDGE, connCompEdgesBytes, edgeIndices.data(), NULL), MC_NO_ERROR);
            ASSERT_GE((uint32_t)edgeIndices.size(), (uint32_t)6); // "6 is the minimum number of indices in a triangle, which is the simplest polygon"

            EXPECT_EQ(mcReleaseConnectedComponents(utest_fixture->myContext, (uint32_t)connComps.size(), connComps.data()), MC_NO_ERROR);
            connComps.clear();
        }
    }
}
