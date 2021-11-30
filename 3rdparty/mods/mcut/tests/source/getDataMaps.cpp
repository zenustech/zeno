#include "utest.h"
#include <mcut/mcut.h>
#include <vector>

enum InputType {
    // i.e. polygon with holes in the output
    REQUIRES_FACE_PARTITION = 0,
    NO_FACE_PARTITION = 1
};

struct TestConfig {
    McFlags dispatchflags;
    InputType inputType;
    bool swapInputs;
};

struct DataMapsQueryTest {
    McContext context_ = MC_NULL_HANDLE;
    std::vector<McConnectedComponent> connComps_;
    std::vector<float> pSrcMeshVertices;
    std::vector<uint32_t> pSrcMeshFaceIndices;
    std::vector<uint32_t> pSrcMeshFaceSizes;
    std::vector<float> pCutMeshVertices;
    std::vector<uint32_t> pCutMeshFaceIndices;
    std::vector<uint32_t> pCutMeshFaceSizes;

    McFlags dispatchflags = 0;
};

#define NUM_TEST_CONFIGS 12

TestConfig testConfigs[] = {
    // NO_FACE_PARTITION
    { MC_DISPATCH_INCLUDE_VERTEX_MAP, InputType::NO_FACE_PARTITION, false },
    { MC_DISPATCH_INCLUDE_FACE_MAP, InputType::NO_FACE_PARTITION, false },
    { MC_DISPATCH_INCLUDE_VERTEX_MAP | MC_DISPATCH_INCLUDE_FACE_MAP, InputType::NO_FACE_PARTITION, false },
    // REQUIRES_FACE_PARTITION
    { MC_DISPATCH_INCLUDE_VERTEX_MAP, InputType::REQUIRES_FACE_PARTITION, false },
    { MC_DISPATCH_INCLUDE_FACE_MAP, InputType::REQUIRES_FACE_PARTITION, false },
    { MC_DISPATCH_INCLUDE_VERTEX_MAP | MC_DISPATCH_INCLUDE_FACE_MAP, InputType::REQUIRES_FACE_PARTITION, false },
    // ** Swap inputs
    // NO_FACE_PARTITION
    { MC_DISPATCH_INCLUDE_VERTEX_MAP, InputType::NO_FACE_PARTITION, true },
    { MC_DISPATCH_INCLUDE_FACE_MAP, InputType::NO_FACE_PARTITION, true },
    { MC_DISPATCH_INCLUDE_VERTEX_MAP | MC_DISPATCH_INCLUDE_FACE_MAP, InputType::NO_FACE_PARTITION, true },
    // REQUIRES_FACE_PARTITION
    { MC_DISPATCH_INCLUDE_VERTEX_MAP, InputType::REQUIRES_FACE_PARTITION, true },
    { MC_DISPATCH_INCLUDE_FACE_MAP, InputType::REQUIRES_FACE_PARTITION, true },
    { MC_DISPATCH_INCLUDE_VERTEX_MAP | MC_DISPATCH_INCLUDE_FACE_MAP, InputType::REQUIRES_FACE_PARTITION, true }
};

void createCubeSMAndtwoTriCM(
    std::vector<float>& pSrcMeshVertices,
    std::vector<uint32_t>& pSrcMeshFaceIndices,
    std::vector<uint32_t>& pSrcMeshFaceSizes,
    std::vector<float>& pCutMeshVertices,
    std::vector<uint32_t>& pCutMeshFaceIndices,
    std::vector<uint32_t>& pCutMeshFaceSizes)
{
    // NOTE: same mesh as hello world
    pSrcMeshVertices = {
        // cube vertices
        -5, -5, 5, // 0
        5, -5, 5, // 1
        5, 5, 5, //2
        -5, 5, 5, //3
        -5, -5, -5, //4
        5, -5, -5, //5
        5, 5, -5, //6
        -5, 5, -5 //7
    };
    pSrcMeshFaceIndices = {
        // cube faces
        0, 1, 2, 3, //0
        7, 6, 5, 4, //1
        1, 5, 6, 2, //2
        0, 3, 7, 4, //3
        3, 2, 6, 7, //4
        4, 5, 1, 0 //5
    };
    pSrcMeshFaceSizes = { // cube face sizes
        4, 4, 4, 4, 4, 4
    };

    // the cut mesh
    // ---------
    pCutMeshVertices = {
        // cut mesh vertices
        -20, -4, 0, //0
        0, 20, 20, //1
        20, -4, 0, //2
        0, 20, -20 //3
    };

    pCutMeshFaceIndices = {
        0, 1, 2, //0
        0, 2, 3 //1
    };
    pCutMeshFaceSizes = {
        3, 3
    };
}

void CreateInputsRequiringFacePartitioning(std::vector<float>& pSrcMeshVertices,
    std::vector<uint32_t>& pSrcMeshFaceIndices,
    std::vector<uint32_t>& pSrcMeshFaceSizes,
    std::vector<float>& pCutMeshVertices,
    std::vector<uint32_t>& pCutMeshFaceIndices,
    std::vector<uint32_t>& pCutMeshFaceSizes)
{
    // NOTE: same mesh as benchmark 8
    pSrcMeshVertices = {
        // tet vertices
        0, 5, 0, // 0
        -5, -5, -5, // 1
        0, -5, 5, //2
        5, -5, -5, //3
    };
    pSrcMeshFaceIndices = {
        // tet faces
        3, 2, 1, //0
        0, 2, 3, //1
        0, 3, 1, //2
        0, 1, 2 //3
    };
    pSrcMeshFaceSizes = { // tet face sizes
        3, 3, 3, 3
    };

    // the cut mesh
    // ---------
    pCutMeshVertices = {
        // cut mesh vertices
        -5, 0, -5, //0
        0, 0, 5, //1
        5, 0, -5, //2
    };

    pCutMeshFaceIndices = {
        0, 1, 2, //0
    };
    pCutMeshFaceSizes = {
        3
    };
}

UTEST_I_SETUP(DataMapsQueryTest)
{
    if (utest_index < (size_t)NUM_TEST_CONFIGS) {
        McResult err = mcCreateContext(&utest_fixture->context_, MC_NULL_HANDLE);
        EXPECT_TRUE(utest_fixture->context_ != nullptr);
        EXPECT_EQ(err, MC_NO_ERROR);

        utest_fixture->dispatchflags = testConfigs[utest_index].dispatchflags;
        switch (testConfigs[utest_index].inputType) {
        case InputType::NO_FACE_PARTITION: {
            if (testConfigs[utest_index].swapInputs) {
                createCubeSMAndtwoTriCM(
                    utest_fixture->pCutMeshVertices,
                    utest_fixture->pCutMeshFaceIndices,
                    utest_fixture->pCutMeshFaceSizes,
                    utest_fixture->pSrcMeshVertices,
                    utest_fixture->pSrcMeshFaceIndices,
                    utest_fixture->pSrcMeshFaceSizes);
            } else {
                createCubeSMAndtwoTriCM(
                    utest_fixture->pSrcMeshVertices,
                    utest_fixture->pSrcMeshFaceIndices,
                    utest_fixture->pSrcMeshFaceSizes,
                    utest_fixture->pCutMeshVertices,
                    utest_fixture->pCutMeshFaceIndices,
                    utest_fixture->pCutMeshFaceSizes);
            }
        } break;
        case InputType::REQUIRES_FACE_PARTITION: {
            if (testConfigs[utest_index].swapInputs) {
                CreateInputsRequiringFacePartitioning(
                    utest_fixture->pCutMeshVertices,
                    utest_fixture->pCutMeshFaceIndices,
                    utest_fixture->pCutMeshFaceSizes,
                    utest_fixture->pSrcMeshVertices,
                    utest_fixture->pSrcMeshFaceIndices,
                    utest_fixture->pSrcMeshFaceSizes);
            } else {
                CreateInputsRequiringFacePartitioning(utest_fixture->pSrcMeshVertices,
                    utest_fixture->pSrcMeshFaceIndices,
                    utest_fixture->pSrcMeshFaceSizes,
                    utest_fixture->pCutMeshVertices,
                    utest_fixture->pCutMeshFaceIndices,
                    utest_fixture->pCutMeshFaceSizes);
            }
        } break;
        }
    }
}

UTEST_I_TEARDOWN(DataMapsQueryTest)
{
    if (utest_index < (size_t)NUM_TEST_CONFIGS) {
        utest_fixture->pSrcMeshVertices.clear();
        utest_fixture->pSrcMeshFaceIndices.clear();
        utest_fixture->pSrcMeshFaceSizes.clear();
        utest_fixture->pCutMeshVertices.clear();
        utest_fixture->pCutMeshFaceIndices.clear();
        utest_fixture->pCutMeshFaceSizes.clear();
        EXPECT_EQ(mcReleaseConnectedComponents(utest_fixture->context_, (uint32_t)utest_fixture->connComps_.size(), utest_fixture->connComps_.data()), MC_NO_ERROR);
        utest_fixture->connComps_.clear();
        EXPECT_EQ(mcReleaseContext(utest_fixture->context_), MC_NO_ERROR);
    }
}

UTEST_I(DataMapsQueryTest, testConfigID, NUM_TEST_CONFIGS)
{
    uint32_t srcMeshVertexCount = (uint32_t)utest_fixture->pSrcMeshVertices.size() / 3;
    uint32_t srcMeshFaceCount = (uint32_t)utest_fixture->pSrcMeshFaceSizes.size();
    uint32_t cutMeshVertexCount = (uint32_t)utest_fixture->pCutMeshVertices.size() / 3;
    uint32_t cutMeshFaceCount = (uint32_t)utest_fixture->pCutMeshFaceSizes.size();

    ASSERT_EQ(mcDispatch(
                  utest_fixture->context_,
                  MC_DISPATCH_VERTEX_ARRAY_FLOAT | utest_fixture->dispatchflags,
                  utest_fixture->pSrcMeshVertices.data(),
                  utest_fixture->pSrcMeshFaceIndices.data(),
                  utest_fixture->pSrcMeshFaceSizes.data(),
                  srcMeshVertexCount,
                  srcMeshFaceCount,
                  utest_fixture->pCutMeshVertices.data(),
                  utest_fixture->pCutMeshFaceIndices.data(),
                  utest_fixture->pCutMeshFaceSizes.data(),
                  cutMeshVertexCount,
                  cutMeshFaceCount),
        MC_NO_ERROR);

    // ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    // Now we query [all] connected components, including the input CCs
    // We want to ensure that we can query vertex- and face-data map for all each of them

    uint32_t numConnComps = 0;
    ASSERT_EQ(mcGetConnectedComponents(utest_fixture->context_, MC_CONNECTED_COMPONENT_TYPE_ALL, 0, NULL, &numConnComps), MC_NO_ERROR);

    utest_fixture->connComps_.resize(numConnComps);

    ASSERT_EQ(mcGetConnectedComponents(utest_fixture->context_, MC_CONNECTED_COMPONENT_TYPE_ALL, (uint32_t)utest_fixture->connComps_.size(), utest_fixture->connComps_.data(), NULL), MC_NO_ERROR);

    for (int i = 0; i < (int)utest_fixture->connComps_.size(); ++i) {
        McConnectedComponent cc = utest_fixture->connComps_[i]; // connected compoenent id

        McConnectedComponentType type = McConnectedComponentType::MC_CONNECTED_COMPONENT_TYPE_ALL;
        ASSERT_EQ(mcGetConnectedComponentData(utest_fixture->context_, cc, MC_CONNECTED_COMPONENT_DATA_TYPE, sizeof(McConnectedComponentType), &type, NULL), MC_NO_ERROR);

        if (utest_fixture->dispatchflags & MC_DISPATCH_INCLUDE_VERTEX_MAP) {
            uint64_t numBytes = 0;

            ASSERT_EQ(mcGetConnectedComponentData(utest_fixture->context_, cc, MC_CONNECTED_COMPONENT_DATA_VERTEX_FLOAT, 0, NULL, &numBytes), MC_NO_ERROR);
            ASSERT_GT((int)numBytes, 0);
            uint32_t ccVertexCount = numBytes / (sizeof(float)*3);

            numBytes = 0;
            ASSERT_EQ(mcGetConnectedComponentData(utest_fixture->context_, cc, MC_CONNECTED_COMPONENT_DATA_VERTEX_MAP, 0, NULL, &numBytes), MC_NO_ERROR);
            ASSERT_EQ(numBytes / sizeof(uint32_t), ccVertexCount);

            std::vector<uint32_t> vertexMap;
            vertexMap.resize(ccVertexCount);

            ASSERT_EQ(mcGetConnectedComponentData(utest_fixture->context_, cc, MC_CONNECTED_COMPONENT_DATA_VERTEX_MAP, vertexMap.size() * sizeof(uint32_t), vertexMap.data(), NULL), MC_NO_ERROR);

            auto testSrcMeshCC = [&]() {
                for (int i = 0; i < (int)ccVertexCount; i++) {

                    const uint32_t imVertexIdxRaw = vertexMap[i];
                    bool vertexIsFromSrcMesh = (imVertexIdxRaw < srcMeshVertexCount);
                    const bool isIntersectionPoint = (imVertexIdxRaw == MC_UNDEFINED_VALUE);

                    if (!isIntersectionPoint) {

                        uint32_t imVertexIdx = imVertexIdxRaw; // actual index value, accounting for offset

                        if (!vertexIsFromSrcMesh) {
                            imVertexIdx = (imVertexIdxRaw - srcMeshVertexCount); // account for offset
                            ASSERT_LT((int)imVertexIdx, (int)cutMeshVertexCount);
                        } else {
                            ASSERT_LT((int)imVertexIdx, (int)srcMeshVertexCount);
                        }
                    }
                }
            };

            auto testPatchCC = [&]() {
                for (int i = 0; i < (int)ccVertexCount; i++) {
                    uint32_t correspondingCutMeshVertex = vertexMap[i];
                    const bool isIntersectionPoint = (correspondingCutMeshVertex == MC_UNDEFINED_VALUE);
                    if (!isIntersectionPoint) {
                        correspondingCutMeshVertex -= srcMeshVertexCount; // account for offset
                        ASSERT_GE((int)correspondingCutMeshVertex, (int)0);
                        ASSERT_LT((int)correspondingCutMeshVertex, (int)cutMeshVertexCount);
                    }
                }
            };

            switch (type) {
            case McConnectedComponentType::MC_CONNECTED_COMPONENT_TYPE_FRAGMENT: { // comes from source-mesh
                testSrcMeshCC();
            } break;
            case McConnectedComponentType::MC_CONNECTED_COMPONENT_TYPE_PATCH: { // comes from cut-mesh
                testPatchCC();
            } break;
            case McConnectedComponentType::MC_CONNECTED_COMPONENT_TYPE_SEAM: {
                // find out where it comes from (source-mesh or cut-mesh)
                McSeamOrigin orig = MC_SEAM_ORIGIN_ALL;
                ASSERT_EQ(mcGetConnectedComponentData(utest_fixture->context_, cc, MC_CONNECTED_COMPONENT_DATA_ORIGIN, sizeof(McSeamOrigin), &orig, NULL), MC_NO_ERROR);

                if (orig == MC_SEAM_ORIGIN_SRCMESH) {
                    testSrcMeshCC();
                } else {
                    ASSERT_TRUE(orig == MC_SEAM_ORIGIN_CUTMESH);
                    testPatchCC();
                }
            } break;
            case McConnectedComponentType::MC_CONNECTED_COMPONENT_TYPE_INPUT: {
                // find out where it comes from (source-mesh or cut-mesh)
                McInputOrigin orig = MC_INPUT_ORIGIN_ALL;
                ASSERT_EQ(mcGetConnectedComponentData(utest_fixture->context_, cc, MC_CONNECTED_COMPONENT_DATA_ORIGIN, sizeof(McSeamOrigin), &orig, NULL), MC_NO_ERROR);

                if (orig == MC_INPUT_ORIGIN_SRCMESH) {
                    testSrcMeshCC();
                } else {
                    ASSERT_TRUE(orig == MC_INPUT_ORIGIN_CUTMESH);
                    testPatchCC();
                }
            } break;
            case MC_CONNECTED_COMPONENT_TYPE_ALL:
            printf("McConnectedComponentType = unknown\n");
            ASSERT_TRUE(false);
            break;
            }
        }

        if (utest_fixture->dispatchflags & MC_DISPATCH_INCLUDE_FACE_MAP) {
            uint64_t numBytes = 0;
            ASSERT_EQ(mcGetConnectedComponentData(utest_fixture->context_, cc, MC_CONNECTED_COMPONENT_DATA_FACE_SIZE, 0, NULL, &numBytes), MC_NO_ERROR);
            uint32_t ccFaceCount = (uint32_t)(numBytes / sizeof(uint32_t));
            ASSERT_GT((int)ccFaceCount, (int)0);

            numBytes = 0;
            ASSERT_EQ(mcGetConnectedComponentData(utest_fixture->context_, cc, MC_CONNECTED_COMPONENT_DATA_FACE_MAP, 0, NULL, &numBytes), MC_NO_ERROR);
            ASSERT_EQ(numBytes / sizeof(uint32_t), ccFaceCount);

            std::vector<uint32_t> faceMap;
            faceMap.resize(ccFaceCount);

            ASSERT_EQ(mcGetConnectedComponentData(utest_fixture->context_, cc, MC_CONNECTED_COMPONENT_DATA_FACE_MAP, faceMap.size() * sizeof(uint32_t), faceMap.data(), NULL), MC_NO_ERROR);

            for (int v = 0; v < (int)ccFaceCount; v++) {
                const uint32_t imFaceIdxRaw = faceMap[v];
                bool faceIsFromSrcMesh = (imFaceIdxRaw < srcMeshFaceCount);

                ASSERT_TRUE(imFaceIdxRaw != MC_UNDEFINED_VALUE); // all face indices are mapped!

                uint32_t imFaceIdx = imFaceIdxRaw; // actual index value, accounting for offset

                if (!faceIsFromSrcMesh) {
                    imFaceIdx = (imFaceIdxRaw - srcMeshFaceCount); // account for offset
                    ASSERT_LT((int)imFaceIdx, (int)cutMeshFaceCount);
                } else {
                    ASSERT_LT((int)imFaceIdx, (int)srcMeshFaceCount);
                }
            }
        }
    }
}

struct FaceAndVertexDataMapsQueryMissingFlagTest {
    McContext context_ = MC_NULL_HANDLE;
    std::vector<McConnectedComponent> connComps_;
    std::vector<float> pSrcMeshVertices;
    std::vector<uint32_t> pSrcMeshFaceIndices;
    std::vector<uint32_t> pSrcMeshFaceSizes;
    std::vector<float> pCutMeshVertices;
    std::vector<uint32_t> pCutMeshFaceIndices;
    std::vector<uint32_t> pCutMeshFaceSizes;
};

UTEST_F_SETUP(FaceAndVertexDataMapsQueryMissingFlagTest)
{

    McResult err = mcCreateContext(&utest_fixture->context_, MC_NULL_HANDLE);
    EXPECT_TRUE(utest_fixture->context_ != nullptr);
    EXPECT_EQ(err, MC_NO_ERROR);

    createCubeSMAndtwoTriCM(
        utest_fixture->pSrcMeshVertices,
        utest_fixture->pSrcMeshFaceIndices,
        utest_fixture->pSrcMeshFaceSizes,
        utest_fixture->pCutMeshVertices,
        utest_fixture->pCutMeshFaceIndices,
        utest_fixture->pCutMeshFaceSizes);
}

UTEST_F_TEARDOWN(FaceAndVertexDataMapsQueryMissingFlagTest)
{
    utest_fixture->pSrcMeshVertices.clear();
    utest_fixture->pSrcMeshFaceIndices.clear();
    utest_fixture->pSrcMeshFaceSizes.clear();
    utest_fixture->pCutMeshVertices.clear();
    utest_fixture->pCutMeshFaceIndices.clear();
    utest_fixture->pCutMeshFaceSizes.clear();
    EXPECT_EQ(mcReleaseConnectedComponents(utest_fixture->context_, (uint32_t)utest_fixture->connComps_.size(), utest_fixture->connComps_.data()), MC_NO_ERROR);
    utest_fixture->connComps_.clear();
    EXPECT_EQ(mcReleaseContext(utest_fixture->context_), MC_NO_ERROR);
}

UTEST_F(FaceAndVertexDataMapsQueryMissingFlagTest, vertexMapFlag)
{
    ASSERT_EQ(mcDispatch(
                  utest_fixture->context_,
                  MC_DISPATCH_VERTEX_ARRAY_FLOAT,
                  utest_fixture->pSrcMeshVertices.data(),
                  utest_fixture->pSrcMeshFaceIndices.data(),
                  utest_fixture->pSrcMeshFaceSizes.data(),
                  (uint32_t)utest_fixture->pSrcMeshVertices.size() / 3,
                  (uint32_t)utest_fixture->pSrcMeshFaceIndices.size() / 4,
                  utest_fixture->pCutMeshVertices.data(),
                  utest_fixture->pCutMeshFaceIndices.data(),
                  utest_fixture->pCutMeshFaceSizes.data(),
                  (uint32_t)utest_fixture->pCutMeshVertices.size() / 3,
                  (uint32_t)utest_fixture->pCutMeshFaceIndices.size() / 3),
        MC_NO_ERROR);

    uint32_t numConnComps = 0;

    ASSERT_EQ(mcGetConnectedComponents(utest_fixture->context_, MC_CONNECTED_COMPONENT_TYPE_ALL, 0, NULL, &numConnComps), MC_NO_ERROR);

    utest_fixture->connComps_.resize(numConnComps);

    ASSERT_EQ(mcGetConnectedComponents(utest_fixture->context_, MC_CONNECTED_COMPONENT_TYPE_ALL, (uint32_t)utest_fixture->connComps_.size(), utest_fixture->connComps_.data(), NULL), MC_NO_ERROR);

    for (int i = 0; i < (int)utest_fixture->connComps_.size(); ++i) {
        McConnectedComponent cc = utest_fixture->connComps_[i]; // connected compoenent id

        uint64_t numBytes = 0;
        ASSERT_EQ(mcGetConnectedComponentData(utest_fixture->context_, cc, MC_CONNECTED_COMPONENT_DATA_VERTEX_MAP, 0, NULL, &numBytes), MC_INVALID_VALUE);
    }
}

UTEST_F(FaceAndVertexDataMapsQueryMissingFlagTest, faceMapFlag)
{
    ASSERT_EQ(mcDispatch(
                  utest_fixture->context_,
                  MC_DISPATCH_VERTEX_ARRAY_FLOAT,
                  utest_fixture->pSrcMeshVertices.data(),
                  utest_fixture->pSrcMeshFaceIndices.data(),
                  utest_fixture->pSrcMeshFaceSizes.data(),
                  (uint32_t)utest_fixture->pSrcMeshVertices.size() / 3,
                  (uint32_t)utest_fixture->pSrcMeshFaceIndices.size() / 4,
                  utest_fixture->pCutMeshVertices.data(),
                  utest_fixture->pCutMeshFaceIndices.data(),
                  utest_fixture->pCutMeshFaceSizes.data(),
                  (uint32_t)utest_fixture->pCutMeshVertices.size() / 3,
                  (uint32_t)utest_fixture->pCutMeshFaceIndices.size() / 3),
        MC_NO_ERROR);

    uint32_t numConnComps = 0;

    ASSERT_EQ(mcGetConnectedComponents(utest_fixture->context_, MC_CONNECTED_COMPONENT_TYPE_ALL, 0, NULL, &numConnComps), MC_NO_ERROR);

    utest_fixture->connComps_.resize(numConnComps);

    ASSERT_EQ(mcGetConnectedComponents(utest_fixture->context_, MC_CONNECTED_COMPONENT_TYPE_ALL, (uint32_t)utest_fixture->connComps_.size(), utest_fixture->connComps_.data(), NULL), MC_NO_ERROR);

    for (int i = 0; i < (int)utest_fixture->connComps_.size(); ++i) {
        McConnectedComponent cc = utest_fixture->connComps_[i]; // connected compoenent id

        uint64_t numBytes = 0;
        ASSERT_EQ(mcGetConnectedComponentData(utest_fixture->context_, cc, MC_CONNECTED_COMPONENT_DATA_FACE_MAP, 0, NULL, &numBytes), MC_INVALID_VALUE);
    }
}
