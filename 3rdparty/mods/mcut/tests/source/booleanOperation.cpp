#include "utest.h"
#include <mcut/mcut.h>

#include <vector>

struct BooleanOperation {
    McContext myContext = MC_NULL_HANDLE;
    std::vector<McConnectedComponent> pConnComps_;
    std::vector<float> srcMeshVertices;
    std::vector<uint32_t> meshFaceIndices;
    std::vector<uint32_t> meshFaceSizes;
};

UTEST_F_SETUP(BooleanOperation)
{
    // create with no flags (default)
    EXPECT_EQ(mcCreateContext(&utest_fixture->myContext, 0), MC_NO_ERROR);
    EXPECT_TRUE(utest_fixture->myContext != nullptr);

    utest_fixture->srcMeshVertices = {
        -1.f, -1.f, 1.f, // 0
        1.f, -1.f, 1.f, // 1
        1.f, -1.f, -1.f, // 2
        -1.f, -1.f, -1.f, //3
        -1.f, 1.f, 1.f, //4
        1.f, 1.f, 1.f, //5
        1.f, 1.f, -1.f, //6
        -1.f, 1.f, -1.f //7
    };

    utest_fixture->meshFaceIndices = {
        3, 2, 1, 0, // bottom
        4, 5, 6, 7, //top
        0, 1, 5, 4, //front
        1, 2, 6, 5, // right
        2, 3, 7, 6, //back
        3, 0, 4, 7 // left
    };

    utest_fixture->meshFaceSizes = { 4, 4, 4, 4, 4, 4 };
}

UTEST_F_TEARDOWN(BooleanOperation)
{
    EXPECT_EQ(mcReleaseConnectedComponents(utest_fixture->myContext, (uint32_t)utest_fixture->pConnComps_.size(), utest_fixture->pConnComps_.data()), MC_NO_ERROR);
    EXPECT_EQ(mcReleaseContext(utest_fixture->myContext), MC_NO_ERROR);
}

// Performing a Boolean "union" operation with the same object, while forgetting to
// pass the appropriate MC_DISPATCH_ENFORCE_GENERAL_POSITION flag.
UTEST_F(BooleanOperation, selfUnionWithoutGeneralPositionEnforcement)
{
    const std::vector<float>& srcMeshVertices = utest_fixture->srcMeshVertices;
    const std::vector<uint32_t>& meshFaceIndices = utest_fixture->meshFaceIndices;
    const std::vector<uint32_t>& meshFaceSizes = utest_fixture->meshFaceSizes;

    const McFlags booleanUnionFlags = MC_DISPATCH_FILTER_FRAGMENT_SEALING_OUTSIDE | MC_DISPATCH_FILTER_FRAGMENT_LOCATION_ABOVE;

    ASSERT_EQ(mcDispatch(utest_fixture->myContext, //
                  MC_DISPATCH_VERTEX_ARRAY_FLOAT | booleanUnionFlags,
                  &srcMeshVertices[0], &meshFaceIndices[0], &meshFaceSizes[0], (uint32_t)(srcMeshVertices.size() / 3), (uint32_t)meshFaceSizes.size(), //
                  &srcMeshVertices[0], &meshFaceIndices[0], &meshFaceSizes[0], (uint32_t)(srcMeshVertices.size() / 3), (uint32_t)meshFaceSizes.size()),
        MC_INVALID_OPERATION);
}

// Performing a Boolean "union" operation with the same object, while allowing general
// position enforcement.
UTEST_F(BooleanOperation, selfUnionWithGeneralPositionEnforcement)
{
    const std::vector<float>& srcMeshVertices = utest_fixture->srcMeshVertices;
    const std::vector<uint32_t>& meshFaceIndices = utest_fixture->meshFaceIndices;
    const std::vector<uint32_t>& meshFaceSizes = utest_fixture->meshFaceSizes;

    const McFlags booleanUnionFlags = MC_DISPATCH_FILTER_FRAGMENT_SEALING_OUTSIDE | MC_DISPATCH_FILTER_FRAGMENT_LOCATION_ABOVE;

    ASSERT_EQ(mcDispatch(utest_fixture->myContext, //
                  MC_DISPATCH_VERTEX_ARRAY_FLOAT | booleanUnionFlags | MC_DISPATCH_ENFORCE_GENERAL_POSITION,
                  &srcMeshVertices[0], &meshFaceIndices[0], &meshFaceSizes[0], (uint32_t)(srcMeshVertices.size() / 3), (uint32_t)meshFaceSizes.size(), //
                  &srcMeshVertices[0], &meshFaceIndices[0], &meshFaceSizes[0], (uint32_t)(srcMeshVertices.size() / 3), (uint32_t)meshFaceSizes.size()),
        MC_NO_ERROR);
}

// Performing a Boolean "diff(A,B)" operation.
UTEST_F(BooleanOperation, differenceA_Not_B)
{
    const std::vector<float>& srcMeshVertices = utest_fixture->srcMeshVertices;
    const std::vector<uint32_t>& meshFaceIndices = utest_fixture->meshFaceIndices;
    const std::vector<uint32_t>& meshFaceSizes = utest_fixture->meshFaceSizes;
    std::vector<float> cutMeshVertices = srcMeshVertices;

    // shifted so that the front-bottom-left vertex is located at (0,0,0) and the centre is at (1,1,1)
    for (int i = 0; i < (int)cutMeshVertices.size(); ++i) {
        cutMeshVertices[i] += 1.f;
    }

    const McFlags booleanA_Not_BFlags = MC_DISPATCH_FILTER_FRAGMENT_SEALING_INSIDE | MC_DISPATCH_FILTER_FRAGMENT_LOCATION_ABOVE;

    ASSERT_EQ(mcDispatch(utest_fixture->myContext, //
                  MC_DISPATCH_VERTEX_ARRAY_FLOAT | booleanA_Not_BFlags,
                  &srcMeshVertices[0], &meshFaceIndices[0], &meshFaceSizes[0], (uint32_t)(srcMeshVertices.size() / 3), (uint32_t)meshFaceSizes.size(), //
                  &cutMeshVertices[0], &meshFaceIndices[0], &meshFaceSizes[0], (uint32_t)(cutMeshVertices.size() / 3), (uint32_t)meshFaceSizes.size()),
        MC_NO_ERROR);

    uint32_t numConnComps;
    ASSERT_EQ(mcGetConnectedComponents(utest_fixture->myContext, MC_CONNECTED_COMPONENT_TYPE_ALL, 0, NULL, &numConnComps), MC_NO_ERROR);

    ASSERT_EQ(uint32_t(3), numConnComps);
}

UTEST_F(BooleanOperation, differenceB_Not_A)
{
    const std::vector<float>& srcMeshVertices = utest_fixture->srcMeshVertices;
    const std::vector<uint32_t>& meshFaceIndices = utest_fixture->meshFaceIndices;
    const std::vector<uint32_t>& meshFaceSizes = utest_fixture->meshFaceSizes;
    std::vector<float> cutMeshVertices = srcMeshVertices;

    // shifted so that the front-bottom-left vertex is located at (0,0,0) and the centre is at (1,1,1)
    for (int i = 0; i < (int)cutMeshVertices.size(); ++i) {
        cutMeshVertices[i] += 1.f;
    }

    const McFlags booleanB_Not_AFlags = MC_DISPATCH_FILTER_FRAGMENT_SEALING_OUTSIDE | MC_DISPATCH_FILTER_FRAGMENT_LOCATION_BELOW;

    ASSERT_EQ(mcDispatch(utest_fixture->myContext, //
                  MC_DISPATCH_VERTEX_ARRAY_FLOAT | booleanB_Not_AFlags,
                  &srcMeshVertices[0], &meshFaceIndices[0], &meshFaceSizes[0], (uint32_t)(srcMeshVertices.size() / 3), (uint32_t)meshFaceSizes.size(), //
                  &cutMeshVertices[0], &meshFaceIndices[0], &meshFaceSizes[0], (uint32_t)(cutMeshVertices.size() / 3), (uint32_t)meshFaceSizes.size()),
        MC_NO_ERROR);

    uint32_t numConnComps;
    ASSERT_EQ(mcGetConnectedComponents(utest_fixture->myContext, MC_CONNECTED_COMPONENT_TYPE_ALL, 0, NULL, &numConnComps), MC_NO_ERROR);

    ASSERT_EQ(numConnComps, uint32_t(3));
}

UTEST_F(BooleanOperation, unionOp)
{
    const std::vector<float>& srcMeshVertices = utest_fixture->srcMeshVertices;
    const std::vector<uint32_t>& meshFaceIndices = utest_fixture->meshFaceIndices;
    const std::vector<uint32_t>& meshFaceSizes = utest_fixture->meshFaceSizes;
    std::vector<float> cutMeshVertices = srcMeshVertices;

    // shifted so that the front-bottom-left vertex is located at (0,0,0) and the centre is at (1,1,1)
    for (int i = 0; i < (int)cutMeshVertices.size(); ++i) {
        cutMeshVertices[i] += 1.f;
    }

    const McFlags booleanB_Not_AFlags = MC_DISPATCH_FILTER_FRAGMENT_SEALING_OUTSIDE | MC_DISPATCH_FILTER_FRAGMENT_LOCATION_ABOVE;

    ASSERT_EQ(mcDispatch(utest_fixture->myContext, //
                  MC_DISPATCH_VERTEX_ARRAY_FLOAT | booleanB_Not_AFlags,
                  &srcMeshVertices[0], &meshFaceIndices[0], &meshFaceSizes[0], (uint32_t)(srcMeshVertices.size() / 3), (uint32_t)meshFaceSizes.size(), //
                  &cutMeshVertices[0], &meshFaceIndices[0], &meshFaceSizes[0], (uint32_t)(cutMeshVertices.size() / 3), (uint32_t)meshFaceSizes.size()),
        MC_NO_ERROR);

    uint32_t numConnComps;
    ASSERT_EQ(mcGetConnectedComponents(utest_fixture->myContext, MC_CONNECTED_COMPONENT_TYPE_ALL, 0, NULL, &numConnComps), MC_NO_ERROR);

    ASSERT_EQ(uint32_t(3), numConnComps);
}

UTEST_F(BooleanOperation, intersectionOp)
{
    const std::vector<float>& srcMeshVertices = utest_fixture->srcMeshVertices;
    const std::vector<uint32_t>& meshFaceIndices = utest_fixture->meshFaceIndices;
    const std::vector<uint32_t>& meshFaceSizes = utest_fixture->meshFaceSizes;
    std::vector<float> cutMeshVertices = srcMeshVertices;

    // shifted so that the front-bottom-left vertex is located at (0,0,0) and the centre is at (1,1,1)
    for (int i = 0; i < (int)cutMeshVertices.size(); ++i) {
        cutMeshVertices[i] += 1.f;
    }

    const McFlags booleanB_Not_AFlags = MC_DISPATCH_FILTER_FRAGMENT_SEALING_INSIDE | MC_DISPATCH_FILTER_FRAGMENT_LOCATION_BELOW;

    ASSERT_EQ(mcDispatch(utest_fixture->myContext, //
                  MC_DISPATCH_VERTEX_ARRAY_FLOAT | booleanB_Not_AFlags,
                  &srcMeshVertices[0], &meshFaceIndices[0], &meshFaceSizes[0], (uint32_t)(srcMeshVertices.size() / 3), (uint32_t)meshFaceSizes.size(), //
                  &cutMeshVertices[0], &meshFaceIndices[0], &meshFaceSizes[0], (uint32_t)(cutMeshVertices.size() / 3), (uint32_t)meshFaceSizes.size()),
        MC_NO_ERROR);

    uint32_t numConnComps;
    ASSERT_EQ(mcGetConnectedComponents(utest_fixture->myContext, MC_CONNECTED_COMPONENT_TYPE_ALL, 0, NULL, &numConnComps), MC_NO_ERROR);

    ASSERT_EQ(numConnComps, uint32_t(3));
}
