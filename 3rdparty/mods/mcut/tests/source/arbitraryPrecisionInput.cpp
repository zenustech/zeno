#if defined(_WIN32)
#define _CRT_SECURE_NO_WARNINGS 1
#endif

#include "utest.h"
#include <mcut/mcut.h>

#include <cctype> // std::isdigit
#include <cstdio>
#include <cstring>
#include <vector>

struct ArbitraryPrecisionInput {
    McContext myContext = MC_NULL_HANDLE;
    std::vector<McConnectedComponent> pConnComps_;
};

UTEST_F_SETUP(ArbitraryPrecisionInput)
{
    // create with no flags (default)
    EXPECT_EQ(mcCreateContext(&utest_fixture->myContext, MC_NULL_HANDLE), MC_NO_ERROR);
    EXPECT_TRUE(utest_fixture->myContext != nullptr);
}

UTEST_F_TEARDOWN(ArbitraryPrecisionInput)
{
    std::vector<McConnectedComponent>& pConnComps_ = utest_fixture->pConnComps_;
    EXPECT_EQ(mcReleaseConnectedComponents(utest_fixture->myContext, (uint32_t)pConnComps_.size(), pConnComps_.data()), MC_NO_ERROR);
    EXPECT_EQ(mcReleaseContext(utest_fixture->myContext), MC_NO_ERROR);
}

UTEST_F(ArbitraryPrecisionInput, dispatchExactCoords)
{
    std::vector<McConnectedComponent>& pConnComps_ = utest_fixture->pConnComps_;

    const char pSrcMeshVertices[] = {
        "0 0 0\n"
        "0 10 0\n"
        "10 0 0"
    }; // exact numbers (x y z\nx y z\n x, .....)
    uint32_t pSrcMeshFaceIndices[] = { 0, 2, 1 };
    uint32_t pSrcMeshFaceSizes[] = { 3 };
    uint32_t numSrcMeshVertices = 3;
    uint32_t numSrcMeshFaces = 1;

    ASSERT_TRUE(pSrcMeshVertices != nullptr);
    ASSERT_TRUE(pSrcMeshFaceIndices != nullptr);
    ASSERT_TRUE(pSrcMeshVertices != nullptr);
    ASSERT_GT((int)numSrcMeshVertices, 2);
    ASSERT_GT((int)numSrcMeshFaces, 0);

    const char pCutMeshVertices[] = {
        "-2.5 7.5 2.5\n"
        "-2.5 7.5 -7.5\n"
        "7.5 7.5 2.5"
    }; // exact numbers ...
    uint32_t pCutMeshFaceIndices[] = { 0, 2, 1 };
    uint32_t pCutMeshFaceSizes[] = { 3 };
    uint32_t numCutMeshVertices = 3;
    uint32_t numCutMeshFaces = 1;

    ASSERT_TRUE(pCutMeshVertices != nullptr);
    ASSERT_TRUE(pCutMeshFaceIndices != nullptr);
    ASSERT_TRUE(pCutMeshFaceSizes != nullptr);
    ASSERT_GT((int)numCutMeshVertices, 2);
    ASSERT_GT((int)numCutMeshFaces, 0);

    // do the cutting
    // --------------
    ASSERT_EQ(mcDispatch(
                  utest_fixture->myContext,
                  MC_DISPATCH_VERTEX_ARRAY_EXACT, // vertex array is a string of numbers
                  pSrcMeshVertices,
                  pSrcMeshFaceIndices,
                  pSrcMeshFaceSizes,
                  numSrcMeshVertices,
                  numSrcMeshFaces,
                  pCutMeshVertices,
                  pCutMeshFaceIndices,
                  pCutMeshFaceSizes,
                  numCutMeshVertices,
                  numCutMeshFaces),
        MC_NO_ERROR);

    uint32_t numConnComps = 0;

    ASSERT_EQ(mcGetConnectedComponents(utest_fixture->myContext, MC_CONNECTED_COMPONENT_TYPE_ALL, 0, NULL, &numConnComps), MC_NO_ERROR);

    if (numConnComps == 0) {
        printf("no connected component found\n");
    }

    pConnComps_.resize(numConnComps);

    ASSERT_EQ(mcGetConnectedComponents(utest_fixture->myContext, MC_CONNECTED_COMPONENT_TYPE_ALL, (uint32_t)pConnComps_.size(), pConnComps_.data(), NULL), MC_NO_ERROR);

    //
    // query connected component data
    //
    for (int i = 0; i < (int)pConnComps_.size(); ++i) {
        McConnectedComponent cc = pConnComps_[i]; // connected compoenent id

        uint64_t vertexCountBytes = 0;
        ASSERT_EQ(mcGetConnectedComponentData(utest_fixture->myContext, cc, MC_CONNECTED_COMPONENT_DATA_VERTEX_COUNT, 0, NULL, &vertexCountBytes), MC_NO_ERROR);
        ASSERT_EQ(vertexCountBytes, sizeof(uint32_t));

        // we can also directly query the number of vertices since we know the number of bytes, which is a constant 4 bytes.
        uint32_t numberOfVertices = 0;
        ASSERT_EQ(mcGetConnectedComponentData(utest_fixture->myContext, cc, MC_CONNECTED_COMPONENT_DATA_VERTEX_COUNT, vertexCountBytes, &numberOfVertices, NULL), MC_NO_ERROR);
        ASSERT_GT((int)numberOfVertices, 0);

        // float
        // -----
        {
            // vertex array
            uint64_t connCompVerticesBytes = 0;
            ASSERT_EQ(mcGetConnectedComponentData(utest_fixture->myContext, cc, MC_CONNECTED_COMPONENT_DATA_VERTEX_FLOAT, 0, NULL, &connCompVerticesBytes), MC_NO_ERROR);

            std::vector<float> vertices;
            uint32_t nfloats = (uint32_t)(connCompVerticesBytes / sizeof(float));
            vertices.resize(nfloats);

            ASSERT_EQ(mcGetConnectedComponentData(utest_fixture->myContext, cc, MC_CONNECTED_COMPONENT_DATA_VERTEX_FLOAT, connCompVerticesBytes, (void*)vertices.data(), NULL), MC_NO_ERROR);

            ASSERT_EQ((uint64_t)numberOfVertices, (uint64_t)(connCompVerticesBytes / (sizeof(float) * 3)));
        }

        // double
        // -----
        {
            // vertex array
            uint64_t connCompVerticesBytes = 0;
            ASSERT_EQ(mcGetConnectedComponentData(utest_fixture->myContext, cc, MC_CONNECTED_COMPONENT_DATA_VERTEX_DOUBLE, 0, NULL, &connCompVerticesBytes), MC_NO_ERROR);

            std::vector<double> vertices;
            uint32_t ndoubles = (uint32_t)(connCompVerticesBytes / sizeof(double));
            vertices.resize(ndoubles);

            ASSERT_EQ(mcGetConnectedComponentData(utest_fixture->myContext, cc, MC_CONNECTED_COMPONENT_DATA_VERTEX_DOUBLE, connCompVerticesBytes, (void*)vertices.data(), NULL), MC_NO_ERROR);

            ASSERT_EQ((uint64_t)numberOfVertices, (uint64_t)(connCompVerticesBytes / (sizeof(double) * 3)));
        }

        // exact numerical strings
        // ----------------------
        {
            // vertex array
            uint64_t connCompVerticesBytes = 0;
            ASSERT_EQ(mcGetConnectedComponentData(utest_fixture->myContext, cc, MC_CONNECTED_COMPONENT_DATA_VERTEX_EXACT, 0, NULL, &connCompVerticesBytes), MC_NO_ERROR);

            std::vector<char> rawVerticesString;
            rawVerticesString.resize(connCompVerticesBytes);

            ASSERT_EQ(mcGetConnectedComponentData(utest_fixture->myContext, cc, MC_CONNECTED_COMPONENT_DATA_VERTEX_EXACT, connCompVerticesBytes, (void*)rawVerticesString.data(), NULL), MC_NO_ERROR);

            auto my_isdigit = [&](unsigned char ch) {
                return ch == '.' || ch == '-' || std::isdigit(ch);
            };

            std::vector<char> x;
            std::vector<char> y;
            std::vector<char> z;

            const char* vptr = reinterpret_cast<const char*>(rawVerticesString.data());
            const char* vptr_ = vptr; // shifted

            for (uint32_t i = 0; i < numberOfVertices * 3; ++i) {

                vptr_ = std::strchr(vptr, ' ');
                std::ptrdiff_t diff = vptr_ - vptr;
                uint64_t srcStrLen = diff + 1; // extra byte for null-char
                if (vptr_ == nullptr) {
                    srcStrLen = strlen(vptr) + 1;
                    ASSERT_TRUE(i == ((numberOfVertices * 3) - 1));
                }

                ASSERT_GT(srcStrLen, 0);

                if ((i % 3) == 0) { // x
                    x.resize(srcStrLen);
                    std::sscanf(vptr, "%s", &x[0]);
                    x.back() = '\0';
                    ASSERT_TRUE(my_isdigit(x[0]));
                } else if ((i % 3) - 1 == 0) { // y
                    y.resize(srcStrLen);
                    std::sscanf(vptr, "%s", &y[0]);
                    y.back() = '\0';
                    ASSERT_TRUE(my_isdigit(y[0]));
                } else if ((i % 3) - 2 == 0) { // z
                    z.resize(srcStrLen);
                    std::sscanf(vptr, "%s", &z[0]);
                    z.back() = '\0';
                    ASSERT_TRUE(my_isdigit(z[0]));
                }

                vptr = vptr_ + 1; // offset so that we point to the start of the next number/line
            }
        }
    }
}
