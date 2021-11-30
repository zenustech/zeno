#include "mcut/mcut.h"
#include <cstring>
#include <inttypes.h> // PRId64
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

// libigl dependencies
#include <Eigen/Core>
#include <igl/list_to_matrix.h>
#include <igl/readOBJ.h>
#include <igl/readPLY.h>
#include <igl/writeOBJ.h>

#define stringize(s) #s
#define XSTR(s) stringize(s)

#define ASSERT(a)                        \
    do {                                 \
        if (0 == (a)) {                  \
            std::fprintf(stderr,         \
                "Assertion failed: %s, " \
                "%d at \'%s\'\n",        \
                __FILE__,                \
                __LINE__,                \
                XSTR(a));                \
            std::abort();                \
        }                                \
    } while (0)

#define mcCheckError(errCode) mcCheckError_(errCode, __FILE__, __LINE__)

void mcCheckError_(McResult err, const char* file, int line);

void mcDebugOutput(McDebugSource source,
    McDebugType type,
    unsigned int id,
    McDebugSeverity severity,
    size_t length,
    const char* message,
    const void* userParam);

void readMesh(const std::string& path, std::vector<double>& V, std::vector<uint32_t>& F, std::vector<uint32_t>& Fsizes);
void writeOBJ(
    const std::string& path,
    const float* ccVertices,
    const int ccVertexCount,
    const uint32_t* ccFaceIndices,
    const uint32_t* faceSizes,
    const uint32_t ccFaceCount);

int main(int argc, char* argv[])
{
    bool help = false;

    for (int i = 0; help == false && i < argc; ++i) {
        if (!strcmp("--help", argv[i]) || !strcmp("-h", argv[i])) {
            help = true;
        }
    }

    if (help || argc < 3) {
        fprintf(stdout, "<exe> <path/to/source-mesh> <path/to/cut-mesh>\n\nSupported file types: obj\n");
        return 1;
    }

    const char* srcMeshFilePath = argv[1];
    const char* cutMeshFilePath = argv[2];

    std::cout << "source-mesh file:" << srcMeshFilePath << std::endl;
    std::cout << "cut-mesh file:" << cutMeshFilePath << std::endl;

    // load meshes
    // -----------
    std::vector<double> srcMeshVertices;
    std::vector<uint32_t> srcMeshFaceIndices;
    std::vector<uint32_t> srcMeshFaceSizes;
    readMesh(srcMeshFilePath, srcMeshVertices, srcMeshFaceIndices, srcMeshFaceSizes);

    printf("src-mesh vertices=%d faces=%d\n", (int)srcMeshVertices.size()/3, (int)srcMeshFaceSizes.size());

    std::vector<double> cutMeshVertices;
    std::vector<uint32_t> cutMeshFaceIndices;
    std::vector<uint32_t> cutMeshFaceSizes;
    readMesh(cutMeshFilePath, cutMeshVertices, cutMeshFaceIndices, cutMeshFaceSizes);

    printf("cut-mesh vertices=%d faces=%d\n", (int)cutMeshVertices.size()/3, (int)cutMeshFaceSizes.size());


    // init dispatch context
    // ---------------------
    McContext context;
#ifdef  NDEBUG
    McResult err = mcCreateContext(&context, MC_NULL_HANDLE);
#else
    McResult err = mcCreateContext(&context, MC_DEBUG);
#endif
    mcCheckError(err);

    // config debug output
    // -----------------------
    uint64_t numBytes = 0;
    McFlags contextFlags;
    err = mcGetInfo(context, MC_CONTEXT_FLAGS, 0, nullptr, &numBytes);
    mcCheckError(err);

    ASSERT(sizeof(McFlags) == numBytes);

    err = mcGetInfo(context, MC_CONTEXT_FLAGS, numBytes, &contextFlags, nullptr);
    mcCheckError(err);

    if (contextFlags & MC_DEBUG) {
        mcDebugMessageCallback(context, mcDebugOutput, nullptr);
        mcDebugMessageControl(context, McDebugSource::MC_DEBUG_SOURCE_ALL, McDebugType::MC_DEBUG_TYPE_ALL, McDebugSeverity::MC_DEBUG_SEVERITY_ALL, true);
    }

    // query and print default numerical configuration settings
    // ------------------------------------
    uint64_t defaultPrec = 0;
    numBytes = 0;
    err = mcGetInfo(context, MC_DEFAULT_PRECISION, 0, nullptr, &numBytes);
    mcCheckError(err);
    ASSERT(sizeof(uint64_t) == numBytes);
    err = mcGetInfo(context, MC_DEFAULT_PRECISION, numBytes, &defaultPrec, nullptr);
    mcCheckError(err);

    fprintf(stdout, "default precision: %" PRId64 "\n", (uint64_t)defaultPrec);

    uint64_t minPrec = 0;
    numBytes = 0;
    err = mcGetInfo(context, MC_PRECISION_MIN, 0, nullptr, &numBytes);
    mcCheckError(err);
    ASSERT(sizeof(uint64_t) == numBytes);

    err = mcGetInfo(context, MC_PRECISION_MIN, numBytes, &minPrec, nullptr);
    mcCheckError(err);

    fprintf(stdout, "min precision: %" PRId64 "\n", (uint64_t)minPrec);

    uint64_t maxPrec = 0;
    numBytes = 0;
    err = mcGetInfo(context, MC_PRECISION_MAX, 0, nullptr, &numBytes);
    mcCheckError(err);
    ASSERT(sizeof(uint64_t) == numBytes);
    err = mcGetInfo(context, MC_PRECISION_MAX, numBytes, &maxPrec, nullptr);
    mcCheckError(err);

    fprintf(stdout, "max precision: %" PRId64 "\n", (uint64_t)maxPrec);

    McFlags defaultRoundingMode = 0;
    numBytes = 0;
    err = mcGetInfo(context, MC_DEFAULT_ROUNDING_MODE, 0, nullptr, &numBytes);
    mcCheckError(err);
    ASSERT(sizeof(McFlags) == numBytes);
    err = mcGetInfo(context, MC_DEFAULT_ROUNDING_MODE, numBytes, &defaultRoundingMode, nullptr);
    mcCheckError(err);

    char roundingModeStr[32];
    switch (defaultRoundingMode) {
    case MC_ROUNDING_MODE_TO_NEAREST:
        sprintf(roundingModeStr, "%s", "TO_NEAREST");
        break;
    case MC_ROUNDING_MODE_TOWARD_ZERO:
        sprintf(roundingModeStr, "%s", "TOWARD_ZERO");
        break;
    case MC_ROUNDING_MODE_TOWARD_POS_INF:
        sprintf(roundingModeStr, "%s", "TOWARD_POS_INF");
        break;
    case MC_ROUNDING_MODE_TOWARD_NEG_INF:
        sprintf(roundingModeStr, "%s", "TOWARD_NEG_INF");
        break;
    default:
        printf("unknown rounding mode\n");
        break;
    }

    fprintf(stdout, "rounding mode: %s\n", (const char*)roundingModeStr);

    // NOTE: at this point (just before mcDispatch) we can change the rounding mode & precision bits if so desired
    // Changing the precision bits will only be effective if MCUT is built with MCUT_WITH_ARBITRARY_PRECISION_NUMBERS

    // do the cutting
    // --------------
    err = mcDispatch(
        context,
        MC_DISPATCH_VERTEX_ARRAY_DOUBLE | MC_DISPATCH_ENFORCE_GENERAL_POSITION,
        // source mesh
        srcMeshVertices.data(),
        srcMeshFaceIndices.data(),
        srcMeshFaceSizes.data(),
        (uint32_t)srcMeshVertices.size() / 3,
        (uint32_t)srcMeshFaceSizes.size(),
        // cut mesh
        cutMeshVertices.data(),
        cutMeshFaceIndices.data(),
        cutMeshFaceSizes.data(),
        (uint32_t)cutMeshVertices.size() / 3,
        (uint32_t)cutMeshFaceSizes.size());

    mcCheckError(err);

    uint32_t numConnComps;
    std::vector<McConnectedComponent> pConnComps;

    // we want to query all available connected components:
    // --> fragments, patches and seamed
    err = mcGetConnectedComponents(context, MC_CONNECTED_COMPONENT_TYPE_ALL, 0, NULL, &numConnComps);
    mcCheckError(err);

    if (numConnComps == 0) {
        printf("no connected components found\n");
        std::exit(0);
    }

    pConnComps.resize(numConnComps);

    err = mcGetConnectedComponents(context, MC_CONNECTED_COMPONENT_TYPE_ALL, (uint32_t)pConnComps.size(), pConnComps.data(), NULL);

    mcCheckError(err);

    //
    // query connected component data
    //
    for (int i = 0; i < (int)pConnComps.size(); ++i) {
        McConnectedComponent connCompId = pConnComps[i]; // connected compoenent id

        printf("connected component: %d\n", i);

        // vertex array
        numBytes = 0;
        err = mcGetConnectedComponentData(context, connCompId, MC_CONNECTED_COMPONENT_DATA_VERTEX_FLOAT, 0, NULL, &numBytes);
        mcCheckError(err);
        uint32_t numberOfVertices = numBytes / (sizeof(float)*3);
        ASSERT(numberOfVertices >= 3);
        std::vector<float> vertices((size_t)numberOfVertices * 3u);

        err = mcGetConnectedComponentData(context, connCompId, MC_CONNECTED_COMPONENT_DATA_VERTEX_FLOAT, numBytes, (void*)vertices.data(), NULL);
        mcCheckError(err);

        printf("vertices: %d\n", (int)vertices.size() / 3);

        // face indices
        numBytes = 0;
        err = mcGetConnectedComponentData(context, connCompId, MC_CONNECTED_COMPONENT_DATA_FACE, 0, NULL, &numBytes);
        mcCheckError(err);

        ASSERT(numBytes > 0);

        std::vector<uint32_t> faceIndices;
        faceIndices.resize(numBytes / sizeof(uint32_t));

        err = mcGetConnectedComponentData(context, connCompId, MC_CONNECTED_COMPONENT_DATA_FACE, numBytes, faceIndices.data(), NULL);
        mcCheckError(err);

        // face sizes
        numBytes = 0;
        err = mcGetConnectedComponentData(context, connCompId, MC_CONNECTED_COMPONENT_DATA_FACE_SIZE, 0, NULL, &numBytes);
        mcCheckError(err);

        ASSERT(numBytes > 0);

        std::vector<uint32_t> faceSizes;
        faceSizes.resize(numBytes / sizeof(uint32_t));

        err = mcGetConnectedComponentData(context, connCompId, MC_CONNECTED_COMPONENT_DATA_FACE_SIZE, numBytes, faceSizes.data(), NULL);
        mcCheckError(err);

        printf("faces: %d\n", (int)faceSizes.size());

        char fnameBuf[512];
        sprintf(fnameBuf, "cc%d.obj", i);

        writeOBJ(fnameBuf,
            (float*)vertices.data(),
            (uint32_t)vertices.size() / 3,
            (uint32_t*)faceIndices.data(),
            (uint32_t*)faceSizes.data(),
            (uint32_t)faceSizes.size());
    }

    // destroy internal data associated with each connected component
    err = mcReleaseConnectedComponents(context, (uint32_t)pConnComps.size(), pConnComps.data());
    mcCheckError(err);

    // destroy context
    err = mcReleaseContext(context);
    mcCheckError(err);

    return 0;
}

void mcDebugOutput(McDebugSource source,
    McDebugType type,
    unsigned int id,
    McDebugSeverity severity,
    size_t length,
    const char* message,
    const void* userParam)
{
    printf("---------------\n");
    printf("Debug message ( %d ), length=%zu\n%s\n--\n", id, length, message);
    printf("userParam=%p\n", userParam);

    switch (source) {
    case MC_DEBUG_SOURCE_API:
        printf("Source: API");
        break;
    case MC_DEBUG_SOURCE_KERNEL:
        printf("Source: Kernel");
        break;
    case MC_DEBUG_SOURCE_ALL:
        break;
    }

    printf("\n");

    switch (type) {
    case MC_DEBUG_TYPE_ERROR:
        printf("Type: Error");
        break;
    case MC_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
        printf("Type: Deprecated Behaviour");
        break;
    case MC_DEBUG_TYPE_OTHER:
        printf("Type: Other");
        break;
    case MC_DEBUG_TYPE_ALL:
        break;
    }

    printf("\n");

    switch (severity) {
    case MC_DEBUG_SEVERITY_HIGH:
        printf("Severity: high");
        break;
    case MC_DEBUG_SEVERITY_MEDIUM:
        printf("Severity: medium");
        break;
    case MC_DEBUG_SEVERITY_LOW:
        printf("Severity: low");
        break;
    case MC_DEBUG_SEVERITY_NOTIFICATION:
        printf("Severity: notification");
        break;
    case MC_DEBUG_SEVERITY_ALL:
        break;
    }

    printf("\n\n");
}

void mcCheckError_(McResult err, const char* file, int line)
{
    std::string error;
    switch (err) {
    case MC_OUT_OF_MEMORY:
        error = "MC_OUT_OF_MEMORY";
        break;
    case MC_INVALID_VALUE:
        error = "MC_INVALID_VALUE";
        break;
    case MC_INVALID_OPERATION:
        error = "MC_INVALID_OPERATION";
        break;
    case MC_NO_ERROR:
        error = "MC_NO_ERROR";
        break;
        case MC_RESULT_MAX_ENUM:
        error = "UNKNOWN";
        break;
    }
    if (err) {
        std::cout << error << " | " << file << " (" << line << ")" << std::endl;
    }
}

// libIGL's "writeOBJ" function fails when dealing with facex with > 4 vertices.
void writeOBJ(
    const std::string& path,
    const float* ccVertices,
    const int ccVertexCount,
    const uint32_t* ccFaceIndices,
    const uint32_t* faceSizes,
    const uint32_t ccFaceCount)
{
    printf("write file: %s\n", path.c_str());

    std::ofstream file(path);

    // write vertices and normals
    for (uint32_t i = 0; i < (uint32_t)ccVertexCount; ++i) {
        double x = ccVertices[(uint64_t)i * 3 + 0];
        double y = ccVertices[(uint64_t)i * 3 + 1];
        double z = ccVertices[(uint64_t)i * 3 + 2];
        file << "v " << x << " " << y << " " << z << std::endl;
    }

    int faceVertexOffsetBase = 0;

    // for each face in CC
    for (uint32_t f = 0; f < ccFaceCount; ++f) {

        int faceSize = faceSizes[f];
        file << "f ";
        // for each vertex in face
        for (int v = 0; (v < faceSize); v++) {
            const int ccVertexIdx = ccFaceIndices[(uint64_t)faceVertexOffsetBase + v];
            file << (ccVertexIdx + 1) << " ";
        } // for (int v = 0; v < faceSize; ++v) {
        file << std::endl;

        faceVertexOffsetBase += faceSize;
    }
}
void readMesh(const std::string& path, std::vector<double>& V, std::vector<uint32_t>& F, std::vector<uint32_t>& Fsizes)
{
    printf("read: %s\n", path.c_str());
    Eigen::MatrixXd Vmat;
    Eigen::MatrixXi Fmat;
    if (path.find(".obj") != std::string::npos) {
        igl::readOBJ(path, Vmat, Fmat);
    }

    for (int i = 0; i < (int)Vmat.rows(); ++i) {
        const Eigen::VectorXd v = Vmat.row(i);
        V.push_back((double)v(0));
        V.push_back((double)v(1));
        V.push_back((double)v(2));
    }

    for (int i = 0; i < (int)Fmat.rows(); ++i) {
        const Eigen::VectorXi f = Fmat.row(i);
        for (int j = 0; j < (int)f.rows(); ++j) {
            F.push_back((uint32_t)f(j));
        }

        Fsizes.push_back((uint32_t)f.rows());
    }
}
