/*
This tutorial shows how to propagate per-face texture coordinates from input meshes and onto the output
connected components after cutting.
*/

#include "mcut/mcut.h"

#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
// libigl dependencies
#include <Eigen/Core>
#include <igl/barycentric_coordinates.h>
#include <igl/barycentric_interpolation.h>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>

#define my_assert(cond)                             \
    if (!(cond))                                    \
    {                                               \
        fprintf(stderr, "MCUT error: %s\n", #cond); \
        std::exit(1);                               \
    }

struct InputMesh
{
    Eigen::MatrixXd corner_normals;
    Eigen::MatrixXi fNormIndices;

    Eigen::MatrixXd UV_V;
    Eigen::MatrixXi UV_F;
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    std::vector<std::tuple<std::string, unsigned, unsigned>> FM;

    // variables for mesh data in a format suited for MCUT
    std::string fpath;                      // path to mesh file
    std::vector<uint32_t> faceSizesArray;   // vertices per face
    std::vector<uint32_t> faceIndicesArray; // face indices
    std::vector<double> vertexCoordsArray;  // vertex coords
};

// basic comparison of doubles
bool compare(double x, double y)
{
    return std::fabs(x - y) < 1e-6;
}

int main()
{
    //  load meshes.
    // -----------------
    InputMesh srcMesh;

    // read file
    srcMesh.fpath = DATA_DIR "/a_.obj"; /* DATA_DIR "/a.obj"*/

    bool srcMeshLoaded = igl::readOBJ(srcMesh.fpath, srcMesh.V, srcMesh.UV_V, srcMesh.corner_normals, srcMesh.F, srcMesh.UV_F, srcMesh.fNormIndices);

    if (!srcMeshLoaded)
    {
        std::fprintf(stderr, "error: could not load source mesh --> %s\n", srcMesh.fpath.c_str());
        std::exit(1);
    }

    // copy vertices
    for (int i = 0; i < srcMesh.V.rows(); ++i)
    {
        const Eigen::Vector3d &v = srcMesh.V.row(i);
        my_assert(v.size() == 3);
        srcMesh.vertexCoordsArray.push_back(v.x());
        srcMesh.vertexCoordsArray.push_back(v.y());
        srcMesh.vertexCoordsArray.push_back(v.z());
    }

    // copy faces
    for (int i = 0; i < srcMesh.F.rows(); ++i)
    {
        const Eigen::VectorXi &f = srcMesh.F.row(i);
        srcMesh.faceIndicesArray.push_back(f.x());
        srcMesh.faceIndicesArray.push_back(f.y());
        srcMesh.faceIndicesArray.push_back(f.z());
        srcMesh.faceSizesArray.push_back((uint32_t)f.rows());
    }

    printf("source mesh:\n\tvertices=%d\n\tfaces=%d\n", (int)srcMesh.V.rows(), (int)srcMesh.F.rows());

    InputMesh cutMesh;

    // read file
    cutMesh.fpath = DATA_DIR "/b_.obj"; /*DATA_DIR "/b.obj";*/
    bool cutMeshLoaded = igl::readOBJ(cutMesh.fpath, cutMesh.V, cutMesh.UV_V, cutMesh.corner_normals, cutMesh.F, cutMesh.UV_F, cutMesh.fNormIndices);

    if (!cutMeshLoaded)
    {
        std::fprintf(stderr, "error: could not load cut mesh --> %s\n", cutMesh.fpath.c_str());
        std::exit(1);
    }

    // copy vertices
    for (int i = 0; i < cutMesh.V.rows(); ++i)
    {
        const Eigen::Vector3d &v = cutMesh.V.row(i);
        cutMesh.vertexCoordsArray.push_back(v.x());
        cutMesh.vertexCoordsArray.push_back(v.y());
        cutMesh.vertexCoordsArray.push_back(v.z());
    }

    // copy faces
    for (int i = 0; i < cutMesh.F.rows(); ++i)
    {
        const Eigen::VectorXi &f = cutMesh.F.row(i);
        cutMesh.faceIndicesArray.push_back(f.x());
        cutMesh.faceIndicesArray.push_back(f.y());
        cutMesh.faceIndicesArray.push_back(f.z());
        cutMesh.faceSizesArray.push_back((uint32_t)f.rows());
    }

    printf("cut mesh:\n\tvertices=%d\n\tfaces=%d\n", (int)cutMesh.V.rows(), (int)cutMesh.F.rows());

    //  create a context
    // -------------------
    McContext context = MC_NULL_HANDLE;
    McResult err = mcCreateContext(&context, MC_DEBUG);

    my_assert(err == MC_NO_ERROR);

    //  do the cutting
    // -----------------
    err = mcDispatch(
        context,
        MC_DISPATCH_VERTEX_ARRAY_DOUBLE | MC_DISPATCH_INCLUDE_VERTEX_MAP | MC_DISPATCH_INCLUDE_FACE_MAP,
        // source mesh
        reinterpret_cast<const void *>(srcMesh.vertexCoordsArray.data()),
        reinterpret_cast<const uint32_t *>(srcMesh.faceIndicesArray.data()),
        srcMesh.faceSizesArray.data(),
        static_cast<uint32_t>(srcMesh.vertexCoordsArray.size() / 3),
        static_cast<uint32_t>(srcMesh.faceSizesArray.size()),
        // cut mesh
        reinterpret_cast<const void *>(cutMesh.vertexCoordsArray.data()),
        cutMesh.faceIndicesArray.data(),
        cutMesh.faceSizesArray.data(),
        static_cast<uint32_t>(cutMesh.vertexCoordsArray.size() / 3),
        static_cast<uint32_t>(cutMesh.faceSizesArray.size()));

    my_assert(err == MC_NO_ERROR);

    //  query the number of available connected component (all types)
    // -------------------------------------------------------------
    uint32_t numConnectedComponents;

    err = mcGetConnectedComponents(context, MC_CONNECTED_COMPONENT_TYPE_ALL, 0, NULL, &numConnectedComponents);
    my_assert(err == MC_NO_ERROR);

    printf("connected components: %d\n", (int)numConnectedComponents);

    if (numConnectedComponents == 0)
    {
        fprintf(stdout, "no connected components found\n");
        exit(0);
    }

    std::vector<McConnectedComponent> connectedComponents(numConnectedComponents, 0);
    err = mcGetConnectedComponents(context, MC_CONNECTED_COMPONENT_TYPE_ALL, (uint32_t)connectedComponents.size(), connectedComponents.data(), NULL);
    my_assert(err == MC_NO_ERROR);

    //  query the data of each connected component from MCUT
    // -------------------------------------------------------

    for (int i = 0; i < (int)numConnectedComponents; ++i)
    {
        McConnectedComponent connComp = connectedComponents[i]; // connected compoenent id

        uint64_t numBytes = 0;

        //  query the ccVertices
        // ------------------------

        numBytes = 0;
        err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_VERTEX_DOUBLE, 0, NULL, &numBytes);
        my_assert(err == MC_NO_ERROR);
        uint32_t ccVertexCount = (uint32_t)(numBytes / (sizeof(double) * 3));

        std::vector<double> ccVertices((size_t)ccVertexCount * 3u, 0);
        err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_VERTEX_DOUBLE, numBytes, (void *)ccVertices.data(), NULL);
        my_assert(err == MC_NO_ERROR);

        //  query the faces
        // -------------------
        numBytes = 0;
        err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_FACE, 0, NULL, &numBytes);
        my_assert(err == MC_NO_ERROR);
        std::vector<uint32_t> ccFaceIndices(numBytes / sizeof(uint32_t), 0);
        err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_FACE, numBytes, ccFaceIndices.data(), NULL);
        my_assert(err == MC_NO_ERROR);

        //  query the face sizes
        // ------------------------
        numBytes = 0;
        err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_FACE_SIZE, 0, NULL, &numBytes);
        my_assert(err == MC_NO_ERROR);
        std::vector<uint32_t> faceSizes(numBytes / sizeof(uint32_t), 0);
        faceSizes.resize(numBytes / sizeof(uint32_t));
        err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_FACE_SIZE, numBytes, faceSizes.data(), NULL);
        my_assert(err == MC_NO_ERROR);

        //  query the vertex map
        // ------------------------

        err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_VERTEX_MAP, 0, NULL, &numBytes);
        my_assert(err == MC_NO_ERROR);
        std::vector<uint32_t> ccVertexMap(numBytes / sizeof(uint32_t));
        err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_VERTEX_MAP, numBytes, ccVertexMap.data(), NULL);
        my_assert(err == MC_NO_ERROR);

        //  query the face map
        // ------------------------
        const uint32_t ccFaceCount = static_cast<uint32_t>(faceSizes.size());
        err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_FACE_MAP, 0, NULL, &numBytes);
        my_assert(err == MC_NO_ERROR);
        std::vector<uint32_t> ccFaceMap(numBytes / sizeof(uint32_t), 0);
        err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_FACE_MAP, numBytes, ccFaceMap.data(), NULL);
        my_assert(err == MC_NO_ERROR);

        //  resolve connected component name
        // ---------------------------------

        // Here we create a name the connected component based on its properties

        // get type
        McConnectedComponentType ccType;
        err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_TYPE, sizeof(McConnectedComponentType), &ccType, NULL);
        my_assert(err == MC_NO_ERROR);

        std::string name;
        McFragmentLocation fragmentLocation = (McFragmentLocation)0;
        McPatchLocation pathLocation = (McPatchLocation)0;
        bool isFragment = false;

        if (ccType == MC_CONNECTED_COMPONENT_TYPE_SEAM)
        {
            name += "seam";
        }
        else if (ccType == MC_CONNECTED_COMPONENT_TYPE_INPUT)
        {
            name += "input";
        }
        else
        {
            isFragment = (ccType == MC_CONNECTED_COMPONENT_TYPE_FRAGMENT);
            name += isFragment ? "frag" : "patch";

            err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_PATCH_LOCATION, sizeof(McPatchLocation), &pathLocation, NULL);
            my_assert(err == MC_NO_ERROR);
            name += pathLocation == MC_PATCH_LOCATION_INSIDE ? ".inside" : (pathLocation == MC_PATCH_LOCATION_OUTSIDE ? ".outside" : ".undefined");

            if (isFragment)
            {

                err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_FRAGMENT_LOCATION, sizeof(McFragmentLocation), &fragmentLocation, NULL);
                my_assert(err == MC_NO_ERROR);
                name += fragmentLocation == MC_FRAGMENT_LOCATION_ABOVE ? ".above" : ".below"; // missing loc="undefined" case

                McFragmentSealType sType = (McFragmentSealType)0;
                err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_FRAGMENT_SEAL_TYPE, sizeof(McFragmentSealType), &sType, NULL);
                my_assert(err == MC_NO_ERROR);
                name += sType == MC_FRAGMENT_SEAL_TYPE_COMPLETE ? ".complete" : ".none";
            }
        }

        bool ccIsFromSrcMesh = (ccType == MC_CONNECTED_COMPONENT_TYPE_FRAGMENT);

        // connected-components is not a fragment && it is a seam
        if (!ccIsFromSrcMesh)
        {
            if (ccType == MC_CONNECTED_COMPONENT_TYPE_SEAM)
            {
                // get origin
                McSeamOrigin ccOrig = (McSeamOrigin)0;
                err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_ORIGIN, sizeof(McSeamOrigin), &ccOrig, NULL);
                my_assert(err == MC_NO_ERROR);

                ccIsFromSrcMesh = (ccOrig == McSeamOrigin::MC_SEAM_ORIGIN_SRCMESH);
                name += ccIsFromSrcMesh ? ".sm" : ".cm";
            }
            else if (ccType == MC_CONNECTED_COMPONENT_TYPE_INPUT)
            {
                McInputOrigin ccOrig = (McInputOrigin)0;
                err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_ORIGIN, sizeof(McInputOrigin), &ccOrig, NULL);
                my_assert(err == MC_NO_ERROR);
                ccIsFromSrcMesh = (ccOrig == McInputOrigin::MC_INPUT_ORIGIN_SRCMESH);
                name += ccIsFromSrcMesh ? ".sm" : ".cm";
            }
        }

        int faceVertexOffsetBase = 0;

        char fnameBuf[64];
        sprintf(fnameBuf, ("OUT_" + name + ".obj").c_str(), i);
        std::string fname(fnameBuf);
        std::string fpath(DATA_DIR "/" + fname);
        printf("write file: %s\n", fpath.c_str());
        std::ofstream file(fpath);

        file << (ccIsFromSrcMesh ? "mtllib a.mtl" : "mtllib b.mtl") << std::endl;

        std::vector<Eigen::Vector2d> ccTexCoords;
        // CC vertex index to texture coordinate index/indices.
        // Its possible to map to more than texture coordinates since such coordinates are specified to per-face.
        std::map<int, std::vector<int>> ccVertexIndexToTexCoordIndices;
        std::vector<uint32_t> ccFaceVertexTexCoordIndices;

        // for each face in CC
        for (int f = 0; f < (int)ccFaceCount; ++f)
        {

            // input mesh face index (which may be offsetted!)
            const uint32_t imFaceIdxRaw = ccFaceMap.at(f); // source- or cut-mesh
            // input mesh face index (actual index value, accounting for offset)
            uint32_t imFaceIdx = imFaceIdxRaw;
            bool faceIsFromSrcMesh = (imFaceIdxRaw < (uint32_t)srcMesh.F.rows());

            if (!faceIsFromSrcMesh)
            {
                imFaceIdx = imFaceIdxRaw - (uint32_t)srcMesh.F.rows(); // accounting for offset
            }

            int faceSize = faceSizes.at(f);

            // for each vertex in face
            for (int v = 0; v < (int)faceSize; ++v)
            {

                const int ccVertexIdx = ccFaceIndices.at((uint64_t)faceVertexOffsetBase + v);
                // input mesh (source mesh or cut mesh) vertex index (which may be offsetted)
                const uint32_t imVertexIdxRaw = ccVertexMap.at(ccVertexIdx);
                bool vertexIsFromSrcMesh = (imVertexIdxRaw < srcMesh.V.rows());
                const bool vertexIsIntersectionPoint = (imVertexIdxRaw == MC_UNDEFINED_VALUE);
                uint32_t imVertexIdx = imVertexIdxRaw; // actual index value, accounting for offset

                if (!vertexIsFromSrcMesh)
                {
                    imVertexIdx = (imVertexIdxRaw - (uint32_t)srcMesh.V.rows()); // account for offset
                }

                const InputMesh *inputMeshPtr = &srcMesh; // assume origin face is from source mesh

                if (!faceIsFromSrcMesh)
                {
                    inputMeshPtr = &cutMesh;
                }

                // the face on which the current cc face came from
                const Eigen::Vector3i &imFace = inputMeshPtr->F.row(imFaceIdx);

                Eigen::Vector2d texCoord;

                if (vertexIsIntersectionPoint)
                { // texture coords unknown and must be computed

                    // interpolate texture coords from source-mesh values

                    // 1. get the origin face of the current cc face

                    double x(ccVertices[((uint64_t)ccVertexIdx * 3) + 0]);
                    double y(ccVertices[((uint64_t)ccVertexIdx * 3) + 1]);
                    double z(ccVertices[((uint64_t)ccVertexIdx * 3) + 2]);

                    // vertices of the origin face
                    const Eigen::Vector3d &a = inputMeshPtr->V.row(imFace.x());
                    const Eigen::Vector3d &b = inputMeshPtr->V.row(imFace.y());
                    const Eigen::Vector3d &c = inputMeshPtr->V.row(imFace.z());

                    // barycentric coords of our intersection point on the origin face
                    Eigen::MatrixXd P;
                    P.resize(1, 3);
                    P << x, y, z;
                    Eigen::MatrixXd A;
                    A.resize(1, 3);
                    A << a.x(), a.y(), a.z();
                    Eigen::MatrixXd B;
                    B.resize(1, 3);
                    B << b.x(), b.y(), b.z();
                    Eigen::MatrixXd C;
                    C.resize(1, 3);
                    C << c.x(), c.y(), c.z();
                    Eigen::MatrixXd L;

                    igl::barycentric_coordinates(P, A, B, C, L);

                    // compute the texture coordinates of our intersection point by interpolation
                    // --------------------------------------------------------------------------

                    // indices of the texture coords that are used by "imFaceIdx"
                    const Eigen::VectorXi &imFaceUVIndices = inputMeshPtr->UV_F.row(imFaceIdx);

                    // texture coordinates of vertices of origin face
                    const Eigen::Vector2d &TCa = inputMeshPtr->UV_V.row(imFaceUVIndices(0));
                    const Eigen::Vector2d &TCb = inputMeshPtr->UV_V.row(imFaceUVIndices(1));
                    const Eigen::Vector2d &TCc = inputMeshPtr->UV_V.row(imFaceUVIndices(2));
                    const Eigen::Vector3d &baryCoords = L.row(0);

                    // interpolate using barycentric coords
                    texCoord = (TCa * baryCoords.x()) + (TCb * baryCoords.y()) + (TCc * baryCoords.z());
                }
                else
                { // texture coords are known must be inferred from input mesh

                    int faceVertexOffset = -1;
                    // for each vertex index in face
                    for (int p = 0; p < (int)imFace.rows(); ++p)
                    {
                        if ((int)imFace(p) == (int)imVertexIdx)
                        {
                            faceVertexOffset = p;
                            break;
                        }
                    }

                    my_assert(faceVertexOffset != -1);

                    int texCoordsIdx = inputMeshPtr->UV_F.row(imFaceIdx)(faceVertexOffset);
                    texCoord = inputMeshPtr->UV_V.row(texCoordsIdx);
                }

                int texCoordIndex = -1;

                std::vector<Eigen::Vector2d>::const_iterator fiter = std::find_if(
                    ccTexCoords.cbegin(), ccTexCoords.cend(),
                    [&](const Eigen::Vector2d &e)
                    { return compare(e.x(), texCoord.x()) && compare(e.y(), texCoord.y()); });

                if (fiter != ccTexCoords.cend())
                {
                    texCoordIndex = (int)std::distance(ccTexCoords.cbegin(), fiter);
                }

                if (texCoordIndex == -1)
                { // tex coord not yet stored for CC vertex in face
                    texCoordIndex = (int)ccTexCoords.size();
                    ccTexCoords.push_back(texCoord);
                }

                ccFaceVertexTexCoordIndices.push_back(texCoordIndex);
            } // for (int v = 0; v < faceSize; ++v) {

            faceVertexOffsetBase += faceSize;
        } // for (int f = 0; f < ccFaceCount; ++f) {

        // save cc mesh to .obj file
        // -------------------------

        // write vertices
        for (int k = 0; k < (int)ccVertexCount; ++k)
        {
            double x = ccVertices[(uint64_t)k * 3 + 0];
            double y = ccVertices[(uint64_t)k * 3 + 1];
            double z = ccVertices[(uint64_t)k * 3 + 2];
            file << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << "v " << x << " " << y << " " << z << std::endl;
        }

        // write tex coords (including duplicates i.e. per face texture coords)

        for (int k = 0; k < (int)ccTexCoords.size(); ++k)
        {
            Eigen::Vector2d uv = ccTexCoords[k];                                                                                        //faceTexCoords[j];
            file << "vt " << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << uv.x() << " " << uv.y() << std::endl; // texcoords have same index as positions
        }

        // write faces
        faceVertexOffsetBase = 0;
        for (int k = 0; k < (int)ccFaceCount; ++k)
        {
            int faceSize = faceSizes.at(k);

            file << "f ";
            for (int j = 0; j < (int)faceSize; ++j)
            {
                const int idx = faceVertexOffsetBase + j;
                const int ccVertexIdx = ccFaceIndices.at(static_cast<size_t>(idx));
                file << (ccVertexIdx + 1) << "/" << ccFaceVertexTexCoordIndices[idx] + 1 << " ";
            }
            file << std::endl;

            faceVertexOffsetBase += faceSize;
        }
    }

    // free connected component data
    // --------------------------------
    err = mcReleaseConnectedComponents(context, 0, NULL);

    my_assert(err == MC_NO_ERROR);

    //  destroy context
    // ------------------
    err = mcReleaseContext(context);

    my_assert(err == MC_NO_ERROR);

    return 0;
}
