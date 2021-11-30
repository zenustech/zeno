/*
This tutorial show how to propagate per-vertex normals (smooth shading) from input meshes and onto the output
connected components after cutting
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
    // variables for reading .obj file data with libigl
    std::vector<std::vector<double>> V, TC, N;
    std::vector<std::vector<int>> F, FTC, FN;
    std::vector<std::tuple<std::string, unsigned, unsigned>> FM;

    // variables for mesh data in a format suited for MCUT
    std::string fpath;                      // path to mesh file
    std::vector<uint32_t> faceSizesArray;   // vertices per face
    std::vector<uint32_t> faceIndicesArray; // face indices
    std::vector<double> vertexCoordsArray;  // vertex coords
};

int main()
{
    // 1. load meshes.
    // -----------------
    InputMesh srcMesh;

    // read file
    srcMesh.fpath = DATA_DIR "/cube.obj";
    bool srcMeshLoaded = igl::readOBJ(srcMesh.fpath, srcMesh.V, srcMesh.TC, srcMesh.N, srcMesh.F, srcMesh.FTC, srcMesh.FN);

    if (!srcMeshLoaded)
    {
        std::fprintf(stderr, "error: could not load source mesh --> %s\n", srcMesh.fpath.c_str());
        std::exit(1);
    }

    // copy vertices
    for (int i = 0; i < (int)srcMesh.V.size(); ++i)
    {
        const std::vector<double> &v = srcMesh.V[i];
        my_assert(v.size() == 3);
        srcMesh.vertexCoordsArray.push_back(v[0]);
        srcMesh.vertexCoordsArray.push_back(v[1]);
        srcMesh.vertexCoordsArray.push_back(v[2]);
    }

    // copy faces
    for (int i = 0; i < (int)srcMesh.F.size(); ++i)
    {
        const std::vector<int> &f = srcMesh.F[i];
        my_assert(f.size() == 3); // we assume triangle meshes for simplicity
        for (int j = 0; j < (int)f.size(); ++j)
        {
            srcMesh.faceIndicesArray.push_back(f[j]);
        }

        srcMesh.faceSizesArray.push_back((uint32_t)f.size());
    }

    printf("source mesh:\n\tvertices=%d\n\tfaces=%d\n", (int)srcMesh.V.size(), (int)srcMesh.F.size());

    InputMesh cutMesh;

    // read file
    cutMesh.fpath = DATA_DIR "/plane.obj";
    bool cutMeshLoaded = igl::readOBJ(cutMesh.fpath, cutMesh.V, cutMesh.TC, cutMesh.N, cutMesh.F, cutMesh.FTC, cutMesh.FN);

    if (!cutMeshLoaded)
    {
        std::fprintf(stderr, "error: could not load source mesh --> %s\n", cutMesh.fpath.c_str());
        std::exit(1);
    }

    // copy vertices
    for (int i = 0; i < (int)cutMesh.V.size(); ++i)
    {
        const std::vector<double> &v = cutMesh.V[i];
        my_assert((int)v.size() == 3);
        cutMesh.vertexCoordsArray.push_back(v[0]);
        cutMesh.vertexCoordsArray.push_back(v[1]);
        cutMesh.vertexCoordsArray.push_back(v[2]);
    }

    // copy faces
    for (int i = 0; i < (int)cutMesh.F.size(); ++i)
    {
        const std::vector<int> &f = cutMesh.F[i];
        my_assert(f.size() == 3);
        for (int j = 0; j < (int)f.size(); ++j)
        {
            cutMesh.faceIndicesArray.push_back(f[j]);
        }

        cutMesh.faceSizesArray.push_back((uint32_t)f.size());
    }

    printf("cut mesh:\n\tvertices=%d\n\tfaces=%d\n", (int)cutMesh.V.size(), (int)cutMesh.F.size());

    // 2. create a context
    // -------------------
    McContext context = MC_NULL_HANDLE;
    McResult err = mcCreateContext(&context, MC_DEBUG);

    my_assert(err == MC_NO_ERROR);

    // 3. do the cutting
    // -----------------
    err = mcDispatch(
        context,
        MC_DISPATCH_VERTEX_ARRAY_DOUBLE | MC_DISPATCH_INCLUDE_VERTEX_MAP | MC_DISPATCH_INCLUDE_FACE_MAP, // We need vertex and face maps to propagate normals
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
    // ----------------------------------------------------------------
    uint32_t numConnComps;

    err = mcGetConnectedComponents(context, MC_CONNECTED_COMPONENT_TYPE_ALL, 0, NULL, &numConnComps);
    my_assert(err == MC_NO_ERROR);

    printf("connected components: %d\n", (int)numConnComps);

    std::vector<McConnectedComponent> connComps(numConnComps, 0);
    err = mcGetConnectedComponents(context, MC_CONNECTED_COMPONENT_TYPE_ALL, (uint32_t)connComps.size(), connComps.data(), NULL);
    my_assert(err == MC_NO_ERROR);

    //  query the data of each connected component from MCUT
    // -------------------------------------------------------

    for (int ci = 0; ci < (int)connComps.size(); ++ci)
    {
        McConnectedComponent connComp = connComps[ci]; // connected component id

        //  query the ccVertices
        // ----------------------

        uint64_t numBytes = 0;
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
        err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_FACE_SIZE, numBytes, faceSizes.data(), NULL);
        my_assert(err == MC_NO_ERROR);

        //  query the vertex map
        // ------------------------

        err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_VERTEX_MAP, 0, NULL, &numBytes);
        my_assert(err == MC_NO_ERROR);
        std::vector<uint32_t> ccVertexMap;
        ccVertexMap.resize(numBytes / sizeof(uint32_t));
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

        //  resolve fragment name
        // -------------------------

        // Here we create a name the connected component based on its properties

        // get type
        McConnectedComponentType ccType;
        err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_TYPE, sizeof(McConnectedComponentType), &ccType, NULL);
        my_assert(err == MC_NO_ERROR);

        std::string name;
        McFragmentLocation fragmentLocation = (McFragmentLocation)0;
        McPatchLocation patchLocation = (McPatchLocation)0;
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

            err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_PATCH_LOCATION, sizeof(McPatchLocation), &patchLocation, NULL);
            my_assert(err == MC_NO_ERROR);
            name += patchLocation == MC_PATCH_LOCATION_INSIDE ? ".inside" : (patchLocation == MC_PATCH_LOCATION_OUTSIDE ? ".outside" : ".undefined");

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
            }
            else if (ccType == MC_CONNECTED_COMPONENT_TYPE_INPUT)
            {
                McInputOrigin ccOrig = (McInputOrigin)0;
                err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_ORIGIN, sizeof(McInputOrigin), &ccOrig, NULL);
                my_assert(err == MC_NO_ERROR);
                ccIsFromSrcMesh = (ccOrig == McInputOrigin::MC_INPUT_ORIGIN_SRCMESH);
            }
            name += ccIsFromSrcMesh ? ".sm" : ".cm";
        }

        int faceVertexOffsetBase = 0;

        std::vector<Eigen::Vector3d> ccVertexNormals(ccVertexCount, Eigen::Vector3d(0., 0., 0.));
        // intersection points do not have a normal value that can be copied (inferred) from an input
        // mesh, it has to be computed by interpolating normals on the origin face in the input mesh.
        // We keep a reference count to compute averaged normal per intersection point.
        std::map<int, int> ccSeamVertexToRefCount;
        std::vector<uint32_t> ccFaceVertexNormalIndices;
        std::vector<int> ccReversedFaces;

        // for each face in CC
        for (int f = 0; f < (int)ccFaceCount; ++f)
        {

            // input mesh face index (which may be offsetted!)
            const uint32_t imFaceIdxRaw = ccFaceMap.at(f); // source- or cut-mesh
            // input mesh face index (actual index value, accounting for offset)
            uint32_t imFaceIdx = imFaceIdxRaw;
            bool faceIsFromSrcMesh = (imFaceIdxRaw < (std::uint32_t)srcMesh.F.size());
            bool flipNormalsOnFace = false;

            if (!faceIsFromSrcMesh)
            {
                imFaceIdx = imFaceIdxRaw - (std::uint32_t)srcMesh.F.size(); // accounting for offset
                flipNormalsOnFace = (isFragment && fragmentLocation == MC_FRAGMENT_LOCATION_ABOVE);
            }

            int faceSize = (int)faceSizes.at(f);

            // for each vertex in face
            for (int v = 0; v < faceSize; ++v)
            {

                const int ccVertexIdx = ccFaceIndices[(uint64_t)faceVertexOffsetBase + v];
                // input mesh (source mesh or cut mesh) vertex index (which may be offsetted)
                const uint32_t imVertexIdxRaw = ccVertexMap.at(ccVertexIdx);
                bool vertexIsFromSrcMesh = (imVertexIdxRaw < (std::uint32_t)srcMesh.V.size());
                const bool isSeamVertex = (imVertexIdxRaw == MC_UNDEFINED_VALUE);
                uint32_t imVertexIdx = imVertexIdxRaw; // actual index value, accounting for offset

                if (!vertexIsFromSrcMesh)
                {
                    imVertexIdx = (imVertexIdxRaw - (std::uint32_t)srcMesh.V.size()); // account for offset
                }

                const InputMesh *inputMeshPtr = &srcMesh; // assume origin face is from source mesh

                if (!faceIsFromSrcMesh)
                {
                    inputMeshPtr = &cutMesh;
                }

                // the face on which the current cc face came from
                const std::vector<int> &imFace = inputMeshPtr->F[imFaceIdx];

                if (isSeamVertex)
                { // normal is unknown and must be computed

                    // interpolate texture coords from source-mesh values

                    // 1. get the origin face of the current cc face

                    double x(ccVertices[((uint64_t)ccVertexIdx * 3u) + 0u]);
                    double y(ccVertices[((uint64_t)ccVertexIdx * 3u) + 1u]);
                    double z(ccVertices[((uint64_t)ccVertexIdx * 3u) + 2u]);

                    // vertices of the origin face
                    const std::vector<double> &a = inputMeshPtr->V[imFace[0]];
                    const std::vector<double> &b = inputMeshPtr->V[imFace[1]];
                    const std::vector<double> &c = inputMeshPtr->V[imFace[2]];

                    // barycentric coords of our intersection point on the origin face
                    Eigen::MatrixXd P;
                    P.resize(1, 3);
                    P << x, y, z;
                    Eigen::MatrixXd A;
                    A.resize(1, 3);
                    A << a[0], a[1], a[2];
                    Eigen::MatrixXd B;
                    B.resize(1, 3);
                    B << b[0], b[1], b[2];
                    Eigen::MatrixXd C;
                    C.resize(1, 3);
                    C << c[0], c[1], c[2];
                    Eigen::MatrixXd L;

                    igl::barycentric_coordinates(P, A, B, C, L);

                    // compute the normal of our intersection point by interpolation
                    // -------------------------------------------------------------

                    // indices of the normals that are used by "imFaceIdx"
                    const std::vector<int> &imFaceNormalIndices = inputMeshPtr->FN[imFaceIdx];
                    my_assert(imFaceNormalIndices.size() == 3);

                    // normals of vertices in origin face
                    const std::vector<double> &Na_ = inputMeshPtr->N[imFaceNormalIndices[0]];
                    const std::vector<double> &Nb_ = inputMeshPtr->N[imFaceNormalIndices[1]];
                    const std::vector<double> &Nc_ = inputMeshPtr->N[imFaceNormalIndices[2]];

                    const Eigen::Vector3d Na(Na_[0], Na_[1], Na_[2]);
                    const Eigen::Vector3d Nb(Nb_[0], Nb_[1], Nb_[2]);
                    const Eigen::Vector3d Nc(Nc_[0], Nc_[1], Nc_[2]);
                    const Eigen::Vector3d baryCoords = L.row(0);

                    // interpolate using barycentric coords
                    Eigen::Vector3d normal = (Na * baryCoords.x()) + (Nb * baryCoords.y()) + (Nc * baryCoords.z()) * (flipNormalsOnFace ? -1.0 : 1.0);

                    ccVertexNormals[ccVertexIdx] += normal;

                    if (ccSeamVertexToRefCount.find(ccVertexIdx) == ccSeamVertexToRefCount.cend())
                    {
                        ccSeamVertexToRefCount[ccVertexIdx] = 1;
                    }
                    else
                    {
                        ccSeamVertexToRefCount[ccVertexIdx] += 1;
                    }
                }
                else
                { // normal must be inferred from input mesh

                    if (ccVertexNormals[ccVertexIdx].norm() == 0)
                    {
                        int faceVertexOffset = -1;
                        // for each vertex index in face
                        for (int i = 0; i < (int)imFace.size(); ++i)
                        {
                            if ((int)imFace[i] == (int)imVertexIdx)
                            {
                                faceVertexOffset = i;
                                break;
                            }
                        }

                        my_assert(faceVertexOffset != -1);

                        int imNormalIdx = inputMeshPtr->FN[imFaceIdx][faceVertexOffset];
                        const std::vector<double> &n = inputMeshPtr->N[imNormalIdx];
                        my_assert(n.size() == 3);
                        Eigen::Vector3d normal = Eigen::Vector3d(n[0], n[1], n[2]) * (flipNormalsOnFace ? -1.0 : 1.0);

                        ccVertexNormals[ccVertexIdx] = normal;
                    }
                }
            } // for (int v = 0; v < faceSize; ++v) {

            faceVertexOffsetBase += faceSize;
        } // for (int f = 0; f < ccFaceCount; ++f) {

        for (std::map<int, int>::const_iterator it = ccSeamVertexToRefCount.cbegin(); it != ccSeamVertexToRefCount.cend(); ++it)
        {
            const int ccSeamVertexIndex = it->first;
            const int refCount = it->second;
            my_assert(refCount >= 1);
            ccVertexNormals[ccSeamVertexIndex] /= (double)refCount; // average
        }

        // save cc mesh to .obj file
        // -------------------------

        char fnameBuf[64];
        sprintf(fnameBuf, ("OUT_" + name + ".obj").c_str(), ci);
        std::string fpath(DATA_DIR "/" + std::string(fnameBuf));

        printf("write file: %s\n", fpath.c_str());

        std::ofstream file(fpath);

        // write vertices and normals

        for (int i = 0; i < (int)ccVertexCount; ++i)
        {
            double x = ccVertices[(uint64_t)i * 3 + 0];
            double y = ccVertices[(uint64_t)i * 3 + 1];
            double z = ccVertices[(uint64_t)i * 3 + 2];
            file << "v " << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << x << " " << y << " " << z << std::endl;

            Eigen::Vector3d n = ccVertexNormals[i];
            file << "vn " << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << n.x() << " " << n.y() << " " << n.z() << std::endl;
        }

        // write faces (with normal indices)

        faceVertexOffsetBase = 0;
        for (int i = 0; i < (int)ccFaceCount; ++i)
        {
            int faceSize = faceSizes.at(i);

            file << "f ";
            for (int j = 0; j < faceSize; ++j)
            {
                const int idx = faceVertexOffsetBase + j;
                const int ccVertexIdx = ccFaceIndices[idx];
                file << (ccVertexIdx + 1) << "//" << (ccVertexIdx + 1) << " ";
            }
            file << std::endl;

            faceVertexOffsetBase += faceSize;
        }
    }

    //  free connected component data
    // --------------------------------
    err = mcReleaseConnectedComponents(context, 0, NULL);

    my_assert(err == MC_NO_ERROR);

    //  destroy context
    // ------------------
    err = mcReleaseContext(context);

    my_assert(err == MC_NO_ERROR);

    return 0;
}
