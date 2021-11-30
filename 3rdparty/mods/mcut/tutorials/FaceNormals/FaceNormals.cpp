/*
This tutorial shows how to propagate per-face normals (flat shading) from input meshes and onto the output 
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

#define my_assert(cond) if(!(cond)){fprintf(stderr, "MCUT error: %s\n", #cond );std::exit(1);}

struct InputMesh {
    // variables for reading .obj file data with libigl
    std::vector<std::vector<double>> V, TC, N;
    std::vector<std::vector<int>> F, FTC, FN;
    std::vector<std::tuple<std::string, unsigned, unsigned>> FM;

    // variables for mesh data in a format suited for MCUT
    std::string fpath; // path to mesh file
    std::vector<uint32_t> faceSizesArray; // vertices per face
    std::vector<uint32_t> faceIndicesArray; // face indices
    std::vector<double> vertexCoordsArray; // vertex coords
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
    srcMesh.fpath = DATA_DIR "/cube.obj";
    bool srcMeshLoaded = igl::readOBJ(srcMesh.fpath, srcMesh.V, srcMesh.TC, srcMesh.N, srcMesh.F, srcMesh.FTC, srcMesh.FN);

    if (!srcMeshLoaded) {
        std::fprintf(stderr, "error: could not load source mesh --> %s\n", srcMesh.fpath.c_str());
        std::exit(1);
    }

    // copy vertices
    for (int i = 0; i < (int)srcMesh.V.size(); ++i) {
        const std::vector<double>& v = srcMesh.V[i];
        my_assert(v.size() == 3);
        srcMesh.vertexCoordsArray.push_back(v[0]);
        srcMesh.vertexCoordsArray.push_back(v[1]);
        srcMesh.vertexCoordsArray.push_back(v[2]);
    }

    // copy faces
    for (int i = 0; i < (int)srcMesh.F.size(); ++i) {
        const std::vector<int>& f = srcMesh.F[i];
        my_assert(f.size() == 3); // we assume triangle meshes for simplicity
        for (int j = 0; j < (int)f.size(); ++j) {
            srcMesh.faceIndicesArray.push_back(f[j]);
        }

        srcMesh.faceSizesArray.push_back((uint32_t)f.size());
    }

    printf("source mesh:\n\tvertices=%d\n\tfaces=%d\n", (int)srcMesh.V.size(), (int)srcMesh.F.size());

    InputMesh cutMesh;

    // read file
    cutMesh.fpath = DATA_DIR "/plane.obj";
    bool cutMeshLoaded = igl::readOBJ(cutMesh.fpath, cutMesh.V, cutMesh.TC, cutMesh.N, cutMesh.F, cutMesh.FTC, cutMesh.FN);

    if (!cutMeshLoaded) {
        std::fprintf(stderr, "error: could not load source mesh --> %s\n", cutMesh.fpath.c_str());
        std::exit(1);
    }

    // copy vertices
    for (int i = 0; i < (int)cutMesh.V.size(); ++i) {
        const std::vector<double>& v = cutMesh.V[i];
        my_assert(v.size() == 3);
        cutMesh.vertexCoordsArray.push_back(v[0]);
        cutMesh.vertexCoordsArray.push_back(v[1]);
        cutMesh.vertexCoordsArray.push_back(v[2]);
    }

    // copy faces
    for (int i = 0; i < (int)cutMesh.F.size(); ++i) {
        const std::vector<int>& f = cutMesh.F[i];
        my_assert(f.size() == 3);
        for (int j = 0; j < (int)f.size(); ++j) {
            cutMesh.faceIndicesArray.push_back(f[j]);
        }

        cutMesh.faceSizesArray.push_back((uint32_t)f.size());
    }

    printf("cut mesh:\n\tvertices=%d\n\tfaces=%d\n", (int)cutMesh.V.size(), (int)cutMesh.F.size());

    //  create an MCUT context
    // -------------------------
    McContext context = MC_NULL_HANDLE;
    McResult err = mcCreateContext(&context, MC_DEBUG);

    my_assert(err == MC_NO_ERROR);

    //  do the cutting
    // -----------------
    err = mcDispatch(
        context,
        MC_DISPATCH_VERTEX_ARRAY_DOUBLE | MC_DISPATCH_INCLUDE_VERTEX_MAP | MC_DISPATCH_INCLUDE_FACE_MAP, // We need vertex and face maps to propagate normals
        // source mesh
        reinterpret_cast<const void*>(srcMesh.vertexCoordsArray.data()),
        reinterpret_cast<const uint32_t*>(srcMesh.faceIndicesArray.data()),
        srcMesh.faceSizesArray.data(),
        static_cast<uint32_t>(srcMesh.vertexCoordsArray.size() / 3),
        static_cast<uint32_t>(srcMesh.faceSizesArray.size()),
        // cut mesh
        reinterpret_cast<const void*>(cutMesh.vertexCoordsArray.data()),
        cutMesh.faceIndicesArray.data(),
        cutMesh.faceSizesArray.data(),
        static_cast<uint32_t>(cutMesh.vertexCoordsArray.size() / 3),
        static_cast<uint32_t>(cutMesh.faceSizesArray.size()));

    my_assert(err == MC_NO_ERROR);

    //  query the number of available connected component (all types)
    // ----------------------------------------------------------------

    uint32_t numConnectedComponents;

    err = mcGetConnectedComponents(context, MC_CONNECTED_COMPONENT_TYPE_ALL, 0, NULL, &numConnectedComponents);
    my_assert(err == MC_NO_ERROR);

    printf("connected components: %d\n", (int)numConnectedComponents);

    if (numConnectedComponents == 0) {
        fprintf(stdout, "no connected components found\n");
        exit(0);
    }

    std::vector<McConnectedComponent> connectedComponents(numConnectedComponents, 0);
    err = mcGetConnectedComponents(context, MC_CONNECTED_COMPONENT_TYPE_ALL, (uint32_t)connectedComponents.size(), connectedComponents.data(), NULL);
    my_assert(err == MC_NO_ERROR);

    //  query the data of each connected component from MCUT
    // -------------------------------------------------------

    for (int i = 0; i < (int)connectedComponents.size(); ++i) {
        McConnectedComponent connComp = connectedComponents[i]; // connected compoene

        //  query the vertices
        // ----------------------

        uint64_t numBytes = 0;
        err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_VERTEX_DOUBLE, 0, NULL, &numBytes);
        my_assert(err == MC_NO_ERROR);

        uint32_t ccVertexCount = (uint32_t)(numBytes / (sizeof(double) * 3));
        std::vector<double> ccVertices((size_t)ccVertexCount * 3u, 0.0);
        err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_VERTEX_DOUBLE, numBytes, (void*)ccVertices.data(), NULL);
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
        std::vector<uint32_t> ccVertexMap(numBytes / sizeof(uint32_t), 0);
        err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_VERTEX_MAP, numBytes, ccVertexMap.data(), NULL);
        my_assert(err == MC_NO_ERROR);

        //  query the face map
        // -----------------------
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

        if (ccType == MC_CONNECTED_COMPONENT_TYPE_SEAM) {
            name += "seam";
        } else if (ccType == MC_CONNECTED_COMPONENT_TYPE_INPUT) {
            name += "input";
        } else {
            isFragment = (ccType == MC_CONNECTED_COMPONENT_TYPE_FRAGMENT);
            name += isFragment ? "frag" : "patch";

            err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_PATCH_LOCATION, sizeof(McPatchLocation), &patchLocation, NULL);
            my_assert(err == MC_NO_ERROR);
            name += patchLocation == MC_PATCH_LOCATION_INSIDE ? ".inside" : (patchLocation == MC_PATCH_LOCATION_OUTSIDE ? ".outside" : ".undefined");

            if (isFragment) {

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
        if (!ccIsFromSrcMesh) {
            if (ccType == MC_CONNECTED_COMPONENT_TYPE_SEAM) {
                // get origin
                McSeamOrigin ccOrig = (McSeamOrigin)0;
                err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_ORIGIN, sizeof(McSeamOrigin), &ccOrig, NULL);
                my_assert(err == MC_NO_ERROR);

                ccIsFromSrcMesh = (ccOrig == McSeamOrigin::MC_SEAM_ORIGIN_SRCMESH);
                name += ccIsFromSrcMesh ? ".sm" : ".cm";
            } else if (ccType == MC_CONNECTED_COMPONENT_TYPE_INPUT) {
                McInputOrigin ccOrig = (McInputOrigin)0;
                err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_ORIGIN, sizeof(McInputOrigin), &ccOrig, NULL);
                my_assert(err == MC_NO_ERROR);
                ccIsFromSrcMesh = (ccOrig == McInputOrigin::MC_INPUT_ORIGIN_SRCMESH);
                name += ccIsFromSrcMesh ? ".sm" : ".cm";
            }
        }

        int faceVertexOffsetBase = 0;

        std::vector<Eigen::Vector3d> ccNormals; // our normals (what we want)

        // CC-vertex-index-to-normal-indices.
        // Its possible to map to more than one normal since such coordinates are specified to per-face.
        std::map<int, std::vector<int>> ccVertexIndexToNormalIndices;
        std::vector<int> ccFaceVertexNormalIndices; // normal indices reference by each face

        // for each face in cc
        for (int f = 0; f < (int)ccFaceCount; ++f) {

            // get the [origin] input-mesh face index (Note: this index may be offsetted
            // to distinguish between source-mesh and cut-mesh faces).
            const uint32_t imFaceIdxRaw = ccFaceMap.at(f); // source- or cut-mesh face index (we don't know yet)
            // ** This is how we infer which mapped indices belong to the source-mesh or the cut-mesh.
            const bool faceIsFromSrcMesh = (imFaceIdxRaw < (uint32_t)srcMesh.F.size());
            bool flipNormalsOnFace = false;
            // Now compute the actual input mesh face index (accounting for offset)
            uint32_t imFaceIdx = imFaceIdxRaw;

            if (!faceIsFromSrcMesh) { // if the current face is from the cut-mesh
                imFaceIdx = (imFaceIdxRaw - (uint32_t)srcMesh.F.size()); // accounting for offset
                // check if we need to flip normals on the face
                flipNormalsOnFace = (isFragment && fragmentLocation == MC_FRAGMENT_LOCATION_ABOVE);
            }

            int faceSize = faceSizes.at(f); /// number of vertices in face

            // for each vertex in face
            for (int v = 0; v < faceSize; ++v) {

                const int ccVertexIdx = ccFaceIndices.at((uint64_t)faceVertexOffsetBase + v);
                const uint32_t imVertexIdxRaw = ccVertexMap.at(ccVertexIdx);
                bool vertexIsFromSrcMesh = (imVertexIdxRaw < srcMesh.V.size());
                const bool isSeamVertex = (imVertexIdxRaw == MC_UNDEFINED_VALUE); // i.e. a vertex along the cut-path (an intersection point)
                uint32_t imVertexIdx = imVertexIdxRaw; // actual index value, accounting for offset

                if (!vertexIsFromSrcMesh) {
                    imVertexIdx = (imVertexIdxRaw - (std::uint32_t)srcMesh.V.size()); // account for offset
                }

                const InputMesh* inputMeshPtr = faceIsFromSrcMesh ? &srcMesh : &cutMesh;

                // the face from which the current cc face came ("birth face")
                const std::vector<int>& imFace = inputMeshPtr->F[imFaceIdx];

                Eigen::Vector3d normal; // the normal of the current vertex

                if (isSeamVertex) { // normal completely unknown and must be computed

                    // interpolate texture coords from input-mesh values
                    // --------------------------------------------------

                    // coordinates of current point
                    double x(ccVertices[((uint64_t)ccVertexIdx * 3u) + 0u]);
                    double y(ccVertices[((uint64_t)ccVertexIdx * 3u) + 1u]);
                    double z(ccVertices[((uint64_t)ccVertexIdx * 3u) + 2u]);

                    // vertices of the origin face (i.e. the face from which the current face came from).
                    // NOTE: we have assumed triangulated input meshes for simplicity. Otherwise, interpolation
                    // will be more complex, which is unnecessary for now.
                    const std::vector<double>& a = inputMeshPtr->V[imFace[0]];
                    const std::vector<double>& b = inputMeshPtr->V[imFace[1]];
                    const std::vector<double>& c = inputMeshPtr->V[imFace[2]];

                    // compute the barycentric coords of our seam vertex on the origin face
                    Eigen::MatrixXd P; // our vertex
                    P.resize(1, 3);
                    P << x, y, z;
                    // the origin face vertices added into Eigen matrices for libIGL
                    Eigen::MatrixXd A;
                    A.resize(1, 3);
                    A << a[0], a[1], a[2];
                    Eigen::MatrixXd B;
                    B.resize(1, 3);
                    B << b[0], b[1], b[2];
                    Eigen::MatrixXd C;
                    C.resize(1, 3);
                    C << c[0], c[1], c[2];

                    // our barycentric coords
                    Eigen::MatrixXd L;

                    igl::barycentric_coordinates(P, A, B, C, L);

                    // compute the normal of our vertex by interpolation
                    // -------------------------------------------------

                    // indices of the normals that are used by the origin face "imFaceIdx"
                    const std::vector<int>& imFaceNormalIndices = inputMeshPtr->FN[imFaceIdx];
                    my_assert(imFaceNormalIndices.size() == 3);

                    // normals of vertices in the origin face
                    const std::vector<double>& Na_ = inputMeshPtr->N[imFaceNormalIndices[0]];
                    const std::vector<double>& Nb_ = inputMeshPtr->N[imFaceNormalIndices[1]];
                    const std::vector<double>& Nc_ = inputMeshPtr->N[imFaceNormalIndices[2]];
                    // simple conversion to Eigen's Vector3d type
                    const Eigen::Vector3d Na(Na_[0], Na_[1], Na_[2]);
                    const Eigen::Vector3d Nb(Nb_[0], Nb_[1], Nb_[2]);
                    const Eigen::Vector3d Nc(Nc_[0], Nc_[1], Nc_[2]);
                    const Eigen::Vector3d baryCoords = L.row(0);

                    // interpolate using barycentric coords
                    normal = (Na * baryCoords.x()) + (Nb * baryCoords.y()) + (Nc * baryCoords.z());

                    // NOTE: if all three vertices have the same normal (index) then there is no need for interpolation
                    // we'd just copy that value from the respective input mesh.

                } else { // the normal is known must be inferred from input mesh

                    // the index of the mapped-to index in the input mesh
                    int imFaceVertexOffset = -1;
                    for (int p = 0; p < (int)imFace.size(); ++p) {
                        if (imFace[p] == (int)imVertexIdx) {
                            imFaceVertexOffset = p;
                            break;
                        }
                    }

                    my_assert(imFaceVertexOffset != -1);

                    // get the normal index of the vertex
                    int imNormalIdx = inputMeshPtr->FN[imFaceIdx][imFaceVertexOffset];
                    // copy the normal value from the input mesh
                    const std::vector<double>& n = inputMeshPtr->N[imNormalIdx];
                    my_assert(n.size() == 3);
                    normal = Eigen::Vector3d(n[0], n[1], n[2]);
                }

                // When MCUT seal's holes, it uses polygons directly from the cut mesh. These polygons
                // may require to be flipped sometimes when holes are filled (e.g. when a fragment is
                // "above"). Thus, we cannot just copy/interpolate the normal from the origin mesh in
                // such cases, we must also flip (negate) it.
                if (flipNormalsOnFace) {
                    normal *= -1.0;
                }

                // Do some book-keeping to prevent us from duplicate the normals that we write to file.
                int normalIndex = -1;

                // has a normal with the same value already been computed?
                std::vector<Eigen::Vector3d>::const_iterator fiter = std::find_if(
                    ccNormals.cbegin(), ccNormals.cend(),
                    [&](const Eigen::Vector3d& e) { return compare(e.x(), normal.x()) && compare(e.y(), normal.y()) && compare(e.z(), normal.z()); });

                if (fiter != ccNormals.cend()) {
                    normalIndex = (int)std::distance(ccNormals.cbegin(), fiter);
                }

                if (normalIndex == -1) { // normal not yet stored for CC vertex in face
                    normalIndex = (int)ccNormals.size();
                    ccNormals.push_back(normal);
                }

                ccFaceVertexNormalIndices.push_back(normalIndex);
            } // for (int v = 0; v < faceSize; ++v) {

            faceVertexOffsetBase += faceSize;
        } // for (int f = 0; f < ccFaceCount; ++f) {

        // save connected component (mesh) to an .obj file
        // -----------------------------------------------

        char fnameBuf[64];
        sprintf(fnameBuf, ("OUT_" + name + ".obj").c_str(), i);
        std::string fpath(DATA_DIR "/" + std::string(fnameBuf));

        printf("write file: %s\n", fpath.c_str());

        std::ofstream file(fpath);

        // write vertices

        for (int p = 0; p < (int)ccVertexCount; ++p) {
            double x = ccVertices[(uint64_t)p * 3 + 0];
            double y = ccVertices[(uint64_t)p * 3 + 1];
            double z = ccVertices[(uint64_t)p * 3 + 2];
            file << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << "v " << x << " " << y << " " << z << std::endl;
        }

        // write normals

        for (int p = 0; p < (int)ccNormals.size(); ++p) {
            Eigen::Vector3d n = ccNormals[p];
            file << "vn " << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << n.x() << " " << n.y() << " " << n.z() << std::endl;
        }

        // write faces (with normal indices)

        faceVertexOffsetBase = 0;
        for (int k = 0; k < (int)ccFaceCount; ++k) {
            int faceSize = faceSizes.at(k);

            file << "f ";
            for (int j = 0; j < faceSize; ++j) {
                const int idx = faceVertexOffsetBase + j;
                const int ccVertexIdx = ccFaceIndices.at(static_cast<size_t>(idx));
                file << (ccVertexIdx + 1) << "//" << ccFaceVertexNormalIndices[idx] + 1 << " ";
            }
            file << std::endl;

            faceVertexOffsetBase += faceSize;
        }
    }

    // 6. free connected component data
    // --------------------------------
    err = mcReleaseConnectedComponents(context, 0, NULL);

    my_assert(err == MC_NO_ERROR);

    // 7. destroy context
    // ------------------
    err = mcReleaseContext(context);

    my_assert(err == MC_NO_ERROR);

    return 0;
}
