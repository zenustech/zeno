/*
This tutorial shows how to query adjacent faces of any face of a connected component.

The tutorial is presented in the context of merging neighouring faces of a 
connected component that share some property (e.g. an ID tag), where 
this property is _derived_ from origin/birth faces. 

A group of faces that share the property and define a connected patch will be merged 
into a single face. This is useful in situations where e.g. one has to triangulate the 
faces of an input mesh before cutting and then recover the untriangulated faces afterwards.

We assume that all faces to be merged are coplanar.
*/

#include "mcut/mcut.h"

#include <cassert>
#include <fstream>
#include <map>
#include <queue>
#include <stdlib.h>
#include <string>
#include <vector>

// libigl dependencies
#include <Eigen/Core>
#include <igl/read_triangle_mesh.h>

#define my_assert(cond) if(!(cond)){fprintf(stderr, "MCUT error: %s\n", #cond );std::exit(1);}

void writeOBJ(
    const std::string& path,
    const double* ccVertices,
    const int ccVertexCount,
    const uint32_t* ccFaceIndices,
    const uint32_t* faceSizes,
    const uint32_t ccFaceCount);
void readOBJ(const std::string& path, std::vector<double>& V, std::vector<uint32_t>& F, std::vector<uint32_t>& Fsizes);
uint32_t getAdjFacesBaseOffset(const uint32_t faceIdx, const uint32_t* faceAdjFacesSizes);
uint32_t getFaceIndicesBaseOffset(const uint32_t faceIdx, const uint32_t* faceSizes);
void mergeAdjacentMeshFacesByProperty(
    std::vector<uint32_t>& meshFaceIndicesOUT,
    std::vector<uint32_t>& meshFaceSizesOUT,
    const std::vector<uint32_t>& meshFaces,
    const std::vector<uint32_t>& meshFaceSizes,
    const std::vector<uint32_t>& meshFaceAdjFace,
    const std::vector<uint32_t>& meshFaceAdjFaceSizes,
    const std::map<int, std::vector<uint32_t>>& tagToMeshFace,
    const std::map<uint32_t, int>& meshFaceToTag);

int main()
{
    std::vector<double> srcMeshVertices;
    std::vector<uint32_t> srcMeshFaceIndices;
    std::vector<uint32_t> srcMeshFaceSizes;
    readOBJ(DATA_DIR "/triangulatedGrid4x4.obj", srcMeshVertices, srcMeshFaceIndices, srcMeshFaceSizes);

    // A map denoting the adjacent faces of the source mesh that share some property
    // (a tag/number) that we will use to merge adjacent faces.

    // Faces in each group are merged into one face/polygon.
    // Groups which are adjacent and share a tag are also merged.

    std::map<uint32_t, int> srcMeshFaceToTag = {
        // Bottom-right quadrant faces
        // group 0
        { 0, 0 },
        { 1, 0 },
        // group 1
        { 2, 1 },
        { 3, 1 },
        // group 2
        { 8, 2 },
        { 9, 2 },
        // group 3
        { 10, 3 },
        { 11, 3 },

        // Top-left quadrant faces

        // group 4
        // For the sake of demonstration, we use the same tag ("0") for group 4 and group 0.
        // This is fine because the triangles/faces of group 0 are not adjacent with any
        // triangles/faces in group 4, which mean group 0 and 4 result in two separate faces
        // after merging.
        { 20, 0 },
        { 21, 0 },
        { 22, 0 },
        { 23, 0 },
        { 28, 0 },
        { 29, 0 },
        { 30, 0 },
        { 31, 0 },

        // Top-right quadrant

        // group 5 (note: this group is merged with group-2)
        { 16, 2 },
        // group 6
        { 17, 0xBEEF }
    };

    std::map<int, std::vector<uint32_t>> tagToSrcMeshFaces;
    for (std::map<uint32_t, int>::const_iterator i = srcMeshFaceToTag.cbegin(); i != srcMeshFaceToTag.cend(); ++i) {
        tagToSrcMeshFaces[i->second].push_back(i->first);
    }

    std::vector<double> cutMeshVertices;
    std::vector<uint32_t> cutMeshFaceIndices;
    std::vector<uint32_t> cutMeshFaceSizes;
    readOBJ(DATA_DIR "/quad.obj", cutMeshVertices, cutMeshFaceIndices, cutMeshFaceSizes);

    // create a context
    // -------------------
    McContext context = MC_NULL_HANDLE;
    McResult err = mcCreateContext(&context, MC_NULL_HANDLE);
    my_assert(err == MC_NO_ERROR);

    //  do the cutting (boolean ops)
    // -------------------------------

    err = mcDispatch(
        context,
        MC_DISPATCH_VERTEX_ARRAY_DOUBLE | MC_DISPATCH_INCLUDE_FACE_MAP,
        // source mesh
        reinterpret_cast<const void*>(srcMeshVertices.data()),
        reinterpret_cast<const uint32_t*>(srcMeshFaceIndices.data()),
        srcMeshFaceSizes.data(),
        static_cast<uint32_t>(srcMeshVertices.size() / 3),
        static_cast<uint32_t>(srcMeshFaceSizes.size()),
        // cut mesh
        reinterpret_cast<const void*>(cutMeshVertices.data()),
        reinterpret_cast<const uint32_t*>(cutMeshFaceIndices.data()),
        cutMeshFaceSizes.data(),
        static_cast<uint32_t>(cutMeshVertices.size() / 3),
        static_cast<uint32_t>(cutMeshFaceSizes.size()));

    my_assert(err == MC_NO_ERROR);

    // query the number of available connected component
    // --------------------------------------------------
    uint32_t numConnComps;
    err = mcGetConnectedComponents(context, MC_CONNECTED_COMPONENT_TYPE_ALL, 0, NULL, &numConnComps);
    my_assert(err == MC_NO_ERROR);

    printf("connected components: %d\n", (int)numConnComps);

    if (numConnComps == 0) {
        fprintf(stdout, "no connected components found\n");
        exit(0);
    }

    my_assert(numConnComps > 0);

    std::vector<McConnectedComponent> connectedComponents(numConnComps, MC_NULL_HANDLE);
    connectedComponents.resize(numConnComps);
    err = mcGetConnectedComponents(context, MC_CONNECTED_COMPONENT_TYPE_ALL, (uint32_t)connectedComponents.size(), connectedComponents.data(), NULL);

    my_assert(err == MC_NO_ERROR);

    // query the data of each connected component from MCUT
    // -------------------------------------------------------

    for (int c = 0; c < (int)numConnComps; ++c) {
        McConnectedComponent connComp = connectedComponents[c];

        McConnectedComponentType type;
        err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_TYPE, sizeof(McConnectedComponentType), &type, NULL);
        my_assert(err == MC_NO_ERROR);

        if (!(type == MC_CONNECTED_COMPONENT_TYPE_INPUT || type == MC_CONNECTED_COMPONENT_TYPE_FRAGMENT)) {
            // we only care about the input source mesh, and the "fragment" connected components
            continue;
        }

        if (type == MC_CONNECTED_COMPONENT_TYPE_INPUT) {
            McInputOrigin origin;
            err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_ORIGIN, sizeof(McInputOrigin), &origin, NULL);
            my_assert(err == MC_NO_ERROR);
            if (origin == MC_INPUT_ORIGIN_CUTMESH) {
                continue; // we only care about the source mesh
            }
        }

        // query the vertices
        // ----------------------

        uint64_t numBytes = 0;
        err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_VERTEX_DOUBLE, 0, NULL, &numBytes);
        my_assert(err == MC_NO_ERROR);
        uint32_t ccVertexCount = (uint32_t)(numBytes / (sizeof(double) * 3));
        std::vector<double> ccVertices((uint64_t)ccVertexCount * 3u, 0);
        err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_VERTEX_DOUBLE, numBytes, (void*)ccVertices.data(), NULL);
        my_assert(err == MC_NO_ERROR);

        // query the faces
        // -------------------

        numBytes = 0;
        err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_FACE, 0, NULL, &numBytes);
        my_assert(err == MC_NO_ERROR);
        std::vector<uint32_t> ccFaceIndices(numBytes / sizeof(uint32_t), 0);
        err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_FACE, numBytes, ccFaceIndices.data(), NULL);
        my_assert(err == MC_NO_ERROR);

        // query the face sizes
        // ------------------------
        numBytes = 0;
        err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_FACE_SIZE, 0, NULL, &numBytes);
        my_assert(err == MC_NO_ERROR);
        std::vector<uint32_t> ccFaceSizes(numBytes / sizeof(uint32_t), 0);
        err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_FACE_SIZE, numBytes, ccFaceSizes.data(), NULL);
        my_assert(err == MC_NO_ERROR);

        const uint32_t ccFaceCount = static_cast<uint32_t>(ccFaceSizes.size());

        {
            char buf[512];
            sprintf(buf, OUTPUT_DIR "/cc%d.obj", c);
            writeOBJ(buf, &ccVertices[0], ccVertexCount, &ccFaceIndices[0], &ccFaceSizes[0], ccFaceCount);
        }

        // query the face map
        // ------------------
        numBytes = 0;
        err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_FACE_MAP, 0, NULL, &numBytes);
        my_assert(err == MC_NO_ERROR);
        std::vector<uint32_t> ccFaceMap(numBytes / sizeof(uint32_t), 0);
        err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_FACE_MAP, numBytes, ccFaceMap.data(), NULL);
        my_assert(err == MC_NO_ERROR);

        // query the face adjacency
        // ------------------------

        numBytes = 0;
        err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_FACE_ADJACENT_FACE, 0, NULL, &numBytes);
        my_assert(err == MC_NO_ERROR);
        std::vector<uint32_t> ccFaceAdjFaces(numBytes / sizeof(uint32_t), 0);
        err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_FACE_ADJACENT_FACE, numBytes, ccFaceAdjFaces.data(), NULL);
        my_assert(err == MC_NO_ERROR);

        // query the face adjacency sizes
        // -------------------------------
        numBytes = 0;
        err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_FACE_ADJACENT_FACE_SIZE, 0, NULL, &numBytes);
        my_assert(err == MC_NO_ERROR);
        std::vector<uint32_t> ccFaceAdjFacesSizes(numBytes / sizeof(uint32_t), 0);
        err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_FACE_ADJACENT_FACE_SIZE, numBytes, ccFaceAdjFacesSizes.data(), NULL);
        my_assert(err == MC_NO_ERROR);

        // resolve mapping between tags and CC faces
        // NOTE: only those CC face whose origin face was tagged will themselves be tagged.
        std::map<int, std::vector<uint32_t>> tagToCcFaces;
        std::map<uint32_t, int> ccFaceToTag;

        for (int ccFaceID = 0; ccFaceID < (int)ccFaceCount; ++ccFaceID) {
            int imFaceID = ccFaceMap[ccFaceID];
            std::map<uint32_t, int>::const_iterator srcMeshFaceToTagIter = srcMeshFaceToTag.find(imFaceID);
            bool faceWasTagged = (srcMeshFaceToTagIter != srcMeshFaceToTag.cend());

            if (faceWasTagged) {
                const int tag = srcMeshFaceToTagIter->second;
                ccFaceToTag[ccFaceID] = tag;
                tagToCcFaces[tag].push_back(ccFaceID);
            }
        }

        for (std::map<uint32_t, int>::const_iterator i = srcMeshFaceToTag.cbegin(); i != srcMeshFaceToTag.cend(); ++i) {
            tagToSrcMeshFaces[i->second].push_back(i->first);
        }

        std::vector<uint32_t> ccFaceIndicesMerged;
        std::vector<uint32_t> ccFaceSizesMerged;

        mergeAdjacentMeshFacesByProperty(
            ccFaceIndicesMerged,
            ccFaceSizesMerged,
            ccFaceIndices,
            ccFaceSizes,
            ccFaceAdjFaces,
            ccFaceAdjFacesSizes,
            tagToCcFaces,
            ccFaceToTag);

        {
            char buf[512];
            sprintf(buf, OUTPUT_DIR "/cc%d-merged.obj", c);
            // NOTE: For the sake of simplicity, we keep unreferenced vertices.
            writeOBJ(buf, &ccVertices[0], ccVertexCount, &ccFaceIndicesMerged[0], &ccFaceSizesMerged[0], (uint32_t)ccFaceSizesMerged.size());
        }
    }

    // free connected component data
    // --------------------------------
    err = mcReleaseConnectedComponents(context, (uint32_t)connectedComponents.size(), connectedComponents.data());
    my_assert(err == MC_NO_ERROR);

    // destroy context
    // ------------------
    err = mcReleaseContext(context);

    my_assert(err == MC_NO_ERROR);

    return 0;
}

void writeOBJ(
    const std::string& path,
    const double* ccVertices,
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

void readOBJ(const std::string& path, std::vector<double>& V, std::vector<uint32_t>& F, std::vector<uint32_t>& Fsizes)
{
    std::vector<std::vector<double>> V_;
    std::vector<std::vector<int>> F_;

    igl::read_triangle_mesh(path, V_, F_);

    for (int i = 0; i < (int)V_.size(); ++i) {
        V.push_back(V_[i][0]);
        V.push_back(V_[i][1]);
        V.push_back(V_[i][2]);
    }

    for (int i = 0; i < (int)F_.size(); ++i) {
        F.push_back((uint32_t)F_[i][0]);
        F.push_back((uint32_t)F_[i][1]);
        F.push_back((uint32_t)F_[i][2]);
        Fsizes.push_back(3);
    }
}

uint32_t getAdjFacesBaseOffset(const uint32_t faceIdx, const uint32_t* faceAdjFacesSizes)
{
    uint32_t baseOffset = 0;
    for (uint32_t f = 0; f < faceIdx; ++f) {
        baseOffset += faceAdjFacesSizes[f];
    }
    return baseOffset;
}

uint32_t getFaceIndicesBaseOffset(const uint32_t faceIdx, const uint32_t* faceSizes)
{
    uint32_t baseOffset = 0;
    for (uint32_t f = 0; f < faceIdx; ++f) {
        baseOffset += faceSizes[f];
    }
    return baseOffset;
};

void mergeAdjacentMeshFacesByProperty(
    std::vector<uint32_t>& meshFaceIndicesOUT,
    std::vector<uint32_t>& meshFaceSizesOUT,
    const std::vector<uint32_t>& meshFaces,
    const std::vector<uint32_t>& meshFaceSizes,
    const std::vector<uint32_t>& meshFaceAdjFace,
    const std::vector<uint32_t>& meshFaceAdjFaceSizes,
    const std::map<int, std::vector<uint32_t>>& tagToMeshFace,
    const std::map<uint32_t, int>& meshFaceToTag)
{
    // for each tag
    for (std::map<int, std::vector<uint32_t>>::const_iterator iter = tagToMeshFace.cbegin(); iter != tagToMeshFace.cend(); ++iter) {

        // NOTE: may contain faces that form disjoint patches i.e. not all faces
        // are merged into one. It is possible to create more than one new face
        // after the merging where the resulting faces after merging are not adjacent.
        std::vector<uint32_t> meshFacesWithSameTag = iter->second; // copy!

        // merge the faces that are adjacent
        std::vector<std::vector<uint32_t>> adjacentFaceLists; // i.e. each element is a patch/collection of adjacent faces

        do {
            adjacentFaceLists.push_back(std::vector<uint32_t>()); // add new patch
            std::vector<uint32_t>& curAdjFaceList = adjacentFaceLists.back();

            // queue of adjacent faces
            std::deque<uint32_t> adjFaceQueue;
            adjFaceQueue.push_back(meshFacesWithSameTag.back()); // start with any
            meshFacesWithSameTag.pop_back();
            do {
                uint32_t cur = adjFaceQueue.front();
                adjFaceQueue.pop_front();
                const int numAdjFaces = meshFaceAdjFaceSizes[cur];
                const int ccFaceAdjFacesBaseOffset = getAdjFacesBaseOffset(cur, meshFaceAdjFaceSizes.data());

                curAdjFaceList.push_back(cur);

                // for each adjacent face of current face
                for (int i = 0; i < numAdjFaces; ++i) {
                    const uint32_t adjFaceID = meshFaceAdjFace[(size_t)ccFaceAdjFacesBaseOffset + i];

                    std::vector<uint32_t>::const_iterator curAdjFaceListIter = std::find(curAdjFaceList.cbegin(), curAdjFaceList.cend(), adjFaceID);
                    bool alreadyAddedToCurAdjFaceList = (curAdjFaceListIter != curAdjFaceList.cend());

                    if (!alreadyAddedToCurAdjFaceList) {

                        // does the adjacent face share a Tag..?
                        std::vector<uint32_t>::const_iterator fiter = std::find(iter->second.cbegin(), iter->second.cend(), adjFaceID);
                        bool haveSharedTag = (fiter != iter->second.cend());

                        if (haveSharedTag) {

                            std::deque<uint32_t>::const_iterator queueIter = std::find(adjFaceQueue.cbegin(), adjFaceQueue.cend(), adjFaceID);
                            bool alreadyAddedToAdjFaceQueue = (queueIter != adjFaceQueue.end());
                            if (!alreadyAddedToAdjFaceQueue) {
                                adjFaceQueue.push_back(adjFaceID); // add it!

                                std::vector<uint32_t>::iterator facesWithSharedTagIter = std::find(meshFacesWithSameTag.begin(), meshFacesWithSameTag.end(), adjFaceID);
                                if (facesWithSharedTagIter != meshFacesWithSameTag.cend()) {
                                    meshFacesWithSameTag.erase(facesWithSharedTagIter); // remove since we have now associated with patch.
                                }
                            }
                        }
                    }
                }

            } while (!adjFaceQueue.empty());
        } while (!meshFacesWithSameTag.empty());

        for (std::vector<std::vector<uint32_t>>::const_iterator adjacentFaceListsIter = adjacentFaceLists.cbegin();
             adjacentFaceListsIter != adjacentFaceLists.cend();
             ++adjacentFaceListsIter) {

            // Unordered list of halfedges which define the boundary of our new
            // face
            std::vector<std::pair<int, int>> halfedgePool;

            for (int f = 0; f < (int)adjacentFaceListsIter->size(); ++f) {

                const uint32_t meshFaceID = adjacentFaceListsIter->at(f);
                const uint32_t meshFaceVertexCount = meshFaceSizes[meshFaceID];
                const uint32_t baseIdx = getFaceIndicesBaseOffset(meshFaceID, meshFaceSizes.data());
                const int numFaceEdges = (int)meshFaceVertexCount; // NOTE: a polygon has the same number of vertices as its edges.

                // for each edge of face
                for (int faceEdgeID = 0; faceEdgeID < numFaceEdges; ++faceEdgeID) {

                    const int srcIdx = faceEdgeID;
                    const int tgtIdx = (faceEdgeID + 1) % meshFaceVertexCount;
                    const uint32_t srcVertexIdx = meshFaces[(size_t)baseIdx + srcIdx];
                    const uint32_t tgtVertexIdx = meshFaces[(size_t)baseIdx + tgtIdx];

                    std::vector<std::pair<int, int>>::iterator fiter = std::find_if(
                        halfedgePool.begin(),
                        halfedgePool.end(),
                        [&](const std::pair<int, int>& elem) {
                            return ((uint32_t)elem.first == srcVertexIdx && (uint32_t)elem.second == tgtVertexIdx) || //
                                ((uint32_t)elem.second == srcVertexIdx && (uint32_t)elem.first == tgtVertexIdx);
                        });

                    const bool opposite_halfedge_exists = (fiter != halfedgePool.cend());

                    if (opposite_halfedge_exists) {
                        halfedgePool.erase(fiter);
                    } else {
                        halfedgePool.emplace_back(srcVertexIdx, tgtVertexIdx);
                    }
                }
            }

            std::map<int, std::vector<int>> vertexToHalfedges;

            for (int i = 0; i < (int)halfedgePool.size(); ++i) {
                std::pair<int, int> halfedge = halfedgePool[i];
                vertexToHalfedges[halfedge.first].push_back(i);
                vertexToHalfedges[halfedge.second].push_back(i);
            }

            std::vector<uint32_t> polygon;
            std::map<int, std::vector<int>>::const_iterator cur;
            std::map<int, std::vector<int>>::const_iterator next = vertexToHalfedges.cbegin(); // could start from any

            do {
                cur = next;
                next = vertexToHalfedges.cend();
                polygon.push_back(cur->first);

                // find next (pick the halfedge whose "source" is the current vertex)
                std::vector<int> halfedges = cur->second;

                for (int i = 0; i < 2; ++i) {
                    std::pair<int, int> edge = halfedgePool[halfedges[i]];
                    if (edge.first == cur->first && std::find(polygon.cbegin(), polygon.cend(), (uint32_t)edge.second) == polygon.cend()) {
                        next = vertexToHalfedges.find(edge.second);
                        my_assert(next != vertexToHalfedges.cend());
                        break;
                    }
                }

            } while (next != vertexToHalfedges.cend());

            meshFaceIndicesOUT.insert(meshFaceIndicesOUT.end(), polygon.cbegin(), polygon.cend());
            meshFaceSizesOUT.push_back((uint32_t)polygon.size());
        }
    }

    // Now we add the untagged faces into the new mesh (the ones which did not need merging)

    for (int meshFaceID = 0; meshFaceID < (int)meshFaceSizes.size(); ++meshFaceID) {
        bool faceWasMerged = meshFaceToTag.find(meshFaceID) != meshFaceToTag.cend();
        if (!faceWasMerged) {
            const uint32_t baseIdx = getFaceIndicesBaseOffset(meshFaceID, meshFaceSizes.data());
            const uint32_t meshFaceVertexCount = meshFaceSizes[meshFaceID];

            for (int i = 0; i < (int)meshFaceVertexCount; ++i) {
                meshFaceIndicesOUT.push_back(meshFaces[(size_t)baseIdx + i]);
            }

            meshFaceSizesOUT.push_back(meshFaceVertexCount);
        }
    }
}
