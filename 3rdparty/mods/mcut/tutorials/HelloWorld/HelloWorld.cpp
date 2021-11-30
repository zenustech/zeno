/*
Simple "hello world" program using MCUT.
Input meshes are defined in-source but output meshes are saved as .off files
*/

#include "mcut/mcut.h"

#include <stdio.h>
#include <stdlib.h>

#include <vector>

#define my_assert(cond)                             \
    if (!(cond))                                    \
    {                                               \
        fprintf(stderr, "MCUT error: %s\n", #cond); \
        std::exit(1);                               \
    }

void writeOFF(
    const char *fpath,
    float *pVertices,
    uint32_t *pFaceIndices,
    uint32_t *pFaceSizes,
    uint32_t numVertices,
    uint32_t numFaces);

int main()
{
    // 1. Create meshes.
    // -----------------

    // the cube
    // --------
    float cubeVertices[] = {
        -5, -5, 5,  // 0
        5, -5, 5,   // 1
        5, 5, 5,    //2
        -5, 5, 5,   //3
        -5, -5, -5, //4
        5, -5, -5,  //5
        5, 5, -5,   //6
        -5, 5, -5   //7
    };
    uint32_t cubeFaces[] = {
        0, 1, 2, 3, //0
        7, 6, 5, 4, //1
        1, 5, 6, 2, //2
        0, 3, 7, 4, //3
        3, 2, 6, 7, //4
        4, 5, 1, 0  //5
    };
    uint32_t cubeFaceSizes[] = {
        4, 4, 4, 4, 4, 4};
    uint32_t numCubeVertices = 8;
    uint32_t numCubeFaces = 6;

    // the cut mesh
    // ---------
    float cutMeshVertices[] = {
        -20, -4, 0, //0
        0, 20, 20,  //1
        20, -4, 0,  //2
        0, 20, -20  //3
    };
    uint32_t cutMeshFaces[] = {
        0, 1, 2, //0
        0, 2, 3  //1
    };
    uint32_t cutMeshFaceSizes[] = {
        3, 3};
    uint32_t numCutMeshVertices = 4;
    uint32_t numCutMeshFaces = 2;

    // 2. create a context
    // -------------------
    McContext context = MC_NULL_HANDLE;
    McResult err = mcCreateContext(&context, MC_NULL_HANDLE);

    if (err != MC_NO_ERROR)
    {
        fprintf(stderr, "could not create context (err=%d)\n", (int)err);
        exit(1);
    }

    // 3. do the magic!
    // ----------------
    err = mcDispatch(
        context,
        MC_DISPATCH_VERTEX_ARRAY_FLOAT,
        cubeVertices,
        cubeFaces,
        cubeFaceSizes,
        numCubeVertices,
        numCubeFaces,
        cutMeshVertices,
        cutMeshFaces,
        cutMeshFaceSizes,
        numCutMeshVertices,
        numCutMeshFaces);

    if (err != MC_NO_ERROR)
    {
        fprintf(stderr, "dispatch call failed (err=%d)\n", (int)err);
        exit(1);
    }

    // 4. query the number of available connected component (all types)
    // -------------------------------------------------------------
    uint32_t numConnComps;
    std::vector<McConnectedComponent> connComps;

    err = mcGetConnectedComponents(context, MC_CONNECTED_COMPONENT_TYPE_ALL, 0, NULL, &numConnComps);

    if (err != MC_NO_ERROR)
    {
        fprintf(stderr, "1:mcGetConnectedComponents(MC_CONNECTED_COMPONENT_TYPE_ALL) failed (err=%d)\n", (int)err);
        exit(1);
    }

    if (numConnComps == 0)
    {
        fprintf(stdout, "no connected components found\n");
        exit(0);
    }

    connComps.resize(numConnComps);

    err = mcGetConnectedComponents(context, MC_CONNECTED_COMPONENT_TYPE_ALL, (uint32_t)connComps.size(), connComps.data(), NULL);

    if (err != MC_NO_ERROR)
    {
        fprintf(stderr, "2:mcGetConnectedComponents(MC_CONNECTED_COMPONENT_TYPE_ALL) failed (err=%d)\n", (int)err);
        exit(1);
    }

    // 5. query the data of each connected component from MCUT
    // -------------------------------------------------------

    for (int i = 0; i < (int)connComps.size(); ++i)
    {
        McConnectedComponent connComp = connComps[i]; // connected compoenent id

        uint64_t numBytes = 0;

        // query the vertices
        // ----------------------

        numBytes = 0;
        err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_VERTEX_FLOAT, 0, NULL, &numBytes);

        if (err != MC_NO_ERROR)
        {
            fprintf(stderr, "1:mcGetConnectedComponentData(MC_CONNECTED_COMPONENT_DATA_VERTEX_FLOAT) failed (err=%d)\n", (int)err);
            exit(1);
        }

        uint32_t numberOfVertices = (uint32_t)(numBytes / (sizeof(float) * 3));

        std::vector<float> vertices(numberOfVertices * 3u);

        err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_VERTEX_FLOAT, numBytes, (void *)vertices.data(), NULL);

        if (err != MC_NO_ERROR)
        {
            fprintf(stderr, "2:mcGetConnectedComponentData(MC_CONNECTED_COMPONENT_DATA_VERTEX_FLOAT) failed (err=%d)\n", (int)err);
            exit(1);
        }

        // query the faces
        // -------------------

        numBytes = 0;
        err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_FACE, 0, NULL, &numBytes);

        if (err != MC_NO_ERROR)
        {
            fprintf(stderr, "1:mcGetConnectedComponentData(MC_CONNECTED_COMPONENT_DATA_FACE) failed (err=%d)\n", (int)err);
            exit(1);
        }

        std::vector<uint32_t> faceIndices;
        faceIndices.resize(numBytes / sizeof(uint32_t));

        err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_FACE, numBytes, faceIndices.data(), NULL);

        if (err != MC_NO_ERROR)
        {
            fprintf(stderr, "2:mcGetConnectedComponentData(MC_CONNECTED_COMPONENT_DATA_FACE) failed (err=%d)\n", (int)err);
            exit(1);
        }

        // query the face sizes
        // ------------------------
        numBytes = 0;
        err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_FACE_SIZE, 0, NULL, &numBytes);
        if (err != MC_NO_ERROR)
        {
            fprintf(stderr, "1:mcGetConnectedComponentData(MC_CONNECTED_COMPONENT_DATA_FACE_SIZE) failed (err=%d)\n", (int)err);
            exit(1);
        }

        std::vector<uint32_t> faceSizes;
        faceSizes.resize(numBytes / sizeof(uint32_t));

        err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_FACE_SIZE, numBytes, faceSizes.data(), NULL);

        if (err != MC_NO_ERROR)
        {
            fprintf(stderr, "2:mcGetConnectedComponentData(MC_CONNECTED_COMPONENT_DATA_FACE_SIZE) failed (err=%d)\n", (int)err);
            exit(1);
        }

        char fnameBuf[32];
        sprintf(fnameBuf, "conncomp%d.off", i);

        // save to mesh file (.off)
        // ------------------------
        writeOFF(fnameBuf,
                 (float *)vertices.data(),
                 (uint32_t *)faceIndices.data(),
                 (uint32_t *)faceSizes.data(),
                 (uint32_t)vertices.size() / 3,
                 (uint32_t)faceSizes.size());
    }

    // 6. free connected component data
    // --------------------------------
    err = mcReleaseConnectedComponents(context, 0, NULL);

    if (err != MC_NO_ERROR)
    {
        fprintf(stderr, "mcReleaseConnectedComponents failed (err=%d)\n", (int)err);
        exit(1);
    }

    // 7. destroy context
    // ------------------
    err = mcReleaseContext(context);

    if (err != MC_NO_ERROR)
    {
        fprintf(stderr, "mcReleaseContext failed (err=%d)\n", (int)err);
        exit(1);
    }

    return 0;
}

// write mesh to .off file
void writeOFF(
    const char *fpath,
    float *pVertices,
    uint32_t *pFaceIndices,
    uint32_t *pFaceSizes,
    uint32_t numVertices,
    uint32_t numFaces)
{
    FILE *file = fopen(fpath, "w");

    if (file == NULL)
    {
        fprintf(stderr, "error: failed to open `%s`", fpath);
        exit(1);
    }

    fprintf(file, "OFF\n");
    fprintf(file, "%d %d %d\n", numVertices, numFaces, 0 /*numEdges*/);
    int i;
    for (i = 0; i < (int)numVertices; ++i)
    {
        float *vptr = pVertices + (i * 3);
        fprintf(file, "%f %f %f\n", vptr[0], vptr[1], vptr[2]);
    }

    int faceBaseOffset = 0;
    for (i = 0; i < (int)numFaces; ++i)
    {
        uint32_t faceVertexCount = pFaceSizes[i];
        fprintf(file, "%d", (int)faceVertexCount);
        int j;
        for (j = 0; j < (int)faceVertexCount; ++j)
        {
            uint32_t *fptr = pFaceIndices + faceBaseOffset + j;
            fprintf(file, " %d", *fptr);
        }
        fprintf(file, "\n");
        faceBaseOffset += faceVertexCount;
    }

    fclose(file);
}
