#if defined(_WIN32)
#define _CRT_SECURE_NO_WARNINGS 1
#endif

#include "utest.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)

// https://stackoverflow.com/questions/735126/are-there-alternate-implementations-of-gnu-getline-interface/735472#735472
/* Modifications, public domain as well, by Antti Haapala, 11/10/17
- Switched to getc on 5/23/19 */
#include <errno.h>
#include <stdint.h>

// if typedef doesn't exist (msvc, blah)
typedef intptr_t ssize_t;

ssize_t getline(char** lineptr, size_t* n, FILE* stream)
{
    size_t pos;
    int c;

    if (lineptr == NULL || stream == NULL || n == NULL) {
        errno = EINVAL;
        return -1;
    }

    c = getc(stream);
    if (c == EOF) {
        return -1;
    }

    if (*lineptr == NULL) {
        *lineptr = (char*)malloc(128);
        if (*lineptr == NULL) {
            return -1;
        }
        *n = 128;
    }

    pos = 0;
    while (c != EOF) {
        if (pos + 1 >= *n) {
            size_t new_size = *n + (*n >> 2);
            if (new_size < 128) {
                new_size = 128;
            }
            char* new_ptr = (char*)realloc(*lineptr, new_size);
            if (new_ptr == NULL) {
                return -1;
            }
            *n = new_size;
            *lineptr = new_ptr;
        }

        ((unsigned char*)(*lineptr))[pos++] = (unsigned char)c;
        if (c == '\n') {
            break;
        }
        c = getc(stream);
    }

    (*lineptr)[pos] = '\0';
    return pos;
}
#endif // #if defined (_WIN32)

bool readLine(FILE* file, char** lineBuf, size_t* len)
{
    while (getline(lineBuf, len, file)) {
        if (strlen(*lineBuf) > 1 && (*lineBuf)[0] != '#') {
            return true;
        }
    }
    return false;
}

extern "C" void readOFF(
    const char* fpath,
    float** pVertices,
    unsigned int** pFaceIndices,
    unsigned int** pFaceSizes,
    unsigned int* numVertices,
    unsigned int* numFaces)
{
    // using "rb" instead of "r" to prevent linefeed conversion
    // See: https://stackoverflow.com/questions/27530636/read-text-file-in-c-with-fopen-without-linefeed-conversion
    FILE* file = fopen(fpath, "rb");

    if (file == NULL) {
        fprintf(stderr, "error: failed to open `%s`", fpath);
        exit(1);
    }

    char* lineBuf = NULL;
    size_t lineBufLen = 0;
    bool lineOk = true;
    int i = 0;

    // file header
    lineOk = readLine(file, &lineBuf, &lineBufLen);

    if (!lineOk) {
        fprintf(stderr, "error: .off file header not found\n");
        exit(1);
    }

    if (strstr(lineBuf, "OFF") == NULL) {
        fprintf(stderr, "error: unrecognised .off file header\n");
        exit(1);
    }

    // #vertices, #faces, #edges
    lineOk = readLine(file, &lineBuf, &lineBufLen);

    if (!lineOk) {
        fprintf(stderr, "error: .off element count not found\n");
        exit(1);
    }

    int nedges = 0;
    sscanf(lineBuf, "%d %d %d", numVertices, numFaces, &nedges);
    *pVertices = (float*)malloc(sizeof(float) * (*numVertices) * 3);
    *pFaceSizes = (unsigned int*)malloc(sizeof(unsigned int) * (*numFaces));

    // vertices
    for (i = 0; i < (float)(*numVertices); ++i) {
        lineOk = readLine(file, &lineBuf, &lineBufLen);

        if (!lineOk) {
            fprintf(stderr, "error: .off vertex not found\n");
            exit(1);
        }

        float x, y, z;
        sscanf(lineBuf, "%f %f %f", &x, &y, &z);

        (*pVertices)[(i * 3) + 0] = x;
        (*pVertices)[(i * 3) + 1] = y;
        (*pVertices)[(i * 3) + 2] = z;
    }
#if _WIN64
    __int64 facesStartOffset = _ftelli64(file);
#else
    long int facesStartOffset = ftell(file);
#endif
    int numFaceIndices = 0;

    // faces
    for (i = 0; i < (int)(*numFaces); ++i) {
        lineOk = readLine(file, &lineBuf, &lineBufLen);

        if (!lineOk) {
            fprintf(stderr, "error: .off file face not found\n");
            exit(1);
        }

        int n; // number of vertices in face
        sscanf(lineBuf, "%d", &n);

        if (n < 3) {
            fprintf(stderr, "error: invalid vertex count in file %d\n", n);
            exit(1);
        }

        (*pFaceSizes)[i] = n;
        numFaceIndices += n;
    }

    (*pFaceIndices) = (unsigned int*)malloc(sizeof(unsigned int) * numFaceIndices);

#if _WIN64
    int err = _fseeki64(file, facesStartOffset, SEEK_SET);
#else
    int err = fseek(file, facesStartOffset, SEEK_SET);
#endif
    if (err != 0) {
        fprintf(stderr, "error: fseek failed\n");
        exit(1);
    }

    int indexOffset = 0;
    for (i = 0; i < (int)(*numFaces); ++i) {

        lineOk = readLine(file, &lineBuf, &lineBufLen);

        if (!lineOk) {
            fprintf(stderr, "error: .off file face not found\n");
            exit(1);
        }

        int n; // number of vertices in face
        sscanf(lineBuf, "%d", &n);

        char* lineBufShifted = lineBuf;
        int j = 0;

        while (j < n) { // parse remaining numbers on lineBuf
            lineBufShifted = strstr(lineBufShifted, " ") + 1; // start of next number

            int val;
            sscanf(lineBufShifted, "%d", &val);

            (*pFaceIndices)[indexOffset + j] = val;
            j++;
        }

        indexOffset += n;
    }

    free(lineBuf);

    fclose(file);
}

extern "C" void writeOFF(
    const char* fpath,
    float* pVertices,
    unsigned int* pFaceIndices,
    unsigned int* pFaceSizes,
    unsigned int* pEdgeIndices,
    unsigned int numVertices,
    unsigned int numFaces,
    unsigned int numEdges)
{
    FILE* file = fopen(fpath, "w");

    if (file == NULL) {
        fprintf(stderr, "error: failed to open `%s`", fpath);
        exit(1);
    }

    fprintf(file, "OFF\n");
    printf("skipped .off file edges %d, %p\n", numEdges, pEdgeIndices);
    //numEdges;
    //pEdgeIndices;
    fprintf(file, "%d %d %d\n", numVertices, numFaces, 0 /*numEdges*/);
    int i;
    for (i = 0; i < (int)numVertices; ++i) {
        float* vptr = pVertices + ((size_t)i * 3);
        fprintf(file, "%f %f %f\n", vptr[0], vptr[1], vptr[2]);
    }
#if 0
  for (i = 0; i < numEdges; ++i) {
    unsigned int* iptr = pEdgeIndices + (i * 2);
    fprintf(file, "%u %u\n", iptr[0], iptr[1]);
  }
#endif
    bool isTriangleMesh = (pFaceSizes == NULL);

    int faceBaseOffset = 0;
    for (i = 0; i < (int)numFaces; ++i) {
        unsigned int faceVertexCount = isTriangleMesh ? 3 : pFaceSizes[i];
        fprintf(file, "%d", (int)faceVertexCount);
        unsigned int j;
        for (j = 0; j < faceVertexCount; ++j) {
            unsigned int* fptr = pFaceIndices + faceBaseOffset + j;
            fprintf(file, " %d", *fptr);
        }
        fprintf(file, "\n");
        faceBaseOffset += faceVertexCount;
    }

    fclose(file);
}

UTEST_MAIN();
