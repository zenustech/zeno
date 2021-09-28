#if 0
/*
 * FTGL - OpenGL font library
 *
 * Copyright (c) 2001-2004 Henry Maddocks <ftgl@opengl.geek.nz>
 * Copyright (c) 2008 Éric Beets <ericbeets@free.fr>
 * Copyright (c) 2008 Sam Hocevar <sam@hocevar.net>
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "config.h"

#include "FTInternals.h"
#include "FTVectoriser.h"

#ifndef CALLBACK
#define CALLBACK
#endif

#if defined __APPLE_CC__ && __APPLE_CC__ < 5465
    typedef GLvoid (*GLUTesselatorFunction) (...);
#elif defined _WIN32 && !defined __CYGWIN__
    typedef GLvoid (CALLBACK *GLUTesselatorFunction) ();
#else
    typedef GLvoid (*GLUTesselatorFunction) ();
#endif


void CALLBACK ftglError(GLenum errCode, FTMesh* mesh)
{
    mesh->Error(errCode);
}


void CALLBACK ftglVertex(void* data, FTMesh* mesh)
{
    FTGL_DOUBLE* vertex = static_cast<FTGL_DOUBLE*>(data);
    mesh->AddPoint(vertex[0], vertex[1], vertex[2]);
}


void CALLBACK ftglCombine(FTGL_DOUBLE coords[3], void* vertex_data[4], GLfloat weight[4], void** outData, FTMesh* mesh)
{
    (void)vertex_data; (void)weight;

    const FTGL_DOUBLE* vertex = static_cast<const FTGL_DOUBLE*>(coords);
    *outData = const_cast<FTGL_DOUBLE*>(mesh->Combine(vertex[0], vertex[1], vertex[2]));
}

void CALLBACK ftglBegin(GLenum type, FTMesh* mesh)
{
    mesh->Begin(type);
}


void CALLBACK ftglEnd(FTMesh* mesh)
{
    mesh->End();
}


FTMesh::FTMesh()
: currentTesselation(0),
    err(0)
{
    tesselationList.reserve(16);
}


FTMesh::~FTMesh()
{
    for(size_t t = 0; t < tesselationList.size(); ++t)
    {
        delete tesselationList[t];
    }

    tesselationList.clear();
}


void FTMesh::AddPoint(const FTGL_DOUBLE x, const FTGL_DOUBLE y, const FTGL_DOUBLE z)
{
    currentTesselation->AddPoint(x, y, z);
}


const FTGL_DOUBLE* FTMesh::Combine(const FTGL_DOUBLE x, const FTGL_DOUBLE y, const FTGL_DOUBLE z)
{
    tempPointList.push_back(FTPoint(x, y,z));
    return static_cast<const FTGL_DOUBLE*>(tempPointList.back());
}


void FTMesh::Begin(GLenum meshType)
{
    currentTesselation = new FTTesselation(meshType);
}


void FTMesh::End()
{
    tesselationList.push_back(currentTesselation);
}


FTTesselation const * FTMesh::Tesselation(size_t index) const
{
    return (index < tesselationList.size()) ? tesselationList[index] : NULL;
}


FTVectoriser::FTVectoriser(const FT_GlyphSlot glyph)
:   contourList(0),
    mesh(0),
    ftContourCount(0),
    contourFlag(0)
{
    if(glyph)
    {
        outline = glyph->outline;

        ftContourCount = outline.n_contours;
        contourList = 0;
        contourFlag = outline.flags;

        ProcessContours();
    }
}


FTVectoriser::~FTVectoriser()
{
    for(size_t c = 0; c < ContourCount(); ++c)
    {
        delete contourList[c];
    }

    delete [] contourList;
    delete mesh;
}


void FTVectoriser::ProcessContours()
{
    short contourLength = 0;
    short startIndex = 0;
    short endIndex = 0;

    contourList = new FTContour*[ftContourCount];

    for(int i = 0; i < ftContourCount; ++i)
    {
        FT_Vector* pointList = &outline.points[startIndex];
        char* tagList = &outline.tags[startIndex];

        endIndex = outline.contours[i];
        contourLength =  (endIndex - startIndex) + 1;

        FTContour* contour = new FTContour(pointList, tagList, contourLength);

        contourList[i] = contour;

        startIndex = endIndex + 1;
    }

    // Compute each contour's parity. FIXME: see if FT_Outline_Get_Orientation
    // can do it for us.
    for(int i = 0; i < ftContourCount; i++)
    {
        FTContour *c1 = contourList[i];

        // 1. Find the leftmost point.
        FTPoint leftmost(65536.0, 0.0);

        for(size_t n = 0; n < c1->PointCount(); n++)
        {
            FTPoint p = c1->Point(n);
            if(p.X() < leftmost.X())
            {
                leftmost = p;
            }
        }

        // 2. Count how many other contours we cross when going further to
        // the left.
        int parity = 0;

        for(int j = 0; j < ftContourCount; j++)
        {
            if(j == i)
            {
                continue;
            }

            FTContour *c2 = contourList[j];

            for(size_t n = 0; n < c2->PointCount(); n++)
            {
                FTPoint p1 = c2->Point(n);
                FTPoint p2 = c2->Point((n + 1) % c2->PointCount());

                /* FIXME: combinations of >= > <= and < do not seem stable */
                if((p1.Y() < leftmost.Y() && p2.Y() < leftmost.Y())
                    || (p1.Y() >= leftmost.Y() && p2.Y() >= leftmost.Y())
                    || (p1.X() > leftmost.X() && p2.X() > leftmost.X()))
                {
                    continue;
                }
                else if(p1.X() < leftmost.X() && p2.X() < leftmost.X())
                {
                    parity++;
                }
                else
                {
                    FTPoint a = p1 - leftmost;
                    FTPoint b = p2 - leftmost;
                    if(b.X() * a.Y() > b.Y() * a.X())
                    {
                        parity++;
                    }
                }
            }
        }

        // 3. Make sure the glyph has the proper parity.
        c1->SetParity(parity);
    }
}


size_t FTVectoriser::PointCount()
{
    size_t s = 0;
    for(size_t c = 0; c < ContourCount(); ++c)
    {
        s += contourList[c]->PointCount();
    }

    return s;
}


FTContour const * FTVectoriser::Contour(size_t index) const
{
    return (index < ContourCount()) ? contourList[index] : NULL;
}


void FTVectoriser::MakeMesh(FTGL_DOUBLE zNormal, int outsetType, float outsetSize)
{
    if(mesh)
    {
        delete mesh;
    }

    mesh = new FTMesh;

    GLUtesselator* tobj = gluNewTess();

    gluTessCallback(tobj, GLU_TESS_BEGIN_DATA,     (GLUTesselatorFunction)ftglBegin);
    gluTessCallback(tobj, GLU_TESS_VERTEX_DATA,    (GLUTesselatorFunction)ftglVertex);
    gluTessCallback(tobj, GLU_TESS_COMBINE_DATA,   (GLUTesselatorFunction)ftglCombine);
    gluTessCallback(tobj, GLU_TESS_END_DATA,       (GLUTesselatorFunction)ftglEnd);
    gluTessCallback(tobj, GLU_TESS_ERROR_DATA,     (GLUTesselatorFunction)ftglError);

    if(contourFlag & ft_outline_even_odd_fill) // ft_outline_reverse_fill
    {
        gluTessProperty(tobj, GLU_TESS_WINDING_RULE, GLU_TESS_WINDING_ODD);
    }
    else
    {
        gluTessProperty(tobj, GLU_TESS_WINDING_RULE, GLU_TESS_WINDING_NONZERO);
    }


    gluTessProperty(tobj, GLU_TESS_TOLERANCE, 0);
    gluTessNormal(tobj, 0.0f, 0.0f, zNormal);
    gluTessBeginPolygon(tobj, mesh);

        for(size_t c = 0; c < ContourCount(); ++c)
        {
            /* Build the */
            switch(outsetType)
            {
                case 1 : contourList[c]->buildFrontOutset(outsetSize); break;
                case 2 : contourList[c]->buildBackOutset(outsetSize); break;
            }
            const FTContour* contour = contourList[c];


            gluTessBeginContour(tobj);
                for(size_t p = 0; p < contour->PointCount(); ++p)
                {
                    const FTGL_DOUBLE* d;
                    switch(outsetType)
                    {
                        case 1: d = contour->FrontPoint(p); break;
                        case 2: d = contour->BackPoint(p); break;
                        case 0: default: d = contour->Point(p); break;
                    }
                    // XXX: gluTessVertex doesn't modify the data but does not
                    // specify "const" in its prototype, so we cannot cast to
                    // a const type.
                    gluTessVertex(tobj, const_cast<GLdouble*>(d), (GLvoid *)d);
                }

            gluTessEndContour(tobj);
        }
    gluTessEndPolygon(tobj);

    gluDeleteTess(tobj);
}

#endif
