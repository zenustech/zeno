/*
 * FTGL - OpenGL font library
 *
 * Copyright (c) 2001-2004 Henry Maddocks <ftgl@opengl.geek.nz>
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

#include <iostream>

#include "FTGL/ftgl.h"

#include "FTInternals.h"
#include "FTExtrudeGlyphImpl.h"
#include "FTVectoriser.h"


//
//  FTGLExtrudeGlyph
//


FTExtrudeGlyph::FTExtrudeGlyph(FT_GlyphSlot glyph, float depth,
                               float frontOutset, float backOutset,
                               bool useDisplayList) :
    FTGlyph(new FTExtrudeGlyphImpl(glyph, depth, frontOutset, backOutset,
                                   useDisplayList))
{}


FTExtrudeGlyph::~FTExtrudeGlyph()
{}


const FTPoint& FTExtrudeGlyph::Render(const FTPoint& pen, int renderMode)
{
    FTExtrudeGlyphImpl *myimpl = dynamic_cast<FTExtrudeGlyphImpl *>(impl);
    return myimpl->RenderImpl(pen, renderMode);
}


//
//  FTGLExtrudeGlyphImpl
//


FTExtrudeGlyphImpl::FTExtrudeGlyphImpl(FT_GlyphSlot glyph, float _depth,
                                       float _frontOutset, float _backOutset,
                                       bool useDisplayList)
:   FTGlyphImpl(glyph),
    vectoriser(0),
    glList(0)
{
    bBox.SetDepth(-_depth);

    if(ft_glyph_format_outline != glyph->format)
    {
        err = 0x14; // Invalid_Outline
        return;
    }

    vectoriser = new FTVectoriser(glyph);

    if((vectoriser->ContourCount() < 1) || (vectoriser->PointCount() < 3))
    {
        delete vectoriser;
        vectoriser = NULL;
        return;
    }

    hscale = glyph->face->size->metrics.x_ppem * 64;
    vscale = glyph->face->size->metrics.y_ppem * 64;
    depth = _depth;
    frontOutset = _frontOutset;
    backOutset = _backOutset;

    if(useDisplayList)
    {
        glList = glGenLists(3);

        /* Front face */
        glNewList(glList + 0, GL_COMPILE);
        RenderFront();
        glEndList();

        /* Back face */
        glNewList(glList + 1, GL_COMPILE);
        RenderBack();
        glEndList();

        /* Side face */
        glNewList(glList + 2, GL_COMPILE);
        RenderSide();
        glEndList();

        delete vectoriser;
        vectoriser = NULL;
    }
}


FTExtrudeGlyphImpl::~FTExtrudeGlyphImpl()
{
    if(glList)
    {
        glDeleteLists(glList, 3);
    }
    else if(vectoriser)
    {
        delete vectoriser;
    }
}


const FTPoint& FTExtrudeGlyphImpl::RenderImpl(const FTPoint& pen,
                                              int renderMode)
{
    glTranslatef(pen.Xf(), pen.Yf(), pen.Zf());
    if(glList)
    {
        if(renderMode & FTGL::RENDER_FRONT)
            glCallList(glList + 0);
        if(renderMode & FTGL::RENDER_BACK)
            glCallList(glList + 1);
        if(renderMode & FTGL::RENDER_SIDE)
            glCallList(glList + 2);
    }
    else if(vectoriser)
    {
        if(renderMode & FTGL::RENDER_FRONT)
            RenderFront();
        if(renderMode & FTGL::RENDER_BACK)
            RenderBack();
        if(renderMode & FTGL::RENDER_SIDE)
            RenderSide();
    }
    glTranslatef(-pen.Xf(), -pen.Yf(), -pen.Zf());

    return advance;
}


void FTExtrudeGlyphImpl::RenderFront()
{
    vectoriser->MakeMesh(1.0, 1, frontOutset);
    glNormal3d(0.0, 0.0, 1.0);

    const FTMesh *mesh = vectoriser->GetMesh();
    for(unsigned int j = 0; j < mesh->TesselationCount(); ++j)
    {
        const FTTesselation* subMesh = mesh->Tesselation(j);
        unsigned int polygonType = subMesh->PolygonType();

        glBegin(polygonType);
            for(unsigned int i = 0; i < subMesh->PointCount(); ++i)
            {
                FTPoint pt = subMesh->Point(i);

                glTexCoord2f(pt.Xf() / hscale,
                             pt.Yf() / vscale);

                glVertex3f(pt.Xf() / 64.0f,
                           pt.Yf() / 64.0f,
                           0.0f);
            }
        glEnd();
    }
}


void FTExtrudeGlyphImpl::RenderBack()
{
    vectoriser->MakeMesh(-1.0, 2, backOutset);
    glNormal3d(0.0, 0.0, -1.0);

    const FTMesh *mesh = vectoriser->GetMesh();
    for(unsigned int j = 0; j < mesh->TesselationCount(); ++j)
    {
        const FTTesselation* subMesh = mesh->Tesselation(j);
        unsigned int polygonType = subMesh->PolygonType();

        glBegin(polygonType);
            for(unsigned int i = 0; i < subMesh->PointCount(); ++i)
            {
                FTPoint pt = subMesh->Point(i);

                glTexCoord2f(subMesh->Point(i).Xf() / hscale,
                             subMesh->Point(i).Yf() / vscale);

                glVertex3f(subMesh->Point(i).Xf() / 64.0f,
                           subMesh->Point(i).Yf() / 64.0f,
                           -depth);
            }
        glEnd();
    }
}


void FTExtrudeGlyphImpl::RenderSide()
{
    int contourFlag = vectoriser->ContourFlag();

    for(size_t c = 0; c < vectoriser->ContourCount(); ++c)
    {
        const FTContour* contour = vectoriser->Contour(c);
        size_t n = contour->PointCount();

        if(n < 2)
        {
            continue;
        }

        glBegin(GL_QUAD_STRIP);
            for(size_t j = 0; j <= n; ++j)
            {
                size_t cur = (j == n) ? 0 : j;
                size_t next = (cur == n - 1) ? 0 : cur + 1;

                FTPoint frontPt = contour->FrontPoint(cur);
                FTPoint nextPt = contour->FrontPoint(next);
                FTPoint backPt = contour->BackPoint(cur);

                FTPoint normal = FTPoint(0.f, 0.f, 1.f) ^ (frontPt - nextPt);
                if(normal != FTPoint(0.0f, 0.0f, 0.0f))
                {
                    glNormal3dv(static_cast<const FTGL_DOUBLE*>(normal.Normalise()));
                }

                glTexCoord2f(frontPt.Xf() / hscale, frontPt.Yf() / vscale);

                if(contourFlag & ft_outline_reverse_fill)
                {
                    glVertex3f(backPt.Xf() / 64.0f, backPt.Yf() / 64.0f, 0.0f);
                    glVertex3f(frontPt.Xf() / 64.0f, frontPt.Yf() / 64.0f, -depth);
                }
                else
                {
                    glVertex3f(backPt.Xf() / 64.0f, backPt.Yf() / 64.0f, -depth);
                    glVertex3f(frontPt.Xf() / 64.0f, frontPt.Yf() / 64.0f, 0.0f);
                }
            }
        glEnd();
    }
}

