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

#include "FTGL/ftgl.h"

#include "FTInternals.h"
#include "FTOutlineGlyphImpl.h"
#include "FTVectoriser.h"


//
//  FTGLOutlineGlyph
//


FTOutlineGlyph::FTOutlineGlyph(FT_GlyphSlot glyph, float outset,
                               bool useDisplayList) :
    FTGlyph(new FTOutlineGlyphImpl(glyph, outset, useDisplayList))
{}


FTOutlineGlyph::~FTOutlineGlyph()
{}


const FTPoint& FTOutlineGlyph::Render(const FTPoint& pen, int renderMode)
{
    FTOutlineGlyphImpl *myimpl = dynamic_cast<FTOutlineGlyphImpl *>(impl);
    return myimpl->RenderImpl(pen, renderMode);
}


//
//  FTGLOutlineGlyphImpl
//


FTOutlineGlyphImpl::FTOutlineGlyphImpl(FT_GlyphSlot glyph, float _outset,
                                       bool useDisplayList)
:   FTGlyphImpl(glyph),
    vectoriser(0),
    glList(0)
{
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

    outset = _outset;

    if(useDisplayList)
    {
        glList = glGenLists(1);
        glNewList(glList, GL_COMPILE);

        DoRender();

        glEndList();

        delete vectoriser;
        vectoriser = NULL;
    }
}


FTOutlineGlyphImpl::~FTOutlineGlyphImpl()
{
    if(glList)
    {
        glDeleteLists(glList, 1);
    }
    else if(vectoriser)
    {
        delete vectoriser;
    }
}


const FTPoint& FTOutlineGlyphImpl::RenderImpl(const FTPoint& pen,
                                              int renderMode)
{
    (void)renderMode;

    glTranslatef(pen.Xf(), pen.Yf(), pen.Zf());
    if(glList)
    {
        glCallList(glList);
    }
    else if(vectoriser)
    {
        DoRender();
    }
    glTranslatef(-pen.Xf(), -pen.Yf(), -pen.Zf());

    return advance;
}


void FTOutlineGlyphImpl::DoRender()
{
    for(unsigned int c = 0; c < vectoriser->ContourCount(); ++c)
    {
        const FTContour* contour = vectoriser->Contour(c);

        glBegin(GL_LINE_LOOP);
            for(unsigned int i = 0; i < contour->PointCount(); ++i)
            {
                FTPoint point = FTPoint(contour->Point(i).X() + contour->Outset(i).X() * outset,
                                        contour->Point(i).Y() + contour->Outset(i).Y() * outset,
                                        0);
                glVertex2f(point.Xf() / 64.0f, point.Yf() / 64.0f);
            }
        glEnd();
    }
}

