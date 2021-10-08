#if 0
/*
 * FTGL - OpenGL font library
 *
 * Copyright (c) 2011 Richard Ulrich <richi@paraeasy.ch>
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
#include "FTTriangleExtractorGlyphImpl.h"
#include "FTVectoriser.h"
// std lib
#include <cassert>

//
//  FTGLTriangleExtractorGlyph
//


FTTriangleExtractorGlyph::FTTriangleExtractorGlyph(FT_GlyphSlot glyph, float outset,
                               std::vector<float>& triangles) :
    FTGlyph(new FTTriangleExtractorGlyphImpl(glyph, outset, triangles))
{}


FTTriangleExtractorGlyph::~FTTriangleExtractorGlyph()
{}


const FTPoint& FTTriangleExtractorGlyph::Render(const FTPoint& pen, int renderMode)
{
    FTTriangleExtractorGlyphImpl *myimpl = dynamic_cast<FTTriangleExtractorGlyphImpl *>(impl);
    return myimpl->RenderImpl(pen, renderMode);
}


//
//  FTGLTriangleExtractorGlyphImpl
//


FTTriangleExtractorGlyphImpl::FTTriangleExtractorGlyphImpl(FT_GlyphSlot glyph, float _outset,
                                       std::vector<float>& triangles)
:   FTGlyphImpl(glyph),
    triangles_(triangles)
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


    hscale = glyph->face->size->metrics.x_ppem * 64;
    vscale = glyph->face->size->metrics.y_ppem * 64;
    outset = _outset;
}


FTTriangleExtractorGlyphImpl::~FTTriangleExtractorGlyphImpl()
{
    delete vectoriser;
}


const FTPoint& FTTriangleExtractorGlyphImpl::RenderImpl(const FTPoint& pen,
                                              int renderMode)
{
    (void)renderMode;

	if(NULL == vectoriser)
		return advance;

    vectoriser->MakeMesh(1.0, 1, outset);
    const FTMesh *mesh = vectoriser->GetMesh();

    for(unsigned int t = 0; t < mesh->TesselationCount(); ++t)
    {
        const FTTesselation* subMesh = mesh->Tesselation(t);
        const unsigned int polygonType = subMesh->PolygonType();

        // convert everything to a single triangle strip.
        // In some cases we have to insert invalid triangles to make a valid strip
        switch(polygonType)
        {
            case GL_TRIANGLE_STRIP:
                AddVertex(pen, subMesh->Point(0));
                for(unsigned int i = 0; i < subMesh->PointCount(); ++i)
                    AddVertex(pen, subMesh->Point(i));
                AddVertex(pen, subMesh->Point(subMesh->PointCount() - 1));
                break;
            case GL_TRIANGLES:
                assert(subMesh->PointCount() % 3 == 0);
                for(unsigned int i = 0; i < subMesh->PointCount(); i += 3)
                {
                    AddVertex(pen, subMesh->Point(i));
                    AddVertex(pen, subMesh->Point(i));
                    AddVertex(pen, subMesh->Point(i+1));
                    AddVertex(pen, subMesh->Point(i+2));
                    AddVertex(pen, subMesh->Point(i+2));
                }
                break;
            case GL_TRIANGLE_FAN:
            {
                const FTPoint& centerPoint = subMesh->Point(0);
                AddVertex(pen, centerPoint);

                for(unsigned int i = 1; i < subMesh->PointCount()-1; ++i)
                {
                    AddVertex(pen, centerPoint);
                    AddVertex(pen, subMesh->Point(i));
                    AddVertex(pen, subMesh->Point(i+1));
                    AddVertex(pen, centerPoint);
                }
                AddVertex(pen, centerPoint);
                break;
            }
            default:
                assert(!"please implement...");
            ;
        }


    }

    return advance;
}

void FTTriangleExtractorGlyphImpl::AddVertex(const FTPoint& pen, const FTPoint& point)
{
    triangles_.push_back(pen.Xf() + point.Xf() / 64.0);
    triangles_.push_back(pen.Yf() + point.Yf() / 64.0);
    triangles_.push_back(pen.Zf());
}

#endif
