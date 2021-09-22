/*
 * FTGL - OpenGL font library
 *
 * Copyright (c) 2001-2004 Henry Maddocks <ftgl@opengl.geek.nz>
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

#include "FTSize.h"


FTSize::FTSize()
:   ftFace(0),
    ftSize(0),
    size(0),
    xResolution(0),
    yResolution(0),
    err(0)
{}


FTSize::~FTSize()
{}


bool FTSize::CharSize(FT_Face* face, unsigned int pointSize, unsigned int xRes, unsigned int yRes)
{
    if(size != pointSize || xResolution != xRes || yResolution != yRes)
    {
        err = FT_Set_Char_Size(*face, 0L, pointSize * 64, xResolution, yResolution);

        if(!err)
        {
            ftFace = face;
            size = pointSize;
            xResolution = xRes;
            yResolution = yRes;
            ftSize = (*ftFace)->size;
        }
    }

    return !err;
}


unsigned int FTSize::CharSize() const
{
    return size;
}


float FTSize::Ascender() const
{
    return ftSize == 0 ? 0.0f : static_cast<float>(ftSize->metrics.ascender) / 64.0f;
}


float FTSize::Descender() const
{
    return ftSize == 0 ? 0.0f : static_cast<float>(ftSize->metrics.descender) / 64.0f;
}


float FTSize::Height() const
{
    if(0 == ftSize)
    {
        return 0.0f;
    }

    if(FT_IS_SCALABLE((*ftFace)))
    {
        return ((*ftFace)->bbox.yMax - (*ftFace)->bbox.yMin) * ((float)ftSize->metrics.y_ppem / (float)(*ftFace)->units_per_EM);
    }
    else
    {
        return static_cast<float>(ftSize->metrics.height) / 64.0f;
    }
}


float FTSize::Width() const
{
    if(0 == ftSize)
    {
        return 0.0f;
    }

    if(FT_IS_SCALABLE((*ftFace)))
    {
        return ((*ftFace)->bbox.xMax - (*ftFace)->bbox.xMin) * (static_cast<float>(ftSize->metrics.x_ppem) / static_cast<float>((*ftFace)->units_per_EM));
    }
    else
    {
        return static_cast<float>(ftSize->metrics.max_advance) / 64.0f;
    }
}


float FTSize::Underline() const
{
    return 0.0f;
}

