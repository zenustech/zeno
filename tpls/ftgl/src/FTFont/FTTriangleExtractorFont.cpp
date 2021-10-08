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
#include "FTTriangleExtractorFontImpl.h"


//
//  FTTriangleExtractorFont
//


FTTriangleExtractorFont::FTTriangleExtractorFont(char const *fontFilePath, std::vector<float>& triangles) :
    FTFont(new FTTriangleExtractorFontImpl(this, fontFilePath, triangles))
{}


FTTriangleExtractorFont::FTTriangleExtractorFont(const unsigned char *pBufferBytes,
                             size_t bufferSizeInBytes, std::vector<float>& triangles) :
    FTFont(new FTTriangleExtractorFontImpl(this, pBufferBytes, bufferSizeInBytes, triangles))
{}


FTTriangleExtractorFont::~FTTriangleExtractorFont()
{}


FTGlyph* FTTriangleExtractorFont::MakeGlyph(FT_GlyphSlot ftGlyph)
{
    FTTriangleExtractorFontImpl *myimpl = dynamic_cast<FTTriangleExtractorFontImpl*>(impl);
    if(!myimpl)
    {
        return NULL;
    }

    return new FTTriangleExtractorGlyph(ftGlyph, myimpl->outset,
                              myimpl->triangles_);
}

//
//  FTTriangleExtractorFontImpl
//


FTTriangleExtractorFontImpl::FTTriangleExtractorFontImpl(FTFont *ftFont, const char* fontFilePath, std::vector<float>& triangles)
: FTFontImpl(ftFont, fontFilePath),
  outset(0.0f),
  triangles_(triangles)
{
    load_flags = FT_LOAD_NO_HINTING;
}


FTTriangleExtractorFontImpl::FTTriangleExtractorFontImpl(FTFont *ftFont,
                                     const unsigned char *pBufferBytes,
                                     size_t bufferSizeInBytes, std::vector<float>& triangles)
: FTFontImpl(ftFont, pBufferBytes, bufferSizeInBytes),
  outset(0.0f),
  triangles_(triangles)
{
    load_flags = FT_LOAD_NO_HINTING;
}


template <typename T>
inline FTPoint FTTriangleExtractorFontImpl::RenderI(const T* string, const int len,
                                          FTPoint position, FTPoint spacing,
                                          int renderMode)
{
    FTPoint tmp = FTFontImpl::Render(string, len,
                                     position, spacing, renderMode);

    return tmp;
}


FTPoint FTTriangleExtractorFontImpl::Render(const char * string, const int len,
                                  FTPoint position, FTPoint spacing,
                                  int renderMode)
{
    return RenderI(string, len, position, spacing, renderMode);
}


FTPoint FTTriangleExtractorFontImpl::Render(const wchar_t * string, const int len,
                                  FTPoint position, FTPoint spacing,
                                  int renderMode)
{
    return RenderI(string, len, position, spacing, renderMode);
}


