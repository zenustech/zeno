/*
 * FTGL - OpenGL font library
 *
 * Copyright (c) 2001-2004 Henry Maddocks <ftgl@opengl.geek.nz>
 * Copyright (c) 2008-2010 Sam Hocevar <sam@hocevar.net>
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
#include "FTPolygonFontImpl.h"


//
//  FTPolygonFont
//


FTPolygonFont::FTPolygonFont(char const *fontFilePath) :
    FTFont(new FTPolygonFontImpl(this, fontFilePath))
{}


FTPolygonFont::FTPolygonFont(const unsigned char *pBufferBytes,
                             size_t bufferSizeInBytes) :
    FTFont(new FTPolygonFontImpl(this, pBufferBytes, bufferSizeInBytes))
{}


FTPolygonFont::~FTPolygonFont()
{}


FTGlyph* FTPolygonFont::MakeGlyph(FT_GlyphSlot ftGlyph)
{
    FTPolygonFontImpl *myimpl = dynamic_cast<FTPolygonFontImpl *>(impl);
    if(!myimpl)
    {
        return NULL;
    }

    return new FTPolygonGlyph(ftGlyph, myimpl->outset,
                              myimpl->useDisplayLists);
}


//
//  FTPolygonFontImpl
//


FTPolygonFontImpl::FTPolygonFontImpl(FTFont *ftFont, const char* fontFilePath)
: FTFontImpl(ftFont, fontFilePath),
  outset(0.0f)
{
    load_flags = FT_LOAD_NO_HINTING;
}


FTPolygonFontImpl::FTPolygonFontImpl(FTFont *ftFont,
                                     const unsigned char *pBufferBytes,
                                     size_t bufferSizeInBytes)
: FTFontImpl(ftFont, pBufferBytes, bufferSizeInBytes),
  outset(0.0f)
{
    load_flags = FT_LOAD_NO_HINTING;
}


template <typename T>
inline FTPoint FTPolygonFontImpl::RenderI(const T* string, const int len,
                                          FTPoint position, FTPoint spacing,
                                          int renderMode)
{
    // Protect GL_POLYGON
    glPushAttrib(GL_POLYGON_BIT);

    // Activate front and back face filling. If the caller wants only
    // front face, it can set proper culling.
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    FTPoint tmp = FTFontImpl::Render(string, len,
                                     position, spacing, renderMode);

    glPopAttrib();

    return tmp;
}


FTPoint FTPolygonFontImpl::Render(const char * string, const int len,
                                  FTPoint position, FTPoint spacing,
                                  int renderMode)
{
    return RenderI(string, len, position, spacing, renderMode);
}


FTPoint FTPolygonFontImpl::Render(const wchar_t * string, const int len,
                                  FTPoint position, FTPoint spacing,
                                  int renderMode)
{
    return RenderI(string, len, position, spacing, renderMode);
}

