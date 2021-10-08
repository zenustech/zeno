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

#include "FTGL/ftgl.h"

#include "FTInternals.h"
#include "FTBitmapFontImpl.h"


//
//  FTBitmapFont
//


FTBitmapFont::FTBitmapFont(char const *fontFilePath) :
    FTFont(new FTBitmapFontImpl(this, fontFilePath))
{}


FTBitmapFont::FTBitmapFont(unsigned char const *pBufferBytes,
                           size_t bufferSizeInBytes) :
    FTFont(new FTBitmapFontImpl(this, pBufferBytes, bufferSizeInBytes))
{}


FTBitmapFont::~FTBitmapFont()
{}


FTGlyph* FTBitmapFont::MakeGlyph(FT_GlyphSlot ftGlyph)
{
    return new FTBitmapGlyph(ftGlyph);
}


//
//  FTBitmapFontImpl
//


template <typename T>
inline FTPoint FTBitmapFontImpl::RenderI(const T* string, const int len,
                                         FTPoint position, FTPoint spacing,
                                         int renderMode)
{
    // Protect glPixelStorei() calls (also in FTBitmapGlyphImpl::RenderImpl)
    glPushClientAttrib(GL_CLIENT_PIXEL_STORE_BIT);

    glPixelStorei(GL_UNPACK_LSB_FIRST, GL_FALSE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    FTPoint tmp = FTFontImpl::Render(string, len,
                                     position, spacing, renderMode);

    glPopClientAttrib();

    return tmp;
}


FTPoint FTBitmapFontImpl::Render(const char * string, const int len,
                                 FTPoint position, FTPoint spacing,
                                 int renderMode)
{
    return RenderI(string, len, position, spacing, renderMode);
}


FTPoint FTBitmapFontImpl::Render(const wchar_t * string, const int len,
                                 FTPoint position, FTPoint spacing,
                                 int renderMode)
{
    return RenderI(string, len, position, spacing, renderMode);
}

