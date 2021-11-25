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
#include "FTPixmapFontImpl.h"


//
//  FTPixmapFont
//


FTPixmapFont::FTPixmapFont(char const *fontFilePath) :
    FTFont(new FTPixmapFontImpl(this, fontFilePath))
{}


FTPixmapFont::FTPixmapFont(const unsigned char *pBufferBytes,
                           size_t bufferSizeInBytes) :
    FTFont(new FTPixmapFontImpl(this, pBufferBytes, bufferSizeInBytes))
{}


FTPixmapFont::~FTPixmapFont()
{}


FTGlyph* FTPixmapFont::MakeGlyph(FT_GlyphSlot ftGlyph)
{
    return new FTPixmapGlyph(ftGlyph);
}


//
//  FTPixmapFontImpl
//


FTPixmapFontImpl::FTPixmapFontImpl(FTFont *ftFont, const char* fontFilePath)
: FTFontImpl(ftFont, fontFilePath)
{
    load_flags = FT_LOAD_NO_HINTING | FT_LOAD_NO_BITMAP;
}


FTPixmapFontImpl::FTPixmapFontImpl(FTFont *ftFont,
                                   const unsigned char *pBufferBytes,
                                   size_t bufferSizeInBytes)
: FTFontImpl(ftFont, pBufferBytes, bufferSizeInBytes)
{
    load_flags = FT_LOAD_NO_HINTING | FT_LOAD_NO_BITMAP;
}


template <typename T>
inline FTPoint FTPixmapFontImpl::RenderI(const T* string, const int len,
                                         FTPoint position, FTPoint spacing,
                                         int renderMode)
{
    // Protect GL_TEXTURE_2D and glPixelTransferf()
    glPushAttrib(GL_ENABLE_BIT | GL_PIXEL_MODE_BIT | GL_COLOR_BUFFER_BIT
                  | GL_POLYGON_BIT);

    // Protect glPixelStorei() calls (made by FTPixmapGlyphImpl::RenderImpl).
    glPushClientAttrib(GL_CLIENT_PIXEL_STORE_BIT);

    // Needed on OSX
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glDisable(GL_TEXTURE_2D);

    GLfloat ftglColour[4];
    glGetFloatv(GL_CURRENT_RASTER_COLOR, ftglColour);

    glPixelTransferf(GL_RED_SCALE, ftglColour[0]);
    glPixelTransferf(GL_GREEN_SCALE, ftglColour[1]);
    glPixelTransferf(GL_BLUE_SCALE, ftglColour[2]);
    glPixelTransferf(GL_ALPHA_SCALE, ftglColour[3]);

    FTPoint tmp = FTFontImpl::Render(string, len,
                                     position, spacing, renderMode);

    glPopClientAttrib();
    glPopAttrib();

    return tmp;
}


FTPoint FTPixmapFontImpl::Render(const char * string, const int len,
                                 FTPoint position, FTPoint spacing,
                                 int renderMode)
{
    return RenderI(string, len, position, spacing, renderMode);
}


FTPoint FTPixmapFontImpl::Render(const wchar_t * string, const int len,
                                 FTPoint position, FTPoint spacing,
                                 int renderMode)
{
    return RenderI(string, len, position, spacing, renderMode);
}

