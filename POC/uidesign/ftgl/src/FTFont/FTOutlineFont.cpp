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
#include "FTOutlineFontImpl.h"


//
//  FTOutlineFont
//


FTOutlineFont::FTOutlineFont(char const *fontFilePath) :
    FTFont(new FTOutlineFontImpl(this, fontFilePath))
{}


FTOutlineFont::FTOutlineFont(const unsigned char *pBufferBytes,
                             size_t bufferSizeInBytes) :
    FTFont(new FTOutlineFontImpl(this, pBufferBytes, bufferSizeInBytes))
{}


FTOutlineFont::~FTOutlineFont()
{}


FTGlyph* FTOutlineFont::MakeGlyph(FT_GlyphSlot ftGlyph)
{
    FTOutlineFontImpl *myimpl = dynamic_cast<FTOutlineFontImpl *>(impl);
    if(!myimpl)
    {
        return NULL;
    }

    return new FTOutlineGlyph(ftGlyph, myimpl->outset,
                              myimpl->useDisplayLists);
}


//
//  FTOutlineFontImpl
//


FTOutlineFontImpl::FTOutlineFontImpl(FTFont *ftFont, const char* fontFilePath)
: FTFontImpl(ftFont, fontFilePath),
  outset(0.0f)
{
    load_flags = FT_LOAD_NO_HINTING;
}


FTOutlineFontImpl::FTOutlineFontImpl(FTFont *ftFont,
                                     const unsigned char *pBufferBytes,
                                     size_t bufferSizeInBytes)
: FTFontImpl(ftFont, pBufferBytes, bufferSizeInBytes),
  outset(0.0f)
{
    load_flags = FT_LOAD_NO_HINTING;
}


template <typename T>
inline FTPoint FTOutlineFontImpl::RenderI(const T* string, const int len,
                                          FTPoint position, FTPoint spacing,
                                          int renderMode)
{
    // Protect GL_TEXTURE_2D, glHint() and GL_LINE_SMOOTH
    glPushAttrib(GL_ENABLE_BIT | GL_HINT_BIT | GL_LINE_BIT
                  | GL_COLOR_BUFFER_BIT);

    glDisable(GL_TEXTURE_2D);
    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_DONT_CARE);

    FTPoint tmp = FTFontImpl::Render(string, len,
                                     position, spacing, renderMode);

    glPopAttrib();

    return tmp;
}


FTPoint FTOutlineFontImpl::Render(const char * string, const int len,
                                  FTPoint position, FTPoint spacing,
                                  int renderMode)
{
    return RenderI(string, len, position, spacing, renderMode);
}


FTPoint FTOutlineFontImpl::Render(const wchar_t * string, const int len,
                                  FTPoint position, FTPoint spacing,
                                  int renderMode)
{
    return RenderI(string, len, position, spacing, renderMode);
}

