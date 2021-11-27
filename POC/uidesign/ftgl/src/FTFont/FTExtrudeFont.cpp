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
#include "FTExtrudeFontImpl.h"


//
//  FTExtrudeFont
//


FTExtrudeFont::FTExtrudeFont(char const *fontFilePath) :
    FTFont(new FTExtrudeFontImpl(this, fontFilePath))
{}


FTExtrudeFont::FTExtrudeFont(const unsigned char *pBufferBytes,
                             size_t bufferSizeInBytes) :
    FTFont(new FTExtrudeFontImpl(this, pBufferBytes, bufferSizeInBytes))
{}


FTExtrudeFont::~FTExtrudeFont()
{}


FTGlyph* FTExtrudeFont::MakeGlyph(FT_GlyphSlot ftGlyph)
{
    FTExtrudeFontImpl *myimpl = dynamic_cast<FTExtrudeFontImpl *>(impl);
    if(!myimpl)
    {
        return NULL;
    }

    return new FTExtrudeGlyph(ftGlyph, myimpl->depth, myimpl->front,
                              myimpl->back, myimpl->useDisplayLists);
}


//
//  FTExtrudeFontImpl
//


FTExtrudeFontImpl::FTExtrudeFontImpl(FTFont *ftFont, const char* fontFilePath)
: FTFontImpl(ftFont, fontFilePath),
  depth(0.0f), front(0.0f), back(0.0f)
{
    load_flags = FT_LOAD_NO_HINTING;
}


FTExtrudeFontImpl::FTExtrudeFontImpl(FTFont *ftFont,
                                     const unsigned char *pBufferBytes,
                                     size_t bufferSizeInBytes)
: FTFontImpl(ftFont, pBufferBytes, bufferSizeInBytes),
  depth(0.0f), front(0.0f), back(0.0f)
{
    load_flags = FT_LOAD_NO_HINTING;
}

