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
#include "FTGlyphImpl.h"


//
//  FTGlyph
//


FTGlyph::FTGlyph(FT_GlyphSlot glyph)
{
    impl = new FTGlyphImpl(glyph);
}


FTGlyph::FTGlyph(FTGlyphImpl *pImpl)
{
    impl = pImpl;
}


FTGlyph::~FTGlyph()
{
    delete impl;
}


float FTGlyph::Advance() const
{
    return impl->Advance();
}


const FTBBox& FTGlyph::BBox() const
{
    return impl->BBox();
}


FT_Error FTGlyph::Error() const
{
    return impl->Error();
}


//
//  FTGlyphImpl
//


FTGlyphImpl::FTGlyphImpl(FT_GlyphSlot glyph, bool useList) : err(0)
{
    (void)useList;

    if(glyph)
    {
        bBox = FTBBox(glyph);
        advance = FTPoint(glyph->advance.x / 64.0f,
                          glyph->advance.y / 64.0f);
    }
}


FTGlyphImpl::~FTGlyphImpl()
{}


float FTGlyphImpl::Advance() const
{
    return advance.Xf();
}


const FTBBox& FTGlyphImpl::BBox() const
{
    return bBox;
}


FT_Error FTGlyphImpl::Error() const
{
    return err;
}

