/*
 * FTGL - OpenGL font library
 *
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

#include <string>

#include "FTGL/ftgl.h"

#include "FTInternals.h"
#include "FTBufferGlyphImpl.h"


//
//  FTGLBufferGlyph
//


FTBufferGlyph::FTBufferGlyph(FT_GlyphSlot glyph, FTBuffer *buffer) :
    FTGlyph(new FTBufferGlyphImpl(glyph, buffer))
{}


FTBufferGlyph::~FTBufferGlyph()
{}


const FTPoint& FTBufferGlyph::Render(const FTPoint& pen, int renderMode)
{
    FTBufferGlyphImpl *myimpl = dynamic_cast<FTBufferGlyphImpl *>(impl);
    return myimpl->RenderImpl(pen, renderMode);
}


//
//  FTGLBufferGlyphImpl
//


FTBufferGlyphImpl::FTBufferGlyphImpl(FT_GlyphSlot glyph, FTBuffer *p)
:   FTGlyphImpl(glyph),
    has_bitmap(false),
    buffer(p)
{
    err = FT_Render_Glyph(glyph, FT_RENDER_MODE_NORMAL);
    if(err || glyph->format != ft_glyph_format_bitmap)
    {
        return;
    }

    bitmap = glyph->bitmap;
    pixels = new unsigned char[bitmap.pitch * bitmap.rows];
    memcpy(pixels, bitmap.buffer, bitmap.pitch * bitmap.rows);

    if(bitmap.width && bitmap.rows)
    {
        has_bitmap = true;
        corner = FTPoint(glyph->bitmap_left, glyph->bitmap_top);
    }
}


FTBufferGlyphImpl::~FTBufferGlyphImpl()
{
    delete[] pixels;
}


const FTPoint& FTBufferGlyphImpl::RenderImpl(const FTPoint& pen, int renderMode)
{
    (void)renderMode;

    if(has_bitmap)
    {
        FTPoint pos(buffer->Pos() + pen + corner);
        int dx = (int)(pos.Xf() + 0.5f);
        int dy = buffer->Height() - (int)(pos.Yf() + 0.5f);
        unsigned char * dest = buffer->Pixels() + dx + dy * buffer->Width();

        for(int y = 0; y < bitmap.rows; y++)
        {
            // FIXME: change the loop bounds instead of doing this test
            if(y + dy < 0 || y + dy >= buffer->Height()) continue;

            if(bitmap.num_grays == 1)
            {
                for(int x = 0; x < bitmap.width; x++)
                {
                    if(x + dx < 0 || x + dx >= buffer->Width()) continue;

                    unsigned char p = pixels[y * bitmap.pitch + x / 8];

                    if((p << (x % 8)) & 0x80)
                    {
                        dest[y * buffer->Width() + x] = 255;
                    }
                }
            }
            else
            {
                for(int x = 0; x < bitmap.width; x++)
                {
                    if(x + dx < 0 || x + dx >= buffer->Width()) continue;

                    unsigned char p = pixels[y * bitmap.pitch + x];

                    if(p)
                    {
                        dest[y * buffer->Width() + x] = p;
                    }
                }
            }
        }
    }

    return advance;
}

