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

#include <math.h>

#include "FTGL/ftgl.h"

#include "FTInternals.h"
#include "FTPixmapGlyphImpl.h"
#include "FTBitmapGlyphImpl.h"


//
//  FTGLPixmapGlyph
//


FTPixmapGlyph::FTPixmapGlyph(FT_GlyphSlot glyph) :
    FTGlyph(NewImpl(glyph))
{}


FTPixmapGlyph::~FTPixmapGlyph()
{}


FTGlyphImpl *FTPixmapGlyph::NewImpl(FT_GlyphSlot glyph)
{
  FTPixmapGlyphImpl *Impl = new FTPixmapGlyphImpl(glyph);
  if (Impl->destWidth && Impl->destHeight)
    return Impl;
  delete Impl;
  return new FTBitmapGlyphImpl(glyph);
}


const FTPoint& FTPixmapGlyph::Render(const FTPoint& pen, int renderMode)
{
    FTPixmapGlyphImpl *myimpl = dynamic_cast<FTPixmapGlyphImpl *>(impl);
    if (myimpl)
      return myimpl->RenderImpl(pen, renderMode);
    FTBitmapGlyphImpl *myimpl_bitmap = dynamic_cast<FTBitmapGlyphImpl *>(impl);
    return myimpl_bitmap->RenderImpl(pen, renderMode);
}


//
//  FTGLPixmapGlyphImpl
//


FTPixmapGlyphImpl::FTPixmapGlyphImpl(FT_GlyphSlot glyph)
:   FTGlyphImpl(glyph),
    destWidth(0),
    destHeight(0),
    data(0)
{
    err = FT_Render_Glyph(glyph, FT_RENDER_MODE_NORMAL);
    if(err || ft_glyph_format_bitmap != glyph->format || glyph->bitmap.num_grays == 1)
    {
        return;
    }

    FT_Bitmap bitmap = glyph->bitmap;

    //check the pixel mode
    //ft_pixel_mode_grays

    int srcWidth = bitmap.width;
    int srcHeight = bitmap.rows;

    destWidth = srcWidth;
    destHeight = srcHeight;

    if(destWidth && destHeight)
    {
        data = new unsigned char[destWidth * destHeight * 2];
        unsigned char* src = bitmap.buffer;

        unsigned char* dest = data + ((destHeight - 1) * destWidth * 2);
        size_t destStep = destWidth * 2 * 2;

        if (FT_PIXEL_MODE_MONO == bitmap.pixel_mode)
        {
            // Convert the 1 bpp bitmap to 8 bpp
            for(int y = 0; y < srcHeight; ++y)
            {
                for(int x = 0; x < srcWidth; ++x)
                {
                    *dest++ = static_cast<unsigned char>(255);
                    // Store 255 if bit (x % 8) is set, store 0 otherwise
                    *dest++ = static_cast<unsigned char>(static_cast<signed char>(src[static_cast<unsigned int>(x) >> 3] << (x & 7)) >> 7);
                }
                dest -= destStep;
                src += bitmap.pitch;
            }
        }
        else
        {
            for(int y = 0; y < srcHeight; ++y)
            {
                for(int x = 0; x < srcWidth; ++x)
                {
                    *dest++ = static_cast<unsigned char>(255);
                    *dest++ = *src++;
                }
                dest -= destStep;
            }
        }

        destHeight = srcHeight;
    }

    pos.X(glyph->bitmap_left);
    pos.Y(srcHeight - glyph->bitmap_top);
}


FTPixmapGlyphImpl::~FTPixmapGlyphImpl()
{
    delete [] data;
}


const FTPoint& FTPixmapGlyphImpl::RenderImpl(const FTPoint& pen,
                                             int renderMode)
{
    if(data)
    {
        float dx, dy;

        dx = floor(pen.Xf() + pos.Xf());
        dy = floor(pen.Yf() - pos.Yf());

        glBitmap(0, 0, 0.0f, 0.0f, dx, dy, (const GLubyte*)0);
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 2);

        glDrawPixels(destWidth, destHeight, GL_LUMINANCE_ALPHA,
                     GL_UNSIGNED_BYTE, (const GLvoid*)data);
        glBitmap(0, 0, 0.0f, 0.0f, -dx, -dy, (const GLubyte*)0);
    }

    return advance;
}

