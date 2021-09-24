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
#include "FTTextureGlyphImpl.h"

#define FTGL_ASSERTS_SHOULD_SOFT_FAIL

#ifdef FTGL_ASSERTS_SHOULD_SOFT_FAIL
#   define FTASSERT_FAIL do {} while(0)
#else
#   define FTASSERT_FAIL do { int *a = (int*)0x0; *a = 0xD15EA5ED; } while(0)
#endif

#define FTASSERT(x) \
    if (!(x)) \
    { \
        static int count = 0; \
        if (count++ < 8) \
            fprintf(stderr, "ASSERTION FAILED (%s:%d): %s\n", \
                    __FILE__, __LINE__, #x); \
        FTASSERT_FAIL; \
        if (count == 8) \
            fprintf(stderr, "\\__ last warning for this assertion\n"); \
    }


//
//  FTGLTextureGlyph
//

FTTextureGlyph::FTTextureGlyph(FT_GlyphSlot glyph, int id, int xOffset,
                               int yOffset, int width, int height) :
    FTGlyph(new FTTextureGlyphImpl(glyph, id, xOffset, yOffset, width, height))
{}


FTTextureGlyph::~FTTextureGlyph()
{}


const FTPoint& FTTextureGlyph::Render(const FTPoint& pen, int renderMode)
{
    FTTextureGlyphImpl *myimpl = dynamic_cast<FTTextureGlyphImpl *>(impl);
    return myimpl->RenderImpl(pen, renderMode);
}


//
//  FTGLTextureGlyphImpl
//

GLint FTTextureGlyphImpl::activeTextureID = 0;

FTTextureGlyphImpl::FTTextureGlyphImpl(FT_GlyphSlot glyph, int id, int xOffset,
                                       int yOffset, int width, int height)
:   FTGlyphImpl(glyph),
    destWidth(0),
    destHeight(0),
    glTextureID(id)
{
    /* FIXME: need to propagate the render mode all the way down to
     * here in order to get FT_RENDER_MODE_MONO aliased fonts.
     */

    err = FT_Render_Glyph(glyph, FT_RENDER_MODE_NORMAL);
    if(err || glyph->format != ft_glyph_format_bitmap)
    {
        return;
    }

    FT_Bitmap      bitmap = glyph->bitmap;

    destWidth  = bitmap.width;
    destHeight = bitmap.rows;

    if(destWidth && destHeight)
    {
        glPushClientAttrib(GL_CLIENT_PIXEL_STORE_BIT);

        glPixelStorei(GL_UNPACK_LSB_FIRST, GL_FALSE);
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

        GLint w,h;

        glBindTexture(GL_TEXTURE_2D, glTextureID);
        glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &w);
        glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &h);

        FTASSERT(xOffset >= 0);
        FTASSERT(yOffset >= 0);
        FTASSERT(destWidth >= 0);
        FTASSERT(destHeight >= 0);
        FTASSERT(xOffset + destWidth <= w);
        FTASSERT(yOffset + destHeight <= h);

        if (yOffset + destHeight > h)
        {
            // We'll only get here if we are soft-failing our asserts. In that
            // case, since the data we're trying to put into our texture is
            // too long, we'll only copy a portion of the image.
            destHeight = h - yOffset;
        }
        if (destHeight >= 0)
        {
            // convert bitmap fonts
            std::vector <unsigned char> data_converted;
            if(bitmap.num_grays == 1)
            {
                bBox = FTBBox(0, 0, 0, destWidth, destHeight, 0);
                data_converted.resize(destWidth * destHeight, 0);
                int n = 0;
                for(int y = 0; y < destHeight; ++y)
                {
                    unsigned char* src = bitmap.pitch < 0
                      ? bitmap.buffer + (y - destHeight + 1) * bitmap.pitch
                      : bitmap.buffer + y * bitmap.pitch;
                    unsigned char c;
                    for(int x = 0; x < destWidth; ++x)
                    {
                        if (x % 8 == 0)
                          c = *src++;
                        data_converted[n++] = ((c >> (7 - (x % 8))) & 1) * 255;
                    }
                }
            }

            glTexSubImage2D(GL_TEXTURE_2D, 0, xOffset, yOffset,
                            destWidth, destHeight, GL_ALPHA, GL_UNSIGNED_BYTE,
                            !data_converted.empty() ? data_converted.data()
                                                    : bitmap.buffer);
        }

        glPopClientAttrib();
    }

//      0
//      +----+
//      |    |
//      |    |
//      |    |
//      +----+
//           1

    uv[0].X(static_cast<float>(xOffset) / static_cast<float>(width));
    uv[0].Y(static_cast<float>(yOffset) / static_cast<float>(height));
    uv[1].X(static_cast<float>(xOffset + destWidth) / static_cast<float>(width));
    uv[1].Y(static_cast<float>(yOffset + destHeight) / static_cast<float>(height));

    corner = FTPoint(glyph->bitmap_left, glyph->bitmap_top);
}


FTTextureGlyphImpl::~FTTextureGlyphImpl()
{}


const FTPoint& FTTextureGlyphImpl::RenderImpl(const FTPoint& pen,
                                              int renderMode)
{
    float dx, dy;

    if(activeTextureID != glTextureID)
    {
        glBindTexture(GL_TEXTURE_2D, (GLuint)glTextureID);
        activeTextureID = glTextureID;
    }

    dx = floor(pen.Xf() + corner.Xf());
    dy = floor(pen.Yf() + corner.Yf());

    glBegin(GL_QUADS);
        glTexCoord2f(uv[0].Xf(), uv[0].Yf());
        glVertex3f(dx, dy, pen.Zf());

        glTexCoord2f(uv[0].Xf(), uv[1].Yf());
        glVertex3f(dx, dy - destHeight, pen.Zf());

        glTexCoord2f(uv[1].Xf(), uv[1].Yf());
        glVertex3f(dx + destWidth, dy - destHeight, pen.Zf());

        glTexCoord2f(uv[1].Xf(), uv[0].Yf());
        glVertex3f(dx + destWidth, dy, pen.Zf());
    glEnd();

    return advance;
}

