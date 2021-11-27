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

#include <wchar.h>

#include "FTGL/ftgl.h"

#include "FTInternals.h"
#include "FTBufferFontImpl.h"


//
//  FTBufferFont
//


FTBufferFont::FTBufferFont(char const *fontFilePath) :
    FTFont(new FTBufferFontImpl(this, fontFilePath))
{}


FTBufferFont::FTBufferFont(unsigned char const *pBufferBytes,
                           size_t bufferSizeInBytes) :
    FTFont(new FTBufferFontImpl(this, pBufferBytes, bufferSizeInBytes))
{}


FTBufferFont::~FTBufferFont()
{}


FTGlyph* FTBufferFont::MakeGlyph(FT_GlyphSlot ftGlyph)
{
    FTBufferFontImpl *myimpl = dynamic_cast<FTBufferFontImpl *>(impl);
    if(!myimpl)
    {
        return NULL;
    }

    return myimpl->MakeGlyphImpl(ftGlyph);
}


//
//  FTBufferFontImpl
//


FTBufferFontImpl::FTBufferFontImpl(FTFont *ftFont, const char* fontFilePath) :
    FTFontImpl(ftFont, fontFilePath),
    buffer(new FTBuffer())
{
    load_flags = FT_LOAD_NO_HINTING | FT_LOAD_NO_BITMAP;

    glGenTextures(BUFFER_CACHE_SIZE, idCache);

    for(int i = 0; i < BUFFER_CACHE_SIZE; i++)
    {
        stringCache[i] = NULL;
        glBindTexture(GL_TEXTURE_2D, idCache[i]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    }

    lastString = 0;
}


FTBufferFontImpl::FTBufferFontImpl(FTFont *ftFont,
                                   const unsigned char *pBufferBytes,
                                   size_t bufferSizeInBytes) :
    FTFontImpl(ftFont, pBufferBytes, bufferSizeInBytes),
    buffer(new FTBuffer())
{
    load_flags = FT_LOAD_NO_HINTING | FT_LOAD_NO_BITMAP;

    glGenTextures(BUFFER_CACHE_SIZE, idCache);

    for(int i = 0; i < BUFFER_CACHE_SIZE; i++)
    {
        stringCache[i] = NULL;
        glBindTexture(GL_TEXTURE_2D, idCache[i]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    }

    lastString = 0;
}


FTBufferFontImpl::~FTBufferFontImpl()
{
    glDeleteTextures(BUFFER_CACHE_SIZE, idCache);

    for(int i = 0; i < BUFFER_CACHE_SIZE; i++)
    {
        if(stringCache[i])
        {
            free(stringCache[i]);
        }
    }

    delete buffer;
}


FTGlyph* FTBufferFontImpl::MakeGlyphImpl(FT_GlyphSlot ftGlyph)
{
    return new FTBufferGlyph(ftGlyph, buffer);
}


bool FTBufferFontImpl::FaceSize(const unsigned int size,
                                const unsigned int res)
{
    for(int i = 0; i < BUFFER_CACHE_SIZE; i++)
    {
        if(stringCache[i])
        {
            free(stringCache[i]);
            stringCache[i] = NULL;
        }
    }

    return FTFontImpl::FaceSize(size, res);
}


static inline GLuint NextPowerOf2(GLuint in)
{
     in -= 1;

     in |= in >> 16;
     in |= in >> 8;
     in |= in >> 4;
     in |= in >> 2;
     in |= in >> 1;

     return in + 1;
}


inline int StringCompare(void const *a, char const *b, int len)
{
    return len < 0 ? strcmp((char const *)a, b)
                   : strncmp((char const *)a, b, len);
}


inline int StringCompare(void const *a, wchar_t const *b, int len)
{
    return len < 0 ? wcscmp((wchar_t const *)a, b)
                   : wcsncmp((wchar_t const *)a, b, len);
}


inline char *StringCopy(char const *s, int len)
{
    if(len < 0)
    {
        return strdup(s);
    }
    else
    {
#ifdef HAVE_STRNDUP
        return strndup(s, len);
#else
        char *s2 = (char*)malloc(len + 1);
        memcpy(s2, s, len);
        s2[len] = 0;
        return s2;
#endif
    }
}


inline wchar_t *StringCopy(wchar_t const *s, int len)
{
    if(len < 0)
    {
#if defined HAVE_WCSDUP
        return wcsdup(s);
#else
        len = (int)wcslen(s);
#endif
    }

    wchar_t *s2 = (wchar_t *)malloc((len + 1) * sizeof(wchar_t));
    memcpy(s2, s, len * sizeof(wchar_t));
    s2[len] = 0;
    return s2;
}


template <typename T>
inline FTPoint FTBufferFontImpl::RenderI(const T* string, const int len,
                                         FTPoint position, FTPoint spacing,
                                         int renderMode)
{
    const float padding = 3.0f;
    int width, height, texWidth, texHeight;
    int cacheIndex = -1;
    bool inCache = false;

    // Protect blending functions and GL_TEXTURE_2D
    glPushAttrib(GL_COLOR_BUFFER_BIT | GL_ENABLE_BIT | GL_TEXTURE_ENV_MODE);

    // Protect glPixelStorei() calls
    glPushClientAttrib(GL_CLIENT_PIXEL_STORE_BIT);

    glEnable(GL_TEXTURE_2D);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

    // Search whether the string is already in a texture we uploaded
    for(int n = 0; n < BUFFER_CACHE_SIZE; n++)
    {
        int i = (lastString + n + BUFFER_CACHE_SIZE) % BUFFER_CACHE_SIZE;

        if(stringCache[i] && !StringCompare(stringCache[i], string, len))
        {
            cacheIndex = i;
            inCache = true;
            break;
        }
    }

    // If the string was not found, we need to put it in the cache and compute
    // its new bounding box.
    if(!inCache)
    {
        // FIXME: this cache is not very efficient. We should first expire
        // strings that are not used very often.
        cacheIndex = lastString;
        lastString = (lastString + 1) % BUFFER_CACHE_SIZE;

        if(stringCache[cacheIndex])
        {
            free(stringCache[cacheIndex]);
        }
        // FIXME: only the first N bytes are copied; we want the first N chars.
        stringCache[cacheIndex] = StringCopy(string, len);
        bboxCache[cacheIndex] = BBox(string, len, FTPoint(), spacing);
    }

    FTBBox bbox = bboxCache[cacheIndex];

    width = static_cast<int>(bbox.Upper().X() - bbox.Lower().X()
                              + padding + padding + 0.5);
    height = static_cast<int>(bbox.Upper().Y() - bbox.Lower().Y()
                               + padding + padding + 0.5);

    texWidth = NextPowerOf2(width);
    texHeight = NextPowerOf2(height);

    glBindTexture(GL_TEXTURE_2D, idCache[cacheIndex]);

    // If the string was not found, we need to render the text in a new
    // texture buffer, then upload it to the OpenGL layer.
    if(!inCache)
    {
        buffer->Size(texWidth, texHeight);
        buffer->Pos(FTPoint(padding, padding) - bbox.Lower());

        advanceCache[cacheIndex] =
              FTFontImpl::Render(string, len, FTPoint(), spacing, renderMode);

        glBindTexture(GL_TEXTURE_2D, idCache[cacheIndex]);

        glPixelStorei(GL_UNPACK_LSB_FIRST, GL_FALSE);
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

        /* TODO: use glTexSubImage2D later? */
        glTexImage2D(GL_TEXTURE_2D, 0, GL_ALPHA, texWidth, texHeight, 0,
                     GL_ALPHA, GL_UNSIGNED_BYTE, (GLvoid *)buffer->Pixels());

        buffer->Size(0, 0);
    }

    FTPoint low = position + bbox.Lower() - FTPoint(padding, padding);
    FTPoint up = position + bbox.Upper() + FTPoint(padding, padding);

    glBegin(GL_QUADS);
        glNormal3f(0.0f, 0.0f, 1.0f);
        glTexCoord2f(0.0f,
                     1.0f / texHeight * (texHeight - height));
        glVertex3f(low.Xf(), up.Yf(), position.Zf());
        glTexCoord2f(0.0f,
                     1.0f);
        glVertex3f(low.Xf(), low.Yf(), position.Zf());
        glTexCoord2f(1.0f / texWidth * width,
                     1.0f);
        glVertex3f(up.Xf(), low.Yf(), position.Zf());
        glTexCoord2f(1.0f / texWidth * width,
                     1.0f / texHeight * (texHeight - height));
        glVertex3f(up.Xf(), up.Yf(), position.Zf());
    glEnd();

    glPopClientAttrib();
    glPopAttrib();

    return position + advanceCache[cacheIndex];
}


FTPoint FTBufferFontImpl::Render(const char * string, const int len,
                                 FTPoint position, FTPoint spacing,
                                 int renderMode)
{
    return RenderI(string, len, position, spacing, renderMode);
}


FTPoint FTBufferFontImpl::Render(const wchar_t * string, const int len,
                                 FTPoint position, FTPoint spacing,
                                 int renderMode)
{
    return RenderI(string, len, position, spacing, renderMode);
}

