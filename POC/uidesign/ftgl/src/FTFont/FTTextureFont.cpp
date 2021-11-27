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

#include <cassert>
#include <string> // For memset

#include "FTGL/ftgl.h"

#include "FTInternals.h"

#include "../FTGlyph/FTTextureGlyphImpl.h"
#include "./FTTextureFontImpl.h"


//
//  FTTextureFont
//


FTTextureFont::FTTextureFont(char const *fontFilePath) :
    FTFont(new FTTextureFontImpl(this, fontFilePath))
{}


FTTextureFont::FTTextureFont(const unsigned char *pBufferBytes,
                             size_t bufferSizeInBytes) :
    FTFont(new FTTextureFontImpl(this, pBufferBytes, bufferSizeInBytes))
{}


FTTextureFont::~FTTextureFont()
{}


FTGlyph* FTTextureFont::MakeGlyph(FT_GlyphSlot ftGlyph)
{
    FTTextureFontImpl *myimpl = dynamic_cast<FTTextureFontImpl *>(impl);
    if(!myimpl)
    {
        return NULL;
    }

    return myimpl->MakeGlyphImpl(ftGlyph);
}


//
//  FTTextureFontImpl
//


static inline GLuint ClampSize(GLuint in, GLuint maxTextureSize)
{
    // Find next power of two
    --in;
    in |= in >> 16;
    in |= in >> 8;
    in |= in >> 4;
    in |= in >> 2;
    in |= in >> 1;
    ++in;

    // Clamp to max texture size
    return in < maxTextureSize ? in : maxTextureSize;
}


FTTextureFontImpl::FTTextureFontImpl(FTFont *ftFont, const char* fontFilePath)
:   FTFontImpl(ftFont, fontFilePath),
    maximumGLTextureSize(0),
    textureWidth(0),
    textureHeight(0),
    glyphHeight(0),
    glyphWidth(0),
    padding(3),
    xOffset(0),
    yOffset(0)
{
    load_flags = FT_LOAD_NO_HINTING | FT_LOAD_NO_BITMAP;
    remGlyphs = numGlyphs = face.GlyphCount();
}


FTTextureFontImpl::FTTextureFontImpl(FTFont *ftFont,
                                     const unsigned char *pBufferBytes,
                                     size_t bufferSizeInBytes)
:   FTFontImpl(ftFont, pBufferBytes, bufferSizeInBytes),
    maximumGLTextureSize(0),
    textureWidth(0),
    textureHeight(0),
    glyphHeight(0),
    glyphWidth(0),
    padding(3),
    xOffset(0),
    yOffset(0)
{
    load_flags = FT_LOAD_NO_HINTING | FT_LOAD_NO_BITMAP;
    remGlyphs = numGlyphs = face.GlyphCount();
}


FTTextureFontImpl::~FTTextureFontImpl()
{
    if(textureIDList.size())
    {
        glDeleteTextures((GLsizei)textureIDList.size(),
                         (const GLuint*)&textureIDList[0]);
    }
}


FTGlyph* FTTextureFontImpl::MakeGlyphImpl(FT_GlyphSlot ftGlyph)
{
    glyphHeight = static_cast<int>(charSize.Height() + 0.5f);
    glyphWidth = static_cast<int>(charSize.Width() + 0.5f);

    if(glyphHeight < 1) glyphHeight = 1;
    if(glyphWidth < 1) glyphWidth = 1;

    if(textureIDList.empty())
    {
        textureIDList.push_back(CreateTexture());
        xOffset = yOffset = padding;
    }

    if(xOffset > (textureWidth - glyphWidth))
    {
        xOffset = padding;
        yOffset += glyphHeight;

        if(yOffset > (textureHeight - glyphHeight))
        {
            textureIDList.push_back(CreateTexture());
            yOffset = padding;
        }
    }

    FTTextureGlyph* tempGlyph = new FTTextureGlyph(ftGlyph, textureIDList[textureIDList.size() - 1],
                                                    xOffset, yOffset, textureWidth, textureHeight);
    xOffset += static_cast<int>(tempGlyph->BBox().Upper().X() - tempGlyph->BBox().Lower().X() + padding + 0.5);

    --remGlyphs;

    return tempGlyph;
}


void FTTextureFontImpl::CalculateTextureSize()
{
    if(!maximumGLTextureSize)
    {
        maximumGLTextureSize = 1024;
        glGetIntegerv(GL_MAX_TEXTURE_SIZE, (GLint*)&maximumGLTextureSize);
        assert(maximumGLTextureSize); // Indicates an invalid OpenGL context
    }

    // Texture width required for numGlyphs glyphs. Will probably not be
    // large enough, but we try to fit as many glyphs in one line as possible
    textureWidth = ClampSize(glyphWidth * numGlyphs + padding * 2,
                             maximumGLTextureSize);

    // Number of lines required for that many glyphs in a line
    int tmp = (textureWidth - (padding * 2)) / glyphWidth;
    tmp = tmp > 0 ? tmp : 1;
    tmp = (numGlyphs + (tmp - 1)) / tmp; // round division up

    // Texture height required for tmp lines of glyphs
    textureHeight = ClampSize(glyphHeight * tmp + padding * 2,
                              maximumGLTextureSize);
}


GLuint FTTextureFontImpl::CreateTexture()
{
    CalculateTextureSize();

    int totalMemory = textureWidth * textureHeight;
    unsigned char* textureMemory = new unsigned char[totalMemory];
    memset(textureMemory, 0, totalMemory);

    GLuint textID;
    glGenTextures(1, (GLuint*)&textID);

    glBindTexture(GL_TEXTURE_2D, textID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_ALPHA, textureWidth, textureHeight,
                 0, GL_ALPHA, GL_UNSIGNED_BYTE, textureMemory);

    delete [] textureMemory;

    return textID;
}


bool FTTextureFontImpl::FaceSize(const unsigned int size, const unsigned int res)
{
    if(!textureIDList.empty())
    {
        glDeleteTextures((GLsizei)textureIDList.size(), (const GLuint*)&textureIDList[0]);
        textureIDList.clear();
        remGlyphs = numGlyphs = face.GlyphCount();
    }

    return FTFontImpl::FaceSize(size, res);
}


template <typename T>
inline FTPoint FTTextureFontImpl::RenderI(const T* string, const int len,
                                          FTPoint position, FTPoint spacing,
                                          int renderMode)
{
    // Protect GL_TEXTURE_2D
    glPushAttrib(GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT | GL_TEXTURE_ENV_MODE);

    glEnable(GL_TEXTURE_2D);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

    FTTextureGlyphImpl::ResetActiveTexture();

    FTPoint tmp = FTFontImpl::Render(string, len,
                                     position, spacing, renderMode);

    glPopAttrib();

    return tmp;
}


FTPoint FTTextureFontImpl::Render(const char * string, const int len,
                                  FTPoint position, FTPoint spacing,
                                  int renderMode)
{
    return RenderI(string, len, position, spacing, renderMode);
}


FTPoint FTTextureFontImpl::Render(const wchar_t * string, const int len,
                                  FTPoint position, FTPoint spacing,
                                  int renderMode)
{
    return RenderI(string, len, position, spacing, renderMode);
}

