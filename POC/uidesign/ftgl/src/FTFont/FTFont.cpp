/*
 * FTGL - OpenGL font library
 *
 * Copyright (c) 2001-2004 Henry Maddocks <ftgl@opengl.geek.nz>
 * Copyright (c) 2008 Sam Hocevar <sam@hocevar.net>
 * Copyright (c) 2008 Daniel Remenak <dtremenak@users.sourceforge.net>
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

#include "FTInternals.h"
#include "FTUnicode.h"

#include "FTFontImpl.h"

#include "FTBitmapFontImpl.h"
#include "FTExtrudeFontImpl.h"
#include "FTOutlineFontImpl.h"
#include "FTPixmapFontImpl.h"
#include "FTPolygonFontImpl.h"
#include "FTTextureFontImpl.h"

#include "FTGlyphContainer.h"
#include "FTFace.h"


//
//  FTFont
//


FTFont::FTFont(char const *fontFilePath)
{
    impl = new FTFontImpl(this, fontFilePath);
}


FTFont::FTFont(const unsigned char *pBufferBytes, size_t bufferSizeInBytes)
{
    impl = new FTFontImpl(this, pBufferBytes, bufferSizeInBytes);
}


FTFont::FTFont(FTFontImpl *pImpl)
{
    impl = pImpl;
}


FTFont::~FTFont()
{
    delete impl;
}


bool FTFont::Attach(const char* fontFilePath)
{
    return impl->Attach(fontFilePath);
}


bool FTFont::Attach(const unsigned char *pBufferBytes, size_t bufferSizeInBytes)
{
    return impl->Attach(pBufferBytes, bufferSizeInBytes);
}


bool FTFont::FaceSize(const unsigned int size, const unsigned int res)
{
    return impl->FaceSize(size, res);
}


unsigned int FTFont::FaceSize() const
{
    return impl->FaceSize();
}


void FTFont::Depth(float depth)
{
    return impl->Depth(depth);
}


void FTFont::Outset(float outset)
{
    return impl->Outset(outset);
}


void FTFont::Outset(float front, float back)
{
    return impl->Outset(front, back);
}


void FTFont::GlyphLoadFlags(FT_Int flags)
{
    return impl->GlyphLoadFlags(flags);
}


bool FTFont::CharMap(FT_Encoding encoding)
{
    return impl->CharMap(encoding);
}


unsigned int FTFont::CharMapCount() const
{
    return impl->CharMapCount();
}


FT_Encoding* FTFont::CharMapList()
{
    return impl->CharMapList();
}


void FTFont::UseDisplayList(bool useList)
{
    return impl->UseDisplayList(useList);
}


float FTFont::Ascender() const
{
    return impl->Ascender();
}


float FTFont::Descender() const
{
    return impl->Descender();
}


float FTFont::LineHeight() const
{
    return impl->LineHeight();
}


FTPoint FTFont::Render(const char * string, const int len,
                       FTPoint position, FTPoint spacing, int renderMode)
{
    return impl->Render(string, len, position, spacing, renderMode);
}


FTPoint FTFont::Render(const wchar_t * string, const int len,
                       FTPoint position, FTPoint spacing, int renderMode)
{
    return impl->Render(string, len, position, spacing, renderMode);
}


float FTFont::Advance(const char * string, const int len, FTPoint spacing)
{
    return impl->Advance(string, len, spacing);
}


float FTFont::Advance(const wchar_t * string, const int len, FTPoint spacing)
{
    return impl->Advance(string, len, spacing);
}


FTBBox FTFont::BBox(const char *string, const int len,
                    FTPoint position, FTPoint spacing)
{
    return impl->BBox(string, len, position, spacing);
}


FTBBox FTFont::BBox(const wchar_t *string, const int len,
                    FTPoint position, FTPoint spacing)
{
    return impl->BBox(string, len, position, spacing);
}


FT_Error FTFont::Error() const
{
    return impl->err;
}


//
//  FTFontImpl
//


FTFontImpl::FTFontImpl(FTFont *ftFont, char const *fontFilePath) :
    face(fontFilePath),
    useDisplayLists(true),
    load_flags(FT_LOAD_DEFAULT),
    intf(ftFont),
    glyphList(0)
{
    err = face.Error();
    if(err == 0)
    {
        glyphList = new FTGlyphContainer(&face);
    }
}


FTFontImpl::FTFontImpl(FTFont *ftFont, const unsigned char *pBufferBytes,
                       size_t bufferSizeInBytes) :
    face(pBufferBytes, bufferSizeInBytes),
    useDisplayLists(true),
    load_flags(FT_LOAD_DEFAULT),
    intf(ftFont),
    glyphList(0)
{
    err = face.Error();
    if(err == 0)
    {
        glyphList = new FTGlyphContainer(&face);
    }
}


FTFontImpl::~FTFontImpl()
{
    if(glyphList)
    {
        delete glyphList;
    }
}


bool FTFontImpl::Attach(const char* fontFilePath)
{
    if(!face.Attach(fontFilePath))
    {
        err = face.Error();
        return false;
    }

    err = 0;
    return true;
}


bool FTFontImpl::Attach(const unsigned char *pBufferBytes,
                        size_t bufferSizeInBytes)
{
    if(!face.Attach(pBufferBytes, bufferSizeInBytes))
    {
        err = face.Error();
        return false;
    }

    err = 0;
    return true;
}


bool FTFontImpl::FaceSize(const unsigned int size, const unsigned int res)
{
    if(glyphList != NULL)
    {
        delete glyphList;
        glyphList = NULL;
    }

    charSize = face.Size(size, res);
    err = face.Error();

    if(err != 0)
    {
        return false;
    }

    glyphList = new FTGlyphContainer(&face);
    return true;
}


unsigned int FTFontImpl::FaceSize() const
{
    return charSize.CharSize();
}


void FTFontImpl::Depth(float depth)
{
    (void)depth;
}


void FTFontImpl::Outset(float outset)
{
    (void)outset;
}


void FTFontImpl::Outset(float front, float back)
{
    (void)front; (void)back;
}


void FTFontImpl::GlyphLoadFlags(FT_Int flags)
{
    load_flags = flags;
}


bool FTFontImpl::CharMap(FT_Encoding encoding)
{
    bool result = glyphList->CharMap(encoding);
    err = glyphList->Error();
    return result;
}


unsigned int FTFontImpl::CharMapCount() const
{
    return face.CharMapCount();
}


FT_Encoding* FTFontImpl::CharMapList()
{
    return face.CharMapList();
}


void FTFontImpl::UseDisplayList(bool useList)
{
    useDisplayLists = useList;
}


float FTFontImpl::Ascender() const
{
    return charSize.Ascender();
}


float FTFontImpl::Descender() const
{
    return charSize.Descender();
}


float FTFontImpl::LineHeight() const
{
    return charSize.Height();
}


template <typename T>
inline FTBBox FTFontImpl::BBoxI(const T* string, const int len,
                                FTPoint position, FTPoint spacing)
{
    FTBBox totalBBox;

    /* Only compute the bounds if string is non-empty. */
    if(string && ('\0' != string[0]))
    {
        // for multibyte - we can't rely on sizeof(T) == character
        FTUnicodeStringItr<T> ustr(string);
        unsigned int thisChar = *ustr++;
        unsigned int nextChar = *ustr;

        if(CheckGlyph(thisChar))
        {
            totalBBox = glyphList->BBox(thisChar);
            totalBBox += position;

            position += FTPoint(glyphList->Advance(thisChar, nextChar), 0.0);
        }

        /* Expand totalBox by each glyph in string */
        for(int i = 1; (len < 0 && *ustr) || (len >= 0 && i < len); i++)
        {
            thisChar = *ustr++;
            nextChar = *ustr;

            if(CheckGlyph(thisChar))
            {
                position += spacing;

                FTBBox tempBBox = glyphList->BBox(thisChar);
                tempBBox += position;
                totalBBox |= tempBBox;

                position += FTPoint(glyphList->Advance(thisChar, nextChar),
                                    0.0);
            }
        }
    }

    return totalBBox;
}


FTBBox FTFontImpl::BBox(const char *string, const int len,
                        FTPoint position, FTPoint spacing)
{
    /* The chars need to be unsigned because they are cast to int later */
    return BBoxI((const unsigned char *)string, len, position, spacing);
}


FTBBox FTFontImpl::BBox(const wchar_t *string, const int len,
                        FTPoint position, FTPoint spacing)
{
    return BBoxI(string, len, position, spacing);
}


template <typename T>
inline float FTFontImpl::AdvanceI(const T* string, const int len,
                                  FTPoint spacing)
{
    float advance = 0.0f;
    FTUnicodeStringItr<T> ustr(string);

    for(int i = 0; (len < 0 && *ustr) || (len >= 0 && i < len); i++)
    {
        unsigned int thisChar = *ustr++;
        unsigned int nextChar = *ustr;

        if(CheckGlyph(thisChar))
        {
            advance += glyphList->Advance(thisChar, nextChar);
        }

        if(nextChar)
        {
            advance += spacing.Xf();
        }
    }

    return advance;
}


float FTFontImpl::Advance(const char* string, const int len, FTPoint spacing)
{
    /* The chars need to be unsigned because they are cast to int later */
    const unsigned char *ustring = (const unsigned char *)string;
    return AdvanceI(ustring, len, spacing);
}


float FTFontImpl::Advance(const wchar_t* string, const int len, FTPoint spacing)
{
    return AdvanceI(string, len, spacing);
}


template <typename T>
inline FTPoint FTFontImpl::RenderI(const T* string, const int len,
                                   FTPoint position, FTPoint spacing,
                                   int renderMode)
{
    // for multibyte - we can't rely on sizeof(T) == character
    FTUnicodeStringItr<T> ustr(string);

    for(int i = 0; (len < 0 && *ustr) || (len >= 0 && i < len); i++)
    {
        unsigned int thisChar = *ustr++;
        unsigned int nextChar = *ustr;

        if(CheckGlyph(thisChar))
        {
            position += glyphList->Render(thisChar, nextChar,
                                          position, renderMode);
        }

        if(nextChar)
        {
            position += spacing;
        }
    }

    return position;
}


FTPoint FTFontImpl::Render(const char * string, const int len,
                           FTPoint position, FTPoint spacing, int renderMode)
{
    return RenderI((const unsigned char *)string,
                   len, position, spacing, renderMode);
}


FTPoint FTFontImpl::Render(const wchar_t * string, const int len,
                           FTPoint position, FTPoint spacing, int renderMode)
{
    return RenderI(string, len, position, spacing, renderMode);
}


bool FTFontImpl::CheckGlyph(const unsigned int characterCode)
{
    if(glyphList->Glyph(characterCode))
    {
        return true;
    }

    unsigned int glyphIndex = glyphList->FontIndex(characterCode);
    FT_GlyphSlot ftSlot = face.Glyph(glyphIndex, load_flags);
    if(!ftSlot)
    {
        err = face.Error();
        return false;
    }

    FTGlyph* tempGlyph = intf->MakeGlyph(ftSlot);
    if(!tempGlyph)
    {
        if(0 == err)
        {
            err = 0x13;
        }

        return false;
    }

    glyphList->Add(tempGlyph, characterCode);

    return true;
}

