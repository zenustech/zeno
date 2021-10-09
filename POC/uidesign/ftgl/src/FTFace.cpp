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

#include "FTFace.h"
#include "FTCleanup.h"
#include "FTLibrary.h"

#include FT_TRUETYPE_TABLES_H

FTFace::FTFace(const char* fontFilePath, bool precomputeKerning)
:   numGlyphs(0),
    fontEncodingList(0),
    kerningCache(0),
    err(0)
{
    const FT_Long DEFAULT_FACE_INDEX = 0;
    ftFace = new FT_Face;

    err = FT_New_Face(*FTLibrary::Instance().GetLibrary(), fontFilePath,
                      DEFAULT_FACE_INDEX, ftFace);
    if(err)
    {
        delete ftFace;
        ftFace = 0;
        return;
    }

    FTCleanup::Instance()->RegisterObject(&ftFace);

    numGlyphs = (*ftFace)->num_glyphs;
    hasKerningTable = (FT_HAS_KERNING((*ftFace)) != 0);

    if(hasKerningTable && precomputeKerning)
    {
        BuildKerningCache();
    }
}


FTFace::FTFace(const unsigned char *pBufferBytes, size_t bufferSizeInBytes,
               bool precomputeKerning)
:   numGlyphs(0),
    fontEncodingList(0),
    kerningCache(0),
    err(0)
{
    const FT_Long DEFAULT_FACE_INDEX = 0;
    ftFace = new FT_Face;

    err = FT_New_Memory_Face(*FTLibrary::Instance().GetLibrary(),
                             (FT_Byte const *)pBufferBytes, (FT_Long)bufferSizeInBytes,
                             DEFAULT_FACE_INDEX, ftFace);
    if(err)
    {
        delete ftFace;
        ftFace = 0;
        return;
    }

    FTCleanup::Instance()->RegisterObject(&ftFace);

    numGlyphs = (*ftFace)->num_glyphs;
    hasKerningTable = (FT_HAS_KERNING((*ftFace)) != 0);

    if(hasKerningTable && precomputeKerning)
    {
        BuildKerningCache();
    }
}


FTFace::~FTFace()
{
    delete[] kerningCache;

    if(ftFace)
    {
        FTCleanup::Instance()->UnregisterObject(&ftFace);

        FT_Done_Face(*ftFace);
        delete ftFace;
        ftFace = 0;
    }
}


bool FTFace::Attach(const char* fontFilePath)
{
    err = FT_Attach_File(*ftFace, fontFilePath);
    return !err;
}


bool FTFace::Attach(const unsigned char *pBufferBytes,
                    size_t bufferSizeInBytes)
{
    FT_Open_Args open;

    open.flags = FT_OPEN_MEMORY;
    open.memory_base = (FT_Byte const *)pBufferBytes;
    open.memory_size = (FT_Long)bufferSizeInBytes;

    err = FT_Attach_Stream(*ftFace, &open);
    return !err;
}


const FTSize& FTFace::Size(const unsigned int size, const unsigned int res)
{
    charSize.CharSize(ftFace, size, res, res);
    err = charSize.Error();

    return charSize;
}


unsigned int FTFace::CharMapCount() const
{
    return (*ftFace)->num_charmaps;
}


FT_Encoding* FTFace::CharMapList()
{
    if(0 == fontEncodingList)
    {
        fontEncodingList = new FT_Encoding[CharMapCount()];
        for(size_t i = 0; i < CharMapCount(); ++i)
        {
            fontEncodingList[i] = (*ftFace)->charmaps[i]->encoding;
        }
    }

    return fontEncodingList;
}


FTPoint FTFace::KernAdvance(unsigned int index1, unsigned int index2)
{
    FTGL_DOUBLE x, y;

    if(!hasKerningTable || !index1 || !index2)
    {
        return FTPoint(0.0, 0.0);
    }

    if(kerningCache && index1 < FTFace::MAX_PRECOMPUTED
        && index2 < FTFace::MAX_PRECOMPUTED)
    {
        x = kerningCache[2 * (index2 * FTFace::MAX_PRECOMPUTED + index1)];
        y = kerningCache[2 * (index2 * FTFace::MAX_PRECOMPUTED + index1) + 1];
        return FTPoint(x, y);
    }

    FT_Vector kernAdvance;
    kernAdvance.x = kernAdvance.y = 0;

    err = FT_Get_Kerning(*ftFace, index1, index2, ft_kerning_unfitted,
                         &kernAdvance);
    if(err)
    {
        return FTPoint(0.0f, 0.0f);
    }

    x = static_cast<float>(kernAdvance.x) / 64.0f;
    y = static_cast<float>(kernAdvance.y) / 64.0f;

    return FTPoint(x, y);
}


FT_GlyphSlot FTFace::Glyph(unsigned int index, FT_Int load_flags)
{
    err = FT_Load_Glyph(*ftFace, index, load_flags);
    if(err)
    {
        return NULL;
    }

    return (*ftFace)->glyph;
}


void FTFace::BuildKerningCache()
{
    FT_Vector kernAdvance;
    kernAdvance.x = 0;
    kernAdvance.y = 0;
    kerningCache = new FTGL_DOUBLE[FTFace::MAX_PRECOMPUTED
                                    * FTFace::MAX_PRECOMPUTED * 2];
    for(unsigned int j = 0; j < FTFace::MAX_PRECOMPUTED; j++)
    {
        for(unsigned int i = 0; i < FTFace::MAX_PRECOMPUTED; i++)
        {
            err = FT_Get_Kerning(*ftFace, i, j, ft_kerning_unfitted,
                                 &kernAdvance);
            if(err)
            {
                delete[] kerningCache;
                kerningCache = NULL;
                return;
            }

            kerningCache[2 * (j * FTFace::MAX_PRECOMPUTED + i)] =
                                static_cast<FTGL_DOUBLE>(kernAdvance.x) / 64.0;
            kerningCache[2 * (j * FTFace::MAX_PRECOMPUTED + i) + 1] =
                                static_cast<FTGL_DOUBLE>(kernAdvance.y) / 64.0;
        }
    }
}

