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

#include "FTGlyphContainer.h"
#include "FTFace.h"
#include "FTCharmap.h"


FTGlyphContainer::FTGlyphContainer(FTFace* f)
:   face(f),
    err(0)
{
    glyphs.push_back(NULL);
    charMap = new FTCharmap(face);
}


FTGlyphContainer::~FTGlyphContainer()
{
    GlyphVector::iterator it;
    for(it = glyphs.begin(); it != glyphs.end(); ++it)
    {
        delete *it;
    }

    glyphs.clear();
    delete charMap;
}


bool FTGlyphContainer::CharMap(FT_Encoding encoding)
{
    bool result = charMap->CharMap(encoding);
    err = charMap->Error();
    return result;
}


unsigned int FTGlyphContainer::FontIndex(const unsigned int charCode) const
{
    return charMap->FontIndex(charCode);
}


void FTGlyphContainer::Add(FTGlyph* tempGlyph, const unsigned int charCode)
{
    charMap->InsertIndex(charCode, glyphs.size());
    glyphs.push_back(tempGlyph);
}


const FTGlyph* const FTGlyphContainer::Glyph(const unsigned int charCode) const
{
    unsigned int index = charMap->GlyphListIndex(charCode);

    return (index < glyphs.size()) ? glyphs[index] : NULL;
}


FTBBox FTGlyphContainer::BBox(const unsigned int charCode) const
{
    return Glyph(charCode)->BBox();
}


float FTGlyphContainer::Advance(const unsigned int charCode,
                                const unsigned int nextCharCode)
{
    unsigned int left = charMap->FontIndex(charCode);
    unsigned int right = charMap->FontIndex(nextCharCode);
    const FTGlyph *glyph = Glyph(charCode);

    if (!glyph)
      return 0.0f;

    return face->KernAdvance(left, right).Xf() + glyph->Advance();
}


FTPoint FTGlyphContainer::Render(const unsigned int charCode,
                                 const unsigned int nextCharCode,
                                 FTPoint penPosition, int renderMode)
{
    unsigned int left = charMap->FontIndex(charCode);
    unsigned int right = charMap->FontIndex(nextCharCode);

    FTPoint kernAdvance = face->KernAdvance(left, right);

    if(!face->Error())
    {
        unsigned int index = charMap->GlyphListIndex(charCode);
        if (index < glyphs.size())
            kernAdvance += glyphs[index]->Render(penPosition, renderMode);
    }

    return kernAdvance;
}

