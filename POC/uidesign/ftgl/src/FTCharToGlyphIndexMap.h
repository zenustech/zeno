/*
 * FTGL - OpenGL font library
 *
 * Copyright (c) 2001-2004 Henry Maddocks <ftgl@opengl.geek.nz>
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

#ifndef    __FTCharToGlyphIndexMap__
#define    __FTCharToGlyphIndexMap__

#include <stdlib.h>

#include "FTGL/ftgl.h"

/**
 * Provides a non-STL alternative to the STL map<unsigned long, unsigned long>
 * which maps character codes to glyph indices inside FTCharmap.
 *
 * Implementation:
 *   - NumberOfBuckets buckets are considered.
 *   - Each bucket has BucketSize entries.
 *   - When the glyph index for the character code C has to be stored, the
 *     bucket this character belongs to is found using 'C div BucketSize'.
 *     If this bucket has not been allocated yet, do it now.
 *     The entry in the bucked is found using 'C mod BucketSize'.
 *     If it is set to IndexNotFound, then the glyph entry has not been set.
 *   - Try to mimic the calls made to the STL map API.
 *
 * Caveats:
 *   - The glyph index is now a signed long instead of unsigned long, so
 *     the special value IndexNotFound (= -1) can be used to specify that the
 *     glyph index has not been stored yet.
 */
class FTCharToGlyphIndexMap
{
    public:
        typedef unsigned long CharacterCode;
        typedef signed long GlyphIndex;

        // XXX: always ensure that 1 << (3 * BucketIdxBits) >= UnicodeValLimit
        static const int BucketIdxBits = 7;
        static const int BucketIdxSize = 1 << BucketIdxBits;
        static const int BucketIdxMask = BucketIdxSize - 1;

        static const CharacterCode UnicodeValLimit = 0x110000;
        static const int IndexNotFound = -1;

        FTCharToGlyphIndexMap()
        {
            Indices = 0;
        }

        virtual ~FTCharToGlyphIndexMap()
        {
            // Free all buckets
            clear();
        }

        inline void clear()
        {
            for(int j = 0; Indices && j < BucketIdxSize; j++)
            {
                for(int i = 0; Indices[j] && i < BucketIdxSize; i++)
                {
                    delete[] Indices[j][i];
                    Indices[j][i] = 0;
                }
                delete[] Indices[j];
                Indices[j] = 0;
            }
            delete[] Indices;
            Indices = 0;
        }

        const GlyphIndex find(CharacterCode c)
        {
            int OuterIdx = (c >> (BucketIdxBits * 2)) & BucketIdxMask;
            int InnerIdx = (c >> BucketIdxBits) & BucketIdxMask;
            int Offset = c & BucketIdxMask;

            if (c >= UnicodeValLimit || !Indices
                 || !Indices[OuterIdx] || !Indices[OuterIdx][InnerIdx])
                return 0;

            GlyphIndex g = Indices[OuterIdx][InnerIdx][Offset];

            return (g != IndexNotFound) ? g : 0;
        }

        void insert(CharacterCode c, GlyphIndex g)
        {
            int OuterIdx = (c >> (BucketIdxBits * 2)) & BucketIdxMask;
            int InnerIdx = (c >> BucketIdxBits) & BucketIdxMask;
            int Offset = c & BucketIdxMask;

            if (c >= UnicodeValLimit)
                return;

            if (!Indices)
            {
                Indices = new GlyphIndex** [BucketIdxSize];
                for(int i = 0; i < BucketIdxSize; i++)
                    Indices[i] = 0;
            }

            if (!Indices[OuterIdx])
            {
                Indices[OuterIdx] = new GlyphIndex* [BucketIdxSize];
                for(int i = 0; i < BucketIdxSize; i++)
                    Indices[OuterIdx][i] = 0;
            }

            if (!Indices[OuterIdx][InnerIdx])
            {
                Indices[OuterIdx][InnerIdx] = new GlyphIndex [BucketIdxSize];
                for(int i = 0; i < BucketIdxSize; i++)
                    Indices[OuterIdx][InnerIdx][i] = IndexNotFound;
            }

            Indices[OuterIdx][InnerIdx][Offset] = g;
        }

    private:
        GlyphIndex*** Indices;
};

#endif  //  __FTCharToGlyphIndexMap__

