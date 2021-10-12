/*
 * FTGL - OpenGL font library
 *
 * Copyright (c) 2011 Richard Ulrich <richi@paraeasy.ch>
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

#ifndef __FTTriangleExtractorFontImpl__
#define __FTTriangleExtractorFontImpl__

#include "FTFontImpl.h"
// std lib
#include <vector>

class FTGlyph;

class FTTriangleExtractorFontImpl : public FTFontImpl
{
    friend class FTTriangleExtractorFont;

    protected:
        FTTriangleExtractorFontImpl(FTFont *ftFont, const char* fontFilePath, std::vector<float>& triangles);

        FTTriangleExtractorFontImpl(FTFont *ftFont, const unsigned char *pBufferBytes,
                          size_t bufferSizeInBytes, std::vector<float>& triangles);

        /**
         * Set the outset distance for the font. Only implemented by
         * FTOutlineFont, FTPolygonFont and FTExtrudeFont
         *
         * @param outset  The outset distance.
         */
        virtual void Outset(float o) { outset = o; }

        virtual FTPoint Render(const char *s, const int len,
                               FTPoint position, FTPoint spacing,
                               int renderMode);

        virtual FTPoint Render(const wchar_t *s, const int len,
                               FTPoint position, FTPoint spacing,
                               int renderMode);


    private:
        /**
         * The outset distance for the font.
         */
        float outset;

        std::vector<float>& triangles_;

        /* Internal generic Render() implementation */
        template <typename T>
        inline FTPoint RenderI(const T *s, const int len,
                               FTPoint position, FTPoint spacing, int mode);
};

#endif // __FTPolygonFontImpl__

