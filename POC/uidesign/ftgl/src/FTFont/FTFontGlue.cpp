/*
 * FTGL - OpenGL font library
 *
 * Copyright (c) 2001-2004 Henry Maddocks <ftgl@opengl.geek.nz>
 * Copyright (c) 2008 Éric Beets <ericbeets@free.fr>
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

#include "FTInternals.h"

static const FTPoint static_ftpoint;
static const FTBBox static_ftbbox;

FTGL_BEGIN_C_DECLS

#define C_TOR(cname, cargs, cxxname, cxxarg, cxxtype) \
    FTGLfont* cname cargs \
    { \
        cxxname *f = new cxxname cxxarg; \
        if(f->Error()) \
        { \
            delete f; \
            return NULL; \
        } \
        FTGLfont *ftgl = (FTGLfont *)malloc(sizeof(FTGLfont)); \
        ftgl->ptr = f; \
        ftgl->type = cxxtype; \
        return ftgl; \
    }

// FTBitmapFont::FTBitmapFont();
C_TOR(ftglCreateBitmapFont, (const char *fontname),
      FTBitmapFont, (fontname), FONT_BITMAP);
C_TOR(ftglCreateBitmapFontFromMem, (const unsigned char *bytes, size_t len),
      FTBitmapFont, (bytes, len), FONT_BITMAP);

// FTBufferFont::FTBufferFont();
C_TOR(ftglCreateBufferFont, (const char *fontname),
      FTBufferFont, (fontname), FONT_BUFFER);
C_TOR(ftglCreateBufferFontFromMem, (const unsigned char *bytes, size_t len),
      FTBufferFont, (bytes, len), FONT_BUFFER);

// FTExtrudeFont::FTExtrudeFont();
C_TOR(ftglCreateExtrudeFont, (const char *fontname),
      FTExtrudeFont, (fontname), FONT_EXTRUDE);
C_TOR(ftglCreateExtrudeFontFromMem, (const unsigned char *bytes, size_t len),
      FTExtrudeFont, (bytes, len), FONT_EXTRUDE);

// FTOutlineFont::FTOutlineFont();
C_TOR(ftglCreateOutlineFont, (const char *fontname),
      FTOutlineFont, (fontname), FONT_OUTLINE);
C_TOR(ftglCreateOutlineFontFromMem, (const unsigned char *bytes, size_t len),
      FTOutlineFont, (bytes, len), FONT_OUTLINE);

// FTPixmapFont::FTPixmapFont();
C_TOR(ftglCreatePixmapFont, (const char *fontname),
      FTPixmapFont, (fontname), FONT_PIXMAP);
C_TOR(ftglCreatePixmapFontFromMem, (const unsigned char *bytes, size_t len),
      FTPixmapFont, (bytes, len), FONT_PIXMAP);

// FTPolygonFont::FTPolygonFont();
C_TOR(ftglCreatePolygonFont, (const char *fontname),
      FTPolygonFont, (fontname), FONT_POLYGON);
C_TOR(ftglCreatePolygonFontFromMem, (const unsigned char *bytes, size_t len),
      FTPolygonFont, (bytes, len), FONT_POLYGON);

// FTTextureFont::FTTextureFont();
C_TOR(ftglCreateTextureFont, (const char *fontname),
      FTTextureFont, (fontname), FONT_TEXTURE);
C_TOR(ftglCreateTextureFontFromMem, (const unsigned char *bytes, size_t len),
      FTTextureFont, (bytes, len), FONT_TEXTURE);

// FTCustomFont::FTCustomFont();
class FTCustomFont : public FTFont
{
public:
    FTCustomFont(char const *fontFilePath, void *p,
                 FTGLglyph * (*makeglyph) (FT_GlyphSlot, void *))
     : FTFont(fontFilePath),
       data(p),
       makeglyphCallback(makeglyph)
    {}

    FTCustomFont(const unsigned char *pBufferBytes, size_t bufferSizeInBytes,
                 void *p, FTGLglyph * (*makeglyph) (FT_GlyphSlot, void *))
     : FTFont(pBufferBytes, bufferSizeInBytes),
       data(p),
       makeglyphCallback(makeglyph)
    {}

    ~FTCustomFont()
    {}

    FTGlyph* MakeGlyph(FT_GlyphSlot slot)
    {
        FTGLglyph *g = makeglyphCallback(slot, data);
        FTGlyph *glyph = g->ptr;
        // XXX: we no longer need g, and no one will free it for us. Not
        // very elegant, and we need to make sure no one else will try to
        // use it.
        free(g);
        return glyph;
    }

private:
    void *data;
    FTGLglyph *(*makeglyphCallback) (FT_GlyphSlot, void *);
};

C_TOR(ftglCreateCustomFont, (char const *fontFilePath, void *data,
                   FTGLglyph * (*makeglyphCallback) (FT_GlyphSlot, void *)),
      FTCustomFont, (fontFilePath, data, makeglyphCallback), FONT_CUSTOM);
C_TOR(ftglCreateCustomFontFromMem, (const unsigned char *bytes, size_t len,
          void *data, FTGLglyph * (*makeglyphCallback) (FT_GlyphSlot, void *)),
      FTCustomFont, (bytes, len, data, makeglyphCallback), FONT_CUSTOM);

#define C_FUN(cret, cname, cargs, cxxerr, cxxname, cxxarg) \
    cret cname cargs \
    { \
        if(!f || !f->ptr) \
        { \
            fprintf(stderr, "FTGL warning: NULL pointer in %s\n", #cname); \
            cxxerr; \
        } \
        return f->ptr->cxxname cxxarg; \
    }

// FTFont::~FTFont();
void ftglDestroyFont(FTGLfont *f)
{
    if(!f || !f->ptr)
    {
        fprintf(stderr, "FTGL warning: NULL pointer in %s\n", __FUNC__);
        return;
    }
    delete f->ptr;
    free(f);
}

// bool FTFont::Attach(const char* fontFilePath);
C_FUN(int, ftglAttachFile, (FTGLfont *f, const char* path),
      return 0, Attach, (path));

// bool FTFont::Attach(const unsigned char *pBufferBytes,
//                     size_t bufferSizeInBytes);
C_FUN(int, ftglAttachData, (FTGLfont *f, const unsigned char *p, size_t s),
      return 0, Attach, (p, s));

// void FTFont::GlyphLoadFlags(FT_Int flags);
C_FUN(void, ftglSetFontGlyphLoadFlags, (FTGLfont *f, FT_Int flags),
      return, GlyphLoadFlags, (flags));

// bool FTFont::CharMap(FT_Encoding encoding);
C_FUN(int, ftglSetFontCharMap, (FTGLfont *f, FT_Encoding enc),
      return 0, CharMap, (enc));

// unsigned int FTFont::CharMapCount();
C_FUN(unsigned int, ftglGetFontCharMapCount, (FTGLfont *f),
      return 0, CharMapCount, ());

// FT_Encoding* FTFont::CharMapList();
C_FUN(FT_Encoding *, ftglGetFontCharMapList, (FTGLfont* f),
      return NULL, CharMapList, ());

// virtual bool FTFont::FaceSize(const unsigned int size,
//                               const unsigned int res = 72);
C_FUN(int, ftglSetFontFaceSize, (FTGLfont *f, unsigned int s, unsigned int r),
      return 0, FaceSize, (s, r > 0 ? r : 72));

// unsigned int FTFont::FaceSize() const;
// XXX: need to call FaceSize() as FTFont::FaceSize() because of FTGLTexture
C_FUN(unsigned int, ftglGetFontFaceSize, (FTGLfont *f),
      return 0, FTFont::FaceSize, ());

// virtual void FTFont::Depth(float depth);
C_FUN(void, ftglSetFontDepth, (FTGLfont *f, float d), return, Depth, (d));

// virtual void FTFont::Outset(float front, float back);
C_FUN(void, ftglSetFontOutset, (FTGLfont *f, float front, float back),
      return, FTFont::Outset, (front, back));

// void FTFont::UseDisplayList(bool useList);
C_FUN(void, ftglSetFontDisplayList, (FTGLfont *f, int l),
      return, UseDisplayList, (l != 0));

// float FTFont::Ascender() const;
C_FUN(float, ftglGetFontAscender, (FTGLfont *f), return 0.f, Ascender, ());

// float FTFont::Descender() const;
C_FUN(float, ftglGetFontDescender, (FTGLfont *f), return 0.f, Descender, ());

// float FTFont::LineHeight() const;
C_FUN(float, ftglGetFontLineHeight, (FTGLfont *f), return 0.f, LineHeight, ());

// void FTFont::BBox(const char* string, float& llx, float& lly, float& llz,
//                   float& urx, float& ury, float& urz);
extern "C++" {
C_FUN(static FTBBox, _ftglGetFontBBox, (FTGLfont *f, char const *s, int len),
      return static_ftbbox, BBox, (s, len));
}

void ftglGetFontBBox(FTGLfont *f, const char* s, int len, float c[6])
{
    FTBBox ret = _ftglGetFontBBox(f, s, len);
    FTPoint lower = ret.Lower(), upper = ret.Upper();
    c[0] = lower.Xf(); c[1] = lower.Yf(); c[2] = lower.Zf();
    c[3] = upper.Xf(); c[4] = upper.Yf(); c[5] = upper.Zf();
}

// float FTFont::Advance(const char* string);
C_FUN(float, ftglGetFontAdvance, (FTGLfont *f, char const *s),
      return 0.0, Advance, (s));

// virtual void Render(const char* string, int renderMode);
extern "C++" {
C_FUN(static FTPoint, _ftglRenderFont, (FTGLfont *f, char const *s, int len,
                                        FTPoint pos, FTPoint spacing, int mode),
      return static_ftpoint, Render, (s, len, pos, spacing, mode));
}

void ftglRenderFont(FTGLfont *f, const char *s, int mode)
{
    _ftglRenderFont(f, s, -1, FTPoint(), FTPoint(), mode);
}

// FT_Error FTFont::Error() const;
C_FUN(FT_Error, ftglGetFontError, (FTGLfont *f), return -1, Error, ());

FTGL_END_C_DECLS

