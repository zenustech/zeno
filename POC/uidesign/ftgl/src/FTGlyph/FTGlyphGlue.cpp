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

#include "FTInternals.h"

static const FTPoint static_ftpoint;
static const FTBBox static_ftbbox;

FTGL_BEGIN_C_DECLS

#define C_TOR(cname, cargs, cxxname, cxxarg, cxxtype) \
    FTGLglyph* cname cargs \
    { \
        cxxname *g = new cxxname cxxarg; \
        if(g->Error()) \
        { \
            delete g; \
            return NULL; \
        } \
        FTGLglyph *ftgl = (FTGLglyph *)malloc(sizeof(FTGLglyph)); \
        ftgl->ptr = g; \
        ftgl->type = cxxtype; \
        return ftgl; \
    }

// FTBitmapGlyph::FTBitmapGlyph();
C_TOR(ftglCreateBitmapGlyph, (FT_GlyphSlot glyph),
      FTBitmapGlyph, (glyph), GLYPH_BITMAP);

// FTBufferGlyph::FTBufferGlyph();
// FIXME: not implemented

// FTExtrudeGlyph::FTExtrudeGlyph();
C_TOR(ftglCreateExtrudeGlyph, (FT_GlyphSlot glyph, float depth,
                   float frontOutset, float backOutset, int useDisplayList),
      FTExtrudeGlyph, (glyph, depth, frontOutset, backOutset, (useDisplayList != 0)),
      GLYPH_EXTRUDE);

// FTOutlineGlyph::FTOutlineGlyph();
C_TOR(ftglCreateOutlineGlyph, (FT_GlyphSlot glyph, float outset,
                               int useDisplayList),
      FTOutlineGlyph, (glyph, outset, (useDisplayList != 0)), GLYPH_OUTLINE);

// FTPixmapGlyph::FTPixmapGlyph();
C_TOR(ftglCreatePixmapGlyph, (FT_GlyphSlot glyph),
      FTPixmapGlyph, (glyph), GLYPH_PIXMAP);

// FTPolygonGlyph::FTPolygonGlyph();
C_TOR(ftglCreatePolygonGlyph, (FT_GlyphSlot glyph, float outset,
                               int useDisplayList),
      FTPolygonGlyph, (glyph, outset, (useDisplayList != 0)), GLYPH_POLYGON);

// FTTextureGlyph::FTTextureGlyph();
C_TOR(ftglCreateTextureGlyph, (FT_GlyphSlot glyph, int id, int xOffset,
                               int yOffset, int width, int height),
      FTTextureGlyph, (glyph, id, xOffset, yOffset, width, height),
      GLYPH_TEXTURE);

// FTCustomGlyph::FTCustomGlyph();
class FTCustomGlyph : public FTGlyph
{
public:
    FTCustomGlyph(FTGLglyph *base, void *p,
                  void (*render) (FTGLglyph *, void *, FTGL_DOUBLE, FTGL_DOUBLE,
                                  int, FTGL_DOUBLE *, FTGL_DOUBLE *),
                  void (*destroy) (FTGLglyph *, void *))
     : FTGlyph((FT_GlyphSlot)0),
       baseGlyph(base),
       data(p),
       renderCallback(render),
       destroyCallback(destroy)
    {}

    ~FTCustomGlyph()
    {
        destroyCallback(baseGlyph, data);
    }

    float Advance() const { return baseGlyph->ptr->Advance(); }

    const FTPoint& Render(const FTPoint& pen, int renderMode)
    {
        FTGL_DOUBLE advancex, advancey;
        renderCallback(baseGlyph, data, pen.X(), pen.Y(), renderMode,
                       &advancex, &advancey);
        advance = FTPoint(advancex, advancey);
        return advance;
    }

    const FTBBox& BBox() const { return baseGlyph->ptr->BBox(); }

    FT_Error Error() const { return baseGlyph->ptr->Error(); }

private:
    FTPoint advance;
    FTGLglyph *baseGlyph;
    void *data;
    void (*renderCallback) (FTGLglyph *, void *, FTGL_DOUBLE, FTGL_DOUBLE,
                            int, FTGL_DOUBLE *, FTGL_DOUBLE *);
    void (*destroyCallback) (FTGLglyph *, void *);
};

C_TOR(ftglCreateCustomGlyph, (FTGLglyph *base, void *data,
         void (*renderCallback) (FTGLglyph *, void *, FTGL_DOUBLE, FTGL_DOUBLE,
                                 int, FTGL_DOUBLE *, FTGL_DOUBLE *),
         void (*destroyCallback) (FTGLglyph *, void *)),
      FTCustomGlyph, (base, data, renderCallback, destroyCallback),
      GLYPH_CUSTOM);

#define C_FUN(cret, cname, cargs, cxxerr, cxxname, cxxarg) \
    cret cname cargs \
    { \
        if(!g || !g->ptr) \
        { \
            fprintf(stderr, "FTGL warning: NULL pointer in %s\n", #cname); \
            cxxerr; \
        } \
        return g->ptr->cxxname cxxarg; \
    }

// FTGlyph::~FTGlyph();
void ftglDestroyGlyph(FTGLglyph *g)
{
    if(!g || !g->ptr)
    {
        fprintf(stderr, "FTGL warning: NULL pointer in %s\n", __FUNC__);
        return;
    }
    delete g->ptr;
    free(g);
}

// const FTPoint& FTGlyph::Render(const FTPoint& pen, int renderMode);
extern "C++" {
C_FUN(static const FTPoint&, _ftglRenderGlyph, (FTGLglyph *g,
                                   const FTPoint& pen, int renderMode),
      return static_ftpoint, Render, (pen, renderMode));
}

void ftglRenderGlyph(FTGLglyph *g, FTGL_DOUBLE penx, FTGL_DOUBLE peny,
                     int renderMode, FTGL_DOUBLE *advancex,
                     FTGL_DOUBLE *advancey)
{
    FTPoint pen(penx, peny);
    FTPoint ret = _ftglRenderGlyph(g, pen, renderMode);
    *advancex = ret.X();
    *advancey = ret.Y();
}

// float FTGlyph::Advance() const;
C_FUN(float, ftglGetGlyphAdvance, (FTGLglyph *g), return 0.0, Advance, ());

// const FTBBox& FTGlyph::BBox() const;
extern "C++" {
C_FUN(static const FTBBox&, _ftglGetGlyphBBox, (FTGLglyph *g),
      return static_ftbbox, BBox, ());
}

void ftglGetGlyphBBox(FTGLglyph *g, float bounds[6])
{
    FTBBox ret = _ftglGetGlyphBBox(g);
    FTPoint lower = ret.Lower(), upper = ret.Upper();
    bounds[0] = lower.Xf(); bounds[1] = lower.Yf(); bounds[2] = lower.Zf();
    bounds[3] = upper.Xf(); bounds[4] = upper.Yf(); bounds[5] = upper.Zf();
}

// FT_Error FTGlyph::Error() const;
C_FUN(FT_Error, ftglGetGlyphError, (FTGLglyph *g), return -1, Error, ());

FTGL_END_C_DECLS

