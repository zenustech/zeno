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

static const FTBBox static_ftbbox;

FTGL_BEGIN_C_DECLS

#define C_TOR(cname, cargs, cxxname, cxxarg, cxxtype) \
    FTGLlayout* cname cargs \
    { \
        cxxname *l = new cxxname cxxarg; \
        if(l->Error()) \
        { \
            delete l; \
            return NULL; \
        } \
        FTGLlayout *ftgl = (FTGLlayout *)malloc(sizeof(FTGLlayout)); \
        ftgl->ptr = l; \
        ftgl->type = cxxtype; \
        return ftgl; \
    }

// FTSimpleLayout::FTSimpleLayout();
C_TOR(ftglCreateSimpleLayout, (), FTSimpleLayout, (), LAYOUT_SIMPLE);

#define C_FUN(cret, cname, cargs, cxxerr, cxxname, cxxarg) \
    cret cname cargs \
    { \
        if(!l || !l->ptr) \
        { \
            fprintf(stderr, "FTGL warning: NULL pointer in %s\n", #cname); \
            cxxerr; \
        } \
        return l->ptr->cxxname cxxarg; \
    }

// FTLayout::~FTLayout();
void ftglDestroyLayout(FTGLlayout *l)
{
    if(!l || !l->ptr)
    {
        fprintf(stderr, "FTGL warning: NULL pointer in %s\n", __FUNC__);
        return;
    }
    delete l->ptr;
    free(l);
}

// virtual FTBBox FTLayout::BBox(const char* string)
extern "C++" {
C_FUN(static FTBBox, _ftglGetLayoutBBox, (FTGLlayout *l, const char *s),
      return static_ftbbox, BBox, (s));
}

void ftglGetLayoutBBox(FTGLlayout *l, const char * s, float c[6])
{
    FTBBox ret = _ftglGetLayoutBBox(l, s);
    FTPoint lower = ret.Lower(), upper = ret.Upper();
    c[0] = lower.Xf(); c[1] = lower.Yf(); c[2] = lower.Zf();
    c[3] = upper.Xf(); c[4] = upper.Yf(); c[5] = upper.Zf();
}

// virtual void FTLayout::Render(const char* string, int renderMode);
C_FUN(void, ftglRenderLayout, (FTGLlayout *l, const char *s, int r),
      return, Render, (s, -1, FTPoint(), r));

// FT_Error FTLayout::Error() const;
C_FUN(FT_Error, ftglGetLayoutError, (FTGLlayout *l), return -1, Error, ());

// void FTSimpleLayout::SetFont(FTFont *fontInit)
void ftglSetLayoutFont(FTGLlayout *l, FTGLfont *font)
{
    if(!l || !l->ptr)
    {
        fprintf(stderr, "FTGL warning: NULL pointer in %s\n", __FUNC__);
        return;
    }
    if(l->type != FTGL::LAYOUT_SIMPLE)
    {
        fprintf(stderr, "FTGL warning: %s not implemented for %d\n",
                        __FUNC__, l->type);
    }
    l->font = font;
    return dynamic_cast<FTSimpleLayout*>(l->ptr)->SetFont(font->ptr);
}

// FTFont *FTSimpleLayout::GetFont()
FTGLfont *ftglGetLayoutFont(FTGLlayout *l)
{
    if(!l || !l->ptr)
    {
        fprintf(stderr, "FTGL warning: NULL pointer in %s\n", __FUNC__);
        return NULL;
    }
    if(l->type != FTGL::LAYOUT_SIMPLE)
    {
        fprintf(stderr, "FTGL warning: %s not implemented for %d\n",
                        __FUNC__, l->type);
    }
    return l->font;
}

#undef C_FUN

#define C_FUN(cret, cname, cargs, cxxerr, cxxname, cxxarg) \
    cret cname cargs \
    { \
        if(!l || !l->ptr) \
        { \
            fprintf(stderr, "FTGL warning: NULL pointer in %s\n", #cname); \
            cxxerr; \
        } \
        if(l->type != FTGL::LAYOUT_SIMPLE) \
        { \
            fprintf(stderr, "FTGL warning: %s not implemented for %d\n", \
                            __FUNC__, l->type); \
            cxxerr; \
        } \
        return dynamic_cast<FTSimpleLayout*>(l->ptr)->cxxname cxxarg; \
    }

// void FTSimpleLayout::SetLineLength(const float LineLength);
C_FUN(void, ftglSetLayoutLineLength, (FTGLlayout *l, const float length),
      return, SetLineLength, (length));

// float FTSimpleLayout::GetLineLength() const
C_FUN(float, ftglGetLayoutLineLength, (FTGLlayout *l),
      return 0.0f, GetLineLength, ());

// void FTSimpleLayout::SetAlignment(const TextAlignment Alignment)
C_FUN(void, ftglSetLayoutAlignment, (FTGLlayout *l, const int a),
      return, SetAlignment, ((FTGL::TextAlignment)a));

// TextAlignment FTSimpleLayout::GetAlignment() const
C_FUN(int, ftglGetLayoutAlignment, (FTGLlayout *l),
      return FTGL::ALIGN_LEFT, GetAlignment, ());
C_FUN(int, ftglGetLayoutAlignement, (FTGLlayout *l),
      return FTGL::ALIGN_LEFT, GetAlignment, ()); // old typo

// void FTSimpleLayout::SetLineSpacing(const float LineSpacing)
C_FUN(void, ftglSetLayoutLineSpacing, (FTGLlayout *l, const float f),
      return, SetLineSpacing, (f));

FTGL_END_C_DECLS

