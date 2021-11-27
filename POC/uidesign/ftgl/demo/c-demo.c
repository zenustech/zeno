/*
 * c-demo.cpp - simple C demo for FTGL, the OpenGL font library
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

#include <math.h> /* sin(), cos() */
#include <stdlib.h> /* exit() */

#if defined HAVE_GL_GLUT_H
#   include <GL/glut.h>
#elif defined HAVE_GLUT_GLUT_H
#   include <GLUT/glut.h>
#else
#   error GLUT headers not present
#endif

#include <FTGL/ftgl.h>

static FTGLfont *font[3];
static int fontindex = 0;
static int lastfps = 0;
static int frames = 0;

/*
 * HaloGlyph is a derivation of FTGLglyph that displays a polygon glyph
 * and a halo of outline glyphs at varying positions and with varying
 * outset values.
 */
struct HaloGlyph
{
    FTGLglyph *subglyph[5];
};

static void RenderHalo(FTGLglyph * baseGlyph, void *data,
                       FTGL_DOUBLE penx, FTGL_DOUBLE peny, int renderMode,
                       FTGL_DOUBLE *advancex, FTGL_DOUBLE *advancey)
{
    struct HaloGlyph *p = (struct HaloGlyph *)data;
    int i;

    glPushMatrix();
    for(i = 0; i < 5; i++)
    {
        glTranslatef(0.0f, 0.0f, -2.0f);
        ftglRenderGlyph(p->subglyph[i], penx, peny, renderMode,
                        advancex, advancey);
    }
    glPopMatrix();

    ftglRenderGlyph(baseGlyph, penx, peny, renderMode, advancex, advancey);
}

static void DestroyHalo(FTGLglyph * baseGlyph, void *data)
{
    struct HaloGlyph *p = (struct HaloGlyph *)data;
    int i;

    for(i = 0; i < 5; i++)
    {
        ftglDestroyGlyph(p->subglyph[i]);
    }

    ftglDestroyGlyph(baseGlyph);
    free(p);
}

static FTGLglyph *MakeHaloGlyph(FT_GlyphSlot slot, void *data)
{
    struct HaloGlyph *p = malloc(sizeof(struct HaloGlyph));
    FTGLglyph *baseGlyph = ftglCreatePolygonGlyph(slot, 0.0f, 1.0f);
    int i;

    for(i = 0; i < 5; i++)
    {
        p->subglyph[i] = ftglCreateOutlineGlyph(slot, i, 1);
    }

    return ftglCreateCustomGlyph(baseGlyph, p, RenderHalo, DestroyHalo);
}

/*
 * Main OpenGL loop: set up lights, apply a few rotation effects, and
 * render text using the current FTGL object.
 */
static void RenderScene(void)
{
    int now = glutGet(GLUT_ELAPSED_TIME);

    float n = (float)now / 20.0f;
    float t1 = sin(n / 80.0f);
    float t2 = sin(n / 50.0f + 1.0f);
    float t3 = sin(n / 30.0f + 2.0f);

    float ambient[4]  = { (t1 + 2.0f) / 3.0f,
                          (t2 + 2.0f) / 3.0f,
                          (t3 + 2.0f) / 3.0f, 0.3f };
    float diffuse[4]  = { 1.0f, 0.9f, 0.9f, 1.0f };
    float specular[4] = { 1.0f, 0.7f, 0.7f, 1.0f };
    float position[4] = { 100.0f, 100.0f, 0.0f, 1.0f };

    float front_ambient[4]  = { 0.7f, 0.7f, 0.7f, 0.0f };

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_LIGHTING);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LINE_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    glPushMatrix();
        glTranslatef(-0.9f, -0.2f, -10.0f);
        glLightfv(GL_LIGHT1, GL_AMBIENT,  ambient);
        glLightfv(GL_LIGHT1, GL_DIFFUSE,  diffuse);
        glLightfv(GL_LIGHT1, GL_SPECULAR, specular);
        glLightfv(GL_LIGHT1, GL_POSITION, position);
        glEnable(GL_LIGHT1);
    glPopMatrix();

    glPushMatrix();
        glMaterialfv(GL_FRONT, GL_AMBIENT, front_ambient);
        glColorMaterial(GL_FRONT, GL_DIFFUSE);
        glTranslatef(0.0f, 0.0f, 20.0f);
        glRotatef(n / 1.11f, 0.0f, 1.0f, 0.0f);
        glRotatef(n / 2.23f, 1.0f, 0.0f, 0.0f);
        glRotatef(n / 3.17f, 0.0f, 0.0f, 1.0f);
        glTranslatef(-260.0f, -0.2f, 0.0f);
        glColor3f(0.0f, 0.0f, 0.0f);
        ftglRenderFont(font[fontindex], "Hello FTGL!", FTGL_RENDER_ALL);
    glPopMatrix();

    glutSwapBuffers();

    frames++;

    if(now - lastfps > 5000)
    {
        fprintf(stderr, "%i frames in 5.0 seconds = %g FPS\n",
                frames, frames * 1000. / (now - lastfps));
        lastfps += 5000;
        frames = 0;
    }
}

/*
 * GLUT key processing function: <esc> quits, <tab> cycles across fonts.
 */
static void ProcessKeys(unsigned char key, int x, int y)
{
    switch(key)
    {
    case 27:
        ftglDestroyFont(font[0]);
        ftglDestroyFont(font[1]);
        ftglDestroyFont(font[2]);
        exit(EXIT_SUCCESS);
        break;
    case '\t':
        fontindex = (fontindex + 1) % 3;
        break;
    }
}

/*
 * Main program entry point: set up GLUT window, load fonts, run GLUT loop.
 */
int main(int argc, char **argv)
{
    char const *file = NULL;

#ifdef FONT_FILE
    file = FONT_FILE;
#else
    if(argc < 2)
    {
        fprintf(stderr, "Usage: %s <font_name.ttf>\n", argv[0]);
        return EXIT_FAILURE;
    }
#endif

    if(argc > 1)
    {
        file = argv[1];
    }

    /* Initialise GLUT stuff */
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(100, 100);
    glutInitWindowSize(640, 480);
    glutCreateWindow("simple FTGL C demo");

    glutDisplayFunc(RenderScene);
    glutIdleFunc(RenderScene);
    glutKeyboardFunc(ProcessKeys);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(90, 640.0f / 480.0f, 1, 1000);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(0.0, 0.0, 640.0f / 2.0f, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

    /* Initialise FTGL stuff */
    font[0] = ftglCreateExtrudeFont(file);
    font[1] = ftglCreateBufferFont(file);
    font[2] = ftglCreateCustomFont(file, NULL, MakeHaloGlyph);
    if(!font[0] || !font[1] || !font[2])
    {
        fprintf(stderr, "%s: could not load font `%s'\n", argv[0], file);
        return EXIT_FAILURE;
    }

    ftglSetFontFaceSize(font[0], 80, 72);
    ftglSetFontDepth(font[0], 10);
    ftglSetFontOutset(font[0], 0, 3);
    ftglSetFontCharMap(font[0], ft_encoding_unicode);

    ftglSetFontFaceSize(font[1], 80, 72);
    ftglSetFontCharMap(font[1], ft_encoding_unicode);

    ftglSetFontFaceSize(font[2], 80, 72);
    ftglSetFontCharMap(font[2], ft_encoding_unicode);

    fprintf(stderr, "Using FTGL version %s\n",
            ftglGetString(FTGL_CONFIG_VERSION));

    /* Run GLUT loop */
    glutMainLoop();

    return EXIT_SUCCESS;
}

