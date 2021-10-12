/*
 * simple.cpp - simple demo for FTGL, the OpenGL font library
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

#include <math.h> // sin(), cos()
#include <stdlib.h> // exit()

#if defined HAVE_GL_GLUT_H
#   include <GL/glut.h>
#elif defined HAVE_GLUT_GLUT_H
#   include <GLUT/glut.h>
#else
#   error GLUT headers not present
#endif

#include <FTGL/ftgl.h>

static FTFont *font[3];
static int fontindex = 0;
static int lastfps = 0;
static int frames = 0;

//
//  FTHaloGlyph is a derivation of FTPolygonGlyph that also displays a
//  halo of FTOutlineGlyph objects at varying positions and with varying
//  outset values.
//
class FTHaloGlyph : public FTPolygonGlyph
{
    public:
        FTHaloGlyph(FT_GlyphSlot glyph) : FTPolygonGlyph(glyph, 0, true)
        {
            for(int i = 0; i < 5; i++)
            {
                subglyph[i] = new FTOutlineGlyph(glyph, i, true);
            }
        }

    private:
        const FTPoint& Render(const FTPoint& pen, int renderMode)
        {
            glPushMatrix();
            for(int i = 0; i < 5; i++)
            {
                glTranslatef(0.0, 0.0, -2.0);
                subglyph[i]->Render(pen, renderMode);
            }
            glPopMatrix();

            return FTPolygonGlyph::Render(pen, renderMode);
        }

        FTGlyph *subglyph[5];
};

//
//  FTHaloFont is a simple FTFont derivation that builds FTHaloGlyph
//  objects.
//
class FTHaloFont : public FTFont
{
    public:
        FTHaloFont(char const *fontFilePath) : FTFont(fontFilePath) {}

    private:
        virtual FTGlyph* MakeGlyph(FT_GlyphSlot slot)
        {
            return new FTHaloGlyph(slot);
        }
};

//
//  Main OpenGL loop: set up lights, apply a few rotation effects, and
//  render text using the current FTGL object.
//
static void RenderScene(void)
{
    int now = glutGet(GLUT_ELAPSED_TIME);

    float n = (float)now / 20.;
    float t1 = sin(n / 80);
    float t2 = sin(n / 50 + 1);
    float t3 = sin(n / 30 + 2);

    float ambient[4]  = { (t1 + 2.0) / 3,
                          (t2 + 2.0) / 3,
                          (t3 + 2.0) / 3, 0.3 };
    float diffuse[4]  = { 1.0, 0.9, 0.9, 1.0 };
    float specular[4] = { 1.0, 0.7, 0.7, 1.0 };
    float position[4] = { 100.0, 100.0, 0.0, 1.0 };

    float front_ambient[4]  = { 0.7, 0.7, 0.7, 0.0 };

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_LIGHTING);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LINE_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    glPushMatrix();
        glTranslatef(-0.9, -0.2, -10.0);
        glLightfv(GL_LIGHT1, GL_AMBIENT,  ambient);
        glLightfv(GL_LIGHT1, GL_DIFFUSE,  diffuse);
        glLightfv(GL_LIGHT1, GL_SPECULAR, specular);
        glLightfv(GL_LIGHT1, GL_POSITION, position);
        glEnable(GL_LIGHT1);
    glPopMatrix();

    glPushMatrix();
        glMaterialfv(GL_FRONT, GL_AMBIENT, front_ambient);
        glColorMaterial(GL_FRONT, GL_DIFFUSE);
        glTranslatef(0.0, 0.0, 20.0);
        glRotatef(n / 1.11, 0.0, 1.0, 0.0);
        glRotatef(n / 2.23, 1.0, 0.0, 0.0);
        glRotatef(n / 3.17, 0.0, 0.0, 1.0);
        glTranslatef(-260.0, -0.2, 0.0);
        glColor3f(1.0, 1.0, 1.0);
        font[fontindex]->Render("Hello FTGL!");
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

//
//  GLUT key processing function: <esc> quits, <tab> cycles across fonts.
//
static void ProcessKeys(unsigned char key, int x, int y)
{
    switch(key)
    {
    case 27:
        delete font[0];
        delete font[1];
        delete font[2];
        exit(EXIT_SUCCESS);
        break;
    case '\t':
        fontindex = (fontindex + 1) % 3;
        break;
    }
}

//
//  Main program entry point: set up GLUT window, load fonts, run GLUT loop.
//
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

    // Initialise GLUT stuff
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(100, 100);
    glutInitWindowSize(640, 480);
    glutCreateWindow("simple FTGL C++ demo");

    glutDisplayFunc(RenderScene);
    glutIdleFunc(RenderScene);
    glutKeyboardFunc(ProcessKeys);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(90, 640.0f / 480.0f, 1, 1000);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(0.0, 0.0, 640.0f / 2.0f, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

    // Initialise FTGL stuff
    font[0] = new FTExtrudeFont(file);
    font[1] = new FTBufferFont(file);
    font[2] = new FTHaloFont(file);

    if(font[0]->Error() || font[1]->Error() || font[2]->Error())
    {
        fprintf(stderr, "%s: could not load font `%s'\n", argv[0], file);
        return EXIT_FAILURE;
    }

    font[0]->FaceSize(80);
    font[0]->Depth(10);
    font[0]->Outset(0, 3);
    font[0]->CharMap(ft_encoding_unicode);

    font[1]->FaceSize(80);
    font[1]->CharMap(ft_encoding_unicode);

    font[2]->FaceSize(80);
    font[2]->CharMap(ft_encoding_unicode);

    fprintf(stderr, "Using FTGL version %s\n",
            FTGL::GetString(FTGL::CONFIG_VERSION));

    // Run GLUT loop
    glutMainLoop();

    return EXIT_SUCCESS;
}

