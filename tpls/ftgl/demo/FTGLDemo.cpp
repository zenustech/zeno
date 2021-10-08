/*                        -*- coding: utf-8 -*-
 * FTGLDemo - simple demo for FTGL, the OpenGL font library
 *
 * Copyright (c) 2001-2004 Henry Maddocks <ftgl@opengl.geek.nz>
 *               2008 Sam Hocevar <sam@hocevar.net>
 *               2008 Éric Beets <ericbeets@free.fr>
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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <sstream>

#if defined HAVE_GL_GLUT_H
#   include <GL/glut.h>
#elif defined HAVE_GLUT_GLUT_H
#   include <GLUT/glut.h>
#else
#   error GLUT headers not present
#endif

#include <FTGL/ftgl.h>

#include "tb.h"

#if !defined FONT_FILE
    // Put your font file here if configure did not find it.
#   define FONT_FILE 0
#endif

#define EDITING 1
#define INTERACTIVE 2

#define FTGL_BITMAP 0
#define FTGL_PIXMAP 1
#define FTGL_OUTLINE 2
#define FTGL_POLYGON 3
#define FTGL_EXTRUDE 4
#define FTGL_TEXTURE 5
#define FTGL_BUFFER 6

char const* fontfile = FONT_FILE;
int current_font = FTGL_EXTRUDE;

GLint w_win = 640, h_win = 480;
int mode = INTERACTIVE;
int carat = 0;

FTSimpleLayout simpleLayout;
FTLayout *layouts[] = { &simpleLayout, NULL };
int currentLayout = 0;
const int NumLayouts = 2;

const float InitialLineLength = 600.0f;

const float OX = -300;
const float OY = 170;

//wchar_t myString[16] = { 0x6FB3, 0x9580};
char myString[4096];

static int const FTGL_NUM_FONTS = 7;
static FTFont* fonts[FTGL_NUM_FONTS];
static FTPixmapFont* infoFont;

static float textures[][48] =
{
    {
        1.0, 1.0, 1.0, 0.7, 0.7, 0.7, 1.0, 1.0, 1.0, 0.7, 0.7, 0.7,
        0.7, 0.7, 0.7, 0.4, 0.4, 0.4, 0.7, 0.7, 0.7, 0.4, 0.4, 0.4,
        1.0, 1.0, 1.0, 0.7, 0.7, 0.7, 1.0, 1.0, 1.0, 0.7, 0.7, 0.7,
        0.7, 0.7, 0.7, 0.4, 0.4, 0.4, 0.7, 0.7, 0.7, 0.4, 0.4, 0.4,
    },
    {
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
        0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
        0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
    }
};

static GLuint textureID[2];

void SetCamera(void);

void setUpLighting()
{
    // Set up lighting.
    float light1_ambient[4]  = { 0.5, 0.5, 0.5, 1.0 };
    float light1_diffuse[4]  = { 1.0, 0.9, 0.9, 1.0 };
    float light1_specular[4] = { 1.0, 0.7, 0.7, 1.0 };
    float light1_position[4] = { 400.0, 400.0, 100.0, 1.0 };
    glLightfv(GL_LIGHT1, GL_AMBIENT,  light1_ambient);
    glLightfv(GL_LIGHT1, GL_DIFFUSE,  light1_diffuse);
    glLightfv(GL_LIGHT1, GL_SPECULAR, light1_specular);
    glLightfv(GL_LIGHT1, GL_POSITION, light1_position);
    glEnable(GL_LIGHT1);

    float front_emission[4] = { 0.5, 0.4, 0.3, 0.0 };
    float front_ambient[4]  = { 0.4, 0.4, 0.4, 0.0 };
    float front_diffuse[4]  = { 0.95, 0.95, 0.8, 0.0 };
    float front_specular[4] = { 0.8, 0.8, 0.8, 0.0 };
    glMaterialfv(GL_FRONT, GL_EMISSION, front_emission);
    glMaterialfv(GL_FRONT, GL_AMBIENT, front_ambient);
    glMaterialfv(GL_FRONT, GL_DIFFUSE, front_diffuse);
    glMaterialfv(GL_FRONT, GL_SPECULAR, front_specular);
    glMaterialf(GL_FRONT, GL_SHININESS, 25.0);
    glColor4fv(front_diffuse);
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_FALSE);

    glColorMaterial(GL_FRONT, GL_DIFFUSE);
    glEnable(GL_COLOR_MATERIAL);

    glEnable(GL_LIGHTING);
}


void setUpFonts(const char* file)
{
    fonts[FTGL_BITMAP] = new FTBitmapFont(file);
    fonts[FTGL_PIXMAP] = new FTPixmapFont(file);
    fonts[FTGL_OUTLINE] = new FTOutlineFont(file);
    fonts[FTGL_POLYGON] = new FTPolygonFont(file);
    fonts[FTGL_EXTRUDE] = new FTExtrudeFont(file);
    fonts[FTGL_TEXTURE] = new FTTextureFont(file);
    fonts[FTGL_BUFFER] = new FTBufferFont(file);

    for(int x = 0; x < FTGL_NUM_FONTS; ++x)
    {
        if(fonts[x]->Error())
        {
            fprintf(stderr, "Failed to open font %s", file);
            exit(1);
        }

        if(!fonts[x]->FaceSize(30))
        {
            fprintf(stderr, "Failed to set size");
            exit(1);
        }

        fonts[x]->Depth(3.);
        fonts[x]->Outset(-.5, 1.5);

        fonts[x]->CharMap(ft_encoding_unicode);
    }

    infoFont = new FTPixmapFont(file);

    if(infoFont->Error())
    {
        fprintf(stderr, "Failed to open font %s", file);
        exit(1);
    }

    infoFont->FaceSize(18);
#if 1
    strcpy(myString, "OpenGL is a powerful software interface for graphics "
           "hardware that allows graphics programmers to produce high-quality "
           "color images of 3D objects.\nabc def ghij klm nop qrs tuv wxyz "
           "ABC DEF GHIJ KLM NOP QRS TUV WXYZ 01 23 45 67 89");
#elif 0
    strcpy(myString, "OpenGL (Open Graphics Library — открытая графическая "
           "библиотека) — спецификация, определяющая независимый от языка "
           "программирования кросс-платформенный программный интерфейс "
           "для написания приложений, использующих двумерную и трехмерную "
           "компьютерную графику.");
#else
    strcpy(myString, "OpenGL™ 是行业领域中最为广泛接纳的 2D/3D 图形 API, "
           "其自诞生至今已催生了各种计算机平台及设备上的数千优秀应用程序。"
           "OpenGL™ 是独立于视窗操作系统或其它操作系统的，亦是网络透明的。"
           "在包含CAD、内容创作、能源、娱乐、游戏开发、制造业、制药业及虚拟"
           "现实等行业领域中， OpenGL™ 帮助程序员实现在 PC、工作站、超级计算"
           "机等硬件设备上的高性能、极具冲击力的高视觉表现力图形处理软件的开"
           "发。");
#endif
}


void renderFontmetrics()
{
    FTBBox bbox;
    float x1, y1, z1, x2, y2, z2;

    // If there is a layout, use it to compute the bbox, otherwise query as
    // a string.
    if(layouts[currentLayout])
        bbox = layouts[currentLayout]->BBox(myString);
    else
        bbox = fonts[current_font]->BBox(myString);

    x1 = bbox.Lower().Xf(); y1 = bbox.Lower().Yf(); z1 = bbox.Lower().Zf();
    x2 = bbox.Upper().Xf(); y2 = bbox.Upper().Yf(); z2 = bbox.Upper().Zf();

    // Draw the bounding box
    glDisable(GL_LIGHTING);
    glEnable(GL_LINE_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE); // GL_ONE_MINUS_SRC_ALPHA

    glColor3f(0.0, 1.0, 0.0);
    // Draw the front face
    glBegin(GL_LINE_LOOP);
        glVertex3f(x1, y1, z1);
        glVertex3f(x1, y2, z1);
        glVertex3f(x2, y2, z1);
        glVertex3f(x2, y1, z1);
    glEnd();
    // Draw the back face
    if(current_font == FTGL_EXTRUDE && z1 != z2)
    {
        glBegin(GL_LINE_LOOP);
            glVertex3f(x1, y1, z2);
            glVertex3f(x1, y2, z2);
            glVertex3f(x2, y2, z2);
            glVertex3f(x2, y1, z2);
        glEnd();
        // Join the faces
        glBegin(GL_LINES);
            glVertex3f(x1, y1, z1);
            glVertex3f(x1, y1, z2);

            glVertex3f(x1, y2, z1);
            glVertex3f(x1, y2, z2);

            glVertex3f(x2, y2, z1);
            glVertex3f(x2, y2, z2);

            glVertex3f(x2, y1, z1);
            glVertex3f(x2, y1, z2);
        glEnd();
    }

    // Render layout-specific metrics
    if(!layouts[currentLayout])
    {
        // There is no layout. Draw the baseline, Ascender and Descender
        glBegin(GL_LINES);
            glColor3f(0.0, 0.0, 1.0);
            glVertex3f(0.0, 0.0, 0.0);
            glVertex3f(fonts[current_font]->Advance(myString), 0.0, 0.0);
            glVertex3f(0.0, fonts[current_font]->Ascender(), 0.0);
            glVertex3f(0.0, fonts[current_font]->Descender(), 0.0);
        glEnd();
    }
    else if(layouts[currentLayout]
             && (dynamic_cast <FTSimpleLayout *>(layouts[currentLayout])))
    {
        float lineWidth = ((FTSimpleLayout *)layouts[currentLayout])->GetLineLength();

        // The layout is a SimpleLayout.  Render guides that mark the edges
        // of the wrap region.
        glColor3f(0.5, 1.0, 1.0);
        glBegin(GL_LINES);
            glVertex3f(0, 10000, 0);
            glVertex3f(0, -10000, 0);
            glVertex3f(lineWidth, 10000, 0);
            glVertex3f(lineWidth, -10000, 0);
        glEnd();
    }

    // Draw the origin
    glTranslatef(-OX, -OY,0);
    glColor3f(1.0, 0.0, 0.0);
    glPointSize(5.0);
    glBegin(GL_POINTS);
        glVertex3f(0.0, 0.0, 0.0);
    glEnd();
    // Draw the axis
    glColor3f(1, 0, 0);
    glBegin(GL_LINES);
        glVertex3f(0,0,0);
        glVertex3f(100,0,0);
    glEnd();
    glColor3f(0, 1, 0);
    glBegin(GL_LINES);
        glVertex3f(0,0,0);
        glVertex3f(0,100,0);
    glEnd();
    glColor3f(0, 0, 1);
    glBegin(GL_LINES);
        glVertex3f(0,0,0);
        glVertex3f(0,0,100);
    glEnd();
}


void renderFontInfo()
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, w_win, 0, h_win);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // draw mode
    glColor3f(1.0, 1.0, 1.0);
    glRasterPos2f(20.0f , h_win - (20.0f + infoFont->Ascender()));

    switch(mode)
    {
        case EDITING:
            infoFont->Render("Edit Mode");
            break;
        case INTERACTIVE:
            break;
    }

    // draw font type
    glRasterPos2i(20 , 20);
    switch(current_font)
    {
        case FTGL_BITMAP:
            infoFont->Render("Bitmap Font");
            break;
        case FTGL_PIXMAP:
            infoFont->Render("Pixmap Font");
            break;
        case FTGL_OUTLINE:
            infoFont->Render("Outline Font");
            break;
        case FTGL_POLYGON:
            infoFont->Render("Polygon Font");
            break;
        case FTGL_EXTRUDE:
            infoFont->Render("Extruded Font");
            break;
        case FTGL_TEXTURE:
            infoFont->Render("Texture Font");
            break;
        case FTGL_BUFFER:
            infoFont->Render("Buffer Font");
            break;
    }

    glRasterPos2f(20.0f , 20.0f + infoFont->LineHeight());
    std::stringstream tmpbuf;
    tmpbuf << fontfile << " size: " << fonts[current_font]->FaceSize();
    infoFont->Render(tmpbuf.str().c_str());

    // If the current layout is a SimpleLayout, output the alignemnt mode
    if(layouts[currentLayout]
        && (dynamic_cast <FTSimpleLayout *>(layouts[currentLayout])))
    {
        glRasterPos2f(20.0f , 20.0f + 2*(infoFont->Ascender() - infoFont->Descender()));
        // Output the alignment mode of the layout
        switch (((FTSimpleLayout *)layouts[currentLayout])->GetAlignment())
        {
            case FTGL::ALIGN_LEFT:
                infoFont->Render("Align Left");
                break;
            case FTGL::ALIGN_RIGHT:
                infoFont->Render("Align Right");
                break;
            case FTGL::ALIGN_CENTER:
                infoFont->Render("Align Center");
                break;
            case FTGL::ALIGN_JUSTIFY:
                infoFont->Render("Align Justified");
                break;
        }
    }
}


void do_display (void)
{
    switch(current_font)
    {
        case FTGL_BITMAP:
        case FTGL_PIXMAP:
        case FTGL_OUTLINE:
            glDisable(GL_TEXTURE_2D);
            break;
        case FTGL_POLYGON:
            glDisable(GL_TEXTURE_2D);
            setUpLighting();
            break;
        case FTGL_EXTRUDE:
            glEnable(GL_DEPTH_TEST);
            glDisable(GL_BLEND);
            glEnable(GL_TEXTURE_2D);
            setUpLighting();
            glBindTexture(GL_TEXTURE_2D, textureID[0]);
            break;
        case FTGL_TEXTURE:
        case FTGL_BUFFER:
            glEnable(GL_TEXTURE_2D);
            glDisable(GL_DEPTH_TEST);
            setUpLighting();
            glNormal3f(0.0, 0.0, 1.0);
            break;
    }
    glTranslatef(OX, OY,0);

    // If you do want to switch the color of bitmaps rendered with glBitmap,
    // you will need to explicitly call glRasterPos (or its ilk) to lock
    // in a changed current color.

    glPushMatrix();
        glColor3f(1.0, 1.0, 1.0);
        int renderMode = FTGL::RENDER_FRONT | FTGL::RENDER_BACK;
        if(layouts[currentLayout])
            layouts[currentLayout]->Render(myString, -1,
                                           FTPoint(), renderMode);
        else
            fonts[current_font]->Render(myString, -1,
                                        FTPoint(), FTPoint(), renderMode);

        if(current_font == FTGL_EXTRUDE)
        {
            glBindTexture(GL_TEXTURE_2D, textureID[1]);
            renderMode = FTGL::RENDER_SIDE;
            if(layouts[currentLayout])
                layouts[currentLayout]->Render(myString, -1,
                                               FTPoint(), renderMode);
            else
                fonts[current_font]->Render(myString, -1,
                                            FTPoint(), FTPoint(), renderMode);
        }
    glPopMatrix();

    glPushMatrix();
        renderFontmetrics();
    glPopMatrix();

    renderFontInfo();
}


void display(void)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    SetCamera();

    switch(current_font)
    {
        case FTGL_BITMAP:
        case FTGL_PIXMAP:
            glRasterPos2i((long)(w_win / 2 + OX), (long)(h_win / 2 + OY));
            glTranslatef(w_win / 2, h_win / 2, 0.0);
            break;
        case FTGL_OUTLINE:
        case FTGL_POLYGON:
        case FTGL_EXTRUDE:
        case FTGL_TEXTURE:
        case FTGL_BUFFER:
            tbMatrix();
            break;
    }

    glPushMatrix();

    do_display();

    glPopMatrix();

    glutSwapBuffers();
}


void myinit(const char* file)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(0.5, 0.5, 0.7, 0.0);
    glColor3f(1.0, 1.0, 1.0);

    glEnable(GL_CULL_FACE);
    glFrontFace(GL_CCW);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glShadeModel(GL_SMOOTH);

    glEnable(GL_POLYGON_OFFSET_LINE);
    glPolygonOffset(1.0, 1.0); // ????

    SetCamera();

    tbInit(GLUT_LEFT_BUTTON);
    tbAnimate(GL_FALSE);

    setUpFonts(file);

    // Configure the SimpleLayout
    simpleLayout.SetLineLength(InitialLineLength);
    simpleLayout.SetFont(fonts[current_font]);

    glGenTextures(2, textureID);

    for(int i = 0; i < 2; i++)
    {
        glBindTexture(GL_TEXTURE_2D, textureID[i]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 4, 4, 0, GL_RGB, GL_FLOAT,
                     textures[i]);
    }
}


void parsekey(unsigned char key, int x, int y)
{
    switch (key)
    {
    case 27:
        exit(0);
        break;
    case 13:
        if(mode == EDITING)
        {
            mode = INTERACTIVE;
        }
        else
        {
            mode = EDITING;
            carat = 0;
        }
        break;
    case '\t':
        // If current layout is a SimpleLayout, change its alignment properties
        if(layouts[currentLayout]
             && (dynamic_cast <FTSimpleLayout *>(layouts[currentLayout])))
        {
            FTSimpleLayout *l = (FTSimpleLayout *)layouts[currentLayout];
            // Decrement the layout
            switch (l->GetAlignment())
            {
            case FTGL::ALIGN_LEFT:
                l->SetAlignment(FTGL::ALIGN_RIGHT);
                break;
            case FTGL::ALIGN_RIGHT:
                l->SetAlignment(FTGL::ALIGN_CENTER);
                break;
            case FTGL::ALIGN_CENTER:
                l->SetAlignment(FTGL::ALIGN_JUSTIFY);
                break;
            case FTGL::ALIGN_JUSTIFY:
                l->SetAlignment(FTGL::ALIGN_LEFT);
                break;
            }
        }
        break;
    default:
        if(mode == INTERACTIVE)
        {
            myString[0] = key;
            myString[1] = 0;
        }
        else
        {
            myString[carat] = key;
            myString[carat + 1] = 0;
            carat = carat > 2000 ? 2000 : carat + 1;
        }
        break;
    }

    glutPostRedisplay();
}


void parseSpecialKey(int key, int x, int y)
{
    FTSimpleLayout *l = NULL;
    unsigned int s;

    // If the currentLayout is a SimpleLayout store a pointer in l
    if(layouts[currentLayout]
        && (dynamic_cast <FTSimpleLayout *>(layouts[currentLayout])))
    {
        l = (FTSimpleLayout *)layouts[currentLayout];
    }

    switch (key)
    {
    case GLUT_KEY_UP:
        current_font = (current_font + 1) % FTGL_NUM_FONTS;
        break;
    case GLUT_KEY_DOWN:
        current_font = (current_font + FTGL_NUM_FONTS - 1) % FTGL_NUM_FONTS;
        break;
    case GLUT_KEY_PAGE_UP:
        currentLayout = (currentLayout + 1) % NumLayouts;
        break;
    case GLUT_KEY_PAGE_DOWN:
        currentLayout = (currentLayout + NumLayouts - 1) % NumLayouts;
        break;
    case GLUT_KEY_HOME:
        /* If the current layout is simple decrement its line length */
        if (l) l->SetLineLength(l->GetLineLength() - 10.0f);
        break;
    case GLUT_KEY_END:
        /* If the current layout is simple increment its line length */
        if (l) l->SetLineLength(l->GetLineLength() + 10.0f);
        break;
    case GLUT_KEY_LEFT:
        s = fonts[current_font]->FaceSize();
        if(s >= 2)
            fonts[current_font]->FaceSize(s - 1);
        break;
    case GLUT_KEY_RIGHT:
        fonts[current_font]->FaceSize(fonts[current_font]->FaceSize() + 1);
        break;
    }

    // If the current layout is a SimpleLayout, update its font.
    if(l)
    {
        l->SetFont(fonts[current_font]);
    }

    glutPostRedisplay();
}


void motion(int x, int y)
{
    tbMotion(x, y);
}

void mouse(int button, int state, int x, int y)
{
    tbMouse(button, state, x, y);
}

void myReshape(int w, int h)
{
    glMatrixMode (GL_MODELVIEW);
    glViewport (0, 0, w, h);
    glLoadIdentity();

    w_win = w;
    h_win = h;
    SetCamera();

    tbReshape(w_win, h_win);
}

void SetCamera(void)
{
    switch(current_font)
    {
        case FTGL_BITMAP:
        case FTGL_PIXMAP:
            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            gluOrtho2D(0, w_win, 0, h_win);
            glMatrixMode(GL_MODELVIEW);
            glLoadIdentity();
            break;
        case FTGL_OUTLINE:
        case FTGL_POLYGON:
        case FTGL_EXTRUDE:
        case FTGL_TEXTURE:
        case FTGL_BUFFER:
            glMatrixMode (GL_PROJECTION);
            glLoadIdentity ();
            gluPerspective(90, (float)w_win / (float)h_win, 1, 1000);
            glMatrixMode(GL_MODELVIEW);
            glLoadIdentity();
            gluLookAt(0.0, 0.0, (float)h_win / 2.0f, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
            break;
    }
}


int main(int argc, char *argv[])
{
#ifndef __APPLE_CC__ // Bloody finder args???
    if (argc == 2)
        fontfile = argv[1];
#endif

    if (!fontfile)
    {
        fprintf(stderr, "A font file must be specified on the command line\n");
        exit(1);
    }

    std::cout
        << "The following interactive commands are available:" << std::endl
        << "\tTAB:         change the alignment of the text" << std::endl
        << "\tENTER:       toggle interactive mode" << std::endl
        << std::endl
        << "\tARROW-UP:    cycle the font type up" << std::endl
        << "\tARROW-DOWN:  cycle the font type down" << std::endl
        << "\tARROW-LEFT:  decrease the font size" << std::endl
        << "\tARROW-RIGHT: increase the font size" << std::endl
        << std::endl
        << "\tPAGE-UP:     cycle the layout up" << std::endl
        << "\tPAGE-DOWN:   cycle the layout down" << std::endl
        << "\tHOME:        decrease line length" << std::endl
        << "\tEND:         increase line length" << std::endl
        << std::endl
        << "\tESC:         quit" << std::endl;

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_RGB | GLUT_DOUBLE | GLUT_MULTISAMPLE);
    glutInitWindowPosition(50, 50);
    glutInitWindowSize(w_win, h_win);
    glutCreateWindow("FTGL TEST");
    glutDisplayFunc(display);
    glutKeyboardFunc(parsekey);
    glutSpecialFunc(parseSpecialKey);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutReshapeFunc(myReshape);
    glutIdleFunc(display);

    myinit(fontfile);

    glutMainLoop();

    return 0;
}

