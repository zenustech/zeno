/*
 * FTGLDemo - advanced demo for FTGL, the OpenGL font library
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

#include "config.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#if defined HAVE_GL_GLUT_H
#   include <GL/glut.h>
#elif defined HAVE_GLUT_GLUT_H
#   include <GLUT/glut.h>
#else
#   error GLUT headers not present
#endif

#include <FTGL/ftgl.h>

#include "tb.h"

// YOU'LL PROBABLY WANT TO CHANGE THESE
#if defined FONT_FILE
    char const *defaultFonts[] = { FONT_FILE };
    const int NumDefaultFonts = 1;
#elif defined __APPLE_CC__
    char const *defaultFonts[] = { "/System/Library/Fonts/Helvetica.dfont",
                                   "/System/Library/Fonts/Geneva.dfont" };
    const int NumDefaultFonts = 2;
#elif defined _WIN32
    char const *defaultFonts[] = { "C:\\WINNT\\Fonts\\arial.ttf" };
    const int NumDefaultFonts = 1;
#else
    // Put your font files here if configure did not find any.
    char const *defaultFonts[] = { };
    const int NumDefaultFonts = 0;
#endif

/* Set this to 1 to build a Mac os app (ignore the command line args). */
#ifndef IGNORE_ARGV
#   define IGNORE_ARGV 0
#endif /* IGNORE_ARGV */

#define EDITING 1
#define INTERACTIVE 2

#define FTGL_BITMAP 0
#define FTGL_PIXMAP 1
#define FTGL_OUTLINE 2
#define FTGL_POLYGON 3
#define FTGL_EXTRUDE 4
#define FTGL_TEXTURE 5
const int NumStyles = 6;

char const * const *fontfiles;
int current_font = FTGL_EXTRUDE;

GLint w_win = 640, h_win = 480;
int mode = INTERACTIVE;
int carat = 0;

FTSimpleLayout simpleLayout;
FTLayout *layouts[] = { &simpleLayout, NULL };
int currentLayout = 0;
const int NumLayouts = 2;

const float InitialLineLength = 300.0f;

const float OX = -100;
const float OY = 200;

//wchar_t myString[16] = { 0x6FB3, 0x9580};
char myString[4096];

int totalFonts;
static FTFont** fonts;
static FTPixmapFont* infoFont;

void SetCamera(void);

inline int GetStyle()
{
    return current_font % NumStyles;
}

inline int GetFace()
{
    return current_font / NumStyles;
}

void setUpLighting()
{
    // Set up lighting.
    float light1_ambient[4]  = { 1.0, 1.0, 1.0, 1.0 };
    float light1_diffuse[4]  = { 1.0, 0.9, 0.9, 1.0 };
    float light1_specular[4] = { 1.0, 0.7, 0.7, 1.0 };
    float light1_position[4] = { -1.0, 1.0, 1.0, 0.0 };
    glLightfv(GL_LIGHT1, GL_AMBIENT,  light1_ambient);
    glLightfv(GL_LIGHT1, GL_DIFFUSE,  light1_diffuse);
    glLightfv(GL_LIGHT1, GL_SPECULAR, light1_specular);
    glLightfv(GL_LIGHT1, GL_POSITION, light1_position);
    glEnable(GL_LIGHT1);

    float light2_ambient[4]  = { 0.2, 0.2, 0.2, 1.0 };
    float light2_diffuse[4]  = { 0.9, 0.9, 0.9, 1.0 };
    float light2_specular[4] = { 0.7, 0.7, 0.7, 1.0 };
    float light2_position[4] = { 1.0, -1.0, -1.0, 0.0 };
    glLightfv(GL_LIGHT2, GL_AMBIENT,  light2_ambient);
    glLightfv(GL_LIGHT2, GL_DIFFUSE,  light2_diffuse);
    glLightfv(GL_LIGHT2, GL_SPECULAR, light2_specular);
    glLightfv(GL_LIGHT2, GL_POSITION, light2_position);
    //glEnable(GL_LIGHT2);

    float front_emission[4] = { 0.3, 0.2, 0.1, 0.0 };
    float front_ambient[4]  = { 0.2, 0.2, 0.2, 0.0 };
    float front_diffuse[4]  = { 0.95, 0.95, 0.8, 0.0 };
    float front_specular[4] = { 0.6, 0.6, 0.6, 0.0 };
    glMaterialfv(GL_FRONT, GL_EMISSION, front_emission);
    glMaterialfv(GL_FRONT, GL_AMBIENT, front_ambient);
    glMaterialfv(GL_FRONT, GL_DIFFUSE, front_diffuse);
    glMaterialfv(GL_FRONT, GL_SPECULAR, front_specular);
    glMaterialf(GL_FRONT, GL_SHININESS, 16.0);
    glColor4fv(front_diffuse);

    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_FALSE);
    glEnable(GL_CULL_FACE);
    glColorMaterial(GL_FRONT, GL_DIFFUSE);
    glEnable(GL_COLOR_MATERIAL);

    glEnable(GL_LIGHTING);
    glShadeModel(GL_SMOOTH);
}


void setUpFonts(int numFontFiles)
{
    // The total number of fonts is styles * faces
    totalFonts = numFontFiles*NumStyles;

    // Allocate an array to hold all fonts
    fonts = new FTFont *[totalFonts];

    // Instantiate and configure named fonts
    for(int i = 0; i < numFontFiles; i++)
    {
        fonts[i*NumStyles + FTGL_BITMAP] = new FTBitmapFont(fontfiles[i]);
        fonts[i*NumStyles + FTGL_PIXMAP] = new FTPixmapFont(fontfiles[i]);
        fonts[i*NumStyles + FTGL_OUTLINE] = new FTOutlineFont(fontfiles[i]);
        fonts[i*NumStyles + FTGL_POLYGON] = new FTPolygonFont(fontfiles[i]);
        fonts[i*NumStyles + FTGL_EXTRUDE] = new FTExtrudeFont(fontfiles[i]);
        fonts[i*NumStyles + FTGL_TEXTURE] = new FTTextureFont(fontfiles[i]);

        for(int x = 0; x < NumStyles; ++x)
        {
            int j = i * NumStyles + x;

            if(fonts[j]->Error())
            {
                fprintf(stderr, "Failed to open font %s\n", fontfiles[i]);
                exit(1);
            }

            if(!fonts[j]->FaceSize(24))
            {
                fprintf(stderr, "Failed to set size\n");
                exit(1);
            }

            fonts[j]->Depth(20);
            fonts[j]->CharMap(ft_encoding_unicode);
        }
    }

    infoFont = new FTPixmapFont(fontfiles[0]);

    if(infoFont->Error())
    {
        fprintf(stderr, "Failed to open font %s\n", fontfiles[0]);
        exit(1);
    }

    infoFont->FaceSize(18);

    strcpy(myString, "OpenGL is a powerful software interface for graphics "
           "hardware that allows graphics programmers to produce high-quality "
           "color images of 3D objects. abcdefghijklmnopqrstuvwxyzABCDEFGHIJKL"
           "MNOPQRSTUVWXYZ0123456789");
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
    glDisable(GL_TEXTURE_2D);
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
    if((GetStyle() == FTGL_EXTRUDE) && (z1 != z2))
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
    else if (layouts[currentLayout]
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
    glColor3f(1.0, 0.0, 0.0);
    glPointSize(5.0);
    glBegin(GL_POINTS);
        glVertex3f(0.0, 0.0, 0.0);
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
    switch(GetStyle())
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
    }

    glRasterPos2f(20.0f , 20.0f + infoFont->Ascender() - infoFont->Descender());
    infoFont->Render(fontfiles[GetFace()]);

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
    switch(GetStyle())
    {
        case FTGL_BITMAP:
        case FTGL_PIXMAP:
        case FTGL_OUTLINE:
            break;
        case FTGL_POLYGON:
            glDisable(GL_BLEND);
            setUpLighting();
            break;
        case FTGL_EXTRUDE:
            glEnable(GL_DEPTH_TEST);
            glDisable(GL_BLEND);
            setUpLighting();
            break;
        case FTGL_TEXTURE:
            glEnable(GL_TEXTURE_2D);
            glDisable(GL_DEPTH_TEST);
            setUpLighting();
            glNormal3f(0.0, 0.0, 1.0);
            break;

    }

    glColor3f(1.0, 1.0, 1.0);
    // If you do want to switch the color of bitmaps rendered with glBitmap,
    // you will need to explicitly call glRasterPos (or its ilk) to lock
    // in a changed current color.

    // If there is an active layout use it to render the font
    if (layouts[currentLayout])
    {
        layouts[currentLayout]->Render(myString);
    }
    else
    {
        fonts[current_font]->Render(myString);
    }

    renderFontmetrics();
    renderFontInfo();
}


void display(void)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    SetCamera();

    switch(GetStyle())
    {
        case FTGL_BITMAP:
        case FTGL_PIXMAP:
            glRasterPos2i((long)(w_win / 2 + OX), (long)(h_win / 2 + OY));
            glTranslatef(w_win / 2 + OX, h_win / 2 + OY, 0.0);
            break;
        case FTGL_OUTLINE:
        case FTGL_POLYGON:
        case FTGL_EXTRUDE:
        case FTGL_TEXTURE:
         glTranslatef(OX, OY, 0);
            tbMatrix();
            break;
    }

    glPushMatrix();

    do_display();

    glPopMatrix();

    glutSwapBuffers();
}


void myinit(int numFontFiles)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(0.13, 0.17, 0.32, 0.0);
    glColor3f(1.0, 1.0, 1.0);

    glEnable(GL_CULL_FACE);
    glFrontFace(GL_CCW);

    glEnable(GL_DEPTH_TEST);

    glEnable(GL_POLYGON_OFFSET_LINE);
    glPolygonOffset(1.0, 1.0); // ????

    SetCamera();

    tbInit(GLUT_LEFT_BUTTON);
    tbAnimate(GL_FALSE);

    setUpFonts(numFontFiles);

    // Configure the SimpleLayout
    simpleLayout.SetLineLength(InitialLineLength);
    simpleLayout.SetFont(fonts[current_font]);
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
        current_font = (GetFace()*NumStyles + (current_font + 1)%NumStyles)%totalFonts;
        break;
    case GLUT_KEY_DOWN:
        current_font = (GetFace()*NumStyles + (current_font + NumStyles - 1)%NumStyles)%totalFonts;
        break;
    case GLUT_KEY_LEFT:
		s = fonts[current_font]->FaceSize();
		if (s >= 2)
			fonts[current_font]->FaceSize(s - 1);
        break;
    case GLUT_KEY_RIGHT:
        fonts[current_font]->FaceSize(fonts[current_font]->FaceSize() + 1);
        break;
    case GLUT_KEY_PAGE_UP:
        current_font = (current_font + NumStyles)%totalFonts;
        break;
    case GLUT_KEY_PAGE_DOWN:
        current_font = (current_font + totalFonts - NumStyles)%totalFonts;
        break;
    case GLUT_KEY_HOME:
        currentLayout = (currentLayout + 1)%NumLayouts;
        break;
    case GLUT_KEY_END:
        currentLayout = (currentLayout + NumLayouts - 1)%NumLayouts;
        break;
    case GLUT_KEY_F1:
    case GLUT_KEY_F10:
        // If the current layout is simple decrement its line length
        if (l) l->SetLineLength(l->GetLineLength() - 10.0f);
        break;
    case GLUT_KEY_F2:
    case GLUT_KEY_F11:
        // If the current layout is simple increment its line length
        if (l) l->SetLineLength(l->GetLineLength() + 10.0f);
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
    switch(GetStyle())
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
    int numFontFiles;

    if((argc >= 2) && !IGNORE_ARGV)
    {
        fontfiles = (char const * const *)argv + 1;
        numFontFiles = argc - 1;
    }
    else
    {
        fontfiles = defaultFonts;
        numFontFiles = NumDefaultFonts;
    }

    if(!fontfiles[0])
    {
        fprintf(stderr, "At least one font file must be specified on the command line\n");
        exit(1);
    }

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

    myinit(numFontFiles);

    glutMainLoop();

    return 0;
}

