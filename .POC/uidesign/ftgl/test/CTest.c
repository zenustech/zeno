/* Small C bindings test program */

#include "config.h"

#if defined HAVE_GL_GLUT_H
#   include <GL/glut.h>
#elif defined HAVE_GLUT_GLUT_H
#   include <GLUT/glut.h>
#else
#   error GLUT headers not present
#endif
#include <FTGL/ftgl.h>

#define ALLOC(ctor, var, arg) \
    var = ctor(arg); \
    if(var == NULL) \
        return 2

int main(int argc, char *argv[])
{
    FTGLfont *f[6];
    char *glutchar = NULL;
    int glutint = 0;
    int i;

    if(argc < 2)
        return 1;

    glutInit(&glutint, &glutchar);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_RGB | GLUT_DOUBLE | GLUT_MULTISAMPLE);
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(150, 150);
    glutCreateWindow("FTGL C test");

    ALLOC(ftglCreateBitmapFont, f[0], argv[1]);
    ALLOC(ftglCreateExtrudeFont, f[1], argv[1]);
    ALLOC(ftglCreateOutlineFont, f[2], argv[1]);
    ALLOC(ftglCreatePixmapFont, f[3], argv[1]);
    ALLOC(ftglCreatePolygonFont, f[4], argv[1]);
    ALLOC(ftglCreateTextureFont, f[5], argv[1]);

    for(i = 0; i < 6; i++)
        ftglRenderFont(f[i], "Hello world", FTGL_RENDER_ALL);

    for(i = 0; i < 6; i++)
        ftglSetFontFaceSize(f[i], 37, 72);

    for(i = 0; i < 6; i++)
        ftglRenderFont(f[i], "Hello world", FTGL_RENDER_ALL);

    for(i = 0; i < 6; i++)
        ftglDestroyFont(f[i]);

    return 0;
}

