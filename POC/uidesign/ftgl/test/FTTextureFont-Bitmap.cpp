//$LIBS_PKG_CONFIG glu gl ftgl fontconfig
//$LIBS -lglut

#include <GL/glut.h>
#include <FTGL/ftgl.h>

FTFont *font;

void display ()
{
  glClear (GL_COLOR_BUFFER_BIT);
  glMatrixMode (GL_PROJECTION);
  glLoadIdentity ();
  gluPerspective (90, 1, 1, 1000);
  glMatrixMode (GL_MODELVIEW);
  glLoadIdentity ();
  gluLookAt (0, 0, 200, 0, 0, 0, 0, 1, 0);
  glPushMatrix ();
  glColor3f (1, 1, 1);
  glTranslatef (-100, 100, 0);
  glEnable (GL_BLEND);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  font->Render ("Test", -1, FTPoint (), FTPoint (), FTGL::RENDER_FRONT | FTGL::RENDER_BACK);
  glPopMatrix ();
  glutSwapBuffers ();
}

int main (int argc, char **argv)
{
  glutInit (&argc, argv);
  glutInitDisplayMode (GLUT_DEPTH | GLUT_RGB | GLUT_DOUBLE | GLUT_MULTISAMPLE);
  glutInitWindowPosition (50, 50);
  glutInitWindowSize (400, 400);
  glutCreateWindow ("FTGL Test");
  glutDisplayFunc (display);
  const char *file = "font_pack/timR12-ISO8859-1.pcf.gz";
  font = new FTTextureFont (file);
  if (font->Error () || !font->FaceSize (17))
    {
      fprintf (stderr, "Failed to open font %s\n", file);
      return 1;
    }
  glutMainLoop ();
}
