#include <z2/UI/UiVisViewport.h>
#include <z2/UI/UiMainWindow.h>


namespace z2::UI {


void UiVisViewport::do_paint() {
    GLint viewport[4];

    glGetIntegerv(GL_VIEWPORT, viewport);
    glViewport(position.x + bbox.x0, position.y + bbox.y0, bbox.nx, bbox.ny);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    paint();

    glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
    glPopMatrix();
}


void UiVisViewport::paint() const {
    glColor3f(0.375f, 0.75f, 1.f);
    glRectf(-.5f, -.5f, .5f, .5f);
}


}
