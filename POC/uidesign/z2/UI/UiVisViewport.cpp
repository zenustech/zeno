#include <z2/UI/UiVisViewport.h>
#include <z2/UI/UiMainWindow.h>
#include <z2/GL/Shader.h>
#include <z2/ds/Mesh.h>


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
    camera->resize(bbox.nx, bbox.ny);

    for (auto const &object: get_parent()->view_results()) {
        auto render = VisRender::make_for(object);
        render->render(camera.get());
    }

    glUseProgram(0);
}


void UiVisViewport::on_event(Event_Mouse e) {
    Widget::on_event(e);

    if (e.btn != 2)
        return;

    if (e.down)
        cur.focus_on(this);
    else
        cur.focus_on(nullptr);
}


void UiVisViewport::on_event(Event_Motion e) {
    Widget::on_event(e);
    if (cur.mmb) {
        float n = (bbox.nx + bbox.ny) / (2 * 1.75f);
        camera->move(cur.dx / n, cur.dy / n, cur.shift);
    }
}


void UiVisViewport::on_event(Event_Scroll e) {
    Widget::on_event(e);
    camera->zoom(e.dy, cur.shift);
}


}
