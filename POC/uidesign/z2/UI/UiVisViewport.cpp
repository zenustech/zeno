#include <glm/gtc/type_ptr.hpp>
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



static GL::Program *make_mesh_shader() {
    static std::unique_ptr<GL::Program> prog;
    if (!prog) {
        GL::Shader vert(GL_VERTEX_SHADER);
        vert.compile(R"(#version 310 core
    )");
        GL::Shader frag(GL_FRAGMENT_SHADER);
        frag.compile(R"(#version 310 core
    )");
        prog = std::make_unique<GL::Program>();
        prog->attach(vert);
        prog->attach(frag);
        prog->link();
    }
    return prog.get();
}


void UiVisViewport::paint() const {
    camera->nx = bbox.nx;
    camera->ny = bbox.ny;
    camera->update();
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadMatrixf(glm::value_ptr(camera->view));
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadMatrixf(glm::value_ptr(camera->proj));

    auto prog = make_mesh_shader();
    prog->use();

    if (auto object = get_parent()->scene->view_result; object.has_value()) {
        auto mesh = std::any_cast<std::shared_ptr<ds::Mesh>>(object);

        glBegin(GL_TRIANGLES);
        for (auto const &poly: mesh->poly) {
            if (poly.num <= 2) continue;
            int first = mesh->loop[poly.start];
            int last = mesh->loop[poly.start + 1];
            for (int l = poly.start + 2; l < poly.start + poly.num; l++) {
                int now = mesh->loop[l];
                glColor3f(1.f, 0.f, 0.f);
                glVertex3fv(mesh->vert[first].data());
                glColor3f(0.f, 1.f, 0.f);
                glVertex3fv(mesh->vert[last].data());
                glColor3f(0.f, 0.f, 1.f);
                glVertex3fv(mesh->vert[now].data());
                last = now;
            }
        }
        glEnd();
    }

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
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
