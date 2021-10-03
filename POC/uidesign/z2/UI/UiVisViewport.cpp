#include <z2/UI/UiVisViewport.h>
#include <z2/UI/UiMainWindow.h>
#include <z2/ds/Mesh.h>
#include <glm/gtc/type_ptr.hpp>


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
    auto object = get_parent()->scene->view_result;

    camera->nx = bbox.nx;
    camera->ny = bbox.ny;
    camera->update();
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadMatrixf(glm::value_ptr(camera->view));
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadMatrixf(glm::value_ptr(camera->proj));

    if (object.has_value()) {
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


}
