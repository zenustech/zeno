#include <z2/UI/UiVisViewport.h>
#include <z2/UI/UiMainWindow.h>
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
    auto view = get_parent()->scene->view_result;

    if (view.has_value()) {
        auto mesh = std::any_cast<std::shared_ptr<ds::Mesh>>(view);

        //glColor3f(0.375f, 0.75f, 1.f);
        glBegin(GL_TRIANGLES);
        for (int i = 0; i < mesh->poly.size(); i++) {
            auto poly = mesh->poly[i];
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
}


}
