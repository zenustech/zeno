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
        vert.compile(R"(#version 120

uniform mat4 u_mvp;
uniform mat4 u_inv_mvp;

attribute vec3 a_position;

varying vec3 v_position;

void main() {
    v_position = a_position;
    gl_Position = u_mvp * vec4(a_position, 1.0);
}
    )");
        GL::Shader frag(GL_FRAGMENT_SHADER);
        frag.compile(R"(#version 120

uniform mat4 u_mvp;
uniform mat4 u_inv_mvp;

varying vec3 v_position;

vec3 pbr(vec3 albedo, float roughness, float metallic, float specular,
    vec3 nrm, vec3 idir, vec3 odir) {

  vec3 hdir = normalize(idir + odir);
  float NoH = max(0, dot(hdir, nrm));
  float NoL = max(0, dot(idir, nrm));
  float NoV = max(0, dot(odir, nrm));
  float VoH = clamp(dot(odir, hdir), 0, 1);
  float LoH = clamp(dot(idir, hdir), 0, 1);

  vec3 f0 = metallic * albedo + (1 - metallic) * 0.16 * specular * specular;
  vec3 fdf = f0 + (1 - f0) * pow(1 - VoH, 5);

  float k = (roughness + 1) * (roughness + 1) / 8;
  float vdf = 0.25 / ((NoV * k + 1 - k) * (NoL * k + 1 - k));

  float alpha2 = max(0, roughness * roughness);
  float denom = 1 - NoH * NoH * (1 - alpha2);
  float ndf = alpha2 / (denom * denom);

  vec3 brdf = fdf * vdf * ndf * f0 + (1 - f0) * albedo;
  return brdf * NoL;
}

vec3 calc_ray_dir(vec3 pos) {
    vec4 vpos = u_mvp * vec4(pos, 1);
    vec2 uv = vpos.xy / vpos.w;
    vec4 ro = u_inv_mvp * vec4(uv, -1, 1);
    vec4 re = u_inv_mvp * vec4(uv, +1, 1);
    vec3 rd = normalize(re.xyz / re.w - ro.xyz / ro.w);
    return rd;
}

void main() {
    vec3 normal = normalize(cross(dFdx(v_position), dFdy(v_position)));

    vec3 light_dir = normalize((u_mvp * vec4(-1, -2, 5, 0)).xyz);
    light_dir = faceforward(light_dir, -light_dir, normal);

    vec3 view_dir = -calc_ray_dir(v_position);
    vec3 color = pbr(vec3(0.8), 0.4, 0.0, 0.5, normal, light_dir, view_dir);
    gl_FragColor = vec4(color, 1.0);
}
    )");
        prog = std::make_unique<GL::Program>();
        prog->attach(vert);
        prog->attach(frag);
        prog->link();
    }
    return prog.get();
}


void UiVisViewport::paint() const {
    camera->resize(bbox.nx, bbox.ny);

    auto prog = make_mesh_shader();
    camera->uniform(prog);

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
