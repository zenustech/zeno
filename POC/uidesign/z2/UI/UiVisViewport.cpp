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

uniform mat4 u_mv;
uniform mat4 u_inv_mv;
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

uniform mat4 u_mv;
uniform mat4 u_inv_mv;
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
    vec3 view_dir = -calc_ray_dir(v_position);
    vec3 normal = normalize(cross(dFdx(v_position), dFdy(v_position)));

    vec3 v_color = vec3(0.96);
    vec3 color = vec3(0.0);
    vec3 light_dir;

    light_dir = normalize((u_inv_mv * vec4(1, 2, 5, 0)).xyz);
    color += vec3(0.45, 0.47, 0.5) * pbr(v_color, 0.19, 0.0, 1.0, normal, light_dir, view_dir);

    light_dir = normalize((u_inv_mv * vec4(-4, -2, 1, 0)).xyz);
    color += vec3(0.3, 0.23, 0.18) * pbr(v_color, 0.14, 0.0, 1.0, normal, light_dir, view_dir);

    light_dir = normalize((u_inv_mv * vec4(3, -5, 2, 0)).xyz);
    color += vec3(0.15, 0.2, 0.22) * pbr(v_color, 0.23, 0.0, 1.0, normal, light_dir, view_dir);

    color *= 1.2;

    //color = pow(clamp(color, 0, 1), vec3(1/2.2));
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

    for (auto const &object: get_parent()->view_results()) {
        auto mesh = std::any_cast<std::shared_ptr<ds::Mesh>>(object);

        std::vector<ztd::vec3f> vertices;
        for (auto const &poly: mesh->poly) {
            if (poly.num <= 2) continue;
            int first = mesh->loop[poly.start];
            int last = mesh->loop[poly.start + 1];
            for (int l = poly.start + 2; l < poly.start + poly.num; l++) {
                int now = mesh->loop[l];
                vertices.push_back(mesh->vert[first]);
                vertices.push_back(mesh->vert[last]);
                vertices.push_back(mesh->vert[now]);
                last = now;
            }
        }
        CHECK_GL(glEnableVertexAttribArray(0));
        CHECK_GL(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
                              sizeof(vertices[0]), vertices.data()));
        CHECK_GL(glDrawArrays(GL_TRIANGLES, 0, vertices.size()));
        CHECK_GL(glDisableVertexAttribArray(0));
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
