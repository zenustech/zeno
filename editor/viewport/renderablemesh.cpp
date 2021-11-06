#include "renderablemesh.h"
#include <QOpenGLShaderProgram>
#include <QOpenGLBuffer>
#include <zeno/math/vec.h>
#include <zeno/types/Mesh.h>

ZENO_NAMESPACE_BEGIN

namespace {

class RenderableMesh final : public Renderable
{
    static std::unique_ptr<QOpenGLShaderProgram> makeShaderProgram() {
        auto program = std::make_unique<QOpenGLShaderProgram>();
        program->addShaderFromSourceCode(QOpenGLShader::Vertex, R"(
uniform mat4 uMVP;

attribute vec3 attrPos;
varying vec3 varyPos;

void main() {
    varyPos = attrPos;
    gl_Position = uMVP * vec4(attrPos, 1.0);
}
)");
        program->addShaderFromSourceCode(QOpenGLShader::Fragment, R"(
uniform mat4 uMVP;
uniform mat4 uInvMVP;
uniform mat4 uInvMV;

varying vec3 varyPos;

vec3 pbr(vec3 albedo, float roughness, float metallic, float specular,
    vec3 nrm, vec3 idir, vec3 odir) {

  vec3 hdir = normalize(idir + odir);
  float NoH = max(0., dot(hdir, nrm));
  float NoL = max(0., dot(idir, nrm));
  float NoV = max(0., dot(odir, nrm));
  float VoH = clamp(dot(odir, hdir), 0., 1.);
  float LoH = clamp(dot(idir, hdir), 0., 1.);

  vec3 f0 = metallic * albedo + (1. - metallic) * 0.16 * specular * specular;
  vec3 fdf = f0 + (1. - f0) * pow(1. - VoH, 5.);

  float k = (roughness + 1.) * (roughness + 1.) / 8.;
  float vdf = 0.25 / ((NoV * k + 1. - k) * (NoL * k + 1. - k));

  float alpha2 = max(0., roughness * roughness);
  float denom = 1. - NoH * NoH * (1. - alpha2);
  float ndf = alpha2 / (denom * denom);

  vec3 brdf = fdf * vdf * ndf * f0 + (1. - f0) * albedo;
  return brdf * NoL;
}

vec3 calc_ray_dir(vec3 pos) {
    vec4 vpos = uMVP * vec4(pos, 1.);
    vec2 uv = vpos.xy / vpos.w;
    vec4 ro = uInvMVP * vec4(uv, -1., 1.);
    vec4 re = uInvMVP * vec4(uv, +1., 1.);
    vec3 rd = normalize(re.xyz / re.w - ro.xyz / ro.w);
    return rd;
}

void main() {
    vec3 view_dir = -calc_ray_dir(varyPos);
    vec3 normal = normalize(cross(dFdx(varyPos), dFdy(varyPos)));

    vec3 v_color = vec3(0.96);
    vec3 color = vec3(0.0);
    vec3 light_dir;

    light_dir = normalize((uInvMV * vec4(1., 2., 5., 0.)).xyz);
    color += vec3(0.45, 0.47, 0.5) * pbr(v_color, 0.19, 0.0, 1.0, normal, light_dir, view_dir);

    light_dir = normalize((uInvMV * vec4(-4., -2., 1., 0.)).xyz);
    color += vec3(0.3, 0.23, 0.18) * pbr(v_color, 0.14, 0.0, 1.0, normal, light_dir, view_dir);

    light_dir = normalize((uInvMV * vec4(3., -5., 2., 0.)).xyz);
    color += vec3(0.15, 0.2, 0.22) * pbr(v_color, 0.23, 0.0, 1.0, normal, light_dir, view_dir);

    color *= 1.2;

    //color = pow(clamp(color, 0., 1.), vec3(1./2.2));
    gl_FragColor = vec4(color, 1.0);
}
)");
        program->link();
        return program;
    }

public:
    virtual ~RenderableMesh() = default;

    std::vector<math::vec3f> vertices;

    RenderableMesh(std::shared_ptr<types::Mesh> const &mesh)
    {
        decltype(auto) vert = mesh->vert.to_vector();
        decltype(auto) loop = mesh->loop.to_vector();
        decltype(auto) poly = mesh->poly.to_vector();

        for (auto const &[p_num, p_start]: poly) {
            if (p_num <= 2) continue;
            int first = loop[p_start];
            int last = loop[p_start + 1];
            for (int l = p_start + 2; l < p_start + p_num; l++) {
                int now = loop[l];
                vertices.push_back(vert[first]);
                vertices.push_back(vert[last]);
                vertices.push_back(vert[now]);
                last = now;
            }
        }
    }

    virtual void render(QDMOpenGLViewport *viewport) override
    {
        static auto program = makeShaderProgram();
        program->bind();

        QOpenGLBuffer attrPos;
        attrPos.create();
        attrPos.setUsagePattern(QOpenGLBuffer::StreamDraw);
        attrPos.bind();
        attrPos.allocate(vertices.data(), vertices.size() * 3 * sizeof(vertices[0]));
        program->enableAttributeArray("attrPos");
        program->setAttributeBuffer("attrPos", GL_FLOAT, 0, 3);

        viewport->glDrawArrays(GL_TRIANGLES, 0, vertices.size());

        program->disableAttributeArray("attrPos");
        attrPos.destroy();

        program->release();
    }
};

}

std::unique_ptr<Renderable> makeRenderableMesh(std::shared_ptr<types::Mesh> const &mesh)
{
    return std::make_unique<RenderableMesh>(mesh);
}

ZENO_NAMESPACE_END
