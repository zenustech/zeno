#include "renderable.h"
#include <QOpenGLShaderProgram>
#include <QOpenGLBuffer>
#include <zeno/ztd/vec.h>

namespace {

class RenderableTestTriangle final : public Renderable
{

    static std::unique_ptr<QOpenGLShaderProgram> makeShaderProgram() {
        auto program = std::make_unique<QOpenGLShaderProgram>();
        program->addShaderFromSourceCode(QOpenGLShader::Vertex, R"(
attribute vec3 attrPos;

void main() {
    gl_Position = vec4(attrPos, 1);
}
)");
        program->addShaderFromSourceCode(QOpenGLShader::Fragment, R"(
void main() {
    gl_FragColor = vec4(vec3(0.8), 1);
}
)");
        program->link();
        return program;
    }

public:
    virtual ~RenderableTestTriangle() = default;

    RenderableTestTriangle()
    {
    }

    virtual void render(QDMOpenGLViewport *viewport) override
    {
        static auto program = makeShaderProgram();
        program->bind();

        std::vector<zeno::ztd::vec3f> vertices = {
            { 0.0f,  0.707f, 0.0f},
            {-0.5f, -0.5f, 0.0f},
            { 0.5f, -0.5f, 0.0f},
        };

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

std::unique_ptr<Renderable> makeRenderableTestTriangle()
{
    return std::make_unique<RenderableTestTriangle>();
}
