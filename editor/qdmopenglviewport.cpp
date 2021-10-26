#include "qdmopenglviewport.h"
#include <GL/gl.h>

QDMOpenGLViewport::QDMOpenGLViewport(QWidget *parent)
    : QOpenGLWidget(parent)
{
    QSurfaceFormat fmt;
    fmt.setSamples(8);
    fmt.setVersion(3, 0);
    fmt.setProfile(QSurfaceFormat::CoreProfile);
    setFormat(fmt);
}

void QDMOpenGLViewport::initializeGL()
{
    QOpenGLFunctions::initializeOpenGLFunctions();

    m_program = std::make_unique<QOpenGLShaderProgram>(this);
    m_program->addShaderFromSourceCode(QOpenGLShader::Vertex, R"(
attribute vec3 attrPos;

void main() {
    gl_Position = vec4(attrPos, 1);
}
)");
    m_program->addShaderFromSourceCode(QOpenGLShader::Fragment, R"(
void main() {
    gl_FragColor = vec4(vec3(0.8), 1);
}
)");
    m_program->link();
}

void QDMOpenGLViewport::resizeGL(int nx, int ny)
{

}

void QDMOpenGLViewport::paintGL()
{
    const qreal retinaScale = devicePixelRatio();
    glViewport(0, 0, width() * retinaScale, height() * retinaScale);

    glClear(GL_COLOR_BUFFER_BIT);

    m_program->bind();

    static const GLfloat vertices[] = {
         0.0f,  0.707f,
        -0.5f, -0.5f,
         0.5f, -0.5f
    };

    auto attrPos = m_program->attributeLocation("attrPos");
    Q_ASSERT(attrPos != -1);
    glEnableVertexAttribArray(attrPos);
    glVertexAttribPointer(attrPos, 2, GL_FLOAT, GL_FALSE, 0, vertices);

    glDrawArrays(GL_TRIANGLES, 0, 3);

    glDisableVertexAttribArray(attrPos);

    m_program->release();
}
