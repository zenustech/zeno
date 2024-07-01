#pragma once

#include "zenovis/Camera.h"
#include "zenovis/Scene.h"
#include "IGraphic.h"
#include "zenovis/ShaderManager.h"
#include "zenovis/opengl/buffer.h"
#include "zenovis/opengl/shader.h"
#include "zenovis/opengl/texture.h"
#include "zenovis/opengl/vao.h"
#include "zenovis/DrawOptions.h"

namespace zenovis {

using opengl::FBO;
using opengl::VAO;
using opengl::Buffer;
using opengl::Texture;
using opengl::RenderObject;
using opengl::Program;

using zeno::vec2i;
using zeno::vec3i;
using zeno::vec3f;

using std::unique_ptr;
using std::make_unique;

static const char * vert_code = R"(
    #version 330 core
    layout (location = 0) in vec2 aPos;
    layout (location = 1) in vec2 aTexCoords;

    out vec2 TexCoords;

    void main()
    {
        TexCoords = aTexCoords;
        gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
    }
)";

static const char* frag_code = R"(
    #version 330 core
    out vec4 FragColor;

    in vec2 TexCoords;

    uniform sampler2D screenTexture;

    void main()
    {
        vec3 col = texture(screenTexture, TexCoords).rgb;
        FragColor = vec4(col, 1.0);
    }
)";

static float quadVertices[] = {   // vertex attributes for a quad that fills the entire screen in Normalized Device Coordinates.
    // positions   // texCoords
    -1.0f,  1.0f,  0.0f, 1.0f,
    -1.0f, -1.0f,  0.0f, 0.0f,
    1.0f, -1.0f,  1.0f, 0.0f,

    -1.0f,  1.0f,  0.0f, 1.0f,
    1.0f, -1.0f,  1.0f, 0.0f,
    1.0f,  1.0f,  1.0f, 1.0f
};

struct FrameBufferRender {
    Scene* scene;

    unique_ptr<FBO> fbo;
    unique_ptr<Texture> picking_texture;
    unique_ptr<Texture> depth_texture;

    unique_ptr<FBO> intermediate_fbo;
    unique_ptr<Texture> screen_depth_tex;
    unique_ptr<Texture> screen_tex;
    
    unique_ptr<VAO> quad_vao;
    unique_ptr<Buffer> quad_vbo;

    int w = 0;
    int h = 0;
    int samples = 16;

    opengl::Program * shader = nullptr;

    explicit FrameBufferRender(Scene* s) : scene(s) {
        shader = scene->shaderMan->compile_program(vert_code, frag_code);
        shader->use();
        shader->set_uniformi("screenTexture", 0);

        quad_vbo = make_unique<Buffer>(GL_ARRAY_BUFFER);
        quad_vao = make_unique<VAO>();

        CHECK_GL(glBindVertexArray(quad_vao->vao));
        CHECK_GL(glBindBuffer(GL_ARRAY_BUFFER, quad_vbo->buf));
        CHECK_GL(glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW));
        CHECK_GL(glEnableVertexAttribArray(0));
        CHECK_GL(glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0));
        CHECK_GL(glEnableVertexAttribArray(1));
        CHECK_GL(glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float))));
    }

    ~FrameBufferRender() {
        destroy_buffers();
    }

    void generate_buffers() {
        // generate framebuffer
        fbo = make_unique<FBO>();
        CHECK_GL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo->fbo));

        // get viewport size
        w = scene->camera->m_nx;
        h = scene->camera->m_ny;

        // generate picking texture
        picking_texture = make_unique<Texture>();
        picking_texture->target = GL_TEXTURE_2D_MULTISAMPLE;
        CHECK_GL(glBindTexture(picking_texture->target, picking_texture->tex));
        CHECK_GL(glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, samples, GL_RGB, w, h, GL_TRUE));
        CHECK_GL(glBindTexture(picking_texture->target, 0));
        CHECK_GL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, picking_texture->tex, 0));

        // generate depth texture
        depth_texture = make_unique<Texture>();
        depth_texture->target = GL_TEXTURE_2D_MULTISAMPLE;
        CHECK_GL(glBindTexture(depth_texture->target, depth_texture->tex));
        CHECK_GL(glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, samples, GL_DEPTH_COMPONENT, w, h, GL_TRUE));
        CHECK_GL(glBindTexture(depth_texture->target, 0));
        CHECK_GL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D_MULTISAMPLE, depth_texture->tex, 0));

        // check fbo
        if(!fbo->complete()) zeno::log_error("fbo error");

        // unbind fbo & texture
        CHECK_GL(glBindTexture(GL_TEXTURE_2D, 0));
        fbo->unbind();

        intermediate_fbo = make_unique<FBO>();
        screen_tex = make_unique<Texture>();
        screen_depth_tex = make_unique<Texture>();
        intermediate_fbo->bind();
        CHECK_GL(glBindTexture(GL_TEXTURE_2D, screen_tex->tex));
        CHECK_GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL));
        CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
        CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
        CHECK_GL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, screen_tex->tex, 0));

        CHECK_GL(glBindTexture(GL_TEXTURE_2D, screen_depth_tex->tex));
        CHECK_GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, w, h, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL));
        CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
        CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
        CHECK_GL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, screen_depth_tex->tex, 0));
        if(!intermediate_fbo->complete()) zeno::log_error("fbo error");
        CHECK_GL(glBindTexture(GL_TEXTURE_2D, 0));
        intermediate_fbo->unbind();
    }

    void destroy_buffers() {
        fbo.reset();
        picking_texture.reset();
        depth_texture.reset();
        intermediate_fbo.reset();
        screen_tex.reset();
        screen_depth_tex.reset();
    }
    void bind() {
        // enable framebuffer writing
        CHECK_GL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo->fbo));
        CHECK_GL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
    }

    void unbind() {
        fbo->unbind();
    }
    void draw_to_screen() {
        // 2. now blit multisampled buffer(s) to normal colorbuffer of intermediate FBO. Image is stored in screenTexture
        CHECK_GL(glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo->fbo));
        CHECK_GL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, intermediate_fbo->fbo));
        CHECK_GL(glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT, GL_NEAREST));

        // 3. now render quad with scene's visuals as its texture image
        CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, 0));
        CHECK_GL(glClearColor(1.0f, 1.0f, 1.0f, 1.0f));
        CHECK_GL(glClear(GL_COLOR_BUFFER_BIT));
        CHECK_GL(glDisable(GL_DEPTH_TEST));

        glDisable(GL_MULTISAMPLE);
        // draw Screen quad
        shader->use();
        CHECK_GL(glBindVertexArray(quad_vao->vao));
        CHECK_GL(glActiveTexture(GL_TEXTURE0));
        CHECK_GL(glBindTexture(GL_TEXTURE_2D, screen_tex->tex)); // use the now resolved color attachment as the quad's texture
        CHECK_GL(glDrawArrays(GL_TRIANGLES, 0, 6));
        glEnable(GL_MULTISAMPLE);
    }
    float getDepth(int x, int y) {
        if (!intermediate_fbo->complete()) return 0;
        intermediate_fbo->bind();
        CHECK_GL(glBindFramebuffer(GL_READ_FRAMEBUFFER, intermediate_fbo->fbo));

        float depth;
        CHECK_GL(glReadPixels(x, h - y - 1, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &depth));

        intermediate_fbo->unbind();
        return depth;
    }
};
}