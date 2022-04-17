#include <zenovis/Camera.h>
#include <zenovis/DepthPass.h>
#include <zenovis/EnvmapManager.h>
#include <zenovis/IGraphic.h>
#include <zenovis/ReflectivePass.h>
#include <zenovis/Scene.h>
#include <zenovis/ShaderManager.h>
#include <zenovis/opengl/buffer.h>
#include <zenovis/opengl/common.h>
#include <zenovis/opengl/vao.h>

namespace zenovis {

Scene::~Scene() = default;

Scene::Scene()
    : camera(std::make_unique<Camera>()),
      shaderMan(std::make_unique<ShaderManager>()) {

    gladLoadGL();

    CHECK_GL(glDepthRangef(0, 30000));
    CHECK_GL(glEnable(GL_BLEND));
    CHECK_GL(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
    CHECK_GL(glEnable(GL_DEPTH_TEST));
    CHECK_GL(glEnable(GL_PROGRAM_POINT_SIZE));
    //CHECK_GL(glEnable(GL_PROGRAM_POINT_SIZE_ARB));
    //CHECK_GL(glEnable(GL_POINT_SPRITE_ARB));
    //CHECK_GL(glEnable(GL_SAMPLE_COVERAGE));
    //CHECK_GL(glEnable(GL_SAMPLE_ALPHA_TO_COVERAGE));
    //CHECK_GL(glEnable(GL_SAMPLE_ALPHA_TO_ONE));
    CHECK_GL(glEnable(GL_MULTISAMPLE));

    vao = std::make_unique<opengl::VAO>();

    auto version = (const char *)glGetString(GL_VERSION);
    zeno::log_info("OpenGL version: {}", version ? version : "(null)");

    envmapMan = std::make_unique<EnvmapManager>(this);
    mDepthPass = std::make_unique<DepthPass>(this);
    mReflectivePass = std::make_unique<ReflectivePass>(this);

    mReflectivePass->initReflectiveMaps(camera->nx, camera->ny);

    /* hudGraphics.push_back(makeGraphicGrid()); */
    /* hudGraphics.push_back(makeGraphicAxis()); */
    //setup_env_map("Default");
}

void Scene::drawSceneDepthSafe(float aspRatio, bool reflect, float isDepthPass,
                               bool _show_grid) {

    //glEnable(GL_BLEND);
    //glBlendFunc(GL_ONE, GL_ONE);
    CHECK_GL(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
    // std::cout<<"camPos:"<<g_camPos.x<<","<<g_camPos.y<<","<<g_camPos.z<<std::endl;
    // std::cout<<"camView:"<<g_camView.x<<","<<g_camView.y<<","<<g_camView.z<<std::endl;
    // std::cout<<"camUp:"<<g_camUp.x<<","<<g_camUp.y<<","<<g_camUp.z<<std::endl;
    //CHECK_GL(glDisable(GL_MULTISAMPLE));
    // CHECK_GL(glClearColor(bgcolor.r, bgcolor.g, bgcolor.b, 0.0f));
    CHECK_GL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

    float range[] = {camera->g_near, 500, 1000, 2000, 8000, camera->g_far};
    for (int i = 5; i >= 1; i--) {
        CHECK_GL(glClearDepth(1));
        CHECK_GL(glClear(GL_DEPTH_BUFFER_BIT));
        camera->proj = glm::perspective(glm::radians(camera->g_fov), aspRatio,
                                        range[i - 1], range[i]);

        for (auto const &gra : graphics) {
            gra->draw(reflect, isDepthPass);
        }
        if (isDepthPass != 1.0f && _show_grid) {
            for (auto const &hudgra : hudGraphics) {
                hudgra->draw(false, 0.0f);
            }
        }
    }
}

void Scene::my_paint_graphics(float samples, float isDepthPass) {

    CHECK_GL(glViewport(0, 0, camera->nx, camera->ny));
    vao->bind();
    camera->m_sample_weight = 1.0f / samples;
    drawSceneDepthSafe(camera->getAspect(), false, isDepthPass, camera->show_grid);
    if (isDepthPass != 1.0 && camera->show_grid) {
        CHECK_GL(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
        draw_small_axis();
    }
    vao->unbind();
}

void Scene::draw_small_axis() { /* TODO: implement this */
}

void Scene::draw(unsigned int target_fbo) {
    /* ZHXX DONT MODIFY ME */
    //mDepthPass->paint_graphics(target_fbo);
    glClearColor(0.2f, 0.3f, 0.4f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    my_paint_graphics(1.0f, 0.0f);
}

std::vector<char> Scene::record_frame_offline() {
    std::vector<char> pixels(camera->nx * camera->ny * 3);

    GLuint fbo, rbo1, rbo2;
    CHECK_GL(glGenRenderbuffers(1, &rbo1));
    CHECK_GL(glGenRenderbuffers(1, &rbo2));
    CHECK_GL(glBindRenderbuffer(GL_RENDERBUFFER, rbo1));
    CHECK_GL(glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA, camera->nx,
                                   camera->ny));
    CHECK_GL(glBindRenderbuffer(GL_RENDERBUFFER, rbo2));
    CHECK_GL(glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F,
                                   camera->nx, camera->ny));

    CHECK_GL(glGenFramebuffers(1, &fbo));
    CHECK_GL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo));
    CHECK_GL(glFramebufferRenderbuffer(
        GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rbo1));
    CHECK_GL(glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                       GL_RENDERBUFFER, rbo2));
    CHECK_GL(glDrawBuffer(GL_COLOR_ATTACHMENT0));

    draw(fbo);

    if (glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER) ==
        GL_FRAMEBUFFER_COMPLETE) {
        CHECK_GL(glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo));
        CHECK_GL(glBlitFramebuffer(0, 0, camera->nx, camera->ny, 0, 0,
                                   camera->nx, camera->ny, GL_COLOR_BUFFER_BIT,
                                   GL_NEAREST));
        CHECK_GL(glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo));
        CHECK_GL(glBindBuffer(GL_PIXEL_PACK_BUFFER, 0));
        CHECK_GL(glReadBuffer(GL_COLOR_ATTACHMENT0));

        CHECK_GL(glReadPixels(0, 0, camera->nx, camera->ny, GL_RGB,
                              GL_UNSIGNED_BYTE, &pixels[0]));
    }

    CHECK_GL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
    CHECK_GL(glBindRenderbuffer(GL_RENDERBUFFER, 0));

    CHECK_GL(glDeleteRenderbuffers(1, &rbo1));
    CHECK_GL(glDeleteRenderbuffers(1, &rbo2));
    CHECK_GL(glDeleteFramebuffers(1, &fbo));

    return pixels;
}

} // namespace zenovis
