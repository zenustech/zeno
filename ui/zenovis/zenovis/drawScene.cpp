#include <zenovis/Camera.h>
#include <zenovis/DepthPass.h>
#include <zenovis/IGraphic.h>
#include <zenovis/Scene.h>
#include <zenovis/opengl/common.h>

namespace zenovis {

Scene::Scene()
    : camera(std::make_unique<Camera>()),
      mDepthPass(std::make_unique<DepthPass>(this)) {
}

Scene::~Scene() = default;

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
    camera->vao->bind();
    camera->m_sample_weight = 1.0f / samples;
    drawSceneDepthSafe(camera->g_aspect, false, isDepthPass, camera->show_grid);
    if (isDepthPass != 1.0 && camera->show_grid) {
        CHECK_GL(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
        draw_small_axis();
    }
    camera->vao->unbind();
}

void Scene::draw_small_axis() { /* TODO: implement this */
}

void Scene::draw() {
    /* ZHXX DONT MODIFY ME */
    mDepthPass->paint_graphics();
}

} // namespace zenovis
