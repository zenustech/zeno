#include <zenovis/RenderEngine.h>
#include <zenovis/Scene.h>
#include <zenovis/DrawOptions.h>
#include <zenovis/zhxx/ZhxxCamera.h>
#include <zenovis/zhxx/ZhxxGraphicsManager.h>
#include <zenovis/zhxx/ZhxxScene.h>
#include <zenovis/opengl/vao.h>

namespace zenovis::zhxx {

struct RenderEngineZhxx : RenderEngine {

    Scene *scene;
    std::unique_ptr<opengl::VAO> vao;
    std::unique_ptr<ZhxxScene> zhxxScene;

    RenderEngineZhxx(Scene *scene_) : scene(scene_) {
        CHECK_GL(glEnable(GL_BLEND));
        CHECK_GL(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
        CHECK_GL(glEnable(GL_DEPTH_TEST));
        CHECK_GL(glEnable(GL_PROGRAM_POINT_SIZE));
        CHECK_GL(glEnable(GL_MULTISAMPLE));
        CHECK_GL(glPixelStorei(GL_UNPACK_ALIGNMENT, 1));
        CHECK_GL(glPixelStorei(GL_PACK_ALIGNMENT, 1));

        vao = std::make_unique<opengl::VAO>();
        zhxxScene = std::make_unique<ZhxxScene>(scene);
    }

    void update() override {
        zeno::log_trace("(zxx) updating {} objects", this->scene->objects.size());
        this->zhxxScene->zxxGraphicsMan->load_objects(this->scene->objects);
    }

    void draw() override {
        this->zhxxScene->camera->copyFromSceneCamera(*this->scene->camera);
        GLint currFbo = 0;
        CHECK_GL(glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &currFbo));
        zeno::log_trace("(zxx) drawing on current frame buffer {}", currFbo);
        CHECK_GL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
        zhxxScene->draw((GLuint)currFbo);
    }
};

}

namespace zenovis {

std::unique_ptr<RenderEngine> makeRenderEngineZhxx(Scene *scene) {
    return std::make_unique<zhxx::RenderEngineZhxx>(scene);
}

}
