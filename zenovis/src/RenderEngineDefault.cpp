#include <zenovis/RenderEngine.h>
#include <zenovis/DrawOptions.h>
#include <zenovis/GraphicsManager.h>
#include <zenovis/opengl/vao.h>

namespace zenovis {

struct RenderEngineDefault : RenderEngine {

    std::unique_ptr<opengl::VAO> vao = std::make_unique<opengl::VAO>();
    std::unique_ptr<GraphicsManager> graphicsMan;
    Scene *scene;

    RenderEngineDefault(Scene *scene_) : scene(scene_) {
        CHECK_GL(glEnable(GL_BLEND));
        CHECK_GL(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
        CHECK_GL(glEnable(GL_DEPTH_TEST));
        CHECK_GL(glEnable(GL_PROGRAM_POINT_SIZE));
        CHECK_GL(glEnable(GL_MULTISAMPLE));
        CHECK_GL(glPixelStorei(GL_UNPACK_ALIGNMENT, 1));
        CHECK_GL(glPixelStorei(GL_PACK_ALIGNMENT, 1));

        graphicsMan = std::make_unique<GraphicsManager>(scene);
    }

    void update() override {
        graphicsMan->load_objects(scene->objects);
    }

    void draw() override {
        CHECK_GL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

        vao->bind();
        for (auto const &gra : graphicsMan->graphics.values<IGraphicDraw>()) {
            gra->draw();
        }
        if (scene->drawOptions->show_grid) {
            for (auto const &hudgra : scene->hudGraphics) {
                hudgra->draw();
            }
        }
        vao->unbind();
    }
};

std::unique_ptr<RenderEngine> makeRenderEngineDefault(Scene *scene) {
    return std::make_unique<RenderEngineDefault>(scene);
}

}
