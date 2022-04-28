#include <zenovis/RenderEngine.h>
#include <zenovis/DrawOptions.h>
#include <zenovis/GraphicsManager.h>
#include <zenovis/opengl/vao.h>

namespace zenovis {

struct RenderEngineBate : RenderEngine {

    std::unique_ptr<opengl::VAO> vao = std::make_unique<opengl::VAO>();
    Scene *scene;

    RenderEngineBate(Scene *scene_) : scene(scene_) {
        CHECK_GL(glEnable(GL_BLEND));
        CHECK_GL(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
        CHECK_GL(glEnable(GL_DEPTH_TEST));
        CHECK_GL(glEnable(GL_PROGRAM_POINT_SIZE));
        CHECK_GL(glEnable(GL_MULTISAMPLE));
        CHECK_GL(glPixelStorei(GL_UNPACK_ALIGNMENT, 1));
        CHECK_GL(glPixelStorei(GL_PACK_ALIGNMENT, 1));
    }

    void draw() override {
        CHECK_GL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

        vao->bind();
        for (auto const &gra : scene->graphicsMan->graphics.values<IGraphicDraw>()) {
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

std::unique_ptr<RenderEngine> makeRenderEngineBate(Scene *scene) {
    return std::make_unique<RenderEngineBate>(scene);
}

}
