#include <zenovis/RenderEngine.h>
#include <zenovis/DrawOptions.h>
#include <zenovis/bate/GraphicsManager.h>
#include <zenovis/bate/EngineData.h>
#include <zenovis/bate/IGraphic.h>
#include <zenovis/opengl/vao.h>

namespace zenovis::bate {

struct RenderEngineBate : RenderEngine {
    std::unique_ptr<opengl::VAO> vao;
    std::unique_ptr<GraphicsManager> graphicsMan;
    std::vector<std::unique_ptr<IGraphicDraw>> hudGraphics;
    Scene *scene;

    RenderEngineBate(Scene *scene_) : scene(scene_) {
        CHECK_GL(glEnable(GL_BLEND));
        CHECK_GL(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
        CHECK_GL(glEnable(GL_DEPTH_TEST));
        CHECK_GL(glEnable(GL_PROGRAM_POINT_SIZE));
        CHECK_GL(glEnable(GL_MULTISAMPLE));
        CHECK_GL(glPixelStorei(GL_UNPACK_ALIGNMENT, 1));
        CHECK_GL(glPixelStorei(GL_PACK_ALIGNMENT, 1));

        vao = std::make_unique<opengl::VAO>();
        graphicsMan = std::make_unique<GraphicsManager>(scene);

        hudGraphics.push_back(makeGraphicGrid(scene));
        hudGraphics.push_back(makeGraphicAxis(scene));
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
            for (auto const &hudgra : hudGraphics) {
                hudgra->draw();
            }
        }
        vao->unbind();
    }
};

}

namespace zenovis {

std::unique_ptr<RenderEngine> makeRenderEngineBate(Scene *scene) {
    return std::make_unique<bate::RenderEngineBate>(scene);
}

}
