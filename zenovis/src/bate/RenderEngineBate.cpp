#include <zenovis/RenderEngine.h>
#include <zenovis/DrawOptions.h>
#include <zenovis/bate/GraphicsManager.h>
#include <zenovis/ObjectsManager.h>
#include <zenovis/DrawOptions.h>
#include <zenovis/bate/IGraphic.h>
#include <zenovis/opengl/vao.h>
#include <zenovis/opengl/scope.h>

namespace zenovis::bate {

struct RenderEngineBate : RenderEngine {
    std::unique_ptr<opengl::VAO> vao;
    std::unique_ptr<GraphicsManager> graphicsMan;
    std::vector<std::unique_ptr<IGraphicDraw>> hudGraphics;
    Scene *scene;

    auto setupState() {
        return std::tuple{
            opengl::scopeGLEnable(GL_BLEND), opengl::scopeGLEnable(GL_DEPTH_TEST),
            opengl::scopeGLEnable(GL_PROGRAM_POINT_SIZE),
            opengl::scopeGLBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA),
        };
    }

    explicit RenderEngineBate(Scene *scene_) : scene(scene_) {
        auto guard = setupState();

        vao = std::make_unique<opengl::VAO>();
        graphicsMan = std::make_unique<GraphicsManager>(scene);

        hudGraphics.push_back(makeGraphicGrid(scene));
        hudGraphics.push_back(makeGraphicAxis(scene));
        hudGraphics.push_back(makeGraphicSelectBox(scene));
    }

    void update() override {
        if (scene->drawOptions->interactive) graphicsMan->graphics.clear();
        graphicsMan->load_objects(scene->objectsMan->pairsShared());
    }

    void draw() override {
        auto guard = setupState();
        CHECK_GL(glClearColor(scene->drawOptions->bgcolor.r, scene->drawOptions->bgcolor.g,
                              scene->drawOptions->bgcolor.b, 0.0f));
        CHECK_GL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

        auto bindVao = opengl::scopeGLBindVertexArray(vao->vao);
        for (auto const &[key, gra] : graphicsMan->graphics.pairs<IGraphicDraw>()) {
            gra->draw();
        }
        if (scene->drawOptions->show_grid) {
            for (auto const &hudgra : hudGraphics) {
                hudgra->draw();
            }
        }
        if (!scene->selected.empty() && scene->drawOptions->handler) {
            CHECK_GL(glClear(GL_DEPTH_BUFFER_BIT));
            scene->drawOptions->handler->draw();
        }
    }
};

static auto definer = RenderManager::registerRenderEngine<RenderEngineBate>("bate");

}
