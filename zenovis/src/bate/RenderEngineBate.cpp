#include <zenovis/RenderEngine.h>
#include <zenovis/DrawOptions.h>
#include <zenovis/bate/GraphicsManager.h>
#include <zenovis/ObjectsManager.h>
#include <zenovis/DrawOptions.h>
#include <zenovis/bate/IGraphic.h>
#include <zenovis/opengl/vao.h>
#include <zenovis/opengl/scope.h>
#include "zenovis/bate/FrameBufferRender.h"

namespace zenovis::bate {
struct RenderEngineBate : RenderEngine {
    std::unique_ptr<opengl::VAO> vao;
    std::unique_ptr<GraphicsManager> graphicsMan;
    std::vector<std::unique_ptr<IGraphicDraw>> hudGraphics;
    std::unique_ptr<IGraphicDraw> primHighlight;
    std::unique_ptr<FrameBufferRender> fbr;
    Scene *scene;
    bool released = false;

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
        fbr = std::make_unique<FrameBufferRender>(scene);

        hudGraphics.push_back(makeGraphicGrid(scene));
        hudGraphics.push_back(makeGraphicAxis(scene));
        hudGraphics.push_back(makeGraphicSelectBox(scene));

        primHighlight = makePrimitiveHighlight(scene);
    }

    void update() override {
        graphicsMan->load_objects(scene->objectsMan->pairsShared());
    }

    void draw(bool record) override {
        if (released) {
            return;
        }
        auto guard = setupState();
        CHECK_GL(glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE));
        glDepthFunc(GL_GREATER);
        CHECK_GL(glClearDepth(0.0));
        CHECK_GL(glClearColor(scene->drawOptions->bgcolor.r, scene->drawOptions->bgcolor.g,
                              scene->drawOptions->bgcolor.b, 0.0f));
        CHECK_GL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
        if (!record) {
            fbr->generate_buffers();
            fbr->bind();
        }

        auto bindVao = opengl::scopeGLBindVertexArray(vao->vao);
        graphicsMan->draw();
//        for (auto const &[key, gra] : graphicsMan->graphics.pairs<IGraphicDraw>()) {
//            gra->draw();
//        }
        primHighlight->draw();
        if (scene->drawOptions->show_grid) {
            for (auto const &hudgra : hudGraphics) {
                hudgra->draw();
            }
            {
                // left-down gizmo axis
                if (hudGraphics.size() > 2) {
                    Camera backup = *scene->camera;
                    scene->camera->focusCamera(0, 0, 0, 10);
                    auto offset = scene->camera->m_block_window? scene->camera->viewport_offset : zeno::vec2i(0, 0);
                    CHECK_GL(glViewport(offset[0], offset[1], scene->camera->m_nx * 0.1, scene->camera->m_ny * 0.1));
                    CHECK_GL(glDisable(GL_DEPTH_TEST));
                    hudGraphics[1]->draw();
                    CHECK_GL(glEnable(GL_DEPTH_TEST));
                    CHECK_GL(glViewport(offset[0], offset[1], scene->camera->m_nx, scene->camera->m_ny));
                    *scene->camera = backup;
                }
            }
        }
        if (!scene->selected.empty() && scene->drawOptions->handler) {
            CHECK_GL(glClear(GL_DEPTH_BUFFER_BIT));
            scene->drawOptions->handler->draw();
        }
        if (!record) {
            fbr->unbind();
            fbr->draw_to_screen();
        }
    }

    void cleanupOptix() override {

    }

    void cleanupWhenExit() override {
        released = true;
        scene->shaderMan = nullptr;
        scene->drawOptions->handler = nullptr;
        vao = nullptr;
        graphicsMan = nullptr;
        hudGraphics.clear();
        primHighlight = nullptr;
        fbr = nullptr;
    }
    float getDepth(int x, int y) override {
        return fbr->getDepth(x, y);
    }
};

static auto definer = RenderManager::registerRenderEngine<RenderEngineBate>("bate");

}
