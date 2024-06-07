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
            auto fuck = getClickedPos(fbr->w / 2, fbr->h / 2);
            if (fuck.has_value()) {
                zeno::log_info("{}", fuck.value());
            }
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
    std::optional<glm::vec3> getClickedPos(int x, int y) override {
        auto depth = fbr->getDepth(x, y);
        if (depth == 0) {
            return {};
        }
        zeno::log_info("depth: {}", depth);

        auto fov = scene->camera->m_fov;
        float cz = scene->camera->m_near / depth;
        auto w = scene->camera->m_nx;
        auto h = scene->camera->m_ny;
        zeno::log_info("{} {} {} {}", x, y, w, h);
        zeno::log_info("fov: {}", fov);
        zeno::log_info("w: {}, h: {}", w, h);
        auto u = (2.0 * x / w) - 1;
        auto v = 1 - (2.0 * y / h);
        zeno::log_info("u: {}, v: {}", u, v);
        auto cy = v * tan(glm::radians(fov) / 2) * cz;
        auto cx = u * tan(glm::radians(fov) / 2) * w / h * cz;
        glm::vec4 cc = {cx, cy, -cz, 1};
        auto wc = glm::inverse(scene->camera->m_view) * cc;
        wc /= wc.w;
        return glm::vec3(wc);
    }
};

static auto definer = RenderManager::registerRenderEngine<RenderEngineBate>("bate");

}
