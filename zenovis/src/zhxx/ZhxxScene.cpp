#include <zenovis/zhxx/ZhxxScene.h>
#include <zenovis/Camera.h>
#include <zenovis/Scene.h>
#include <zenovis/zhxx/ZhxxDrawOptions.h>
#include <zenovis/zhxx/ZhxxCamera.h>
#include <zenovis/zhxx/DepthPass.h>
#include <zenovis/zhxx/EnvmapManager.h>
#include <zenovis/zhxx/ZhxxGraphicsManager.h>
#include <zenovis/zhxx/ReflectivePass.h>
#include <zenovis/ShaderManager.h>
#include <zenovis/opengl/buffer.h>
#include <zenovis/opengl/common.h>
#include <zenovis/opengl/vao.h>
#include <zeno/utils/scope_exit.h>
#include <map>

namespace zenovis::zhxx {

ZhxxScene::~ZhxxScene() = default;

ZhxxScene::ZhxxScene(Scene *visScene_)
    : camera(std::make_unique<ZhxxCamera>()), visScene(visScene_) {

    CHECK_GL(glDepthRangef(0, 30000));
    CHECK_GL(glEnable(GL_BLEND));
    CHECK_GL(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
    CHECK_GL(glEnable(GL_DEPTH_TEST));
    CHECK_GL(glEnable(GL_PROGRAM_POINT_SIZE));
    //CHECK_GL(glEnable(GL_PROGRAM_POINT_SIZE));
    //CHECK_GL(glEnable(GL_POINT_SPRITE));
    //CHECK_GL(glEnable(GL_SAMPLE_COVERAGE));
    //CHECK_GL(glEnable(GL_SAMPLE_ALPHA_TO_COVERAGE));
    //CHECK_GL(glEnable(GL_SAMPLE_ALPHA_TO_ONE));
    CHECK_GL(glEnable(GL_MULTISAMPLE));
    CHECK_GL(glPixelStorei(GL_UNPACK_ALIGNMENT, 1));
    CHECK_GL(glPixelStorei(GL_PACK_ALIGNMENT, 1));

    lightCluster = std::make_unique<LightCluster>(this);

    vao = std::make_unique<opengl::VAO>();
    zxxDrawOptions = std::make_unique<ZhxxDrawOptions>();

    zxxGraphicsMan = std::make_unique<ZhxxGraphicsManager>(this);
    envmapMan = std::make_unique<EnvmapManager>(this);
    mDepthPass = std::make_unique<DepthPass>(this);
    mReflectivePass = std::make_unique<ReflectivePass>(this);

    mReflectivePass->initReflectiveMaps(camera->m_nx, camera->m_ny);

    //setup_env_map("Default");
}

//zeno::PolymorphicVector<std::vector<IGraphic *>> ZhxxScene::graphics() const {
    //zeno::PolymorphicVector<std::vector<IGraphic *>> gras;
    //gras.reserve(graphicsMan->graphics.size());
    //for (auto const &[key, val] : graphicsMan->graphics) {
        //gras.push_back(val.get());
    //}
    //return gras;
//}

static glm::mat4 MakeInfReversedZProjRH(float fovY_radians, float aspectWbyH, float zNear)
{
    float f = 1.0f / tan(fovY_radians / 2.0f);
    return glm::mat4(
        f / aspectWbyH, 0.0f,  0.0f,  0.0f,
                  0.0f,    f,  0.0f,  0.0f,
                  0.0f, 0.0f,  0.0f, -1.0f,
                  0.0f, 0.0f, zNear,  0.0f);
}

void ZhxxScene::drawSceneDepthSafe(bool reflect, bool isDepthPass) {
    auto aspRatio = camera->getAspect();

    //glEnable(GL_BLEND);
    //glBlendFunc(GL_ONE, GL_ONE);
    /* CHECK_GL(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)); */
    // std::cout<<"camPos:"<<g_camPos.x<<","<<g_camPos.y<<","<<g_camPos.z<<std::endl;
    // std::cout<<"camView:"<<g_camView.x<<","<<g_camView.y<<","<<g_camView.z<<std::endl;
    // std::cout<<"camUp:"<<g_camUp.x<<","<<g_camUp.y<<","<<g_camUp.z<<std::endl;
    //CHECK_GL(glDisable(GL_MULTISAMPLE));
    // CHECK_GL(glClearColor(bgcolor.r, bgcolor.g, bgcolor.b, 0.0f));
    CHECK_GL(glClearColor(visScene->drawOptions->bgcolor.r, visScene->drawOptions->bgcolor.g, visScene->drawOptions->bgcolor.b, 0.0f));
    CHECK_GL(glClear(GL_COLOR_BUFFER_BIT));

    // float range[] = {camera->m_near, 500, 1000, 2000, 8000, camera->m_far};
    // for (int i = 5; i >= 1; i--) {
    //     CHECK_GL(glClearDepth(1));
    //     CHECK_GL(glClear(GL_DEPTH_BUFFER_BIT));
    //     camera->proj = glm::perspective(glm::radians(camera->m_fov), aspRatio,
    //                                     range[i - 1], range[i]);

    //     {
    //         zeno::scope_modify unused1{zxxDrawOptions->passReflect, reflect};
    //         zeno::scope_modify unused2{zxxDrawOptions->passIsDepthPass, isDepthPass};
    //         for (auto const &gra : getGraphics()) {
    //             gra->draw();
    //         }
    //     }
    //     if (!isDepthPass && visScene->drawOptions->show_grid) {
    //         zeno::scope_modify unused3{zxxDrawOptions->passReflect, false};
    //         zeno::scope_modify unused4{zxxDrawOptions->passIsDepthPass, false};
    //         for (auto const &hudgra : hudGraphics) {
    //             hudgra->draw();
    //         }
    //     }
        CHECK_GL(glClearDepth(0));
        CHECK_GL(glClear(GL_DEPTH_BUFFER_BIT));
        glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_GEQUAL);
        camera->proj = MakeInfReversedZProjRH(glm::radians(camera->m_fov), aspRatio, 0.01f);
        {
            zeno::scope_modify unused1{zxxDrawOptions->passReflect, reflect};
            zeno::scope_modify unused2{zxxDrawOptions->passIsDepthPass, isDepthPass};
            for (auto const &gra : getGraphics()) {
                gra->draw();
            }
        }
        if (!isDepthPass && visScene->drawOptions->show_grid) {
            zeno::scope_modify unused3{zxxDrawOptions->passReflect, false};
            zeno::scope_modify unused4{zxxDrawOptions->passIsDepthPass, false};
            for (auto const &hudgra : hudGraphics) {
                hudgra->draw();
            }
        }
    // }
}

void ZhxxScene::fast_paint_graphics() {
    CHECK_GL(glViewport(0, 0, camera->m_nx, camera->m_ny));
    vao->bind();
    camera->m_sample_weight = 1.0f;
    /* CHECK_GL(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)); */
    CHECK_GL(glClearColor(visScene->drawOptions->bgcolor.r, visScene->drawOptions->bgcolor.g, visScene->drawOptions->bgcolor.b, 0.0f));
    CHECK_GL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

    zeno::scope_modify unused1{zxxDrawOptions->passReflect, false};
    zeno::scope_modify unused2{zxxDrawOptions->passIsDepthPass, false};
    for (auto const &gra : getGraphics()) {
        gra->draw();
    }
    if (visScene->drawOptions->show_grid) {
        for (auto const &hudgra : hudGraphics) {
            hudgra->draw();
        }
        draw_small_axis();
    }
    vao->unbind();
}

void ZhxxScene::my_paint_graphics(int samples, bool isDepthPass) {

    CHECK_GL(glViewport(0, 0, camera->m_nx, camera->m_ny));
    vao->bind();
    camera->m_sample_weight = 1.0f / (float)samples;
    drawSceneDepthSafe(false, isDepthPass);
    if (!isDepthPass && visScene->drawOptions->show_grid) {
        draw_small_axis();
    }
    vao->unbind();
}

void ZhxxScene::draw_small_axis() { /* TODO: implement this */
}

std::vector<ZhxxIGraphicDraw *> ZhxxScene::getGraphics() {
    std::vector<ZhxxIGraphicDraw *> ret;
    for (auto *gra: this->zxxGraphicsMan->graphics.values<ZhxxIGraphicDraw>()) {
        ret.push_back(gra);
    }
    return ret;
}

bool ZhxxScene::anyGraphicHasMaterial() {
    for (auto const &gra : getGraphics()) {
        if (gra->hasMaterial())
            return true;
    }
    return false;
}

void ZhxxScene::draw(unsigned int target_fbo) {
    if (!anyGraphicHasMaterial()) {
        CHECK_GL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, target_fbo));
        fast_paint_graphics();
        return;
    }

    //CHECK_GL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
    /* ZHXX DONT MODIFY ME */
    lightCluster->clearLights();
    for (auto const &litgra: zxxGraphicsMan->graphics.values<ZhxxIGraphicLight>()) {
        litgra->addToScene();  // inside this will call lightCluster->addLight()
    }

    mDepthPass->paint_graphics(target_fbo);
}

std::vector<char> ZhxxScene::record_frame_offline(int hdrSize, int rgbComps) {
    auto hdrType = std::map<int, int>{
        {1, GL_UNSIGNED_BYTE},
        {2, GL_HALF_FLOAT},
        {4, GL_FLOAT},
    }.at(hdrSize);
    auto rgbType = std::map<int, int>{
        {1, GL_RED},
        {2, GL_RG},
        {3, GL_RGB},
        {4, GL_RGBA},
    }.at(rgbComps);

    std::vector<char> pixels(camera->m_nx * camera->m_ny * rgbComps * hdrSize);

    GLuint fbo, rbo1, rbo2;
    CHECK_GL(glGenRenderbuffers(1, &rbo1));
    CHECK_GL(glGenRenderbuffers(1, &rbo2));
    CHECK_GL(glBindRenderbuffer(GL_RENDERBUFFER, rbo1));
    CHECK_GL(glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA, camera->m_nx,
                                   camera->m_ny));
    CHECK_GL(glBindRenderbuffer(GL_RENDERBUFFER, rbo2));
    CHECK_GL(glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F,
                                   camera->m_nx, camera->m_ny));

    CHECK_GL(glGenFramebuffers(1, &fbo));
    CHECK_GL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo));
    CHECK_GL(glFramebufferRenderbuffer(
        GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rbo1));
    CHECK_GL(glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                       GL_RENDERBUFFER, rbo2));
    CHECK_GL(glDrawBuffer(GL_COLOR_ATTACHMENT0));
    CHECK_GL(glClearColor(visScene->drawOptions->bgcolor.r, visScene->drawOptions->bgcolor.g,
                          visScene->drawOptions->bgcolor.b, 0.0f));
    CHECK_GL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

    draw(fbo);

    if (glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE) {
        CHECK_GL(glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo));
        CHECK_GL(glBlitFramebuffer(0, 0, camera->m_nx, camera->m_ny, 0, 0,
                                   camera->m_nx, camera->m_ny, GL_COLOR_BUFFER_BIT,
                                   GL_NEAREST));
        CHECK_GL(glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo));
        CHECK_GL(glBindBuffer(GL_PIXEL_PACK_BUFFER, 0));
        CHECK_GL(glReadBuffer(GL_COLOR_ATTACHMENT0));

        CHECK_GL(glReadPixels(0, 0, camera->m_nx, camera->m_ny, rgbType,
                              hdrType, pixels.data()));
    }

    CHECK_GL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
    CHECK_GL(glBindRenderbuffer(GL_RENDERBUFFER, 0));

    CHECK_GL(glDeleteRenderbuffers(1, &rbo1));
    CHECK_GL(glDeleteRenderbuffers(1, &rbo2));
    CHECK_GL(glDeleteFramebuffers(1, &fbo));

    return pixels;
}

} // namespace zenovis
