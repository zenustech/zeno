#include <zenovis/Scene.h>
#include <zenovis/Camera.h>
#include <zeno/core/Session.h>
#include <zeno/extra/GlobalComm.h>
#include <zeno/funcs/ObjectGeometryInfo.h>
#include <zeno/utils/envconfig.h>
#include <zeno/types/UserData.h>
#include <zenovis/DrawOptions.h>
#include <zenovis/RenderEngine.h>
#include <zenovis/ShaderManager.h>
#include <zenovis/ObjectsManager.h>
#include <zenovis/opengl/buffer.h>
#include <zenovis/opengl/common.h>
#include <zenovis/opengl/scope.h>
#ifdef ZENO_ENABLE_OPTIX
    #include "../xinxinoptix/xinxinoptixapi.h"
#endif
#include <cstdlib>
#include <map>

namespace zenovis {

void Scene::loadGLAPI(void *procaddr) {
    int res = gladLoadGLLoader((GLADloadproc)procaddr);
    if (res < 0)
        zeno::log_error("failed to load OpenGL via GLAD: {}", res);
}

Scene::~Scene() = default;

Scene::Scene()
    : camera(std::make_unique<Camera>()),
      drawOptions(std::make_unique<DrawOptions>()),
      shaderMan(std::make_unique<ShaderManager>()),
      objectsMan(std::make_unique<ObjectsManager>()),
      renderMan(std::make_unique<RenderManager>(this)) {

    /* gl has been removed from optix scene.
    auto version = (const char *)glGetString(GL_VERSION);
    zeno::log_info("OpenGL version: {}", version ? version : "(null)");
    */

    if (zeno::envconfig::get("OPTX"))
        switchRenderEngine("optx");
    else if (zeno::envconfig::get("ZHXX"))
        switchRenderEngine("zhxx");
    else
        switchRenderEngine("bate");
}

void Scene::cleanUpScene()
{
    if (objectsMan)
        objectsMan->clear_objects();
    if (renderMan && renderMan->getEngine())
        renderMan->getEngine()->update();
}

void Scene::switchRenderEngine(std::string const &name) {
    renderMan->switchDefaultEngine(name);
}

void* Scene::getOptixImg(int& w, int& h)
{
#ifdef ZENO_ENABLE_OPTIX
    return xinxinoptix::optixgetimg(w, h);
#else
    return nullptr;
#endif
}

bool Scene::cameraFocusOnNode(std::string const &nodeid, zeno::vec3f &center, float &radius) {
    for (auto const &[key, ptr]: this->objectsMan->pairs()) {
        if (nodeid == key.substr(0, key.find_first_of(':'))) {
            return zeno::objectGetFocusCenterRadius(ptr, center, radius);
        }
    }
    zeno::log_warn("cannot focus: node with id {} not found, did you tagged VIEW on it?", nodeid);
    return false;
}

bool Scene::loadFrameObjects(int frameid) {
    auto &ud = zeno::getSession().userData();
    ud.set2<int>("frameid", std::move(frameid));

    const auto& cbLoadObjs = [this](std::map<std::string, std::shared_ptr<zeno::IObject>> const& objs) -> bool {
        return this->objectsMan->load_objects(objs);
    };
    bool isFrameValid = false;
    bool inserted = zeno::getSession().globalComm->load_objects(frameid, cbLoadObjs, isFrameValid);
    if (!isFrameValid)
        return false;

    renderMan->getEngine()->update();
    return inserted;
}

void Scene::draw() {
    if (renderMan->getDefaultEngineName() != "optx")
    {
        //CHECK_GL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
        auto offset = camera->m_block_window? camera->viewport_offset : zeno::vec2i(0, 0);
        CHECK_GL(glViewport(offset[0], offset[1], camera->m_nx, camera->m_ny));
        //CHECK_GL(glClearColor(drawOptions->bgcolor.r, drawOptions->bgcolor.g, drawOptions->bgcolor.b, 0.0f));
    }

    zeno::log_trace("scene redraw {}x{}", camera->m_nx, camera->m_ny);
    renderMan->getEngine()->draw();
}

std::vector<char> Scene::record_frame_offline(int hdrSize, int rgbComps) {
    zeno::log_trace("offline draw {}x{}x{}x{}", camera->m_nx, camera->m_ny, rgbComps, hdrSize);
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

    bool bOptix = renderMan->getDefaultEngineName() == "optx";
    if (bOptix)
    {
        draw();
        return std::vector<char>();
    }

    std::vector<char> pixels(camera->m_nx * camera->m_ny * rgbComps * hdrSize);

    //GLint zerofbo = 0, zerorbo = 0;
    //CHECK_GL(glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &zerofbo));
    //CHECK_GL(glGetIntegerv(GL_RENDERBUFFER_BINDING, &zerorbo));
    //printf("%d\n", zerofbo);
    //printf("%d\n", zerorbo);

    auto fbo = opengl::scopeGLGenFramebuffer();
    auto rbo1 = opengl::scopeGLGenRenderbuffer();
    auto rbo2 = opengl::scopeGLGenRenderbuffer();

    {
        auto bindFbo = opengl::scopeGLBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);

        CHECK_GL(glBindRenderbuffer(GL_RENDERBUFFER, rbo1));
        CHECK_GL(glRenderbufferStorageMultisample(GL_RENDERBUFFER, drawOptions->msaa_samples, GL_RGBA, camera->m_nx, camera->m_ny));
        CHECK_GL(glBindRenderbuffer(GL_RENDERBUFFER, rbo2));
        CHECK_GL(glRenderbufferStorageMultisample(GL_RENDERBUFFER, drawOptions->msaa_samples, GL_DEPTH_COMPONENT32, camera->m_nx, camera->m_ny));
        CHECK_GL(glBindRenderbuffer(GL_RENDERBUFFER, 0));

        CHECK_GL(glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rbo1));
        CHECK_GL(glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo2));
        CHECK_GL(glClearColor(drawOptions->bgcolor.r, drawOptions->bgcolor.g,
                              drawOptions->bgcolor.b, 0.0f));
        CHECK_GL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

        {
            auto bindDrawBuf = opengl::scopeGLDrawBuffer(GL_COLOR_ATTACHMENT0);
            draw();
        }

        if (glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE) {
            int nx = camera->m_nx;
            int ny = camera->m_ny;
            // normal buffer as intermedia
            auto sfbo = opengl::scopeGLGenFramebuffer();
            auto srbo1 = opengl::scopeGLGenRenderbuffer();
            auto srbo2 = opengl::scopeGLGenRenderbuffer();
            CHECK_GL(glBindRenderbuffer(GL_RENDERBUFFER, srbo1));
            CHECK_GL(glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA, nx, ny));
            CHECK_GL(glBindRenderbuffer(GL_RENDERBUFFER, srbo2));
            CHECK_GL(glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32, nx, ny));

            auto bindReadSFbo = opengl::scopeGLBindFramebuffer(GL_DRAW_FRAMEBUFFER, sfbo);
            CHECK_GL(glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER,
                                               GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, srbo1));
            CHECK_GL(glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER,
                                               GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, srbo2));

            auto bindDrawBuf = opengl::scopeGLDrawBuffer(GL_COLOR_ATTACHMENT0);

            CHECK_GL(glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo));
            CHECK_GL(glBlitFramebuffer(0, 0, camera->m_nx, camera->m_ny, 0, 0,
                                       camera->m_nx, camera->m_ny, GL_COLOR_BUFFER_BIT,
                                       GL_NEAREST));
            CHECK_GL(glBindFramebuffer(GL_READ_FRAMEBUFFER, sfbo));

            auto bindPackBuffer = opengl::scopeGLBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
            auto bindPackAlignment = opengl::scopeGLPixelStorei(GL_PACK_ALIGNMENT, 1);
            auto bindRead = opengl::scopeGLReadBuffer(GL_COLOR_ATTACHMENT0);

            CHECK_GL(glReadPixels(0, 0, camera->m_nx, camera->m_ny, rgbType,
                                  hdrType, pixels.data()));
        } else {
            zeno::log_error("failed to complete framebuffer");
        }
    }

    return pixels;
}

} // namespace zenovis
