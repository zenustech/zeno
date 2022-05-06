#include <zenovis/Scene.h>
#include <zenovis/Camera.h>
#include <zeno/core/Session.h>
#include <zeno/extra/GlobalComm.h>
#include <zeno/types/UserData.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zenovis/DrawOptions.h>
#include <zenovis/RenderEngine.h>
#include <zenovis/ShaderManager.h>
#include <zenovis/ObjectsManager.h>
#include <zenovis/opengl/buffer.h>
#include <zenovis/opengl/common.h>
#include <zenovis/opengl/vao.h>
#include <zeno/utils/scope_exit.h>
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
      objectsMan(std::make_unique<ObjectsManager>()) {

    auto version = (const char *)glGetString(GL_VERSION);
    zeno::log_info("OpenGL version: {}", version ? version : "(null)");

    if (std::getenv("ZENO_ZHXX"))
        switchRenderEngine("zhxx");
    else
        switchRenderEngine("bate");
}

void Scene::switchRenderEngine(std::string const &name) {
    std::map<std::string, std::function<void()>>{
    {"bate", [&] { renderEngine = makeRenderEngineBate(this); }},
    {"zhxx", [&] { renderEngine = makeRenderEngineZhxx(this); }},
    }.at(name)();
}

static bool calcObjectCenterRadius(zeno::IObject *ptr, zeno::vec3f &center, float &radius) {
    if (auto obj = dynamic_cast<zeno::PrimitiveObject *>(ptr)) {
        auto [bmin, bmax] = primBoundingBox(obj);
        auto delta = bmax - bmin;
        radius = std::max({delta[0], delta[1], delta[2]}) * 0.5f;
        center = (bmin + bmax) * 0.5f;
        return true;
    }
    return false;
}

bool Scene::cameraFocusOnNode(std::string const &nodeid) {
    for (auto const &[key, ptr]: this->objectsMan->pairs()) {
        if (nodeid == key.substr(0, key.find_first_of(':'))) {
            zeno::vec3f center;
            float radius;
            if (calcObjectCenterRadius(ptr, center, radius))
                this->camera->focusCamera(center[0], center[1], center[2], radius);
            else
                return false;
        }
    }
    zeno::log_debug("cannot focus: node with id {} not found, did you tagged VIEW on it?", nodeid);
    return false;
}

void Scene::loadFrameObjects(int frameid) {
    auto const *viewObjs = zeno::getSession().globalComm->getViewObjects(frameid);
    if (viewObjs) {
        zeno::log_trace("load_objects: {} objects at frame {}", viewObjs->size(), frameid);
        this->objectsMan->load_objects(viewObjs->m_curr);
    } else {
        zeno::log_trace("load_objects: no objects at frame {}", frameid);
        this->objectsMan->load_objects({});
    }
    renderEngine->update();
}

void Scene::draw() {
    CHECK_GL(glViewport(0, 0, camera->m_nx, camera->m_ny));
    CHECK_GL(glClearColor(drawOptions->bgcolor.r, drawOptions->bgcolor.g, drawOptions->bgcolor.b, 0.0f));
    CHECK_GL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));

    renderEngine->draw();
}

std::vector<char> Scene::record_frame_offline(int hdrSize, int rgbComps) {
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
    CHECK_GL(glClearColor(drawOptions->bgcolor.r, drawOptions->bgcolor.g,
                          drawOptions->bgcolor.b, 0.0f));
    CHECK_GL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

    CHECK_GL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo));
    draw();

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
