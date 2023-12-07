#include <zeno/utils/log.h>
#include <zenovis/Camera.h>
#include <zenovis/Scene.h>
#include <zenovis/Session.h>
#include <zenovis/DrawOptions.h>
#include <zenovis/bate/GraphicsManager.h>
#include <zenovis/bate/IGraphic.h>
#include <zeno/utils/format.h>
#include <stb_image_write.h>
#ifdef ZENO_ENABLE_OPTIX
#include "ChiefDesignerEXR.h"
#else
#include <tinyexr.h>
#endif
#include <functional>
#include <map>
#include <utility>
#include "zeno/core/Session.h"

namespace zenovis {

struct Session::Impl {
    std::unique_ptr<Scene> scene = std::make_unique<Scene>();

    //std::unordered_map<std::shared_ptr<zeno::IObject>, std::unique_ptr<IGraphic>>
        //new_graphics;
    //std::unordered_map<std::shared_ptr<zeno::IObject>, std::unique_ptr<IGraphic>>
        //graphics;

    //std::vector<std::function<void()>> render_tasks;

    int curr_frameid = 0;
};

Session::Session() : impl(std::make_unique<Impl>()) {
}

Session::~Session() = default;

Scene* Session::get_scene() const {
    return impl->scene.get();
}

void Session::set_safe_frames(bool bLock, int nx, int ny) {
    impl->scene->camera->set_safe_frames(bLock, nx, ny);
}

float Session::get_safe_frames() const {
    return impl->scene->camera->get_safe_frames();
}

bool Session::is_lock_window() const {
    return impl->scene->camera->is_locked_window();
}

void Session::set_window_size(int nx, int ny) {
    impl->scene->camera->setResolution(nx, ny);
}

void Session::set_window_size(int nx, int ny, zeno::vec2i offset) {
    impl->scene->camera->setResolution(nx, ny);
    impl->scene->camera->viewport_offset = offset;
}

std::tuple<int, int> Session::get_window_size() {
    return {
        impl->scene->camera->m_nx,
        impl->scene->camera->m_ny,
    };
}

zeno::vec2i Session::get_viewportOffset()
{
    return impl->scene->camera->viewport_offset;
}

void Session::set_show_grid(bool show_grid) {
    impl->scene->drawOptions->show_grid = show_grid;
}

void Session::set_uv_mode(bool enable) {
    impl->scene->drawOptions->uv_mode = enable;
}

void Session::set_num_samples(int num_samples) {
    // TODO
}

void Session::set_normal_check(bool check) {
    impl->scene->drawOptions->normal_check = check;
}

void Session::set_render_wireframe(bool render_wireframe) {
    impl->scene->drawOptions->render_wireframe = render_wireframe;
}

void Session::set_enable_gi(bool enable_gi) {
    impl->scene->drawOptions->enable_gi = enable_gi;
}

void Session::set_smooth_shading(bool smooth) {
    impl->scene->drawOptions->smooth_shading = smooth;
}

void Session::new_frame() {
    impl->scene->draw();
    //for (auto const &task: impl->render_tasks) {
        //task();
    //}
    //impl->render_tasks.clear();
}

//void Session::new_frame_offline(std::string path, int nsamples) {
    ////impl->render_tasks.push_back([this, path] {
    //auto newpath = zeno::format("{}/{:06d}.png", path, impl->curr_frameid);
    ////zeno::log_info("saving screen {}x{} to {}", impl->scene->camera->m_nx,
                   ////impl->scene->camera->m_ny, newpath);
    //do_screenshot(newpath, "png", nsamples);
    ////});
//}

void Session::do_screenshot(std::string path, std::string type, bool bOptix) {
    auto hdrSize = std::map<std::string, int>{
        {"png", 1},
        {"jpg", 1},
        {"bmp", 1},
        {"exr", 4},
        {"hdr", 4},
    }.at(type);
    auto nx = impl->scene->camera->m_nx;
    auto ny = impl->scene->camera->m_ny;

    auto &ud = zeno::getSession().userData();
    if (bOptix)
    {
        ud.set2("optix_image_path", path);
        if (!ud.has("optix_image_path")) {
            return;
        }
    }

    std::vector<char> pixels = impl->scene->record_frame_offline(hdrSize, 3);

    if (pixels.empty()) {
        return;
    }
    zeno::log_info("saving screenshot {}x{} to {}", nx, ny, path);

    std::map<std::string, std::function<void()>>{
    {"png", [&] {
        stbi_flip_vertically_on_write(true);
        stbi_write_png(path.c_str(), nx, ny, 3, pixels.data(), 0);
    }},
    {"jpg", [&] {
        stbi_flip_vertically_on_write(true);
        stbi_write_jpg(path.c_str(), nx, ny, 3, pixels.data(), 100);
    }},
    {"bmp", [&] {
        stbi_flip_vertically_on_write(true);
        stbi_write_bmp(path.c_str(), nx, ny, 3, pixels.data());
    }},
    {"exr", [&] {
        for (int line = 0; line < ny / 2; ++line) {
            std::swap_ranges(pixels.begin() + hdrSize * 3 * nx * line,
                             pixels.begin() + hdrSize * 3 * nx * (line + 1),
                             pixels.begin() + hdrSize * 3 * nx * (ny - line - 1));
        }
        const char *err = nullptr;
#ifdef ZENO_ENABLE_OPTIX
        using namespace zeno::ChiefDesignerEXR;
#endif
        int ret = SaveEXR((float *)pixels.data(), nx, ny, 3, 1, path.c_str(), &err);
        if (ret != 0) {
            if (err) {
                zeno::log_error("failed to perform SaveEXR to {}: {}", path, err);
                FreeEXRErrorMessage(err);
            }
        }
    }},
    {"hdr", [&] {
        stbi_flip_vertically_on_write(true);
        stbi_write_hdr(path.c_str(), impl->scene->camera->m_nx,
                       impl->scene->camera->m_ny, 3, (float *)pixels.data());
    }},
    }.at(type)();
}

void Session::look_perspective() {
    impl->scene->camera->updateMatrix();
}

void Session::look_to_dir(float cx, float cy, float cz,
                          float dx, float dy, float dz,
                          float ux, float uy, float uz) {
    impl->scene->camera->placeCamera({cx, cy, cz}, {dx, dy, dz}, {ux, uy, uz});
}

void Session::set_background_color(float r, float g, float b) {
    impl->scene->drawOptions->bgcolor = glm::vec3(r, g, b);
}

std::tuple<float, float, float> Session::get_background_color() {
    auto c = impl->scene->drawOptions->bgcolor;
    return {c[0], c[1], c[2]};
}

bool Session::focus_on_node(std::string const &nodeid, zeno::vec3f &center, float &radius) {
    return impl->scene->cameraFocusOnNode(nodeid, center, radius);
}

void Session::set_curr_frameid(int frameid) {
    impl->curr_frameid = std::max(frameid, 0);
}

int Session::get_curr_frameid() {
    return impl->curr_frameid;
}

bool Session::load_objects() {
    return impl->scene->loadFrameObjects(impl->curr_frameid);
}

void Session::set_render_engine(std::string const &name) {
    impl->scene->switchRenderEngine(name);
}

void Session::set_handler(std::shared_ptr<IGraphicHandler> &handler) {
    impl->scene->drawOptions->handler = handler;
}

void Session::load_opengl_api(void *procaddr) {
    Scene::loadGLAPI(procaddr);
}
} // namespace zenovis
