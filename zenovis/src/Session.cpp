#include <zeno/utils/log.h>
#include <zenovis/Camera.h>
#include <zenovis/Scene.h>
#include <zenovis/Session.h>
#include <zenovis/DrawOptions.h>
#include <zenovis/bate/GraphicsManager.h>
#include <zenovis/bate/IGraphic.h>
#include <zeno/utils/format.h>
#include <stb_image_write.h>
#include <tinyexr.h>
#include <functional>
#include <map>

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

void Session::set_window_size(int nx, int ny) {
    impl->scene->camera->m_nx = nx;
    impl->scene->camera->m_ny = ny;
}

void Session::set_show_grid(bool show_grid) {
    impl->scene->drawOptions->show_grid = show_grid;
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

void Session::new_frame_offline(std::string path) {
    //impl->render_tasks.push_back([this, path] {
    auto newpath = zeno::format("{}/{:06d}.png", path, impl->curr_frameid);
    //zeno::log_info("saving screen {}x{} to {}", impl->scene->camera->m_nx,
                   //impl->scene->camera->m_ny, newpath);
    do_screenshot(newpath, "png");
    //});
}

void Session::do_screenshot(std::string path, std::string type) {
    auto hdrSize = std::map<std::string, int>{
        {"png", 1},
        {"jpg", 1},
        {"bmp", 1},
        {"exr", 4},
        {"hdr", 4},
    }.at(type);
    auto nx = impl->scene->camera->m_nx;
    auto ny = impl->scene->camera->m_ny;

    zeno::log_info("saving screenshot {}x{} to {}", nx, ny, path);
    std::vector<char> pixels = impl->scene->record_frame_offline(hdrSize, 3);

    stbi_flip_vertically_on_write(true);
    std::map<std::string, std::function<void()>>{
    {"png", [&] {
        stbi_write_png(path.c_str(), nx, ny, 3, pixels.data(), 0);
    }},
    {"jpg", [&] {
        stbi_write_jpg(path.c_str(), nx, ny, 3, pixels.data(), 80);
    }},
    {"bmp", [&] {
        stbi_write_bmp(path.c_str(), nx, ny, 3, pixels.data());
    }},
    {"exr", [&] {
        const char *err = nullptr;
        int ret = SaveEXR((float *)pixels.data(), nx, ny, 3, 1, path.c_str(), &err);
        if (ret != TINYEXR_SUCCESS) {
            if (err) {
                zeno::log_error("failed to perform SaveEXR to {}: {}", path, err);
                FreeEXRErrorMessage(err);
            }
        }
    }},
    {"hdr", [&] {
        stbi_write_hdr(path.c_str(), impl->scene->camera->m_nx,
                       impl->scene->camera->m_ny, 3, (float *)pixels.data());
    }},
    }.at(type)();
}

void Session::look_perspective(float cx, float cy, float cz, float theta,
                               float phi, float radius, float fov,
                               bool ortho_mode) {
    impl->scene->camera->lookCamera(cx, cy, cz, theta, phi, radius, ortho_mode ? 0.f : fov);
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

void Session::load_objects() {
    impl->scene->loadFrameObjects(impl->curr_frameid);
}

void Session::set_render_engine(std::string const &name) {
    impl->scene->switchRenderEngine(name);
}

void Session::load_opengl_api(void *procaddr) {
    Scene::loadGLAPI(procaddr);
}
} // namespace zenovis
