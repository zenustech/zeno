#include <stb_image_write.h>
#include <unordered_map>
#include <zeno/utils/log.h>
#include <zenovis/Camera.h>
#include <zenovis/Scene.h>
#include <zenovis/Session.h>
#include <zenovis/makeGraphic.h>
#include <zenovis/opengl/common.h>

namespace zenovis {

using namespace zeno;

struct Session::Impl {
    std::unique_ptr<Scene> scene = std::make_unique<Scene>();

    std::unordered_map<std::shared_ptr<IObject>, std::unique_ptr<IGraphic>>
        new_graphics;
    std::unordered_map<std::shared_ptr<IObject>, std::unique_ptr<IGraphic>>
        graphics;

    int curr_frameid = 0;
};

Session::Session() : impl(std::make_unique<Impl>()) {
}

Session::~Session() = default;

void Session::load_objects(
    std::vector<std::shared_ptr<zeno::IObject>> const &objs) {
    impl->new_graphics.clear();
    for (auto const &obj : objs) {
        log_trace("load_object: got object {}", obj.get());
        auto it = impl->graphics.find(obj);
        if (it != impl->graphics.end()) {
            log_trace("load_object: cache hit graphics {}", it->second.get());
            impl->new_graphics.emplace(it->first, std::move(it->second));
            impl->graphics.erase(it);
            continue;
        }
        auto ig = makeGraphic(impl->scene.get(), obj);
        log_trace("load_object: fresh load graphics {}", ig.get());
        if (!ig)
            continue;
        impl->new_graphics.emplace(obj, std::move(ig));
    }
    std::swap(impl->new_graphics, impl->graphics);
    impl->new_graphics.clear();
}

void Session::set_window_size(int nx, int ny) {
    impl->scene->camera->nx = nx;
    impl->scene->camera->ny = ny;
}

void Session::set_show_grid(bool show_grid) {
    impl->scene->camera->show_grid = show_grid;
}

void Session::set_num_samples(int num_samples) {
    impl->scene->camera->setNumSamples(num_samples);
}

void Session::set_normal_check(bool check) {
    impl->scene->camera->normal_check = check;
}

void Session::set_render_wireframe(bool render_wireframe) {
    impl->scene->camera->render_wireframe = render_wireframe;
}

void Session::new_frame() {
    impl->scene->draw();
}

void Session::new_frame_offline(std::string path) {
    char buf[1024];
    sprintf(buf, "%s/%06d.png", path.c_str(), impl->curr_frameid);
    zeno::log_info("saving screen {}x{} to {}", impl->scene->camera->nx,
                    impl->scene->camera->ny, buf);
    do_screenshot(buf);
}

void Session::do_screenshot(std::string path) {
    std::vector<char> pixels = impl->scene->record_frame_offline();
    stbi_flip_vertically_on_write(true);
    stbi_write_png(path.c_str(), impl->scene->camera->nx,
                   impl->scene->camera->ny, 3, &pixels[0], 0);
}

void Session::look_perspective(double cx, double cy, double cz, double theta,
                               double phi, double radius, double fov,
                               bool ortho_mode) {
    impl->scene->camera->look_perspective(cx, cy, cz, theta, phi, radius, fov,
                                          ortho_mode);
}

void Session::set_perspective(std::array<double, 16> viewArr,
                              std::array<double, 16> projArr) {
    impl->scene->camera->set_perspective(viewArr, projArr);
}

void Session::set_background_color(float r, float g, float b) {
    impl->scene->camera->bgcolor = glm::vec3(r, g, b);
}

void Session::set_curr_frameid(int frameid) {
    impl->curr_frameid = std::max(frameid, 0);
}

int Session::get_curr_frameid() {
    return impl->curr_frameid;
}

} // namespace zenovis
