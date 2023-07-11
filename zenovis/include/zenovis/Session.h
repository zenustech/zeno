#pragma once

#include <array>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include <set>
#include <zeno/core/IObject.h>
#include <zeno/utils/disable_copy.h>
#include <zeno/utils/vec.h>
#include <zenovis/Scene.h>
#include <zenovis/bate/IGraphic.h>

namespace zenovis {

struct Session : zeno::disable_copy {
    struct Impl;

    std::unique_ptr<Impl> impl;

    Session();
    ~Session();

    void new_frame();
    bool load_objects();
    void lock_window_size(bool bLock, int nx, int ny);
    void set_safe_frames(bool bLock, int nx, int ny);
    float get_safe_frames() const;
    bool is_lock_window() const;
    void set_window_size(int nx, int ny);
    std::tuple<int, int> get_window_size();
    void set_curr_frameid(int frameid);
    int get_curr_frameid();
    void set_show_grid(bool flag);
    void look_perspective(float cx, float cy, float cz, float theta,
                          float phi, float radius, float fov,
                          bool ortho_mode, float aperture, float focalPlaneDistance);
    void look_to_dir(float cx, float cy, float cz,
                     float dx, float dy, float dz,
                     float ux, float uy, float uz);
    void do_screenshot(std::string path, std::string type);
    //void new_frame_offline(std::string path, int nsamples);
    void set_background_color(float r, float g, float b);
    std::tuple<float, float, float> get_background_color();
    void set_num_samples(int num_samples);
    void set_enable_gi(bool enable_gi);
    void set_smooth_shading(bool smooth);
    void set_normal_check(bool check);
    void set_render_wireframe(bool render_wireframe);
    void set_render_engine(std::string const &name);
    void set_handler(std::shared_ptr<IGraphicHandler> &handler);
    bool focus_on_node(std::string const &nodeid, zeno::vec3f &center, float &radius);
    static void load_opengl_api(void *procaddr);
    Scene* get_scene() const;
};

} // namespace zenovis
