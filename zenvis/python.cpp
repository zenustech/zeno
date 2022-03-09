#include "stdafx.hpp"
#include "main.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

namespace zenvis {

void initialize();
void finalize();
void new_frame();
void clear_graphics();
void auto_gc_frame_data(int nkeep);
std::vector<int> get_valid_frames_list();
void load_file(std::string name, std::string ext, std::string path, int frameid);
void set_window_size(int nx, int ny);
void set_curr_frameid(int frameid);
int get_curr_frameid();
double get_solver_interval();
double get_render_fps();
void set_show_grid(bool flag);
void look_perspective(
    double cx, double cy, double cz,
    double theta, double phi, double radius,
    double fov, bool ortho_mode);
void set_perspective(
    std::array<double, 16> viewArr,
    std::array<double, 16> projArr);
void do_screenshot(std::string path);
void new_frame_offline(std::string path);
void set_background_color(float r, float g, float b);
std::tuple<float, float, float> get_background_color();
void set_smooth_shading(bool smooth);
void set_render_wireframe(bool render_wireframe);
void setup_env_map(std::string name);

};

PYBIND11_MODULE(pylib_zenvis, m) {
    m.def("initialize", zenvis::initialize);
    m.def("finalize", zenvis::finalize);
    m.def("new_frame", zenvis::new_frame);
    m.def("set_window_size", zenvis::set_window_size);
    m.def("set_curr_frameid", zenvis::set_curr_frameid);
    m.def("get_curr_frameid", zenvis::get_curr_frameid);
    m.def("get_solver_interval", zenvis::get_solver_interval);
    m.def("get_render_fps", zenvis::get_render_fps);
    m.def("look_perspective", zenvis::look_perspective);
    m.def("set_perspective", zenvis::set_perspective);
    m.def("clear_graphics", zenvis::clear_graphics);
    m.def("auto_gc_frame_data", zenvis::auto_gc_frame_data);
    m.def("get_valid_frames_list", zenvis::get_valid_frames_list);
    m.def("load_file", zenvis::load_file);
    m.def("do_screenshot", zenvis::do_screenshot);
    m.def("set_show_grid", zenvis::set_show_grid);
    m.def("new_frame_offline", zenvis::new_frame_offline);
    m.def("set_background_color", zenvis::set_background_color);
    m.def("get_background_color", zenvis::get_background_color);
    m.def("set_smooth_shading", zenvis::set_smooth_shading);
    m.def("set_render_wireframe", zenvis::set_render_wireframe);
    m.def("setup_env_map", zenvis::setup_env_map);
}


/****\

server -> client: (per-frame)

{frameid}:{solver_frameid}:{solver_interval}:{jpegData}

client -> server: (per-mouse-event)

{nx}:{ny}:{cx}:{cy}:{cz}:{theta}:{phi}:{radius}:{fov}:{ortho_mode}:{set_frameid}

^^^NVM: our web enginneer never get hired in to handle zenwebvis..^^^

\****/
