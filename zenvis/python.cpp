#include "stdafx.hpp"
#include "main.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

namespace zenvis {

void initialize();
void finalize();
void new_frame();
void set_window_size(int nx_, int ny_);
void set_curr_frameid(int frameid);
int get_curr_frameid();
int get_solver_interval();
int get_render_fps();
void look_perspective(
    double cx, double cy, double cz,
    double theta, double phi, double radius,
    double fov, bool ortho_mode);
void set_perspective(
    std::array<double, 16> viewArr,
    std::array<double, 16> projArr);

};

PYBIND11_MODULE(libzenvis, m) {
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
}
