#pragma once

#include <string>
#include <array>
#include <vector>
#include <tuple>

namespace zenvis {
extern void clearReflectMask();
void setLightHight(float h);
void setLight(float x, float y, float z);
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
void set_num_samples(int num_samples);
void set_smooth_shading(bool smooth);
void set_normal_check(bool check);
void set_render_wireframe(bool render_wireframe);
unsigned int setup_env_map(std::string name);
void setLightData(
  int index,
  std::tuple<float, float, float> dir,
  float height,
  float softness,
  std::tuple<float, float, float> tint,
  std::tuple<float, float, float> color,
  float intensity
);
int getLightCount();
void addLight();
std::tuple<
  std::tuple<float, float, float>,
  float,
  float,
  std::tuple<float, float, float>,
  std::tuple<float, float, float>,
  float
> getLight(int i);
}
