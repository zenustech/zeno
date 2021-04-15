#include "stdafx.hpp"
#include "main.hpp"
#include "server.hpp"
#include "frames.hpp"
#include "IGraphic.hpp"
#include <Hg/FPSCounter.hpp>
#include <Hg/IPC/SharedMemory.hpp>
#include <sstream>
#include <cstdlib>

namespace zenvis {

std::unique_ptr<Server> Server::_instance;

int curr_frameid = -1;

static int nx = 960, ny = 800;

static double last_xpos, last_ypos;
static double theta = 0.0, phi = 0.0, radius = 3.0, fov = 60.0;
static bool mmb_pressed, shift_pressed, ortho_mode;
static glm::dvec3 center;

void set_program_uniforms(Program *pro) {
  double cos_t = glm::cos(theta), sin_t = glm::sin(theta);
  double cos_p = glm::cos(phi), sin_p = glm::sin(phi);
  glm::dvec3 back(cos_t * sin_p, sin_t, -cos_t * cos_p);
  glm::dvec3 up(-sin_t * sin_p, cos_t, sin_t * cos_p);

  auto view = glm::lookAt(center - back * radius, center, up);
  auto proj = glm::perspective(glm::radians(fov), nx * 1.0 / ny, 0.05, 500.0);
  if (ortho_mode) {
    view = glm::lookAt(center - back, center, up);
    proj = glm::ortho(-radius * nx / ny, radius * nx / ny, -radius, radius,
                      -100.0, 100.0);
  }

  pro->use();

  auto pers = proj * view;
  pro->set_uniform("mVP", pers);
  pro->set_uniform("mInvVP", glm::inverse(pers));
  pro->set_uniform("mView", view);
  pro->set_uniform("mProj", proj);

  pro->set_uniform("light.dir", glm::normalize(glm::vec3(1, 2, 3)));
  pro->set_uniform("light.color", glm::vec3(1, 1, 1));
}

static void key_callback(GLFWwindow *window, int key, int scancode, int action,
                         int mods) {
  if (key == GLFW_KEY_TAB && action == GLFW_RELEASE)
    ortho_mode = !ortho_mode;
  if (key == GLFW_KEY_ESCAPE)
    glfwSetWindowShouldClose(window, GLFW_TRUE);
}

static void click_callback(GLFWwindow *window, int button, int action,
                           int mods) {
  if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
    mmb_pressed = action == GLFW_PRESS;
    shift_pressed = !!(mods & GLFW_MOD_SHIFT);
  }

  if (mmb_pressed)
    glfwGetCursorPos(window, &last_xpos, &last_ypos);
}

static void scroll_callback(GLFWwindow *window, double xoffset,
                            double yoffset) {
  radius *= glm::pow(0.89, yoffset);
}

static void motion_callback(GLFWwindow *window, double xpos, double ypos) {
  double dx = (xpos - last_xpos) / nx, dy = (ypos - last_ypos) / ny;

  if (mmb_pressed && !shift_pressed) {
    theta = glm::clamp(theta - dy * M_PI, -M_PI / 2, M_PI / 2);
    phi = phi + dx * M_PI;
  }

  if (mmb_pressed && shift_pressed) {
    double cos_t = glm::cos(theta), sin_t = glm::sin(theta);
    double cos_p = glm::cos(phi), sin_p = glm::sin(phi);
    glm::dvec3 back(cos_t * sin_p, sin_t, -cos_t * cos_p);
    glm::dvec3 up(-sin_t * sin_p, cos_t, sin_t * cos_p);
    auto right = glm::cross(up, back);
    up = glm::cross(back, right);
    glm::dvec3 delta = glm::normalize(right) * dx + glm::normalize(up) * dy;
    center = center + delta * radius;
  }

  last_xpos = xpos;
  last_ypos = ypos;
}

static std::unique_ptr<VAO> vao;

void initialize() {
  glewInit();

  CHECK_GL(glEnable(GL_BLEND));
  CHECK_GL(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
  CHECK_GL(glEnable(GL_DEPTH_TEST));
  CHECK_GL(glEnable(GL_PROGRAM_POINT_SIZE));

  vao = std::make_unique<VAO>();
}

static void draw_contents(void) {
  CHECK_GL(glClearColor(0.3f, 0.2f, 0.1f, 0.0f));
  CHECK_GL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
  vao->bind();
  for (auto const &gra: graphics) {
    gra->draw();
  }
  vao->unbind();
  CHECK_GL(glFlush());
}

static hg::FPSCounter solverFPS(glfwGetTime, 1);
static hg::FPSCounter renderFPS(glfwGetTime, 10);

static char titleBuf[512];

static void update_title() {
  sprintf(titleBuf, "frame %d | %.1f fps | %.02f spf\n",
      curr_frameid, renderFPS.fps(), solverFPS.interval());
}

void finalize() {
  vao = nullptr;
}

void new_frame() {
  auto &server = Server::get();

  server.poll_init();
  if (curr_frameid >= server.frameid) {
    curr_frameid = server.frameid - 1;
    server.poll();
    if (server.frameid - 1 != curr_frameid) {
      solverFPS.tick();
    }
  }

  update_frame_graphics();
  renderFPS.tick();
  update_title();

  CHECK_GL(glViewport(0, 0, nx, ny));
  draw_contents();

  curr_frameid++;
  if (curr_frameid < 0) curr_frameid = 0;
}

void set_window_size(int nx_, int ny_) {
  nx = nx_;
  ny = ny_;
}

void set_curr_frameid(int frameid) {
  curr_frameid = frameid;
}

int get_curr_frameid() {
  return curr_frameid;
}

double get_render_fps() {
  return renderFPS.fps();
}

double get_solver_interval() {
  return solverFPS.interval();
}

}
