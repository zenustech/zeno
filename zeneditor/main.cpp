#include <Hg/OpenGL/stdafx.hpp>
#include <Hg/IOUtils.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace hg::OpenGL;

#include <imgui/imgui.h>
#include <imnodes/imnodes.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>
#include "NodeEditor.h"
#include <iostream>
#include <cassert>

namespace zeneditor {

static GLFWwindow *window;
static int nx = 1024, ny = 768;

static void error_callback(int error, const char *msg) {
  fprintf(stderr, "error %d: %s\n", error, msg);
}

static void key_callback(GLFWwindow *window, int key, int scancode, int action,
                         int mods) {
  if (key == GLFW_KEY_ESCAPE)
    glfwSetWindowShouldClose(window, GLFW_TRUE);
}

static void click_callback(GLFWwindow *window, int button, int action,
                           int mods) {
}

static void scroll_callback(GLFWwindow *window, double xoffset,
                            double yoffset) {
}

static void motion_callback(GLFWwindow *window, double xpos, double ypos) {
}

std::unique_ptr<NodeEditor> editor;

void load_descs(std::string const &descs) {
  assert(editor);
  editor->load_descriptors(descs);
}

void initialize() {
  glfwInit();
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);

  window = glfwCreateWindow(nx, ny, "noeditor", NULL, NULL);
  glfwMakeContextCurrent(window);
  glfwSetErrorCallback(error_callback);
  glfwSetKeyCallback(window, key_callback);
  glfwSetMouseButtonCallback(window, click_callback);
  glfwSetCursorPosCallback(window, motion_callback);
  glfwSetScrollCallback(window, scroll_callback);

  glewInit();
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 330 core");
  imnodes::Initialize();

  editor = std::make_unique<NodeEditor>();
}

std::map<std::string, std::string> new_frame() {
  std::map<std::string, std::string> ret;

  if (glfwWindowShouldClose(window)) {
    ret["close"] = "1";
    return ret;
  }

  glfwPollEvents();

  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  glfwGetFramebufferSize(window, &nx, &ny);
  CHECK_GL(glViewport(0, 0, nx, ny));
  CHECK_GL(glClearColor(0.3f, 0.2f, 0.1f, 0.0f));
  CHECK_GL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

  ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(nx, ny), ImGuiCond_FirstUseEver);
  ImGui::Begin("Node Editor");

  if (ImGui::Button("Refresh Descs")) {
    ret["refresh"] = "1";
  }

  static char pathBuf[1024] = "/tmp/a.json";
  ImGui::SameLine();
  ImGui::SetNextItemWidth(24 * ImGui::GetFontSize());
  ImGui::InputText("", pathBuf, sizeof(pathBuf));

  ImGui::SameLine();
  if (ImGui::Button("Save Graph")) {
    ret["save"] = pathBuf;
  }

  ImGui::SameLine();
  if (ImGui::Button("Load Graph")) {
    ret["load"] = pathBuf;
  }

  ImGui::SameLine();
  if (ImGui::Button("Execute Graph")) {
    std::stringstream ss;
    editor->dump_graph(ss);
    ret["execute"] = ss.str();
  }

  ImGui::SameLine();
  {
    static int max_frames = 1;
    ImGui::SetNextItemWidth(6 * ImGui::GetFontSize());
    ImGui::DragInt("Frames", &max_frames, 1);
    max_frames = std::max(0, max_frames);
    std::stringstream ss;
    ss << max_frames;
    ret["exec_nframes"] = ss.str();
  }

  editor->draw();
  ImGui::End();

  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
  glfwSwapBuffers(window);

  return ret;
}

void finalize() {
  imnodes::Shutdown();
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  glfwDestroyWindow(window);
  glfwTerminate();
}

NodeEditor::FGraphType save_graph() {
  return editor->save_graph();
}

void load_graph(NodeEditor::FGraphType const &graph) {
  return editor->load_graph(graph);
}

}


PYBIND11_MODULE(libzeneditor, m) {
    m.def("initialize", zeneditor::initialize);
    m.def("load_descs", zeneditor::load_descs);
    m.def("new_frame", zeneditor::new_frame);
    m.def("finalize", zeneditor::finalize);
    m.def("save_graph", zeneditor::save_graph);
    m.def("load_graph", zeneditor::load_graph);
}
