#include "MyShader.hpp"
#include "glad/glad.h"
#include "stdafx.hpp"
#include "main.hpp"
#include "IGraphic.hpp"
#include <Hg/FPSCounter.hpp>
#include <Hg/OpenGL/stdafx.hpp>
#include <sstream>
#include <cstdlib>
#include <cmath>
#include <array>
#include <stb_image_write.h>
#include <zeno/utils/logger.h>
#include <Hg/OpenGL/stdafx.hpp>
#include "zenvisapi.hpp"

namespace zenvis {


int curr_frameid = -1;

static int num_samples = 16;
static bool show_grid = true;
static bool smooth_shading = false;
static bool normal_check = false;
bool render_wireframe = false;

static int nx = 960, ny = 800;

static glm::vec3 bgcolor(0.23f, 0.23f, 0.23f);

static double last_xpos, last_ypos;
static glm::vec3 center;

static glm::mat4x4 view(1), proj(1);
static glm::mat4x4 gizmo_view(1), gizmo_proj(1);
static float point_scale = 1.f;
static float camera_radius = 1.f;
static float grid_scale = 1.f;
static float grid_blend = 0.f;

void set_perspective(
    std::array<double, 16> viewArr,
    std::array<double, 16> projArr)
{
  std::memcpy(glm::value_ptr(view), viewArr.data(), viewArr.size());
  std::memcpy(glm::value_ptr(proj), projArr.data(), projArr.size());
}

void look_perspective(
    double cx, double cy, double cz,
    double theta, double phi, double radius,
    double fov, bool ortho_mode) {
  center = glm::vec3(cx, cy, cz);

  point_scale = ny / (50.f * tanf(fov*0.5f*3.1415926f/180.0f));

  double cos_t = glm::cos(theta), sin_t = glm::sin(theta);
  double cos_p = glm::cos(phi), sin_p = glm::sin(phi);
  glm::vec3 back(cos_t * sin_p, sin_t, -cos_t * cos_p);
  glm::vec3 up(-sin_t * sin_p, cos_t, sin_t * cos_p);

  if (ortho_mode) {
    view = glm::lookAt(center - back, center, up);
    proj = glm::ortho(-radius * nx / ny, radius * nx / ny, -radius, radius,
                      -100.0, 100.0);
  } else {
    view = glm::lookAt(center - back * (float)radius, center, up);
    proj = glm::perspective(glm::radians(fov), nx * 1.0 / ny, 0.05, 20000.0 * std::max(1.0f, (float)radius / 10000.f));
  }
  camera_radius = radius;
  float level = std::fmax(std::log(radius) / std::log(5) - 1.0, -1);
  grid_scale = std::pow(5, std::floor(level));
  auto ratio_clamp = [](float value, float lower_bound, float upper_bound) {
      float ratio = (value - lower_bound) / (upper_bound - lower_bound);
      return fmin(fmax(ratio, 0.0), 1.0);
  };
  grid_blend = ratio_clamp(level - std::floor(level), 0.8, 1.0);
  center = glm::vec3(0, 0, 0);
  radius = 5.0;
  gizmo_view = glm::lookAt(center - back, center, up);
  gizmo_proj = glm::ortho(-radius * nx / ny, radius * nx / ny, -radius, radius,
                      -100.0, 100.0);
}

void set_program_uniforms(Program *pro) {
  pro->use();

  auto pers = proj * view;
  pro->set_uniform("mVP", pers);
  pro->set_uniform("mInvVP", glm::inverse(pers));
  pro->set_uniform("mView", view);
  pro->set_uniform("mProj", proj);
  pro->set_uniform("mInvView", glm::inverse(view));
  pro->set_uniform("mInvProj", glm::inverse(proj));
  pro->set_uniform("mPointScale", point_scale);
  pro->set_uniform("mSmoothShading", smooth_shading);
  pro->set_uniform("mNormalCheck", normal_check);
  pro->set_uniform("mCameraRadius", camera_radius);
  pro->set_uniform("mCameraCenter", center);
  pro->set_uniform("mGridScale", grid_scale);
  pro->set_uniform("mGridBlend", grid_blend);
}

static std::unique_ptr<VAO> vao;
static std::unique_ptr<IGraphic> grid;
static std::unique_ptr<IGraphic> axis;
static glm::vec3 light;
extern glm::vec3 getLight()
{
  return light;
}
void setLight(float x, float y, float z)
{
  light = glm::vec3(x,y,z);
}
std::unique_ptr<IGraphic> makeGraphicGrid();
std::unique_ptr<IGraphic> makeGraphicAxis();
void initialize() {
  gladLoadGL();
  
  setLight(1,1,0);
  auto version = (const char *)glGetString(GL_VERSION);
  zeno::log_info("OpenGL version: {}", version ? version : "(null)");

  CHECK_GL(glEnable(GL_BLEND));
  CHECK_GL(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
  CHECK_GL(glEnable(GL_DEPTH_TEST));
  CHECK_GL(glEnable(GL_PROGRAM_POINT_SIZE));
  //CHECK_GL(glEnable(GL_PROGRAM_POINT_SIZE_ARB));
  //CHECK_GL(glEnable(GL_POINT_SPRITE_ARB));
  //CHECK_GL(glEnable(GL_SAMPLE_COVERAGE));
  //CHECK_GL(glEnable(GL_SAMPLE_ALPHA_TO_COVERAGE));
  //CHECK_GL(glEnable(GL_SAMPLE_ALPHA_TO_ONE));
  CHECK_GL(glEnable(GL_MULTISAMPLE));
  vao = std::make_unique<VAO>();
  grid = makeGraphicGrid();
  axis = makeGraphicAxis();
  //setup_env_map("Default");
}

static void draw_small_axis() {
  glm::mat4x4 backup_view = view;
  glm::mat4x4 backup_proj = proj;
  view = gizmo_view;
  proj = gizmo_proj;
  CHECK_GL(glViewport(0, 0, nx * 0.1, ny * 0.1));
  CHECK_GL(glDisable(GL_DEPTH_TEST));
  axis->draw();
  CHECK_GL(glEnable(GL_DEPTH_TEST));
  CHECK_GL(glViewport(0, 0, nx, ny));
  view = backup_view;
  proj = backup_proj;
}




static void my_paint_graphics() {
  CHECK_GL(glViewport(0, 0, nx, ny));
  CHECK_GL(glClearColor(bgcolor.r, bgcolor.g, bgcolor.b, 0.0f));
  CHECK_GL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
  vao->bind();
  for (auto const &gra: current_graphics()) {
    gra->draw();
  }
  if (show_grid) {
      axis->draw();
      grid->draw();
      draw_small_axis();
  }
  vao->unbind();
}


static bool enable_hdr = true;
/* BEGIN ZHXX HAPPY */
namespace {

auto qvert = R"(
#version 330 core
const vec2 quad_vertices[4] = vec2[4]( vec2( -1.0, -1.0), vec2( 1.0, -1.0), vec2( -1.0, 1.0), vec2( 1.0, 1.0));
void main()
{
    gl_Position = vec4(quad_vertices[gl_VertexID], 0.0, 1.0);
}
)";
auto qfrag = R"(#version 330 core
// #extension GL_EXT_gpu_shader4 : enable
// hdr_adaptive.fs

const mat3x3 ACESInputMat = mat3x3
(
    0.59719, 0.35458, 0.04823,
    0.07600, 0.90834, 0.01566,
    0.02840, 0.13383, 0.83777
);

// ODT_SAT => XYZ => D60_2_D65 => sRGB
const mat3x3 ACESOutputMat = mat3x3
(
     1.60475, -0.53108, -0.07367,
    -0.10208,  1.10813, -0.00605,
    -0.00327, -0.07276,  1.07602
);

vec3 RRTAndODTFit(vec3 v)
{
    vec3 a = v * (v + 0.0245786f) - 0.000090537f;
    vec3 b = v * (0.983729f * v + 0.4329510f) + 0.238081f;
    return a / b;
}

vec3 ACESFitted(vec3 color, float gamma)
{
    color = color * ACESInputMat;

    // Apply RRT and ODT
    color = RRTAndODTFit(color);

    color = color * ACESOutputMat;

    // Clamp to [0, 1]
  	color = clamp(color, 0.0, 1.0);
    
    color = pow(color, vec3(1. / gamma));

    return color;
}


uniform sampler2DRect hdr_image;
out vec4 oColor;
void main(void)
{
	int i;
	float lum[25];
	vec2 tex_scale = vec2(1.0);
	for (i = 0; i < 25; i++)
	{
		vec2 tc = (2.0 * gl_FragCoord.xy + 3.5 * vec2(i % 5 - 2, i / 5 - 2));
		vec3 col = texture2DRect(hdr_image, tc).rgb;
    lum[i] = dot(col, vec3(0.3, 0.59, 0.11));
	} 
	// Calculate weighted color of region
	float kernelLuminance = (
		(1.0 * (lum[0] + lum[4] + lum[20] + lum[24])) +
		(4.0 * (lum[1] + lum[3] + lum[5] + lum[9] +
		lum[15] + lum[19] + lum[21] + lum[23])) +
		(7.0 * (lum[2] + lum[10] + lum[14] + lum[22])) +
		(16.0 * (lum[6] + lum[8] + lum[16] + lum[18])) +
		(26.0 * (lum[7] + lum[11] + lum[13] + lum[17])) +
		(41.0 * lum[12])
	) / 273.0;
	// Compute the corresponding exposure
	float exposure = sqrt(8.0 / (kernelLuminance + 0.25));
	// Apply the exposure to this texel
  //oColor.rgb = 1.0 - exp2(-texture2DRect(hdr_image, gl_FragCoord.xy).rgb * exposure);
	//oColor.a = 1.0f;
	oColor = vec4(texture2DRect(hdr_image, gl_FragCoord.xy).rgb, 1.0);
  
}
)";
hg::OpenGL::Program* tmProg=nullptr;
GLuint msfborgb=0, msfbod=0, tonemapfbo=0;
int oldnx, oldny;
GLuint texRect=0, regularFBO = 0;
GLuint emptyVAO=0;
void ScreenFillQuad(GLuint tex)
{
  if(emptyVAO==0)
    glGenVertexArrays(1, &emptyVAO); 
  CHECK_GL(glViewport(0, 0, nx, ny));
  CHECK_GL(glClearColor(bgcolor.r, bgcolor.g, bgcolor.b, 0.0f));
  CHECK_GL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
  tmProg->use();
  tmProg->set_uniformi("hdr_image",0);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_RECTANGLE, tex);

  glEnableVertexAttribArray(0);
  glBindVertexArray(emptyVAO); 
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  glDisableVertexAttribArray(0);
  glUseProgram(0);
}
static void paint_graphics(GLuint target_fbo = 0) {
  if(enable_hdr && tmProg==nullptr)
  {
    std::cout<<"compiling zhxx hdr program"<<std::endl;
    tmProg = compile_program(qvert, qfrag);
    if (!tmProg) {
    std::cout<<"failed to compile zhxx hdr program, giving up"<<std::endl;
        enable_hdr = false;
    }
  }
    if (!enable_hdr) {
        CHECK_GL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, target_fbo));
        return my_paint_graphics();
    }
  
  if(msfborgb==0||oldnx!=nx||oldny!=ny)
  {
    if(msfborgb!=0)
    {
      CHECK_GL(glDeleteRenderbuffers(1, &msfborgb));
    }
    CHECK_GL(glGenRenderbuffers(1, &msfborgb));
    CHECK_GL(glBindRenderbuffer(GL_RENDERBUFFER, msfborgb));
    /* begin cihou mesa */
    int max_num_samples = num_samples;
    CHECK_GL(glGetIntegerv(GL_MAX_INTEGER_SAMPLES, &max_num_samples));
    num_samples = std::min(num_samples, max_num_samples);
    printf("num samples: %d\n", num_samples);
    /* end cihou mesa */
    CHECK_GL(glRenderbufferStorageMultisample(GL_RENDERBUFFER, num_samples, GL_RGBA32F, nx, ny));
    
  
    if(msfbod!=0)
    {
      CHECK_GL(glDeleteRenderbuffers(1, &msfbod));
    }
    CHECK_GL(glGenRenderbuffers(1, &msfbod));
    CHECK_GL(glBindRenderbuffer(GL_RENDERBUFFER, msfbod));
    CHECK_GL(glRenderbufferStorageMultisample(GL_RENDERBUFFER, num_samples, GL_DEPTH_COMPONENT24, nx, ny));

    
    if(tonemapfbo!=0)
    {
      CHECK_GL(glDeleteFramebuffers(1, &tonemapfbo));
    }
    CHECK_GL(glGenFramebuffers(1, &tonemapfbo));


    if(regularFBO!=0)
    {
      CHECK_GL(glDeleteFramebuffers(1, &regularFBO));
    }
    CHECK_GL(glGenFramebuffers(1, &regularFBO));
    if(texRect!=0)
    {
      CHECK_GL(glDeleteTextures(1, &texRect));
    }
    CHECK_GL(glGenTextures(1, &texRect));
    CHECK_GL(glBindTexture(GL_TEXTURE_RECTANGLE, texRect));
    {
        CHECK_GL(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
        CHECK_GL(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
        CHECK_GL(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
        CHECK_GL(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));

        CHECK_GL(glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGBA32F, nx, ny, 0, GL_RGBA, GL_FLOAT, nullptr));
    }
    CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, regularFBO));
    CHECK_GL(glBindTexture(GL_TEXTURE_RECTANGLE, texRect));
    CHECK_GL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                    GL_TEXTURE_RECTANGLE, texRect, 0));
    CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, 0));

    oldnx = nx;
    oldny = ny;
    
  }

    
  
  CHECK_GL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, tonemapfbo));
  CHECK_GL(glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER,
                GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, msfborgb));
  CHECK_GL(glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER,
                GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, msfbod));
  CHECK_GL(glDrawBuffer(GL_COLOR_ATTACHMENT0));


  my_paint_graphics();
  CHECK_GL(glBindFramebuffer(GL_READ_FRAMEBUFFER, tonemapfbo));
  CHECK_GL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, regularFBO));
  glBlitFramebuffer(0, 0, nx, ny, 0, 0, nx, ny, GL_COLOR_BUFFER_BIT, GL_NEAREST);

  //CHECK_GL(glBindFramebuffer(GL_READ_FRAMEBUFFER, regularFBO));
  CHECK_GL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, target_fbo));
  ScreenFillQuad(texRect);
  //glBlitFramebuffer(0, 0, nx, ny, 0, 0, nx, ny, GL_COLOR_BUFFER_BIT, GL_NEAREST);
  //drawScreenQuad here:
  CHECK_GL(glFlush());
}
}
/* END ZHXX HAPPY */

static double get_time() {
  static auto start = std::chrono::system_clock::now();
  auto now = std::chrono::system_clock::now();
  std::chrono::duration<double> diff = now - start;
  return diff.count();
}

static hg::FPSCounter solverFPS(get_time, 1);
static hg::FPSCounter renderFPS(get_time, 10);

void finalize() {
  vao = nullptr;
}

void new_frame() {
  my_paint_graphics();
  renderFPS.tick();
}

void set_window_size(int nx_, int ny_) {
  nx = nx_;
  ny = ny_;
}

void set_curr_frameid(int frameid) {
  curr_frameid = std::max(frameid, 0);
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

void set_show_grid(bool flag) {
    show_grid = flag;
}

std::vector<char> record_frame_offline() {
    std::vector<char> pixels(nx * ny * 3);

    GLuint fbo, rbo1, rbo2;
    CHECK_GL(glGenRenderbuffers(1, &rbo1));
    CHECK_GL(glGenRenderbuffers(1, &rbo2));
    CHECK_GL(glBindRenderbuffer(GL_RENDERBUFFER, rbo1));
    CHECK_GL(glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA, nx, ny));
    CHECK_GL(glBindRenderbuffer(GL_RENDERBUFFER, rbo2));
    CHECK_GL(glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, nx, ny));

    CHECK_GL(glGenFramebuffers(1, &fbo));
    CHECK_GL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo));
    CHECK_GL(glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER,
                GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rbo1));
    CHECK_GL(glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER,
                GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo2));
    CHECK_GL(glDrawBuffer(GL_COLOR_ATTACHMENT0));

    paint_graphics(fbo);

    if (glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE) {
        CHECK_GL(glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo));
        CHECK_GL(glBlitFramebuffer(0, 0, nx, ny, 0, 0, nx, ny, GL_COLOR_BUFFER_BIT, GL_NEAREST));
        CHECK_GL(glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo));
        CHECK_GL(glBindBuffer(GL_PIXEL_PACK_BUFFER, 0));
        CHECK_GL(glReadBuffer(GL_COLOR_ATTACHMENT0));

        CHECK_GL(glReadPixels(0, 0, nx, ny, GL_RGB, GL_UNSIGNED_BYTE, &pixels[0]));
    }

    CHECK_GL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
    CHECK_GL(glBindRenderbuffer(GL_RENDERBUFFER, 0));

    CHECK_GL(glDeleteRenderbuffers(1, &rbo1));
    CHECK_GL(glDeleteRenderbuffers(1, &rbo2));
    CHECK_GL(glDeleteFramebuffers(1, &fbo));

    return pixels;
}

void new_frame_offline(std::string path) {
    char buf[1024];
    sprintf(buf, "%s/%06d.png", path.c_str(), curr_frameid);
    printf("saving screen %dx%d to %s\n", nx, ny, buf);

    std::vector<char> pixels = record_frame_offline();
    stbi_flip_vertically_on_write(true);
    stbi_write_png(buf, nx, ny, 3, &pixels[0], 0);
}

void do_screenshot(std::string path) {
    std::vector<char> pixels = record_frame_offline();
    stbi_flip_vertically_on_write(true);
    stbi_write_png(path.c_str(), nx, ny, 3, &pixels[0], 0);
}

void set_background_color(float r, float g, float b) {
    bgcolor = glm::vec3(r, g, b);
}

std::tuple<float, float, float> get_background_color() {
    return {bgcolor.r, bgcolor.g, bgcolor.b};
}

void set_smooth_shading(bool smooth) {
    smooth_shading = smooth;
}
void set_normal_check(bool check) {
    normal_check = check;
}

void set_render_wireframe(bool render_wireframe_) {
    render_wireframe = render_wireframe_;
}

void set_num_samples(int num_samples_) {
    num_samples = num_samples_;
}
}
