#include "MyShader.hpp"
#include "stdafx.hpp"
#include "main.hpp"
#include "IGraphic.hpp"
#include <Hg/FPSCounter.hpp>
#include <sstream>
#include <cstdlib>
#include <cmath>
#include <array>
#include <stb_image_write.h>
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

std::unique_ptr<IGraphic> makeGraphicGrid();
std::unique_ptr<IGraphic> makeGraphicAxis();

void initialize() {
  gladLoadGL();
  setup_env_map("forest");
  auto version = (const char *)glGetString(GL_VERSION);
  printf("OpenGL version: %s\n", version ? version : "null");

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
  for (auto const &[key, gra]: current_frame_data()->graphics) {
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


uniform samplerRect hdr_image;
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


auto SHFrag = R"(
  #version 140
  //Constants cn from equation 12 in [1]
const float c1 = 0.429043;
const float c2 = 0.511664;
const float c3 = 0.743125;
const float c4 = 0.886227;
const float c5 = 0.247708;

//First 9 spherical harmonics coefficients from equation 3 in [1]
const float Y00 = 0.282095;
const float Y1n = 0.488603; // 3 direction dependent values
const float Y2n = 1.092548; // 3 direction dependent values
const float Y20 = 0.315392;
const float Y22 = 0.546274;
#define PI 3.14159
#define TWO_PI (2.0 * PI)
#define HALF_PI (0.5 * PI)

#define GAMMA 2.2
#define INV_GAMMA (1.0/GAMMA)
uniform sampler2D skybox;

vec3 gamma(vec3 col){
	return pow(col, vec3(INV_GAMMA));
}

vec3 inv_gamma(vec3 col){
	return pow(col, vec3(GAMMA));
}

float saturate(float x){
    return max(0.0, min(x, 1.0));
}

vec3 getRadiance(vec3 r){
    float u = atan(r.x, r.z) * ( 1 / PI) * 0.5 + 0.5;
    float v = 1.0 - (asin(r.y) * (2 / PI) * 0.5 + 0.5);
    vec3 col = texture(skybox, vec2(u,v)).rgb;
    //// Add some bloom to the environment
    // col += 0.5 * pow(col, vec3(2));
    return col;
}
out vec4 fragColor;
void main()
{
  vec4 col = vec4(0);
        
  //  Store an equirectangular projection of the environment map. Subsequent code will
  //  overwrite specific pixels to store the SH matrices and state flags but this 
  //  should not be visible in the final render.
  vec2 fragCoord = gl_FragCoord;
  vec2 iResolution = vec2(1024, 1024);
  vec2 texCoord = fragCoord.xy / iResolution.xy;
  vec2 thetaphi = ((texCoord * 2.0) - vec2(1.0)) * vec2(PI, HALF_PI); 
  vec3 rayDir = vec3( cos(thetaphi.y) * cos(thetaphi.x), 
                      -sin(thetaphi.y), 
                      cos(thetaphi.y) * sin(thetaphi.x));

  col = vec4(getRadiance(rayDir), 1.0);
  //Ensure radiance is not 0
  col.x = max(col.x, 1e-5);
  col.y = max(col.y, 1e-5);
  col.z = max(col.z, 1e-5);

  
  //------------------------------------------------------------------------------------
  //----------------------------------- BUG? -------------------------------------------
  //------- It should be fragCoord.x < 3.0 because we write and read 3 matrices --------
  //------- But that will give the wrong result and I can't figure out why -------------
  //------------------------------------------------------------------------------------
  //------------------------------------------------------------------------------------

  if(fragCoord.x < 4.0 && fragCoord.y < 4.0){

      //Coefficient values to accumulate
      vec3 L00 = vec3(0);
      vec3 L1_1 = vec3(0);
      vec3 L10 = vec3(0);
      vec3 L11 = vec3(0);

      vec3 L2_2 = vec3(0);
      vec3 L2_1 = vec3(0);
      vec3 L20 = vec3(0);
      vec3 L21 = vec3(0);
      vec3 L22 = vec3(0);

      //To make the sampling rate scalable and independent of the cubemap dimensions, 
      //we can sample a set number of equidistant directions on a sphere. While this is 
      //not doable for all number of directions, a good approximation is the Fibonacci 
      //spiral on a sphere.

      //From [4]
      //Golden angle in radians
      float phi = PI * (3.0 - sqrt(5.0));
      
      //The loop should not run every frame but Windows FPS drops anyway. 
      //This seems to have fixed it
      float sampleCount = 2048;
      

      for(float i = float(ZERO); i < sampleCount; i++){

          float y = 1.0 - (i / sampleCount) * 2.0;
          //Radius at y
          float radius = sqrt(1.0 - y * y);  

          //Golden angle increment
          float theta = phi * i;

          float x = cos(theta) * radius;
          float z = sin(theta) * radius;

          //Sample directiion
          vec3 dir = normalize(vec3(x, y, z));

          //Envronment map value in the direction (interpolated)
          vec3 radiance = getRadiance(dir);

          //Accumulate value weighted by spherical harmonic coefficient in the direction
          L00 += radiance * Y00;
          L1_1 += radiance * Y1n * dir.y;
          L10 += radiance * Y1n * dir.z;
          L11 += radiance * Y1n * dir.x;
          L2_2 += radiance * Y2n * dir.x * dir.y;
          L2_1 += radiance * Y2n * dir.y * dir.z;
          L20 += radiance * Y20 * (3.0 * pow(dir.z, 2.0) - 1.0);
          L21 += radiance * Y2n * dir.x * dir.z;
          L22 += radiance * Y22 * (pow(dir.x, 2.0) - pow(dir.y, 2.0));
      }

      //Scale the sum of coefficents on a sphere
      float factor = 4.0*PI / sampleCount;

      L00 *= factor;
      L1_1 *= factor;
      L10 *= factor;
      L11 *= factor;
      L2_2 *= factor;
      L2_1 *= factor;
      L20 *= factor;
      L21 *= factor;
      L22 *= factor;

      //Write three 4x4 matrices to bufferB
      //GLSL matrices are column major
      int idxM = int(fragCoord.y-0.5);

      if(fragCoord.x == 0.5){
          mat4 redMatrix;
          redMatrix[0] = vec4(c1*L22.r, c1*L2_2.r, c1*L21.r, c2*L11.r);
          redMatrix[1] = vec4(c1*L2_2.r, -c1*L22.r, c1*L2_1.r, c2*L1_1.r);
          redMatrix[2] = vec4(c1*L21.r, c1*L2_1.r, c3*L20.r, c2*L10.r);
          redMatrix[3] = vec4(c2*L11.r, c2*L1_1.r, c2*L10.r, c4*L00.r-c5*L20.r);
          col = redMatrix[idxM];
      }

      if(fragCoord.x == 1.5){
          mat4 grnMatrix;
          grnMatrix[0] = vec4(c1*L22.g, c1*L2_2.g, c1*L21.g, c2*L11.g);
          grnMatrix[1] = vec4(c1*L2_2.g, -c1*L22.g, c1*L2_1.g, c2*L1_1.g);
          grnMatrix[2] = vec4(c1*L21.g, c1*L2_1.g, c3*L20.g, c2*L10.g);
          grnMatrix[3] = vec4(c2*L11.g, c2*L1_1.g, c2*L10.g, c4*L00.g-c5*L20.g);
          col = grnMatrix[idxM];
      }

      if(fragCoord.x == 2.5){
          mat4 bluMatrix;
          bluMatrix[0] = vec4(c1*L22.b, c1*L2_2.b, c1*L21.b, c2*L11.b);
          bluMatrix[1] = vec4(c1*L2_2.b, -c1*L22.b, c1*L2_1.b, c2*L1_1.b);
          bluMatrix[2] = vec4(c1*L21.b, c1*L2_1.b, c3*L20.b, c2*L10.b);
          bluMatrix[3] = vec4(c2*L11.b, c2*L1_1.b, c2*L10.b, c4*L00.b-c5*L20.b);
          col = bluMatrix[idxM];
      }
  }
  
  fragColor = col;
  
}

)";



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
  paint_graphics();
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
