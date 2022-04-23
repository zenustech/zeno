#pragma once

#include <memory>
#include <vector>
#include <zeno/utils/disable_copy.h>

namespace zenovis {

struct Camera;
struct ShaderManager;
struct EnvmapManager;
struct GraphicsManager;
struct LightCluster;
struct Light;
struct IGraphic;
struct DepthPass;
struct ReflectivePass;

namespace opengl {
struct VAO;
}

struct Scene : zeno::disable_copy {
    std::unique_ptr<Camera> camera;
    std::unique_ptr<LightCluster> lightCluster;
    std::vector<std::unique_ptr<IGraphic>> hudGraphics;

    std::unique_ptr<ShaderManager> shaderMan;
    std::unique_ptr<EnvmapManager> envmapMan;
    std::unique_ptr<GraphicsManager> graphicsMan;
    std::unique_ptr<DepthPass> mDepthPass;
    std::unique_ptr<ReflectivePass> mReflectivePass;
    std::unique_ptr<opengl::VAO> vao;

    Scene();
    ~Scene();

    std::vector<IGraphic *> graphics() const;

    void fast_paint_graphics();
    void drawSceneDepthSafe(bool reflect, bool isDepthPass);
    void my_paint_graphics(int samples, bool isDepthPass);
    void draw_small_axis();
    std::vector<char> record_frame_offline();
    bool anyGraphicHasMaterial();

    void draw(unsigned int target_fbo = 0);
};

} // namespace zenovis
