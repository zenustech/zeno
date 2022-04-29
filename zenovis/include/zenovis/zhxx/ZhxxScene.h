#pragma once

#include <memory>
#include <vector>
#include <zeno/utils/disable_copy.h>
#include <zeno/utils/PolymorphicVector.h>

namespace zenovis::opengl {
struct VAO;
}

namespace zenovis {
struct Scene;
struct GraphicsManager;
struct ShaderManager;
}

namespace zenovis::zhxx {

struct ZhxxDrawOptions;
struct ZhxxCamera;
struct EnvmapManager;
struct LightCluster;
struct ZhxxLight;
struct ZhxxIGraphic;
struct ZhxxIGraphicDraw;
struct ZhxxGraphicsManager;
struct DepthPass;
struct ReflectivePass;

struct ZhxxScene : zeno::disable_copy {
    std::unique_ptr<ZhxxCamera> camera;
    std::unique_ptr<ZhxxDrawOptions> zxxDrawOptions;
    std::unique_ptr<LightCluster> lightCluster;
    std::unique_ptr<ZhxxGraphicsManager> zxxGraphicsMan;
    std::unique_ptr<EnvmapManager> envmapMan;
    std::vector<std::unique_ptr<ZhxxIGraphicDraw>> hudGraphics;
    std::unique_ptr<DepthPass> mDepthPass;
    std::unique_ptr<ReflectivePass> mReflectivePass;
    std::unique_ptr<opengl::VAO> vao;
    Scene *visScene;

    explicit ZhxxScene(Scene *visScene);
    ~ZhxxScene();

    void fast_paint_graphics();
    void drawSceneDepthSafe(bool reflect, bool isDepthPass);
    void my_paint_graphics(int samples, bool isDepthPass);
    void draw_small_axis();
    std::vector<char> record_frame_offline(int hdrSize = 1, int rgbComps = 3);
    std::vector<ZhxxIGraphicDraw *> getGraphics();
    bool anyGraphicHasMaterial();

    void draw(unsigned int target_fbo = 0);
};

} // namespace zenovis
