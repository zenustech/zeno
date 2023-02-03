#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/utils/log.h>

#define TINYEXR_IMPLEMENTATION
#include <tinyexr.h>

namespace zeno {
struct WriteVAT : INode {
    std::vector<std::shared_ptr<PrimitiveObject>> prims;
    virtual void apply() override {
        int frameid;
        if (has_input("frameid")) {
            frameid = get_param<int>("frameid");
        } else {
            frameid = getGlobalState()->frameid;
        }
        int frameStart = get_param<int>("frameStart");
        int frameEnd = get_param<int>("frameEnd");
        int frameCount = frameEnd - frameStart + 1;
        if (frameid == frameStart) {
            prims.resize(frameCount);
        }
        auto prim = std::dynamic_pointer_cast<PrimitiveObject>(get_input<PrimitiveObject>("prim")->clone());
        prims[frameid - frameStart] = prim;
        if (frameid == frameEnd) {
            std::string path = get_param<std::string>("path");

            int maxTris = 0;
            for (const auto &prim: prims) {
                maxTris = std::max(maxTris, (int)prim->tris.size());
            }
            int width = maxTris * 3;
            int height = frameCount;
            std::vector<zeno::vec3f> image;
            image.resize(width * height);
            for (auto i = 0; i < prims.size(); i++) {
                auto prim = prims[i];
                for (auto j = 0; j < prim->tris.size(); j++) {
                    const auto & tri = prim->tris[j];
                    size_t index = width * i + j * 3;
                    image[index + 0] = prim->verts[tri[0]];
                    image[index + 1] = prim->verts[tri[1]];
                    image[index + 2] = prim->verts[tri[2]];
                }
            }
            const char* err;
            int32_t ret = SaveEXR(
                    reinterpret_cast<float*>( image.data() ),
                    width,
                    height,
                    3, // num components
                    static_cast<int32_t>( true ), // save_as_fp16
                    path.c_str(),
                    &err );

            if( ret != TINYEXR_SUCCESS ) {
                zeno::log_error("VAT: tinyexr saveImage error: {}", err);
            } else {
                zeno::log_info("VAT: tinyexr saveImage success!");
            }
        }
    }
};

ZENDEFNODE(WriteVAT, {
    {
        {"prim"},
        {"frameid"},
    },
    {},
    {
        {"writepath", "path", ""},
        {"int", "frameStart", "0"},
        {"int", "frameEnd", "100"},
    },
    {"primitive"},
});
} // namespace zeno
