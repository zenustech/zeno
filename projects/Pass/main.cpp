#include <zeno/zeno.h>
#include <zeno/utils/memory.h>
#include <zeno/utils/disable_copy.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/LightObject.h>
#include <zeno/types/CameraObject.h>
#include "common.h"
#include <functional>

namespace zeno {
namespace {

struct PassImageGL : IObject {
    struct Impl {
        GLuint tex_color{};
        GLuint tex_depth{};
        GLuint fbo{};
        GLuint rbo_color{};
        GLuint rbo_depth{};
        GLuint nx{}, ny{};

        explicit Impl(int nx_, int ny_) : nx(nx_), ny(ny_) {
            CHECK_GL(glGenTextures(1, &tex_color));
            CHECK_GL(glGenFramebuffers(1, &fbo));
            CHECK_GL(glGenRenderbuffers(1, &rbo_color));
            CHECK_GL(glGenRenderbuffers(1, &rbo_depth));

            CHECK_GL(glBindTexture(GL_TEXTURE_2D, tex_color));
            CHECK_GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, nx, ny, 0, GL_RGBA, GL_FLOAT, nullptr));
            CHECK_GL(glBindTexture(GL_TEXTURE_2D, tex_depth));
            CHECK_GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, nx, ny, 0, GL_RED, GL_FLOAT, nullptr));
            CHECK_GL(glBindTexture(GL_TEXTURE_2D, 0));

            CHECK_GL(glBindRenderbuffer(GL_RENDERBUFFER, rbo_color));
            CHECK_GL(glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA16F, nx, ny));
            CHECK_GL(glBindRenderbuffer(GL_RENDERBUFFER, rbo_depth));
            CHECK_GL(glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F, nx, ny));
            CHECK_GL(glBindRenderbuffer(GL_RENDERBUFFER, 0));

            CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, fbo));
            CHECK_GL(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rbo_color));
            CHECK_GL(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo_depth));
            CHECK_GL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_color, 0));
            CHECK_GL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, tex_depth, 0));
            CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, 0));
        }
        
        void draw_body(std::function<void()> callback) {
            CHECK_GL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo));
            CHECK_GL(glDrawBuffer(GL_COLOR_ATTACHMENT0));
            CHECK_GL(glViewport(0, 0, nx, ny));
            callback();
            CHECK_GL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
        }

        Impl(Impl const &) {
            throw "not implemented";
        }

        ~Impl() {
            CHECK_GL(glDeleteTextures(1, &tex_color));
            CHECK_GL(glDeleteTextures(1, &tex_depth));
            CHECK_GL(glDeleteFramebuffers(1, &fbo));
            CHECK_GL(glDeleteRenderbuffers(1, &rbo_color));
            CHECK_GL(glDeleteRenderbuffers(1, &rbo_depth));
        }
    };

    copiable_unique_ptr<Impl> impl;
    
    template <class ...Ts>
    void emplace(Ts &&...ts) {
        impl = std::make_unique<Impl>(std::forward<Ts>(ts)...);
    }
};

struct ForwardPass : INode {
    virtual void apply() override {
        auto image = get_input<PassImageGL>("image");
        auto objects = has_input("objects")
            ? get_input<ListObject>("objects")
            : std::make_shared<ListObject>();
        auto lights = has_input("lights")
            ? get_input<ListObject>("lights")
            : std::make_shared<ListObject>();
        auto materials = has_input("materials")
            ? get_input<ListObject>("materials")
            : std::make_shared<ListObject>();
        auto camera = get_input<CameraObject>("camera");
        auto bgcolor = get_input2<vec3f>("bgcolor");
        auto bgalpha = get_input2<float>("bgalpha");
        auto nsamples = get_input2<int>("nsamples");

        image->impl->draw_body([&] {
            CHECK_GL(glClearColor(bgcolor[0], bgcolor[1], bgcolor[2], bgalpha));
            CHECK_GL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
            // zhxx: draw(objects) here;
        });

        set_output("image", std::move(image));
    }
};

ZENO_DEFNODE(ForwardPass)({
    {
        {"image", "image"},
        {"list", "objects"},
        {"list", "lights"},
        {"list", "materials"},
        {"camera", "camera"},
        {"vec3f", "bgcolor", "0,0,0"},
        {"float", "bgalpha", "0"},
        {"int", "nsamples", "1"},
    },
    {
        {"image", "image"},
    },
    {},
    {"pass"},
});

}
}
