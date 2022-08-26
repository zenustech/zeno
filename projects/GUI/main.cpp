#include <SDL2/SDL.h>
#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/extra/MethodCaller.h>
#include <zeno/utils/log.h>
#include <zeno/utils/zeno_p.h>
#include <glad/glad.h>
#include "glutils.h"

namespace zeno {
namespace {

struct ZGL_PrimAsTriBuff : INode {
    void apply() override {
        throw "TODO";
    }
};
ZENO_DEFNODE(ZGL_PrimAsTriBuff)({
    {
        {"PrimitiveObject", "prim"},
    },
    {
        {"PrimitiveObject", "buff"},
    },
    {},
    {"GUI"},
});

struct ZGL_VboObject : IObjectClone<ZGL_VboObject> {
    struct Impl {
        GLuint i;

        Impl() = default;
        Impl(Impl &&) = delete;

        ~Impl() {
            if (i)
                CHECK_GL(glDeleteBuffers(1, &i));
        }
    };

    std::shared_ptr<Impl> impl;

    void create() {
        impl = std::make_shared<Impl>();
        CHECK_GL(glCreateBuffers(1, &impl->i));
    }

    int handle() const {
        return impl ? impl->i : 0;
    }
};

struct ZGL_VboFromBuff : INode {
    void apply() override {
        auto vbo = std::make_shared<ZGL_VboObject>();
        vbo->create();

        auto buff = get_input<PrimitiveObject>("buff");
        auto &arr = buff->verts.values;
        CHECK_GL(glBindBuffer(GL_ARRAY_BUFFER, vbo->handle()));
        CHECK_GL(glBufferData(GL_ARRAY_BUFFER, arr.size() * sizeof(arr[0]), arr.data(), GL_STATIC_DRAW));
        CHECK_GL(glEnableVertexAttribArray(0));
        CHECK_GL(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, reinterpret_cast<void *>(static_cast<uintptr_t>(0))));
        CHECK_GL(glBindBuffer(GL_ARRAY_BUFFER, 0));

        set_output("vbo", std::move(vbo));
    }
};
ZENO_DEFNODE(ZGL_VboFromBuff)({
    {
        {"PrimitiveObject", "buff"},
    },
    {
        {"ZGL_VboObject", "vbo"},
    },
    {},
    {"GUI"},
});

struct ZGL_SetPointSize : INode {
    void apply() override {
        auto size = get_input2<float>("size");
        CHECK_GL(glPointSize(size));
    }
};
ZENO_DEFNODE(ZGL_SetPointSize)({
    {
        {"float", "size", "1.0"},
    },
    {},
    {},
    {"GUI"},
});

struct ZGL_DrawVboArrays : INode {
    void apply() override {
        auto vbo = get_input<ZGL_VboObject>("vbo");
        auto count = get_input2<int>("count");
        auto mode = safe_at(std::map<std::string, int>{
            {"TRIANGLES", GL_TRIANGLES},
            {"LINES", GL_LINES},
            {"POINTS", GL_POINTS},
        }, get_input2<std::string>("mode"), "draw mode");
        CHECK_GL(glBindBuffer(GL_ARRAY_BUFFER, vbo->handle()));
        CHECK_GL(glDrawArrays(mode, 0, count));
        CHECK_GL(glBindBuffer(GL_ARRAY_BUFFER, 0));
    }
};
ZENO_DEFNODE(ZGL_DrawVboArrays)({
    {
        {"ZGL_VboObject", "vbo"},
        {"int", "count", "1"},
        {"enum TRIANGLES LINES POINTS", "mode", "POINTS"},
    },
    {},
    {},
    {"GUI"},
});

struct ZGL_ClearColor : INode {
    void apply() override {
        auto color = get_input2<vec3f>("color");
        auto alpha = get_input2<float>("alpha");
        CHECK_GL(glClearColor(color[0], color[1], color[2], alpha));
        CHECK_GL(glClear(GL_COLOR_BUFFER_BIT));
    }
};
ZENO_DEFNODE(ZGL_ClearColor)({
    {
        {"vec3f", "color", "0.2,0.3,0.4"},
        {"float", "alpha", "1.0"},
    },
    {},
    {},
    {"GUI"},
});

struct ZGL_Main : INode {
    void apply() override {
        auto title = get_input2<std::string>("title");
        auto resx = get_input2<int>("resx");
        auto resy = get_input2<int>("resy");
        auto callbacks = get_input<DictObject>("callbacks");

        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
        SDL_Init(SDL_INIT_EVERYTHING);

        SDL_Window *window = SDL_CreateWindow(title.c_str(), SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                              resx, resy, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);
        SDL_GL_CreateContext(window);
        SDL_GL_SetSwapInterval(1);
        gladLoadGL();
        MethodCaller(callbacks, "on_init", {}).call();

        bool quit = false;
        while (!quit) {
            SDL_Event event;
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_KEYDOWN) {
                    std::string key = SDL_GetKeyName(event.key.keysym.sym);
                    log_info("key pressed {}", key);
                    MethodCaller(callbacks, "on_keydown", {}).set2("key", key).call();
                } else if (event.type == SDL_QUIT) {
                    quit = true;
                }
            }
            MethodCaller(callbacks, "on_draw", {}).call();
            SDL_GL_SwapWindow(window);
            float dt = MethodCaller(callbacks, "calc_dt", {}).get2<float>("ret", 1.f / 60.f);
            if (dt > 0)
                SDL_Delay(static_cast<unsigned int>(dt * 1000));
        }

        MethodCaller(callbacks, "on_exit", {}).call();
        SDL_DestroyWindow(window);
    }
};
ZENO_DEFNODE(ZGL_Main)({
    {
        {"string", "title", "ZenoPlay"},
        {"int", "resx", "400"},
        {"int", "resy", "300"},
        {"DictObject", "callbacks"},
    },
    {},
    {},
    {"GUI"},
});

}
}
