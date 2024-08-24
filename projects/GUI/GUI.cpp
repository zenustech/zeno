#include <SDL2/SDL.h>
#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/DummyObject.h>
#include <zeno/types/DictObject.h>
#include <zeno/extra/MethodCaller.h>
#include <zeno/extra/TempNode.h>
#include <zeno/utils/log.h>
#include <zeno/utils/zeno_p.h>
#include <glad/glad.h>
#include "glutils.h"
#include <set>

namespace zeno {
namespace {

struct ZGL_PrimAsBuff : INode {
    void apply() override {
        auto mode = get_input2<std::string>("mode");
        if (mode == "POINTS") {
            auto buff = get_input<PrimitiveObject>("prim");
            set_output2("count", (int)buff->verts.size());
            set_output("buff", std::move(buff));
        } else if (mode == "TRIANGLES") {
            auto buff = temp_node("PrimSepTriangles")
                .set("prim", get_input<PrimitiveObject>("prim"))
                .set2("smoothNormal", get_input2<bool>("smoothNormal"))
                .set2("keepTriFaces", false)
                .get<PrimitiveObject>("prim");
                ;
            set_output2("count", (int)buff->verts.size());
            set_output("buff", std::move(buff));
        } else if (mode == "LINES") {
            throw makeError<UnimplError>();
        } else {
            throw makeError<KeyError>(mode, "mode");
        }
    }
};
ZENO_DEFNODE(ZGL_PrimAsBuff)({
    {
        {gParamType_Primitive, "prim"},
        {"enum ALL TRIANGLES LINES POINTS", "mode", "POINTS"},
        {gParamType_Bool, "smoothNormal"},
    },
    {
        {gParamType_Primitive, "buff"},
        {gParamType_Int, "count"},
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
        CHECK_GL(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void *)0));
        CHECK_GL(glBindBuffer(GL_ARRAY_BUFFER, 0));

        set_output("vbo", std::move(vbo));
    }
};
ZENO_DEFNODE(ZGL_VboFromBuff)({
    {
        {gParamType_Primitive, "buff"},
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
        auto mode = get_input2<std::string>("mode");
        if (mode == "PointSize") {
            CHECK_GL(glPointSize(size));
        } else if (mode == "LineWidth") {
            CHECK_GL(glLineWidth(size));
        } else {
            throw makeError<KeyError>(mode, "mode");
        }
    }
};
ZENO_DEFNODE(ZGL_SetPointSize)({
    {
        {gParamType_Float, "size", "1.0"},
        {"enum PointSize LineWidth", "mode", "PointSize"},
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
        {gParamType_Int, "count", "1"},
        {"enum TRIANGLES LINES POINTS", "mode", "POINTS"},
    },
    {},
    {},
    {"GUI"},
});

struct ZGL_ClearColor : INode {
    void apply() override {
        auto color = get_input2<zeno::vec3f>("color");
        auto alpha = get_input2<float>("alpha");
        CHECK_GL(glClearColor(color[0], color[1], color[2], alpha));
        CHECK_GL(glClear(GL_COLOR_BUFFER_BIT));
    }
};
ZENO_DEFNODE(ZGL_ClearColor)({
    {
        {gParamType_Vec3f, "color", "0.2,0.3,0.4"},
        {gParamType_Float, "alpha", "1.0"},
    },
    {},
    {},
    {"GUI"},
});

struct ZGL_StateData : IObjectClone<ZGL_StateData> {
    std::set<std::string> pressed_keys;
};

struct ZGL_StateGetKeys : INode {
    void apply() override {
        auto state = get_input<ZGL_StateData>("state");
        auto type = get_input2<std::string>("type");
        if (type == "STRING") {
            std::string keys;
            for (auto const &key: state->pressed_keys) {
                keys += key;
                keys += ' ';
            }
            if (!keys.empty()) keys.pop_back();
            set_output2("keys", std::move(keys));
        } else if (type == "DICT") {
            auto keys = std::make_shared<DictObject>();
            for (auto const &key: state->pressed_keys) {
                keys->lut.emplace(key, std::make_shared<DummyObject>());
            }
            set_output("keys", std::move(keys));
        } else {
            auto keys = std::make_shared<ListObject>();
            for (auto const &key: state->pressed_keys) {
                keys->push_back(std::make_shared<StringObject>(key));
            }
            set_output("keys", std::move(keys));
        }
    }
};

ZENO_DEFNODE(ZGL_StateGetKeys)({
    {
        {"state"},
        {"enum DICT LIST STRING", "type", "DICT"},
    },
    {
        {"keys"},
    },
    {},
    {"GUI"},
});

struct ZGL_Main : INode {
    void apply() override {
        auto title = get_input2<std::string>("title");
        auto resx = get_input2<int>("resx");
        auto resy = get_input2<int>("resy");
        auto quitOnEsc = get_input2<int>("quitOnEsc");
        auto callbacks = get_input<DictObject>("callbacks");
        auto state = std::make_shared<ZGL_StateData>();

        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
        SDL_Init(SDL_INIT_EVERYTHING);

        SDL_Window *window = SDL_CreateWindow(title.c_str(), SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                              resx, resy, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);
        SDL_GL_CreateContext(window);
        SDL_GL_SetSwapInterval(1);
        if (gladLoadGLLoader((GLADloadproc)SDL_GL_GetProcAddress) < 0) {
            throw makeError("failed to initialize glad from SDL OpenGL context");
        }
        MethodCaller(callbacks, "on_init", {}).call();

        bool quit = false;
        while (!quit) {
            SDL_Event event;
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_KEYDOWN) {
                    std::string key = SDL_GetKeyName(event.key.keysym.sym);
                    log_debug("key pressed [{}]", key);
                    MethodCaller(callbacks, "on_keydown", {}).set2("key", key).call();
                    state->pressed_keys.insert(key);
                    if (quitOnEsc && key == "Escape")
                        quit = true;
                } else if (event.type == SDL_KEYUP) {
                    std::string key = SDL_GetKeyName(event.key.keysym.sym);
                    log_debug("key released [{}]", key);
                    MethodCaller(callbacks, "on_keyup", {}).set2("key", key).call();
                    state->pressed_keys.erase(key);
                } else if (event.type == SDL_QUIT) {
                    quit = true;
                }
            }
            MethodCaller(callbacks, "on_update", {}).set("state", state).call();
            MethodCaller(callbacks, "on_draw", {}).call();
            SDL_GL_SwapWindow(window);
            float dt = MethodCaller(callbacks, "calc_dt", {}).get2<float>("ret", 1.f / 60.f);
            if (dt > 0)
                SDL_Delay((unsigned int)(dt * 1000));
        }

        MethodCaller(callbacks, "on_exit", {}).call();
        SDL_DestroyWindow(window);
    }
};
ZENO_DEFNODE(ZGL_Main)({
    {
        {gParamType_String, "title", "ZenoPlay"},
        {gParamType_Int, "resx", "400"},
        {gParamType_Int, "resy", "300"},
        {gParamType_Bool, "quitOnEsc", "1"},
        {gParamType_Dict,"callbacks"},
    },
    {},
    {},
    {"GUI"},
});

}
}
