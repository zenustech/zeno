#include <zenovis/opengl/common.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace zenovis::opengl {

namespace {
constexpr uint32_t kContextApiWgl = 1;
constexpr uint32_t kContextApiGlx = 2;
constexpr uint32_t kContextApiEgl = 3;
constexpr uint32_t kContextApiCgl = 4;
constexpr uint32_t kContextApiUnknown = 5;
}

OpenGLContextHandle current_context_handle() noexcept {
#if defined(_WIN32)
    using GetCurrentContext = HGLRC(WINAPI *)();
    static const auto getCurrentContext = [] {
        const auto module = GetModuleHandleW(L"opengl32.dll");
        return module ? reinterpret_cast<GetCurrentContext>(
                            GetProcAddress(module, "wglGetCurrentContext"))
                      : nullptr;
    }();
    if (getCurrentContext)
        return {reinterpret_cast<void *>(getCurrentContext()), kContextApiWgl};
#else
    using GetCurrentContext = void *(*)();
#if defined(__APPLE__)
    static const auto getCurrentContext = reinterpret_cast<GetCurrentContext>(
        dlsym(RTLD_DEFAULT, "CGLGetCurrentContext"));
    if (getCurrentContext)
        return {getCurrentContext(), kContextApiCgl};
#else
    static const auto getGlxCurrentContext = reinterpret_cast<GetCurrentContext>(
        dlsym(RTLD_DEFAULT, "glXGetCurrentContext"));
    if (getGlxCurrentContext) {
        if (auto *context = getGlxCurrentContext())
            return {context, kContextApiGlx};
    }

    static const auto getEglCurrentContext = reinterpret_cast<GetCurrentContext>(
        dlsym(RTLD_DEFAULT, "eglGetCurrentContext"));
    if (getEglCurrentContext) {
        if (auto *context = getEglCurrentContext())
            return {context, kContextApiEgl};
    }
#endif
#endif

    if (glad_glGetString && glad_glGetString(GL_VERSION))
        return {reinterpret_cast<void *>(uintptr_t{1}), kContextApiUnknown};
    return {};
}

} // namespace zenovis::opengl
