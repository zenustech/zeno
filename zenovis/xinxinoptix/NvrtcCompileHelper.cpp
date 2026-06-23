#include "NvrtcCompileProtocol.h"

#include <nvrtc.h>
#include <sutil/sutil.h>

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN 1
#endif
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#endif

namespace sutil {

std::vector<const char*>& getIncFileTab()
{
    static std::vector<const char*> ret;
    return ret;
}

std::vector<const char*>& getIncPathTab()
{
    static std::vector<const char*> ret;
    return ret;
}

const char* lookupIncFile(const char* name)
{
    auto const& pathtab = getIncPathTab();
    auto it = std::find(pathtab.begin(), pathtab.end(), std::string_view(name));
    return getIncFileTab().at(it - pathtab.begin());
}

} // namespace sutil

namespace {

void checkNvrtc(nvrtcResult result, const char* expr)
{
    if (result != NVRTC_SUCCESS) {
        throw std::runtime_error(std::string(expr) + " failed: " + nvrtcGetErrorString(result));
    }
}

struct NvrtcProgramGuard {
    nvrtcProgram program = nullptr;

    ~NvrtcProgramGuard()
    {
        if (program != nullptr) {
            nvrtcDestroyProgram(&program);
        }
    }

    nvrtcProgram* put()
    {
        return &program;
    }
};

bool compileShader(
    std::string& compiled,
    std::string& log,
    const zeno_nvrtc_ipc::Request& request)
{
    NvrtcProgramGuard prog;
    checkNvrtc(
        nvrtcCreateProgram(
            prog.put(),
            request.source.c_str(),
            request.name.c_str(),
            static_cast<int>(sutil::getIncFileTab().size()),
            sutil::getIncFileTab().data(),
            sutil::getIncPathTab().data()),
        "nvrtcCreateProgram");

    std::vector<const char*> options;
    options.reserve(request.options.size());
    for (const auto& option : request.options) {
        options.push_back(option.c_str());
    }

    const nvrtcResult compile_result = nvrtcCompileProgram(prog.program, static_cast<int>(options.size()), options.data());

    size_t log_size = 0;
    checkNvrtc(nvrtcGetProgramLogSize(prog.program, &log_size), "nvrtcGetProgramLogSize");
    if (log_size > 1) {
        log.resize(log_size);
        checkNvrtc(nvrtcGetProgramLog(prog.program, log.data()), "nvrtcGetProgramLog");
    }

    if (compile_result != NVRTC_SUCCESS) {
        return false;
    }

    bool using_ir = false;
    for (const auto& option : request.options) {
        if (option == "--optix-ir") {
            using_ir = true;
            break;
        }
    }

    size_t code_size = 0;
    if (using_ir) {
        checkNvrtc(nvrtcGetOptiXIRSize(prog.program, &code_size), "nvrtcGetOptiXIRSize");
        compiled.resize(code_size);
        checkNvrtc(nvrtcGetOptiXIR(prog.program, compiled.data()), "nvrtcGetOptiXIR");
    } else {
        checkNvrtc(nvrtcGetPTXSize(prog.program, &code_size), "nvrtcGetPTXSize");
        compiled.resize(code_size);
        checkNvrtc(nvrtcGetPTX(prog.program, compiled.data()), "nvrtcGetPTX");
    }

    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 1 && argc != 3) {
        std::cerr << "usage: zeno_nvrtc_helper [request response]\n";
        return 2;
    }

#if defined(_WIN32)
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
    SetHandleInformation(GetStdHandle(STD_INPUT_HANDLE), HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(GetStdHandle(STD_OUTPUT_HANDLE), HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(GetStdHandle(STD_ERROR_HANDLE), HANDLE_FLAG_INHERIT, 0);
#endif

    zeno_nvrtc_ipc::Response response;
    try {
        const auto request = argc == 3
            ? zeno_nvrtc_ipc::readRequest(argv[1])
            : zeno_nvrtc_ipc::readRequest(std::cin);
        response.success = compileShader(response.data, response.log, request);
    } catch (const std::exception& e) {
        response.success = false;
        response.log = e.what();
    }

    try {
        if (argc == 3) {
            zeno_nvrtc_ipc::writeResponse(argv[2], response);
        } else {
            zeno_nvrtc_ipc::writeResponse(std::cout, response);
            std::cout.flush();
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 3;
    }

    return 0;
}
