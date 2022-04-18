cmake_minimum_required(VERSION 3.12)
project(ZFX LANGUAGES CXX)

option(ZFX_PRINT_IR "Print generated IR in log" ON)
option(ZFX_ENABLE_CUDA "Build ZFX with CUDA support" ON)

set(CMAKE_CXX_STANDARD 17)

add_library(ZFX STATIC
# ls {,include/zfx/}*{,/*}.{h,cpp} | grep -v main.cpp
AST.h
ConstantFold.cpp
ConstParametrize.cpp
ControlCheck.cpp
DemoteMathFuncs.cpp
DetectNewSymbols.cpp
EmitAssembly.cpp
ExpandFunctions.cpp
GlobalLocalize.cpp
KillUnreachable.cpp
MergeIdentical.cpp
ReassignGlobals.cpp
ReassignParameters.cpp
include/zfx/utils.h
include/zfx/x64.h
include/zfx/zfx.h
IR.h
IRVisitor.h
Lexical.h
LowerAccess.cpp
LowerAST.cpp
LowerAST.h
LowerMath.cpp
Parser.cpp
RegisterAllocation.cpp
SaveMathRegisters.cpp
Statement.h
Stmts.h
SymbolCheck.cpp
Tokenizer.cpp
TypeCheck.cpp
Visitors.h
x64/Assembler.cpp
x64/Executable.h
x64/SIMDBuilder.h
zfx.cpp
    )
target_include_directories(ZFX PUBLIC include)
if (ZFX_PRINT_IR)
    target_compile_definitions(ZFX PRIVATE -DZFX_PRINT_IR)
endif()
if (ZFX_ENABLE_CUDA)
    target_sources(ZFX PRIVATE cuda/Assembler.cpp)
endif()

#if (ZFX_ENABLE_CUDA)
#    find_package(CUDAToolkit REQUIRED)
#    set(CMAKE_CUDA_ARCHITECTURES 75)  # WXL may need change this
#    add_executable(ZFXtest cuda/test_main.cpp)
#    link_directories(/opt/cuda/lib64)  # WXL may need change this
#    include_directories(/opt/cuda/include)  # WXL may need change this
#    target_link_libraries(ZFXtest PUBLIC CUDA::cuda_driver CUDA::nvrtc)
#else()
#    add_executable(ZFXtest x64/test_main.cpp)
#endif()
#target_link_libraries(ZFXtest PRIVATE ZFX)
