cmake_minimum_required(VERSION 3.12)
project(ZFX LANGUAGES CXX)

option(ZFX_ENABLE_CUDA "Build ZFX with CUDA support" OFF)
if (ZFX_ENABLE_CUDA)
    link_directories(/opt/cuda/lib64)
    include_directories(/opt/cuda/include)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)

if (CMAKE_BUILD_TYPE EQUAL Debug)
    option(ZFX_PRINT_IR "Print IR for the ZFX compiler" OFF)
else()
    option(ZFX_PRINT_IR "Print IR for the ZFX compiler" ON)
endif()

add_library(ZFX
# ls {,include/zfx/}*{,/*}.{h,cpp} | grep -v main.cpp
AST.h
ConstParametrize.cpp
EmitAssembly.cpp
ExpandFunctions.cpp
GlobalLocalize.cpp
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
if (ZFX_PRINT_IR)
    target_compile_definitions(ZFX PRIVATE -DZFX_PRINT_IR)
endif()
target_include_directories(ZFX PUBLIC include)
if (ZFX_ENABLE_CUDA)
    target_sources(ZFX PRIVATE cuda/Assembler.cpp)
    target_link_libraries(ZFX PUBLIC cuda nvrtc)
endif()

if (ZFX_ENABLE_CUDA)
    add_executable(ZFXtest cuda/test_main.cpp)
else()
    add_executable(ZFXtest x64/test_main.cpp)
endif()
target_link_libraries(ZFXtest PRIVATE ZFX)
