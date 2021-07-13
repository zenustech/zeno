cmake_minimum_required(VERSION 3.10)
project(ZFX LANGUAGES C CXX)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)

add_library(ZFX
# ls {,include/zfx/}*{,/*}.{h,cpp} | cat
AST.h
common.h
include/zfx/x64.h
include/zfx/zfx.h
IR.h
IRVisitor.h
Lexical.h
LowerAST.cpp
LowerAST.h
main.cpp
Parser.cpp
Statement.h
Stmts.h
Tokenizer.cpp
visitors/Clone.cpp
visitors/EmitAssembly.cpp
visitors/ExpandFunctions.cpp
Visitors.h
visitors/LowerAccess.cpp
visitors/LowerMath.cpp
visitors/RegisterAllocation.cpp
visitors/TypeCheck.cpp
x64/Assembler.cpp
x64/Executable.h
x64/SIMDBuilder.h
ZFX.cpp
    )
target_compile_definitions(ZFX PRIVATE -DZFX_PRINT_IR)
target_include_directories(ZFX PUBLIC include)
target_include_directories(ZFX PRIVATE .)

add_executable(ZFXtest main.cpp)
target_link_libraries(ZFXtest ZFX)
