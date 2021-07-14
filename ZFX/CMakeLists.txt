cmake_minimum_required(VERSION 3.10)
project(ZFX LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

if (CMAKE_BUILD_TYPE EQUAL Debug)
    option(ZFX_PRINT_IR "Print IR for the ZFX compiler" OFF)
else()
    option(ZFX_PRINT_IR "Print IR for the ZFX compiler" ON)
endif()

add_library(ZFX
# ls {,include/zfx/}*{,/*}.{h,cpp} | cat
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

add_executable(ZFXtest main.cpp)
target_link_libraries(ZFXtest ZFX)
