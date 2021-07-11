cmake_minimum_required(VERSION 3.10)
project(ZFX LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

add_executable(ZFX
# ls *{,/*}.{h,cpp} | cat
AST.h
common.h
IR.h
IRVisitor.h
Lexical.h
LowerAST.cpp
Parser.cpp
Statement.h
Stmts.h
Tokenizer.cpp
visitors/Clone.cpp
visitors/EmitAssembly.cpp
Visitors.h
visitors/LowerAccess.cpp
visitors/LowerMath.cpp
x64/Assembler.cpp
x64/Executable.h
x64/Program.h
x64/SIMDBuilder.h
ZFX.cpp
    )
target_compile_definitions(ZFX PRIVATE -DZFX_PRINT_IR)
target_include_directories(ZFX PRIVATE .)
