cmake_minimum_required(VERSION 3.10)
project(ZFX LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

add_executable(ZFX
# ls *{,/*}.{h,cpp} | cat
assembler/Assembler.cpp
assembler/Executable.h
assembler/Program.h
assembler/SIMDBuilder.h
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
ZFX.cpp
    )
target_include_directories(ZFX PRIVATE .)
