//
// Created by admin on 2022/5/19.
//
#pragma once

#include <vector>
#include <iostream>
/*
 * zfx only three types $ is the Parameter @ is the symbol, The third is the built-in function type
 * compare by dimension
 * */
namespace zfx {
    enum class TypeKind {
        Parameter,
        Symbol,
        Function
    };
    class ParaType;
    class SymbolType;
    class FunctionType;
    class Type {
      public:
      Type(TypeKind kind, int dim) : kind(kind), dim(dim) {}

      virtual std::string ToString();

      virtual bool LE(const Type &type) const;

      virtual bool isFunction();

      virtual bool isParaType();

      virtual bool isFunctionType();

      virtual ~Type() = default;
      public:
        int dim;
        TypeKind kind;
    };
    bool operator== (const Type &lhs, const Type &rhs) const{
         if (lhs.kind != rhs.kind) {
             return false;
         }  else {
             return lhs.dim == rhs.dim;
         }
         return false;
    }

    class ParaType : public Type{
      public:
        ParaType(TypeKind kind, int dim) : Type(kind, dim) {

        }

        bool isFunctionType() override final {
            return false;
        }

        ~ParaType() = default;
    };

    class SymbolType : public Type {
      public:

        bool LE(const Type &rhs) override {
            return this->dim < rhs.dim;
        }

        ~SymbolType() = default;
    };

    class FunctionType : public Type {
      public:

        std::string ToString() override {

        }

        ~FunctionType() = default;
    };
}
