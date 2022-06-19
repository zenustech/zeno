//
// Created by admin on 2022/5/7.
//
//试验性的将zfx转成x64汇编代码
#pragma once

#include <map>
#include <memory>
namespace zfx::x64 {
    /*
    struct Executable {

    };

    struct Assembler {
        std::map<std::string, std::unique_ptr<Executable>> cache;

    };
     */

    enum AsmOpCode {
        jmp=0,
        je,
        jne,
        jle,
        jl,
        jge,
        jg,
        jbe,
        jb,
        jae,
        ja,

        sete=20,
        setne,
        setl,
        setle,
        setg,
        setge,

        //8字节指令
        movq=40,
        addq,
        subq,
        mulq,
        imulq,
        divq,
        idivq,
        negq,
        incq,
        decq,
        xorq,
        orq,
        andq,
        notq,
        leaq,
        callq,
        retq,
        pushq,
        popq,
        cmpq,

        //4字节指令
        movl=80,
        addl,
        subl,
        mull,
        imull,
        divl,
        idivl,
        negl,
        incl,
        decl,
        xorl,
        orl,
        andl,
        notl,
        leal,
        calll,
        retl,
        pushl,
        popl,
        cmpl,

        //2字节指令
        movw=120,
        addw,
        subw,
        mulw,
        imulw,
        divw,
        idivw,
        negw,
        incw,
        decw,
        xorw,
        orw,
        andw,
        notw,
        leaw,
        callw,
        retw,
        pushw,
        popw,
        cmpw,

        //单字节指令
        movb=160,
        addb,
        subb,
        mulb,   //无符号乘
        imulb,  //有符号乘
        divb,   //无符号除
        idivb,  //有符号除
        negb,
        incb,
        decb,
        xorb,
        orb,
        andb,
        notb,
        leab,
        callb,
        retb,
        pushb,
        popb,
        cmpb,

        //SSE指令
        movsd = 200,
        addsd,
        subsd,
        mulsd,
        divsd,
        sqrtsd,
        maxsd,
        minsd,
        cmpsd,
        comisd,
        ucomisd,

        cvttsd2si = 240,   //double 到long或int都可以，会导致截断, 第一个操作数可以是内存
        cvtsi2sdq,         //从long到double


        //伪指令
        declVar = 300,     //变量声明
        reload,            //重新装载被溢出到内存的变量到寄存器
        tailRecursive,     //尾递归的函数调用
        tailCall,          //尾调用
        tailRecursiveJmp,  //尾递归产生的jmp指令，操作数是一个基本块，是序曲下的第一个基本块。
        tailCallJmp,       //尾调用产生的jmp指令，操作数是一个字符串（标签）
    };
    /*
     * 操作数的类型
     * */
    enum OprandKind {
        varIndex,  //变量下标
        returnSlot,//用于存放返回值的位置
        bb,//调准指令指向的基本快
        function,//函数调用
        stringConst,//字符串常量

    };
    class OpCodeHelper {
      public:
        static bool isReturn(AsmOpCode op) {
               return op == AsmOpcode::retb || op == AsmOpcode::retw || op == AsmOpcode.retl || op == AsmOpcode::retq;
        }

        static bool isJump(AsmOpCode op) {
            return op < AsmOpcode;
        }

        static bool isMov() {

        }

    };

    class Oprand{
      public:
        OprandKind kind;
        std::string name;

        //判断操作数是否相同
        bool isSame()
        virtual std::string toString() {

        }
    };

    class Inst {
      public:
        AsmOpCode op;
        static uint32_t index;//下标
        uint32_t  numOpRands;
        std::string comment;
        Inst(AsmOpCode op, uint32_t numOpRands) : op(op), numOpRands(numOpRands) {
            index++;
        }

        virtual bool is_Inst_0() {
            return false;
        }

        virtual bool is_Inst_1() {
            return false;
        }

        virtual bool is_Inst_2() {
            return false;
        }
        virtual std::string toString() = 0;
    };


    //没有操作数的指令
    class Inst_0 : public Inst{
      public:
        //
    };

    class Inst_1 : public Inst {
      public:
        std::shared_ptr<Oprand> oprand;
        bool is_Inst1() override {
            return true;
        }

        std::string toString() override {
            auto str =
        }
    };

    class Inst_2 : public Inst {
      public:
        std::shared_ptr<Oprand> oprand1;
        std::shared_ptr<Oprand> oprand2;

        bool isInst_2() override {
            return true;
        }

        std::string toString() override {

        }

    };

    class BasicBlock {
      public:
        std::vector<std::shared_ptr<Inst>> insts;//基本快内的指令
        int32_t funcIndex{-1};//函数编号
        int32_t bbIndex {-1};//基本块编号
        //是否有其他快跳转到该块
        bool isDestination{false};

        std::string getName() {

        }

        std::string toString() {
            std::string str;

            return str;
        }
    };
    //变量活跃性分析结果
    struct LivenessResult {
        std::map<std::shared_ptr<Inst>, std::set<uint32_t>> liveVars;
        std::map<std::shared_ptr<BasicBlock>, std::set<uint32_t>> initalVars;
    };
    class AsmModule {
      public:

        /*
         * 输出代表一个模块的asm文件字符串
         * */
        std::string toString() {
            std::string str;

            return std::move(str);
        }
    };
    //当前汇编依靠直接便令Ast生成，并没有用到自定义ir
    class AsmGenerator : public AstVisitor {
      public:
        std::shared_ptr<AsmModule> asmModule;

        //存放一些临时变量
        //std::shared_ptr<TempStates> s;
        std::shared_ptr<Oprand> returnSlot;
        AsmGenerator() {
            this->asmModule = std::make_shared<AsmModule>();
            this->returnSlot = std::make_shared<Oprand>();
        }

        //接下来就是用访问者模式生成asm
        std::any visitVariable(Variable& variable, std::string prefix) override {

        }


    };
}
