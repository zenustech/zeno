#ifndef __TREE_H__
#define __TREE_H__

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <cmath>
#include <memory>
#include <zeno/core/common.h>


enum nodeType {
    UNDEFINE = 0,
    NUMBER,             //数字
    BOOLTYPE,
    FUNC,               //函数
    FOUROPERATIONS,     //四则运算+ - * / %
    STRING,             //字符串
    ZENVAR,
    COMPOP,             //操作符
    CONDEXP,            //条件表达式
    ARRAY,
    MATRIX,
    COMPVISIT,          //访问元素分量，比如vec.x vec.y vec.z
    PLACEHOLDER,
    DECLARE,            //变量定义
    ASSIGNMENT,           //赋值
    IF,
    FOR,
    FOREACH,
    WHILE,
    DOWHILE,
    CODEBLOCK,          //多个语法树作为children的代码块
    JUMP,
    VARIABLETYPE,       //变量类型，比如int vector3 float string等
};

enum operatorVals {
    UNDEFINE_OP = 0,
    DEFAULT_FUNCVAL,

    //四则运算 nodeType对应FOUROPERATIONS
    PLUS,
    MINUS,
    MUL,
    DIV,
    //函数 nodeType对应FUNC
    SIN,
    SINH,
    COS,
    COSH,
    ABS,

    //以下仅针对变量
    AssignTo,
    AddAssign,
    MulAssign,
    SubAssign,
    DivAssign,

    JUMP_RETURN,
    JUMP_CONTINUE,
    JUMP_BREAK,

    TYPE_INT,
    TYPE_INT_ARR,   //仅针对一维数组
    TYPE_FLOAT,
    TYPE_FLOAT_ARR,
    TYPE_STRING,
    TYPE_STRING_ARR,
    TYPE_VECTOR2,
    TYPE_VECTOR3,
    TYPE_VECTOR4,
    TYPE_MATRIX2,
    TYPE_MATRIX3,
    TYPE_MATRIX4,

    AutoIncreaseFirst,
    AutoIncreaseLast,
    AutoDecreaseFirst,
    AutoDecreaseLast,
    Indexing,
    AttrMark,
    BulitInVar,     //$F, $FPS, $T
};

enum TokenMatchCase {
    Match_Nothing,
    Match_LeftPAREN,
    Match_Exactly,      //fully match
};

struct ZfxASTNode {
    enum operatorVals opVal;
    enum nodeType type;

    std::vector<std::shared_ptr<ZfxASTNode>> children;
    std::weak_ptr<ZfxASTNode> parent;

    zeno::zvariant value;

    TokenMatchCase func_match = Match_Nothing;
    TokenMatchCase paren_match = Match_Nothing;

    bool isParenthesisNode = false;
    bool isParenthesisNodeComplete = false;
    bool bCompleted = false;
};

struct FuncContext {
    std::string nodePath;
};

std::string getOperatorString(nodeType type, operatorVals op);
operatorVals funcName2Enum(std::string func);

std::shared_ptr<ZfxASTNode> newNode(nodeType type, operatorVals op, std::vector<std::shared_ptr<ZfxASTNode>> Children);
std::shared_ptr<ZfxASTNode> newNumberNode(float value);
void addChild(std::shared_ptr<ZfxASTNode> spNode, std::shared_ptr<ZfxASTNode> spChild);

void print_syntax_tree(std::shared_ptr<ZfxASTNode> root, int depth, std::string& printContent);
float calc_syntax_tree(std::shared_ptr<ZfxASTNode> root);

void currFuncNamePos(std::shared_ptr<ZfxASTNode> root, std::string& name, int& pos);  //当前函数名及处于第几个参数
void preOrderVec(std::shared_ptr<ZfxASTNode> root, std::vector<std::shared_ptr<ZfxASTNode>>& tmplist);

bool checkparentheses(std::string& exp, int& addleft, int& addright);
#endif