#ifndef __TREE_H__
#define __TREE_H__

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <cmath>
#include <memory>
#include <zeno/core/common.h>
#include <zeno/core/INode.h>
#include <glm/glm.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <zeno/types/PrimitiveObject.h>


namespace zeno {

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
    PLACEHOLDER,
    DECLARE,            //变量定义
    ASSIGNMENT,           //赋值
    IF,
    FOR,
    FOREACH,
    FOREACH_ATTR,
    EACH_ATTRS,
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
    MOD,
    OR,
    AND,

    //函数 nodeType对应FUNC
    SIN,
    SINH,
    COS,
    COSH,
    ABS,

    //比较符号
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Equal,
    NotEqual,

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
    COMPVISIT,      //a.x, a.y, a.z
    BulitInVar,     //$F, $FPS, $T
};

using zfxintarr = std::vector<int>;
using zfxfloatarr = std::vector<float>;
using zfxstringarr = std::vector<std::string>;

using zfxvariant = std::variant<int, float, std::string,
    zfxintarr, zfxfloatarr, zfxstringarr,
    glm::vec2, glm::vec3, glm::vec4, 
    glm::mat2, glm::mat3, glm::mat4>;

enum TokenMatchCase {
    Match_Nothing,
    Match_LeftPAREN,
    Match_Exactly,      //fully match
};

enum ZfxRunOver {
    RunOver_Points,
    RunOver_Face,
    RunOver_Geom,
};

struct ZfxASTNode {
    std::string code;

    enum operatorVals opVal;
    enum nodeType type;

    std::vector<std::shared_ptr<ZfxASTNode>> children;
    std::weak_ptr<ZfxASTNode> parent;

    zfxvariant value;

    TokenMatchCase func_match = Match_Nothing;
    TokenMatchCase paren_match = Match_Nothing;

    bool isParenthesisNode = false;
    bool isParenthesisNodeComplete = false;
    bool bCompleted = false;
    bool AttrAssociateVar = false;      //属性相关变量的赋值，比如int a = @P.y,  b=  @N.x; 后续要被剔除。
    bool bOverridedStmt = false;        //会被if条件覆盖的stmt，不会嵌入到foreach属性循环里。
    bool bOverridedIfLoop = false;      //会被if条件覆盖的if/while/for语句。
    bool bAttr = false;                 //属性值，@P @N @Cd
    int sortOrderNum = 0;               //用于先后顺序排序的值
};

struct ZfxContext
{
    /* in */ std::shared_ptr<IObject> spObject;
    /* in */ std::weak_ptr<INode> spNode;
    /* in */ std::string code;
    /* in */ bool bParsingAttr = false;     //刚parse的语法树还需要对带有属性的进行处理，在外面套一层foreach循环以遍历属性
    /* in */ ZfxRunOver runover = RunOver_Points;
    /* out */ std::string printContent;
    /* out */ operatorVals jumpFlag;
};

std::string getOperatorString(nodeType type, operatorVals op);
operatorVals funcName2Enum(std::string func);

std::shared_ptr<ZfxASTNode> newNode(nodeType type, operatorVals op, std::vector<std::shared_ptr<ZfxASTNode>> Children);
std::shared_ptr<ZfxASTNode> newNumberNode(float value);
void addChild(std::shared_ptr<ZfxASTNode> spNode, std::shared_ptr<ZfxASTNode> spChild);
void appendChild(std::shared_ptr<ZfxASTNode> spNode, std::shared_ptr<ZfxASTNode> spChild);

void print_syntax_tree(std::shared_ptr<ZfxASTNode> root, int depth, std::string& printContent);
float calc_syntax_tree(std::shared_ptr<ZfxASTNode> root);

void printSyntaxTree(std::shared_ptr<ZfxASTNode> root, std::string original_code);

void currFuncNamePos(std::shared_ptr<ZfxASTNode> root, std::string& name, int& pos);  //当前函数名及处于第几个参数
void preOrderVec(std::shared_ptr<ZfxASTNode> root, std::vector<std::shared_ptr<ZfxASTNode>>& tmplist);

bool checkparentheses(std::string& exp, int& addleft, int& addright);

void findAllZenVar(std::shared_ptr<ZfxASTNode> root, std::set<std::string>& vars);

int markOrder(std::shared_ptr<ZfxASTNode> root, int startIndex);

void removeAstNode(std::shared_ptr<ZfxASTNode> root);

std::shared_ptr<ZfxASTNode> clone(std::shared_ptr<ZfxASTNode> spNode);

std::string decompile(std::shared_ptr<ZfxASTNode> root, const std::string& indent = "");

}

#endif