#ifndef __TREE_H__
#define __TREE_H__

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <cmath>
#include <memory>
#include <zeno/core/common.h>
#include <glm/glm.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/geo/kdsearch.h>


namespace zeno {

enum nodeType {
    UNDEFINE = 0,
    NUMBER,             //数字
    BOOLTYPE,
    FUNC,               //函数
    FOUROPERATIONS,     //四则运算+ - * / %
    UNARY_EXP,           //一元运算符
    NEGATIVE,           //取负号
    STRING,             //字符串
    ZENVAR,
    ATTR_VAR,           //attr value
    COMPOP,             //操作符
    CONDEXP,            //条件表达式
    ARRAY,
    PLACEHOLDER,
    DECLARE,            //变量定义
    ASSIGNMENT,           //赋值
    ATTR_VISIT,      //a.x, a.y, a.z, param("...").value  ref("./node").x
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
    NOT,    //非，即!
    NEG,    //取负

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
    TYPE_VECTOR2_ARR,
    TYPE_VECTOR3,
    TYPE_VECTOR3_ARR,
    TYPE_VECTOR4,
    TYPE_VECTOR4_ARR,
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
using zfxvec2arr = std::vector<glm::vec2>;
using zfxvec3arr = std::vector<glm::vec3>;
using zfxvec4arr = std::vector<glm::vec4>;

enum LValueType {
    LVal_NodeParam,
    LVal_Dollar,
    LVal_ZfxVar
};

struct ZfxLValue {
    std::string name;
    std::variant<ParamPrimitive, ParamObject> var;  //only for LVal_NodeParam
    LValueType type;
};

using zfxvariant = std::variant<int, float, std::string, ZfxLValue,
    zfxintarr, zfxfloatarr, zfxstringarr,
    zfxvec2arr, zfxvec3arr, zfxvec4arr,
    glm::vec2, glm::vec3, glm::vec4, 
    glm::mat2, glm::mat3, glm::mat4>;

using ZfxVector = std::variant<
    std::vector<int>,
    std::vector<float>,
    std::vector<std::string>,
    std::vector<ZfxLValue>,
    std::vector<zfxintarr>,
    std::vector<zfxfloatarr>,
    std::vector<zfxstringarr>,
    std::vector<zfxvec2arr>,
    std::vector<zfxvec3arr>,
    std::vector<zfxvec4arr>,
    std::vector<glm::vec2>,
    std::vector<glm::vec3>,
    std::vector<glm::vec4>,
    std::vector<glm::mat2>,
    std::vector<glm::mat3>,
    std::vector<glm::mat4>
>;



struct ZfxVariable
{
    ZfxVector value;  //如果是属性变量(bAttr=true)，那这个容器的大小就是runover（点线面）的元素个数，否则就是size=1，也可以单值表示所有属性的等值。

    bool bAttr = false;     //是否与属性关联（好像没什么用）
    bool bAttrUpdated = false;      //ZfxVariable也记录属性值（比如@P, @N @ptnum等），此标记记录在zfx执行中，属性值是否修改了
    bool bArray = false;    //int[] float[] vectorN[]这种都array，而vec3这些不算

    ZfxVariable() {}
    ZfxVariable(ZfxVector&& var) {
        value = var;
    }
    ZfxVariable(const ZfxVector& var) {
        value = var;
    }

    size_t size() const {
        return std::visit([&](const auto& vec)->size_t {
            return vec.size();
        }, value);
    }
};

using VariableTable = std::map<std::string, ZfxVariable>;
using ZfxVarRef = VariableTable::const_iterator;
using ZfxElemFilter = std::vector<char>;

using ZfxVecVar = std::variant<
    std::vector<int>,
    std::vector<float>,
    std::vector<std::string>,
    std::vector<ZfxLValue>,
    std::vector<glm::vec2>,
    std::vector<glm::vec3>,
    std::vector<glm::vec4>,
    std::vector<glm::mat2>,
    std::vector<glm::mat3>,
    std::vector<glm::mat4>,
    std::vector<zfxintarr>,
    std::vector<zfxfloatarr>,
    std::vector<zfxstringarr>>;


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

struct ZfxParamConstrain
{
    std::string constrain_param;    /*专门用于参数ui约束调整的场景，如果不为空，即为要约束的参数*/
    bool bInput = true;
    bool update_nodeparam_prop = false;      //参数属性是否更新了，比如可见性可用性
};

class KdTree;

struct PointCloud
{
    std::shared_ptr<zeno::KdTree> pTree;
    std::vector<vec3f> testPoints;
    float radius = 0.f;
    int maxpoints = 0;
};

class NodeImpl;

struct ZfxContext
{
    /* in */ zany spObject;
    /* in */ NodeImpl* spNode;
    /* in */ std::string code;
    /* in */ GeoAttrGroup runover = ATTR_POINT;
    /* in */ bool bSingleFmla = false;      //只是计算一个表达式
    /**/     std::vector<PointCloud> pchandles;
    /**/     VariableTable* zfxVariableTbl = nullptr;
    /* inout */ ZfxParamConstrain param_constrain;
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