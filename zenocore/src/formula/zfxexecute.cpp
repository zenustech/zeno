#include <zeno/formula/zfxexecute.h>
#include <zeno/core/Session.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/core/Graph.h>
#include <zeno/extra/GraphException.h>
#include "zfxscanner.h"
#include "zfxparser.hpp"
#include <regex>
#include <zeno/core/ReferManager.h>
#include <zeno/core/FunctionManager.h>


namespace zeno
{

ZENO_API ZfxExecute::ZfxExecute(const std::string& code, const std::string& nodepath)
    : m_location(0)
    , m_code(code)
    , m_context(nullptr)
{

}

ZENO_API ZfxExecute::~ZfxExecute()
{
}

ZENO_API int ZfxExecute::parse() {
    std::stringstream inStream;
    std::stringstream outStream;
    ZfxScanner scanner(inStream, outStream, *this);
    ZfxParser parser(scanner, *this);
    m_location = 0;
    inStream << m_code << std::endl;
    int ret = parser.parse();
    return ret;
}

std::shared_ptr<ZfxASTNode> ZfxExecute::makeNewNode(nodeType type, operatorVals op, std::vector<std::shared_ptr<ZfxASTNode>> children) {
    auto pNode = newNode(type, op, children);
    return pNode;
}

std::shared_ptr<ZfxASTNode> ZfxExecute::makeNewNumberNode(float value) {
    auto pNode = newNumberNode(value);
    return pNode;
}

std::shared_ptr<ZfxASTNode> ZfxExecute::makeBoolNode(bool bVal) {
    std::shared_ptr<ZfxASTNode> n = std::make_shared<ZfxASTNode>();
    if (!n)
    {
        exit(0);
    }
    n->type = BOOLTYPE;
    n->opVal = UNDEFINE_OP;
    n->value = bVal;
    return n;
}

std::shared_ptr<ZfxASTNode> ZfxExecute::makeStringNode(std::string text) {
    std::shared_ptr<ZfxASTNode> spNode = std::make_shared<ZfxASTNode>();
    spNode->type = STRING;
    spNode->opVal = UNDEFINE_OP;
    spNode->value = text.substr(1, text.length() - 2);
    return spNode;
}

std::shared_ptr<ZfxASTNode> ZfxExecute::makeZfxVarNode(std::string text, operatorVals op) {
    std::shared_ptr<ZfxASTNode> spNode = std::make_shared<ZfxASTNode>();
    spNode->type = ZENVAR;
    spNode->opVal = op;
    spNode->value = text;
    return spNode;
}

std::shared_ptr<ZfxASTNode> ZfxExecute::makeTypeNode(std::string text, bool bArray) {
    std::shared_ptr<ZfxASTNode> spNode = std::make_shared<ZfxASTNode>();
    spNode->type = VARIABLETYPE;
    spNode->value = text;
    if (text == "int") {
        spNode->opVal = bArray ? TYPE_INT_ARR : TYPE_INT;
    }
    else if (text == "float") {
        spNode->opVal = bArray ? TYPE_FLOAT_ARR : TYPE_FLOAT;
    }
    else if (text == "string") {
        spNode->opVal = bArray ? TYPE_STRING_ARR : TYPE_STRING;
    }
    else if (text == "vector2") {
        spNode->opVal = TYPE_VECTOR2;
    }
    else if (text == "vector3") {
        spNode->opVal = TYPE_VECTOR3;
    }
    else if (text == "vector4") {
        spNode->opVal = TYPE_VECTOR4;
    }
    else if (text == "matrix2") {
        spNode->opVal = TYPE_MATRIX2;
    }
    else if (text == "matrix3") {
        spNode->opVal = TYPE_MATRIX3;
    }
    else if (text == "matrix4") {
        spNode->opVal = TYPE_MATRIX4;
    }
    return spNode;
}

std::shared_ptr<ZfxASTNode> ZfxExecute::makeComponentVisit(std::shared_ptr<ZfxASTNode> pVarNode, std::string component) {
    std::shared_ptr<ZfxASTNode> childNode = std::make_shared<ZfxASTNode>();
    childNode->value = component;
    pVarNode->children.push_back(childNode);
    return pVarNode;
}

std::shared_ptr<ZfxASTNode> ZfxExecute::makeQuoteStringNode(std::string text) {
    std::shared_ptr<ZfxASTNode> spNode = std::make_shared<ZfxASTNode>();
    spNode->type = STRING;
    spNode->opVal = UNDEFINE_OP;
    spNode->value = text.substr(1);
    return spNode;
}

void ZfxExecute::setASTResult(std::shared_ptr<ZfxASTNode> pNode) {
    m_root = pNode;
}

ZENO_API void ZfxExecute::printSyntaxTree()
{
    std::string printContent = "\noriginal code: " + m_code + '\n';
    if (!m_root) {
        printContent += "parser failed";
    }
    else {
        print_syntax_tree(m_root, 0, printContent);
    }
    zeno::log_info(printContent);
}

std::shared_ptr<ZfxASTNode> ZfxExecute::makeEmptyNode() {
    std::shared_ptr<ZfxASTNode> n = std::make_shared<ZfxASTNode>();
    if (!n)
    {
        exit(0);
    }
    n->type = PLACEHOLDER;
    n->value = 0;
    return n;
}

unsigned int ZfxExecute::location() const {
    return m_location;
}

void ZfxExecute::increaseLocation(unsigned int loc, char* txt) {
    m_location += loc;
}

}