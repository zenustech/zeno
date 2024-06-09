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

std::shared_ptr<ZfxASTNode> ZfxExecute::makeStringNode(std::string text) {
    std::shared_ptr<ZfxASTNode> spNode = std::make_shared<ZfxASTNode>();
    spNode->type = STRING;
    spNode->opVal = UNDEFINE_OP;
    spNode->value = text.substr(1, text.length() - 2);
    return spNode;
}

std::shared_ptr<ZfxASTNode> ZfxExecute::makeZenVarNode(std::string text) {
    std::shared_ptr<ZfxASTNode> spNode = std::make_shared<ZfxASTNode>();
    spNode->type = ZENVAR;
    spNode->opVal = UNDEFINE_OP;
    if (!text.empty())
        spNode->value = text.substr(1);
    else
        spNode->value = text;
    return spNode;
}

std::shared_ptr<ZfxASTNode> ZfxExecute::makeZfxVarNode(std::string text) {
    std::shared_ptr<ZfxASTNode> spNode = std::make_shared<ZfxASTNode>();
    spNode->type = ZENVAR;
    spNode->opVal = UNDEFINE_OP;
    spNode->value = text;
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
    m_vecCommands.push_back(pNode);
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