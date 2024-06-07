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
    return nullptr;
}

std::shared_ptr<ZfxASTNode> ZfxExecute::makeNewNumberNode(float value) {
    return nullptr;
}

std::shared_ptr<ZfxASTNode> ZfxExecute::makeStringNode(std::string text) {
    return nullptr;
}

std::shared_ptr<ZfxASTNode> ZfxExecute::makeZenVarNode(std::string text) {
    return nullptr;
}

std::shared_ptr<ZfxASTNode> ZfxExecute::makeQuoteStringNode(std::string text) {
    return nullptr;
}

std::shared_ptr<ZfxASTNode> ZfxExecute::makeEmptyNode() {
    return nullptr;
}

unsigned int ZfxExecute::location() const {
    return m_location;
}

void ZfxExecute::increaseLocation(unsigned int loc, char* txt) {
    m_location += loc;
}

}