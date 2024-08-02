#include <zeno/formula/formula.h>
#include <zeno/core/Session.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/core/Graph.h>
#include <zeno/extra/GraphException.h>
#include "scanner.h"
#include "parser.hpp"
#include <regex>
#include <zeno/core/ReferManager.h>
#include "reflect/reflection.generated.hpp"

using namespace zeno;

static std::map<std::string, std::string> funcDescription({ 
    //函数名，参数个数\n函数签名\n描述\nUSage:\nExample\n---
    {"sin", "3\nfloat sin(float degrees)\nReturn the sine of argument\nUsage:\nExample:\n---"},
    {"cos", "3\nfloat cos(float degrees)\nReturn the cosine of argument\nUsage:\nExample:\n---"},
    {"sinh", "3\nfloat sinh(float degrees)\nReturn the sinh of argument\nUsage:\nExample:\n---"},
    {"cosh", "3\nfloat cosh(float degrees)\nReturn the cosh of argument\nUsage:\nExample:\n---"},
    {"abs", "3\nfloat abs(float degrees)\nReturn the abs of argument\nUsage:\nExample:\n---"},
    });

ZENO_API Formula::Formula(const std::string& formula)
    : m_location(0)
    , m_formula(formula)
    , m_rootNode(nullptr)
{
}

ZENO_API Formula::~Formula()
{
}

ZENO_API int Formula::parse() {
    std::stringstream inStream;
    std::stringstream outStream;
    Scanner scanner(inStream, outStream, *this);
    Parser parser(scanner, *this);
    m_location = 0;
    inStream << m_formula << std::endl;
    int ret = parser.parse();
    return ret;
}

void Formula::clear() {
    m_location = 0;
}

void Formula::setResult(float res) {
    m_result = res;
}

float Formula::getResult() const {
    return m_result;
}

int Formula::getFrameNum() {
    int frame = zeno::getSession().globalState->getFrameId();
    return frame;
}

float Formula::getFps() {
    return 234;
}

float Formula::getPI() {
    return 3.14;
}

std::string Formula::str() const {
    std::stringstream s;
    return s.str();
}

void Formula::callFunction(const std::string& funcname) {

}

float Formula::callRef(const std::string& ref) {
    //the refer param
    int sPos = ref.find_last_of('/');
    std::string param = ref.substr(sPos + 1, ref.size() - sPos - 2);
    //remove " 
    std::string path = ref.substr(1, sPos - 1);
    //apply the referenced node
    auto pNode = zeno::getSession().mainGraph->getNodeByPath(path);
    if (!pNode) {
        zeno::log_error("reference {} error", path);
        return NAN;
    }
    std::string uuid_path = zeno::objPathToStr(pNode->get_uuid_path());
    std::regex rgx("(\\.x|\\.y|\\.z|\\.w)$");
    std::string paramName = std::regex_replace(param, rgx, "");
    if (zeno::getSession().referManager->isReferSelf(uuid_path, paramName))
    {
        zeno::log_error("{} refer loop", path);
        return NAN;
    }
    if (pNode->requireInput(param))
    {
        //refer float
        bool bExist = true;
        zeno::ParamPrimitive primparam = pNode->get_input_prim_param(param, &bExist);
        if (!bExist)
            return NAN;
        return zeno_get<float>(primparam.result);
    }
    else
    {
        //vec refer
        if (param == paramName)
        {
            zeno::log_error("reference param {} error", param);
            return NAN;
        }
        if (pNode->requireInput(paramName))
        {
            std::string vecStr = param.substr(param.size() - 1, 1);
            int idx = vecStr == "x" ? 0 : vecStr == "y" ? 1 : vecStr == "z" ? 2 : 3;
            bool bExist = true;
            zeno::ParamPrimitive primparam = pNode->get_input_prim_param(param, &bExist);
            if (!bExist)
                return NAN;

            switch (primparam.type)
            {
                case zeno::types::gParamType_Vec2f:
                case zeno::types::gParamType_Vec2i:
                {
                    auto vec = zeno_get<zeno::vec2f>(primparam.result);
                    if (idx < vec.size())
                        return vec[idx];
                    break;
                }
                case zeno::types::gParamType_Vec3f:
                case zeno::types::gParamType_Vec3i:
                {
                    auto vec = zeno_get<zeno::vec3f>(primparam.result);
                    if (idx < vec.size())
                        return vec[idx];
                    break;
                }
                case zeno::types::gParamType_Vec4f:
                case zeno::types::gParamType_Vec4i:
                {
                    auto vec = zeno_get<zeno::vec4f>(primparam.result);
                    if (idx < vec.size())
                        return vec[idx];
                    break;
                }
            }
        }
    }
    zeno::log_error("reference {} error", path);
    return NAN;
}

void Formula::increaseLocation(unsigned int loc, char* txt) {
    m_location += loc;
}

unsigned int Formula::location() const {
    return m_location;
}

ZENO_API std::shared_ptr<struct node> Formula::getRoot()
{
    return m_rootNode;
}

void Formula::setRoot(std::shared_ptr<struct node> root)
{
    m_rootNode = root;
}

std::shared_ptr<struct node> Formula::makeNewNode(nodeType type, operatorVals op, std::vector<std::shared_ptr<struct node>> children)
{
    m_rootNode = newNode(type, op, children);
    return m_rootNode;
}

std::shared_ptr<node> Formula::makeStringNode(std::string text)
{
    std::shared_ptr<node> spNode = std::make_shared<node>();
    spNode->type = STRING;
    spNode->opVal = UNDEFINE_OP;
    spNode->value = text.substr(1, text.length() - 2);
    return spNode;
}

std::shared_ptr<node> Formula::makeQuoteStringNode(std::string text)
{
    std::shared_ptr<node> spNode = std::make_shared<node>();
    spNode->type = STRING;
    spNode->opVal = UNDEFINE_OP;
    spNode->value = text.substr(1);
    return spNode;
}

std::shared_ptr<struct node> Formula::makeNewNumberNode(float value)
{
    m_rootNode = newNumberNode(value);
    return m_rootNode;
}

std::shared_ptr<struct node> Formula::makeEmptyNode()
{
    std::shared_ptr<struct node> n = std::make_shared<struct node>();
    if (!n)
    {
        exit(0);
    }
    n->type = PLACEHOLDER;
    n->value = 0;
    return n;
}

ZENO_API void Formula::printSyntaxTree()
{
    printf("\n");
    printf("original formula: %s\n", m_formula.c_str());
    print_syntax_tree(m_rootNode, 0);
    printf("\n");
}

ZENO_API std::optional<std::tuple<std::string, std::string, int>> Formula::getCurrFuncDescription()
{
    //printSyntaxTree();
    std::string funcName = "";
    int paramPos = 0;
    currFuncNamePos(m_rootNode, funcName, paramPos);
    auto it = funcDescription.find(funcName);
    if (it != funcDescription.end()) {
        return std::optional<std::tuple<std::string, std::string, int>>(std::make_tuple(funcName, it->second, paramPos));
    }
    return nullopt;
}

ZENO_API std::vector<std::string> Formula::getHintList(std::string originTxt, std::string& candidateTxt)
{
    std::vector<std::string> list;
    std::smatch match;
    std::reverse(originTxt.begin(), originTxt.end());
    if (std::regex_search(originTxt, match, std::regex("([0-9a-zA-Z]*[a-zA-Z])")) && match.size() == 2) {
        std::string resStr = match[1].str();
        if (originTxt.substr(0, resStr.size()) == resStr) {
            std::reverse(resStr.begin(), resStr.end());
            candidateTxt = resStr;
            for (auto& [k, v] : funcDescription) {
                if (k.substr(0, resStr.size()) == resStr) {
                    list.push_back(k);
                }
            }
        }
    }
    return list;
}
