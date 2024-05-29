#include <zeno/formula/formula.h>
#include <zeno/core/Session.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/core/Graph.h>
#include <zeno/extra/GraphException.h>
#include "scanner.h"
#include "parser.hpp"
#include <regex>
#include <zeno/core/ReferManager.h>

using namespace zeno;

static std::map<std::string, std::string> funcDescription({ 
    {"sin", "float sin(float degrees)\nReturn the sine of argument\nUsage:\nExample:\n---"},
    {"cos", "float cos(float degrees)\nReturn the cosine of argument\nUsage:\nExample:\n---"},
    {"sinh", "float sinh(float degrees)\nReturn the sinh of argument\nUsage:\nExample:\n---"},
    {"cosh", "float cosh(float degrees)\nReturn the cosh of argument\nUsage:\nExample:\n---"},
    {"abs", "float abs(float degrees)\nReturn the abs of argument\nUsage:\nExample:\n---"},
    });

ZENO_API Formula::Formula(const std::string& formula)
    : m_location(0)
    , m_formula(formula)
    , m_rootNode(nullptr)
    , m_leftParenthesesAdded(0)
    , m_rightParenthesesAdded(0)
{
    checkparentheses(m_formula, m_leftParenthesesAdded, m_rightParenthesesAdded);
}

ZENO_API Formula::~Formula()
{
    free_syntax_tree(m_rootNode);
}

ZENO_API int Formula::parse() {
    std::stringstream inStream;
    std::stringstream outStream;
    Scanner scanner(inStream, outStream, *this);
    Parser parser(scanner, *this);
    m_location = 0;
    inStream << m_formula << std::endl;
    int ret = parser.parse();
    if (ret == 0) {
        m_result = calc_syntax_tree(m_rootNode);
    }
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
    std::string key = pNode->m_uuid + "/" + param;
    if (zeno::getSession().referManager->isReferSelf(key))
    {
        zeno::log_error("{} refer loop", path);
        return NAN;
    }
    if (auto spParam = pNode->get_input_param(param))
    {
        //refer float
        if (pNode->requireInput(spParam))
            return objectToLiterial<float>(spParam->result);
    }
    else
    {
        //vec refer
        std::regex rgx("(\\.x|\\.y|\\.z|\\.w)$");
        if (!std::regex_search(param, rgx))
        {
            zeno::log_error("reference param {} error", param);
            return NAN;
        }
        std::string name = std::regex_replace(param, rgx, "");
        if (auto spParam = pNode->get_input_param(name))
        {
            if (pNode->requireInput(spParam))
            {
                std::string vecStr = param.substr(param.size() - 1, 1);
                int idx = vecStr == "x" ? 0 : vecStr == "y" ? 1 : vecStr == "z" ? 2 : 3;
                switch (spParam->type)
                {
                case Param_Vec2f:
                case Param_Vec2i:
                {
                    auto vec = objectToLiterial<zeno::vec2f>(spParam->result);
                    if (idx < vec.size())
                        return vec[idx];
                    break;
                }
                case Param_Vec3f:
                case Param_Vec3i:
                {
                    auto vec = objectToLiterial<zeno::vec3f>(spParam->result);
                    if (idx < vec.size())
                        return vec[idx];
                    break;
                }
                case Param_Vec4f:
                case Param_Vec4i:
                {
                    auto vec = objectToLiterial<zeno::vec4f>(spParam->result);
                    if (idx < vec.size())
                        return vec[idx];
                    break;
                }
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

struct node* Formula::getRoot()
{
    return m_rootNode;
}

void Formula::setRoot(struct node* root)
{
    m_rootNode = root;
}

struct node* Formula::makeNewNode(nodeType type, operatorVals op, struct node* left, struct node* right)
{
    m_rootNode = newNode(type, op, left, right);
    return m_rootNode;
}

struct node* Formula::makeNewNumberNode(float value)
{
    m_rootNode = newNumberNode(value);
    return m_rootNode;
}

ZENO_API void Formula::printSyntaxTree()
{
    print_syntax_tree(m_rootNode, 0);
}

ZENO_API void Formula::freeSyntaxTree()
{
    free_syntax_tree(m_rootNode);
    m_rootNode = nullptr;
}

ZENO_API std::optional<std::pair<std::string, std::string>> Formula::getCurrFuncDescription()
{
    //printSyntaxTree();
    auto it = funcDescription.find(currFuncName(m_rootNode, m_rightParenthesesAdded));
    if (it != funcDescription.end()) {
        return *it;
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
