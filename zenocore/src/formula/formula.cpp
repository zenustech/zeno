#include <zeno/formula/formula.h>
#include <zeno/core/Session.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/core/Graph.h>
#include <zeno/extra/GraphException.h>
#include "scanner.h"
#include "parser.hpp"
#include <regex>
#include <zeno/core/FunctionManager.h>


using namespace zeno;

ZENO_API Formula::Formula(const std::string& formula, const std::string& nodepath)
    : m_location(0)
    , m_formula(formula)
    , m_nodepath(nodepath)
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
    //apply the referenced ZfxASTNode
    auto pNode = zeno::getSession().mainGraph->getNodeByPath(path);
    if (!pNode) {
        zeno::log_error("reference {} error", path);
        return NAN;
    }
    std::string uuid_path = zeno::objPathToStr(pNode->get_uuid_path());
    std::regex rgx("(\\.x|\\.y|\\.z|\\.w)$");
    std::string paramName = std::regex_replace(param, rgx, "");
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

ZENO_API std::shared_ptr<ZfxASTNode> Formula::getASTResult()
{
    return m_rootNode;
}

std::shared_ptr<ZfxASTNode> Formula::makeNewNode(nodeType type, operatorVals op, std::vector<std::shared_ptr<ZfxASTNode>> children)
{
    auto pNode = newNode(type, op, children);
    return pNode;
}

std::shared_ptr<ZfxASTNode> Formula::makeStringNode(std::string text)
{
    std::shared_ptr<ZfxASTNode> spNode = std::make_shared<ZfxASTNode>();
    spNode->type = STRING;
    spNode->opVal = UNDEFINE_OP;
    spNode->value = text.substr(1, text.length() - 2);
    return spNode;
}

std::shared_ptr<ZfxASTNode> Formula::makeZenVarNode(std::string text)
{
    std::shared_ptr<ZfxASTNode> spNode = std::make_shared<ZfxASTNode>();
    spNode->type = ZENVAR;
    spNode->opVal = UNDEFINE_OP;
    if (!text.empty())
        spNode->value = text.substr(1);
    else
        spNode->value = text;
    return spNode;
}

std::shared_ptr<ZfxASTNode> Formula::makeQuoteStringNode(std::string text)
{
    std::shared_ptr<ZfxASTNode> spNode = std::make_shared<ZfxASTNode>();
    spNode->type = STRING;
    spNode->opVal = UNDEFINE_OP;
    spNode->value = text.substr(1);
    return spNode;
}

std::shared_ptr<ZfxASTNode> Formula::makeNewNumberNode(float value)
{
    auto pNode = newNumberNode(value);
    return pNode;
}

std::shared_ptr<ZfxASTNode> Formula::makeEmptyNode()
{
    std::shared_ptr<ZfxASTNode> n = std::make_shared<ZfxASTNode>();
    if (!n)
    {
        exit(0);
    }
    n->type = PLACEHOLDER;
    n->value = 0;
    return n;
}

void Formula::setASTResult(std::shared_ptr<ZfxASTNode> pNode)
{
    m_rootNode = pNode;
}

void Formula::debugASTNode(std::shared_ptr<ZfxASTNode> pNode) {
    int j;
    j = 0;
}

ZENO_API void Formula::printSyntaxTree()
{
    zeno::log_info("--------------------------");
    zeno::log_info("original formula: {}", m_formula);
    std::string printContent;
    print_syntax_tree(m_rootNode, 0, printContent);
    zeno::log_info(printContent);
    zeno::log_info("--------------------------");
}

ZENO_API formula_tip_info Formula::getRecommandTipInfo() const
{
    formula_tip_info ret;
    ret.type = FMLA_NO_MATCH;
    std::vector<std::shared_ptr<ZfxASTNode>> preorderVec;
    preOrderVec(m_rootNode, preorderVec);
    if (preorderVec.size() != 0)
    {
        //按照先序遍历，得到最后的叶节点就是当前编辑光标对应的语法树项。
        auto last = preorderVec.back();
        do {
            //因为推荐仅针对函数，所以只需遍历当前节点及其父节点，找到函数节点即可。
            if (last->type == FUNC) {
                std::string funcprefix = std::get<std::string>(last->value);
                if (Match_Nothing == last->func_match) {
                    //仅仅有（潜在的）函数名，还没有括号。
                    std::vector<std::string> candidates = zeno::getSession().funcManager->getCandidates(funcprefix, true);
                    if (!candidates.empty())
                    {
                        ret.func_candidats = candidates;
                        ret.prefix = funcprefix;
                        ret.type = FMLA_TIP_FUNC_CANDIDATES;
                    }
                    else {
                        ret.func_candidats.clear();
                        ret.type = FMLA_TIP_FUNC_CANDIDATES;
                    }
                    break;
                }
                else if (Match_LeftPAREN == last->func_match) {
                    bool bExist = false;
                    FUNC_INFO info = zeno::getSession().funcManager->getFuncInfo(funcprefix);
                    if (!info.name.empty()) {
                        if (info.name == "ref") {
                            if (last->children.size() == 1 && last->children[0] &&
                                last->children[0]->type == nodeType::STRING) {
                                const std::string& refcontent = std::get<std::string>(last->children[0]->value);

                                if (refcontent == "") {
                                    ret.ref_candidates.push_back({ "/", /*TODO: icon*/"" });
                                    ret.type = FMLA_TIP_REFERENCE;
                                    break;
                                }

                                auto idx = refcontent.rfind('/');
                                auto graphpath = refcontent.substr(0, idx);
                                auto nodepath = refcontent.substr(idx + 1);

                                if (graphpath.empty()) {
                                    // "/" "/m" 这种，只有推荐词 /main （不考虑引用asset的情况）
                                    std::string mainstr = "main";
                                    if (mainstr.find(nodepath) != std::string::npos) {
                                        ret.ref_candidates.push_back({ "main", /*TODO: icon*/"" });
                                        ret.prefix = nodepath;
                                        ret.type = FMLA_TIP_REFERENCE;
                                        break;
                                    }
                                    else {
                                        ret.type = FMLA_NO_MATCH;
                                        break;
                                    }
                                }

                                ret = getNodesByPath(m_nodepath, graphpath, nodepath);
                                break;
                            }
                        }
                        else {
                            ret.func_args.func = info;
                            //TODO: 参数位置高亮
                            ret.func_args.argidx = last->children.size();
                            ret.type = FMLA_TIP_FUNC_ARGS;
                        }
                    }
                    else {
                        ret.type = FMLA_NO_MATCH;
                    }
                    break;
                }
                else if (Match_Exactly == last->func_match) {
                    ret.type = FMLA_NO_MATCH;
                }
            }
            else if (last->type == ZENVAR) {
                const std::string& varprefix = std::get<std::string>(last->value);
                std::vector<std::string> candidates = zeno::getSession().funcManager->getCandidates(varprefix, false);
                if (!candidates.empty()) {
                    ret.func_candidats = candidates;
                    ret.prefix = varprefix;
                    ret.type = FMLA_TIP_FUNC_CANDIDATES;
                }
                else {
                    ret.func_candidats.clear();
                    ret.type = FMLA_NO_MATCH;
                }
            }
            last = last->parent.lock();
        } while (last);
    }
    return ret;
}
