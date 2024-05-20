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

Formula::Formula(const std::string& formula)
    : m_location(0)
    , m_formula(formula)
{
}

int Formula::parse(float& result) {
    std::stringstream inStream;
    std::stringstream outStream;
    Scanner scanner(inStream, outStream, *this);
    Parser parser(scanner, *this);
    m_location = 0;
    inStream << m_formula << std::endl;
    int ret = parser.parse();
    result = m_result;
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

void Formula::increaseLocation(unsigned int loc) {
    m_location += loc;
    //cout << "increaseLocation(): " << loc << ", total = " << m_location << endl;
}

unsigned int Formula::location() const {
    return m_location;
}
