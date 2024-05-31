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
        return std::get<float>(primparam.result);
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
                case Param_Vec2f:
                case Param_Vec2i:
                {
                    auto vec = std::get<zeno::vec2f>(primparam.result);
                    if (idx < vec.size())
                        return vec[idx];
                    break;
                }
                case Param_Vec3f:
                case Param_Vec3i:
                {
                    auto vec = std::get<zeno::vec3f>(primparam.result);
                    if (idx < vec.size())
                        return vec[idx];
                    break;
                }
                case Param_Vec4f:
                case Param_Vec4i:
                {
                    auto vec = std::get<zeno::vec4f>(primparam.result);
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

void Formula::increaseLocation(unsigned int loc) {
    m_location += loc;
    //cout << "increaseLocation(): " << loc << ", total = " << m_location << endl;
}

unsigned int Formula::location() const {
    return m_location;
}
