#include <zeno/formula/formula.h>
#include <zeno/core/Session.h>
#include <zeno/extra/GlobalState.h>
#include "scanner.h"
#include "parser.hpp"

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
    return 0.;
}

void Formula::increaseLocation(unsigned int loc) {
    m_location += loc;
    //cout << "increaseLocation(): " << loc << ", total = " << m_location << endl;
}

unsigned int Formula::location() const {
    return m_location;
}
