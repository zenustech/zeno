#ifndef __ZEN_FORMULA_H__
#define __ZEN_FORMULA_H__

#include <vector>
#include <sstream>
#include <memory>

namespace zeno {

class Formula
{
public:
    Formula(const std::string& formula);
    
    /**
     * Run parser. Results are stored inside.
     * \returns 0 on success, 1 on failure
     */
    int parse(float& result);

    /**
     * Clear AST
     */
    void clear();
    
    /**
     * Print AST
     */
    std::string str() const;
    
    /**
     * This is needed so that Scanner and Parser can call some
     * methods that we want to keep hidden from the end user.
     */

    // Used internally by Scanner YY_USER_ACTION to update location indicator
    void increaseLocation(unsigned int loc);

    void callFunction(const std::string& funcname);

    float callRef(const std::string& ref);

    void setResult(float res);

    int getFrameNum();
    float getFps();
    float getPI();

    // Used to get last Scanner location. Used in error messages.
    unsigned int location() const;

private:
    float getResult() const;

    unsigned int m_location;          // Used by scanner
    std::string m_formula;
    float m_result;
};

}

#endif // __ZEN_FORMULA_H__
