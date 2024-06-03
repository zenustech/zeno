#ifndef __ZEN_FORMULA_H__
#define __ZEN_FORMULA_H__

#include <vector>
#include <sstream>
#include <memory>
#include <regex>
#include <optional>
#include <zeno/utils/api.h>
#include "syntax_tree.h"

namespace zeno {

class Formula
{
public:
    ZENO_API Formula(const std::string& formula);
    ZENO_API ~Formula();
    /**
     * Run parser. Results are stored inside.
     * \returns 0 on success, 1 on failure
     */
    ZENO_API int parse();

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
    void increaseLocation(unsigned int loc, char* txt);

    void callFunction(const std::string& funcname);

    float callRef(const std::string& ref);

    void setResult(float res);
    float getResult() const;

    int getFrameNum();
    float getFps();
    float getPI();

    // Used to get last Scanner location. Used in error messages.
    unsigned int location() const;

    //syntax_tree
    ZENO_API std::shared_ptr<struct node> getRoot();
    void setRoot(std::shared_ptr<struct node> root);
    std::shared_ptr<struct node> makeNewNode(nodeType type, operatorVals op, std::vector<std::shared_ptr<struct node>> children);
    std::shared_ptr<struct node> makeNewNumberNode(float value);
    std::shared_ptr<node> makeStringNode(std::string text);
    std::shared_ptr<node> makeQuoteStringNode(std::string text);
    std::shared_ptr<struct node> makeEmptyNode();
    ZENO_API void printSyntaxTree();
    ZENO_API std::optional<std::tuple<std::string, std::string, int>> getCurrFuncDescription();
    //regex
    ZENO_API std::vector<std::string> getHintList(std::string originTxt, std::string& candidateTxt);

private:

    unsigned int m_location;          // Used by scanner
    std::string m_formula;
    float m_result;

    //syntax_tree
    std::shared_ptr<struct node> m_rootNode;
};

}

#endif __ZEN_FORMULA_H__
