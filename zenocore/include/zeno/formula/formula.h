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
    struct node* getRoot();
    void setRoot(struct node* root);
    struct node* makeNewNode(nodeType type, operatorVals op, struct node* left, struct node* right);
    struct node* makeNewNumberNode(float value);
    struct node* makeEmptyNode();
    ZENO_API void printSyntaxTree();
    ZENO_API void freeSyntaxTree();
    ZENO_API std::optional<std::pair<std::string, std::string>> getCurrFuncDescription();
    //regex
    ZENO_API std::vector<std::string> getHintList(std::string originTxt, std::string& candidateTxt);

private:

    unsigned int m_location;          // Used by scanner
    std::string m_formula;
    float m_result;

    //syntax_tree
    struct node* m_rootNode;
    int m_leftParenthesesAdded;
    int m_rightParenthesesAdded;
};

}

#endif __ZEN_FORMULA_H__
