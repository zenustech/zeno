#ifndef __ZFX_EXECUTE_H__
#define __ZFX_EXECUTE_H__

#include <vector>
#include <memory>
#include <regex>
#include <zeno/core/data.h>
#include "syntax_tree.h"

namespace zeno {

class ZfxExecute
{
public:
    ZENO_API ZfxExecute(const std::string& code, const std::string& nodepath);
    ZENO_API ~ZfxExecute();

    ZENO_API int parse();

    std::shared_ptr<ZfxASTNode> makeNewNode(nodeType type, operatorVals op, std::vector<std::shared_ptr<ZfxASTNode>> children);
    std::shared_ptr<ZfxASTNode> makeNewNumberNode(float value);
    std::shared_ptr<ZfxASTNode> makeStringNode(std::string text);
    std::shared_ptr<ZfxASTNode> makeZfxVarNode(std::string text, operatorVals op = UNDEFINE_OP);
    std::shared_ptr<ZfxASTNode> makeQuoteStringNode(std::string text);
    std::shared_ptr<ZfxASTNode> makeComponentVisit(std::shared_ptr<ZfxASTNode> pVarNode, std::string component);
    std::shared_ptr<ZfxASTNode> makeEmptyNode();
    void setASTResult(std::shared_ptr<ZfxASTNode> pNode);
    ZENO_API void ZfxExecute::printSyntaxTree();

    // Used to get last Scanner location. Used in error messages.
    unsigned int location() const;
    // Used internally by Scanner YY_USER_ACTION to update location indicator
    void increaseLocation(unsigned int loc, char* txt);

private:
    unsigned int m_location;          // Used by scanner
    std::string m_code;
    FuncContext* m_context;
    std::shared_ptr<ZfxASTNode> m_root;
};

}


#endif