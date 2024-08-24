#ifndef __ZFX_SCANNER_H__
#define __ZFX_SCANNER_H__


/**
 * Generated Flex class name is yyFlexLexer by default. If we want to use more flex-generated
 * classes we should name them differently. See scanner.l prefix option.
 * 
 * Unfortunately the implementation relies on this trick with redefining class name
 * with a preprocessor macro. See GNU Flex manual, "Generating C++ Scanners" section
 */
#if ! defined(yyFlexLexerOnce)
#undef yyFlexLexer
#define yyFlexLexer ZFX_FlexLexer // the trick with prefix; no namespace here :(
#include <FlexLexer.h>
#endif

// Scanner method signature is defined by this macro. Original yylex() returns int.
// Sinice Bison 3 uses symbol_type, we must change returned type. We also rename it
// to something sane, since you cannot overload return type.
#undef YY_DECL
#define YY_DECL zeno::ZfxParser::symbol_type zeno::ZfxScanner::get_next_token()

#include "zfxparser.hpp" // this is needed for symbol_type

namespace zeno {

// Forward declare interpreter to avoid include. Header is added inimplementation file.
class ZfxExecute; 
    
class ZfxScanner : public yyFlexLexer {
public:
    ZfxScanner(std::istream& arg_yyin, std::ostream& arg_yyout, ZfxExecute& driver)
        : yyFlexLexer(arg_yyin, arg_yyout)
        , m_driver(driver) {}
    virtual ~ZfxScanner() {}
    virtual zeno::ZfxParser::symbol_type get_next_token();
        
private:
    ZfxExecute& m_driver;
};

}

#endif