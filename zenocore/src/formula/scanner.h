#ifndef ZENFORMULA_SCANNER_H
#define ZENFORMULA_SCANNER_H


/**
 * Generated Flex class name is yyFlexLexer by default. If we want to use more flex-generated
 * classes we should name them differently. See scanner.l prefix option.
 * 
 * Unfortunately the implementation relies on this trick with redefining class name
 * with a preprocessor macro. See GNU Flex manual, "Generating C++ Scanners" section
 */
#if ! defined(yyFlexLexerOnce)
#undef yyFlexLexer
#define yyFlexLexer Zeno_FlexLexer // the trick with prefix; no namespace here :(
#include <FlexLexer.h>
#endif

// Scanner method signature is defined by this macro. Original yylex() returns int.
// Sinice Bison 3 uses symbol_type, we must change returned type. We also rename it
// to something sane, since you cannot overload return type.
#undef YY_DECL
#define YY_DECL zeno::Parser::symbol_type zeno::Scanner::get_next_token()

#include "parser.hpp" // this is needed for symbol_type

namespace zeno {

// Forward declare interpreter to avoid include. Header is added inimplementation file.
class Formula; 
    
class Scanner : public yyFlexLexer {
public:
    Scanner(std::istream& arg_yyin, std::ostream& arg_yyout, Formula &driver) 
        : yyFlexLexer(arg_yyin, arg_yyout)
        , m_driver(driver) {}
    virtual ~Scanner() {}
    virtual zeno::Parser::symbol_type get_next_token();
        
private:
    Formula &m_driver;
};

}

#endif