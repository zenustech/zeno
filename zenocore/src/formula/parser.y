
%skeleton "lalr1.cc" /* -*- C++ -*- */
%require "3.0"
%defines
%define api.parser.class { Parser }

%define api.token.constructor
%define api.value.type variant
%define parse.assert
%define api.namespace { zeno }
%code requires
{
    #include <iostream>
    #include <string>
    #include <vector>
    #include <cmath>
    #include <cstdlib>
    #include <ctime>
    #include <vector>
    #include <memory>
    #include <zeno/formula/syntax_tree.h>

    using namespace std;

    namespace zeno {
        class Scanner;
        class Formula;
    }
}

// Bison calls yylex() function that must be provided by us to suck tokens
// from the scanner. This block will be placed at the beginning of IMPLEMENTATION file (cpp).
// We define this function here (function! not method).
// This function is called only inside Bison, so we make it static to limit symbol visibility for the linker
// to avoid potential linking conflicts.
%code top
{
    #include <iostream>
    #include "scanner.h"
    #include "parser.hpp"
    #include <zeno/formula/formula.h>
    #include <zeno/formula/syntax_tree.h>
    #include "location.hh"
    
    // yylex() arguments are defined in parser.y
    static zeno::Parser::symbol_type yylex(zeno::Scanner &scanner, zeno::Formula &driver) {
        return scanner.get_next_token();
    }
    
    // you can accomplish the same thing by inlining the code using preprocessor
    // x and y are same as in above static function
    // #define yylex(x, y) scanner.get_next_token()
    
    using namespace zeno;

}

/*定义parser传给scanner的参数*/
%lex-param { zeno::Scanner &scanner }
%lex-param { zeno::Formula &driver }

/*定义driver传给parser的参数*/
%parse-param { zeno::Scanner &scanner }
%parse-param { zeno::Formula &driver }

%locations
%define parse.trace
%define parse.error verbose

/*通过zeno::Parser::make_XXX(loc)给token添加前缀*/
%define api.token.prefix {TOKEN_}

%token RPAREN
%token <string>IDENTIFIER
%token <float>NUMBER
%token EOL
%token END 0
%token FRAME
%token FPS
%token PI
%token COMMA
%token LITERAL
%token FUNC
%token UNCOMPSTR
%token DOLLAR
%token VARNAME

%left ADD "+"
%left SUB "-"
%left MUL "*"
%left DIV "/"

//%nonassoc ABS "|"

%nonassoc NEG // 负号具有最高优先级但没有结合性

%left <string>LPAREN

%type <std::shared_ptr<ZfxASTNode>> exp calclist factor term func-content zenvar parencontent
%type <std::vector<std::shared_ptr<ZfxASTNode>>> funcargs
%type <string> LITERAL FUNC UNCOMPSTR DOLLAR VARNAME RPAREN

%start calclist

%%
calclist: %empty{}|calclist exp EOL {
    $$ = $2;
    driver.setASTResult($$);
};

exp: factor             { $$ = $1; }
    | exp ADD factor    { 
            std::vector<std::shared_ptr<ZfxASTNode>>children({$1, $3});
            $$ = driver.makeNewNode(FOUROPERATIONS, PLUS, children);
            }
    | exp SUB factor    {
            std::vector<std::shared_ptr<ZfxASTNode>>children({$1, $3});
            $$ = driver.makeNewNode(FOUROPERATIONS, MINUS, children);
            }
    ;

factor: term            { $$ = $1; }
    | factor MUL term   { 
            std::vector<std::shared_ptr<ZfxASTNode>>children({$1, $3});
            $$ = driver.makeNewNode(FOUROPERATIONS, MUL, children);
            }
    | factor DIV term {
            std::vector<std::shared_ptr<ZfxASTNode>>children({$1, $3});
            $$ = driver.makeNewNode(FOUROPERATIONS, DIV, children); 
            }
    ;

zenvar: VARNAME { $$ = driver.makeZenVarNode($1); }
    | DOLLAR { $$ = driver.makeZenVarNode(""); };

funcargs: exp            { $$ = std::vector<std::shared_ptr<ZfxASTNode>>({$1}); }
    | funcargs COMMA exp { $1.push_back($3); $$ = $1; }

func-content: LPAREN funcargs RPAREN { 
        $$ = driver.makeNewNode(FUNC, DEFAULT_FUNCVAL, $2);
        $$->isParenthesisNodeComplete = true;
        $$->func_match = Match_Exactly;
    }
    | LPAREN funcargs { 
        $$ = driver.makeNewNode(FUNC, DEFAULT_FUNCVAL, $2);
        $$->isParenthesisNodeComplete = false;
        $$->func_match = Match_LeftPAREN;
    }
    | %empty {
        $$ = driver.makeNewNode(FUNC, DEFAULT_FUNCVAL, {});
        $$->func_match = Match_Nothing;
    };

parencontent: LPAREN exp RPAREN { $$ = $2; $$->paren_match = Match_Exactly; }
    /* 括号不全的情况不予以识别，否则parser无法识别全括号的情况
    | LPAREN exp { driver.debugASTNode($2); $$ = $2; $$->paren_match = Match_LeftPAREN; }
    | LPAREN { $$ = driver.makeEmptyNode(); $$->paren_match = Match_LeftPAREN; };
    */

term: NUMBER            { $$ = driver.makeNewNumberNode($1); }
    | LITERAL           { $$ = driver.makeStringNode($1); }
    | UNCOMPSTR         { $$ = driver.makeQuoteStringNode($1); }
    | parencontent      { $$ = $1; }
    | SUB exp %prec NEG { $$ = driver.makeNegativeNode($2); }
    | zenvar            { $$ = $1; }
    | FUNC func-content  { 
        $$ = $2;
        $$->opVal = DEFAULT_FUNCVAL;
        $$->type = FUNC;
        $$->value = $1;
        $$->isParenthesisNode = true;
    }
    | %empty { /*$$ = driver.makeEmptyNode();*/ }
    ;
%%

// Bison expects us to provide implementation - otherwise linker complains
void zeno::Parser::error(const location &loc , const std::string &message) {
        
        // Location should be initialized inside scanner action, but is not in this example.
        // Let's grab location directly from driver class.
	// cout << "Error: " << message << endl << "Location: " << loc << endl;

    cout << "Error: " << message << endl << "Error location: " << driver.location() << endl;
}
