
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

%token <string>RPAREN
%token <string>IDENTIFIER
%token <float>NUMBER
%token EOL
%token END 0
%token FRAME
%token FPS
%token PI
%token COMMA
%token SIN
%token SINH
%token COS
%token COSH
%token ABS
%token REF
%token LITERAL
%token RAND
%token WORD
%token FUNC

%left ADD "+"
%left SUB "-"
%left MUL "*"
%left DIV "/"

//%nonassoc ABS "|"

%nonassoc NEG // 负号具有最高优先级但没有结合性

%left <string>LPAREN

%type <struct node*> exp calclist factor term unaryfunc funccontent funcargs farg// zenvar func 
%type <string> LITERAL FUNC

%start calclist

%%
calclist: %empty{}|calclist exp EOL {
    $$ = $2;
    //driver.setResult($$);
};

exp: factor             { $$ = $1; }
    | exp ADD factor    { $$ = driver.makeNewNode(FOUROPERATIONS, PLUS, $1, $3); }
    | exp SUB factor    { $$ = driver.makeNewNode(FOUROPERATIONS, MINUS, $1, $3); }
    ;

factor: term            { $$ = $1; }
    | factor MUL term   { $$ = driver.makeNewNode(FOUROPERATIONS, MUL, $1, $3); }
    | factor DIV term {
        float wtf = $3->value;
        if (wtf == 0) {
            /*error(wtf, "zero divide");*/
            YYABORT;
        }
        $$ = driver.makeNewNode(FOUROPERATIONS, DIV, $1, $3); 
    }
    | factor MUL   { $$ = driver.makeNewNode(FOUROPERATIONS, MUL, $1, driver.makeNewNumberNode(NAN)); }
    | factor DIV   { $$ = driver.makeNewNode(FOUROPERATIONS, DIV, $1, driver.makeNewNumberNode(NAN)); }
    ;

//zenvar: FRAME { $$ = driver.getFrameNum(); }
//    | FPS { $$ = driver.getFps(); }
//    | PI { $$ = driver.getPI(); }
//    ;

//func: RAND LPAREN RPAREN {
//        std::srand(std::time(nullptr)); // use current time as seed for random generator
//        int random_value = std::rand();
//        $$ = (float)random_value / (RAND_MAX + 1u);
//    }
//    ;

/* 一元函数 */
unaryfunc: SIN LPAREN exp RPAREN { $$ = driver.makeNewNode(UNARY_FUNC, SIN, $3, nullptr); }
    | SIN LPAREN RPAREN { $$ = driver.makeNewNode(UNARY_FUNC, SIN, nullptr, nullptr); }
    | SINH LPAREN exp RPAREN { $$ = driver.makeNewNode(UNARY_FUNC, SINH, $3, nullptr); }
    | SINH LPAREN RPAREN { $$ = driver.makeNewNode(UNARY_FUNC, SINH, nullptr, nullptr); }
    | COS LPAREN exp RPAREN { $$ = driver.makeNewNode(UNARY_FUNC, COS, $3, nullptr); }
    | COS LPAREN RPAREN { $$ = driver.makeNewNode(UNARY_FUNC, COS, nullptr, nullptr); }
    | COSH LPAREN exp RPAREN { $$ = driver.makeNewNode(UNARY_FUNC, COSH, $3, nullptr); }
    | COSH LPAREN RPAREN { $$ = driver.makeNewNode(UNARY_FUNC, COSH, nullptr, nullptr); }
    | ABS LPAREN exp RPAREN { $$ = driver.makeNewNode(UNARY_FUNC, ABS, $3, nullptr); }
    | ABS LPAREN RPAREN { $$ = driver.makeNewNode(UNARY_FUNC, ABS, nullptr, nullptr); }
    //| REF LPAREN LITERAL RPAREN { $$ = driver.callRef($3); }
    ;

funcargs: exp            { $$ = driver.makeNewNode(FUNC_ARG, DEFAULT_OPVAL, $1, nullptr); }
    | funcargs COMMA exp {  }

funccontent: LPAREN funcargs RPAREN { }
    | LPAREN funcargs { }
    | %empty {}

term: NUMBER            { $$ = driver.makeNewNumberNode($1); }
    | WORD              { $$ = driver.makeNewNumberNode(NAN);}
    | LPAREN exp RPAREN { $2->isParenthesisNode = true; $$ = $2; }
    | SUB exp %prec NEG { $2->value = -1 * $2->value; $$ = $2; }
    //| zenvar { $$ = $1; }
    //| func { $$ = $1; }
    | unaryfunc { $$ = $1; }
    | FUNC funccontent {}
    | %empty { $$ = driver.makeEmptyNode(); }
    ;
%%

// Bison expects us to provide implementation - otherwise linker complains
void zeno::Parser::error(const location &loc , const std::string &message) {
        
        // Location should be initialized inside scanner action, but is not in this example.
        // Let's grab location directly from driver class.
	// cout << "Error: " << message << endl << "Location: " << loc << endl;

    cout << "Error: " << message << endl << "Error location: " << driver.location() << endl;
    //报错释放所有节点
    driver.freeSyntaxTree();
}
