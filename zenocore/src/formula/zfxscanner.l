%top{
    #include <stdint.h>
}

%{
    #include <iostream>
    #include <cstdlib>
    #include "zfxscanner.h"
    #include <zeno/formula/zfxexecute.h>
    #include "zfxparser.hpp"
    #include "location.hh"

    static zeno::location loc;    //声明位置实例

    using namespace std;

    // Original yyterminate() macro returns int. Since we're using Bison 3 variants
    // as tokens, we must redefine it to change type from `int` to `Parser::semantic_type`
    #define yyterminate() zeno::ZfxParser::make_END(zeno::location());

    // This will track current scanner location.
    // Action is called when length of the token is known.
    #define YY_USER_ACTION m_driver.increaseLocation(yyleng, yytext);
    
    // !!!WARNING!!!
    // Location API is used, but the location is not initialized, 'cause I'm lazy. When making
    // a token with make_{something} method you can pass detailed token location. Current location
    // is accessible with m_driver.location() method. All puzzle elements are there - just
    // pass location value in every action code block below. I'm going to waste more time writing
    // this excuse than putting this boilerplate below...
    //
    // Location class can be found in location.hh and posistion.hh files. It's just a bit too much
    // boilerplate for this small example. Bummer.
%}

%option nodefault
%option noyywrap
%option c++
%option yyclass="ZfxScanner"
%option prefix="ZFX_"

/*正则式的各种简写，使程序有层次感而显得明晰 */
string  \"[^\n"]+\"
uncompstring \"[^\n"]*
ws      [ \t]+
alpha   [A-Za-z]
dig     [0-9]
/*name    (\$)({alpha}|{dig}|[_.\-/$])*/
num1    {dig}+\.?([eE][-]?{dig}+)?
num2    {dig}*\.{dig}+([eE][-]?{dig}+)?
number  {num1}|{num2}
frame   \$F
fps     \$FPS
pi      \$PI
varname [A-Za-z][A-Za-z_0-9]*
dollar \$
zenvarname \$[A-Za-z_]+
compareop (<|<=|>=|>)
semicolon ;
equalto \=

%%
%{
  // C++ 兼容的词法分析器的规则，step函数把位置的起始值设置为与结束值相等，这样位置就指向了上一个极少的结束位置。
  loc.step();
%}

{number}    {
                return ZfxParser::make_NUMBER(std::strtof(yytext,0),loc); // strtof函数将字符串转换为浮点数
            }

"//".* |
[ \t]   {
            /* 跳过注释和空白符号 */
            // step函数把位置的起始值设置为与结束值相等，这样位置就指向了上一个极少的结束位置。
            // 由于注释和空白符号识别后并不会返回，而前一个step的调用是在上一次yylex返回时，所以此处需要手动更新记号的起始位置
            loc.step();
        }

\n      {
            loc.lines(yyleng);  //使用lines函数来更新位置信息中的符号
            loc.step();
            //return zeno::ZfxParser::make_EOL(zeno::location());
        }

"+"       { return zeno::ZfxParser::make_ADD(loc); }
"-"       { return zeno::ZfxParser::make_SUB(loc); }
"*"       { return zeno::ZfxParser::make_MUL(loc); }
"/"       { return zeno::ZfxParser::make_DIV(loc); }
"("       { return zeno::ZfxParser::make_LPAREN(yytext,loc); }
")"       { return zeno::ZfxParser::make_RPAREN(yytext,loc); }
","       { return zeno::ZfxParser::make_COMMA(loc); }
"?"       { return zeno::ZfxParser::make_QUESTION(yytext,loc); }
":"       { return zeno::ZfxParser::make_COLON(yytext,loc); }
"\."      { return zeno::ZfxParser::make_DOT(yytext,loc); }
"{"       { return zeno::ZfxParser::make_LBRACKET(yytext,loc); }
"}"       { return zeno::ZfxParser::make_RBRACKET(yytext,loc); }
{string}  { return zeno::ZfxParser::make_LITERAL(yytext, loc); }
{uncompstring} { return zeno::ZfxParser::make_UNCOMPSTR(yytext, loc); }
{varname} { return zeno::ZfxParser::make_VARNAME(yytext, loc); }
{semicolon} { return zeno::ZfxParser::make_SEMICOLON(yytext, loc); }
{equalto}   { return zeno::ZfxParser::make_EQUALTO(yytext, loc); }
{dollar}  { return zeno::ZfxParser::make_DOLLAR(yytext, loc); }
{zenvarname} { return zeno::ZfxParser::make_DOLLARVARNAME(yytext, loc); }
{compareop}  { return zeno::ZfxParser::make_COMPARE(yytext, loc); }
<<EOF>>   { return yyterminate(); }
.         {
             cout << "ZfxScanner: unknown character [" << yytext << "]" << endl;
          }

%%
