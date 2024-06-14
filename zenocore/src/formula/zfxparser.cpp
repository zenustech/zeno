// A Bison parser, made by GNU Bison 3.8.2.

// Skeleton implementation for Bison LALR(1) parsers in C++

// Copyright (C) 2002-2015, 2018-2021 Free Software Foundation, Inc.

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

// As a special exception, you may create a larger work that contains
// part or all of the Bison parser skeleton and distribute that work
// under terms of your choice, so long as that work isn't itself a
// parser generator using the skeleton or a modified version thereof
// as a parser skeleton.  Alternatively, if you modify or redistribute
// the parser skeleton itself, you may (at your option) remove this
// special exception, which will cause the skeleton and the resulting
// Bison output files to be licensed under the GNU General Public
// License without this special exception.

// This special exception was added by the Free Software Foundation in
// version 2.2 of Bison.

// DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
// especially those whose name start with YY_ or yy_.  They are
// private implementation details that can be changed or removed.

// "%code top" blocks.
#line 38 "zfxparser.y"

    #include <iostream>
    #include "zfxscanner.h"
    #include "zfxparser.hpp"
    #include <zeno/formula/zfxexecute.h>
    #include <zeno/formula/syntax_tree.h>
    #include "location.hh"

    static zeno::ZfxParser::symbol_type yylex(zeno::ZfxScanner &scanner, zeno::ZfxExecute &driver) {
        return scanner.get_next_token();
    }

    using namespace zeno;

#line 54 "zfxparser.cpp"




#include "zfxparser.hpp"




#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> // FIXME: INFRINGES ON USER NAME SPACE.
#   define YY_(msgid) dgettext ("bison-runtime", msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(msgid) msgid
# endif
#endif


// Whether we are compiled with exception support.
#ifndef YY_EXCEPTIONS
# if defined __GNUC__ && !defined __EXCEPTIONS
#  define YY_EXCEPTIONS 0
# else
#  define YY_EXCEPTIONS 1
# endif
#endif

#define YYRHSLOC(Rhs, K) ((Rhs)[K].location)
/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

# ifndef YYLLOC_DEFAULT
#  define YYLLOC_DEFAULT(Current, Rhs, N)                               \
    do                                                                  \
      if (N)                                                            \
        {                                                               \
          (Current).begin  = YYRHSLOC (Rhs, 1).begin;                   \
          (Current).end    = YYRHSLOC (Rhs, N).end;                     \
        }                                                               \
      else                                                              \
        {                                                               \
          (Current).begin = (Current).end = YYRHSLOC (Rhs, 0).end;      \
        }                                                               \
    while (false)
# endif


// Enable debugging if requested.
#if YYDEBUG

// A pseudo ostream that takes yydebug_ into account.
# define YYCDEBUG if (yydebug_) (*yycdebug_)

# define YY_SYMBOL_PRINT(Title, Symbol)         \
  do {                                          \
    if (yydebug_)                               \
    {                                           \
      *yycdebug_ << Title << ' ';               \
      yy_print_ (*yycdebug_, Symbol);           \
      *yycdebug_ << '\n';                       \
    }                                           \
  } while (false)

# define YY_REDUCE_PRINT(Rule)          \
  do {                                  \
    if (yydebug_)                       \
      yy_reduce_print_ (Rule);          \
  } while (false)

# define YY_STACK_PRINT()               \
  do {                                  \
    if (yydebug_)                       \
      yy_stack_print_ ();                \
  } while (false)

#else // !YYDEBUG

# define YYCDEBUG if (false) std::cerr
# define YY_SYMBOL_PRINT(Title, Symbol)  YY_USE (Symbol)
# define YY_REDUCE_PRINT(Rule)           static_cast<void> (0)
# define YY_STACK_PRINT()                static_cast<void> (0)

#endif // !YYDEBUG

#define yyerrok         (yyerrstatus_ = 0)
#define yyclearin       (yyla.clear ())

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab
#define YYRECOVERING()  (!!yyerrstatus_)

#line 10 "zfxparser.y"
namespace  zeno  {
#line 154 "zfxparser.cpp"

  /// Build a parser object.
   ZfxParser :: ZfxParser  (zeno::ZfxScanner &scanner_yyarg, zeno::ZfxExecute &driver_yyarg)
#if YYDEBUG
    : yydebug_ (false),
      yycdebug_ (&std::cerr),
#else
    :
#endif
      scanner (scanner_yyarg),
      driver (driver_yyarg)
  {}

   ZfxParser ::~ ZfxParser  ()
  {}

   ZfxParser ::syntax_error::~syntax_error () YY_NOEXCEPT YY_NOTHROW
  {}

  /*---------.
  | symbol.  |
  `---------*/



  // by_state.
   ZfxParser ::by_state::by_state () YY_NOEXCEPT
    : state (empty_state)
  {}

   ZfxParser ::by_state::by_state (const by_state& that) YY_NOEXCEPT
    : state (that.state)
  {}

  void
   ZfxParser ::by_state::clear () YY_NOEXCEPT
  {
    state = empty_state;
  }

  void
   ZfxParser ::by_state::move (by_state& that)
  {
    state = that.state;
    that.clear ();
  }

   ZfxParser ::by_state::by_state (state_type s) YY_NOEXCEPT
    : state (s)
  {}

   ZfxParser ::symbol_kind_type
   ZfxParser ::by_state::kind () const YY_NOEXCEPT
  {
    if (state == empty_state)
      return symbol_kind::S_YYEMPTY;
    else
      return YY_CAST (symbol_kind_type, yystos_[+state]);
  }

   ZfxParser ::stack_symbol_type::stack_symbol_type ()
  {}

   ZfxParser ::stack_symbol_type::stack_symbol_type (YY_RVREF (stack_symbol_type) that)
    : super_type (YY_MOVE (that.state), YY_MOVE (that.location))
  {
    switch (that.kind ())
    {
      case symbol_kind::S_TRUE: // TRUE
      case symbol_kind::S_FALSE: // FALSE
      case symbol_kind::S_61_bool_stmt: // bool-stmt
      case symbol_kind::S_67_array_mark: // array-mark
        value.YY_MOVE_OR_COPY< bool > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_NUMBER: // NUMBER
        value.YY_MOVE_OR_COPY< float > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_60_assign_op: // assign-op
        value.YY_MOVE_OR_COPY< operatorVals > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_56_zfx_program: // zfx-program
      case symbol_kind::S_57_multi_statements: // multi-statements
      case symbol_kind::S_58_general_statement: // general-statement
      case symbol_kind::S_59_array_or_exp: // array-or-exp
      case symbol_kind::S_62_assign_statement: // assign-statement
      case symbol_kind::S_63_jump_statement: // jump-statement
      case symbol_kind::S_arrcontent: // arrcontent
      case symbol_kind::S_66_array_stmt: // array-stmt
      case symbol_kind::S_68_only_declare: // only-declare
      case symbol_kind::S_69_declare_statement: // declare-statement
      case symbol_kind::S_70_code_block: // code-block
      case symbol_kind::S_71_if_statement: // if-statement
      case symbol_kind::S_72_for_begin: // for-begin
      case symbol_kind::S_73_for_condition: // for-condition
      case symbol_kind::S_74_for_step: // for-step
      case symbol_kind::S_76_loop_statement: // loop-statement
      case symbol_kind::S_77_exp_statement: // exp-statement
      case symbol_kind::S_compareexp: // compareexp
      case symbol_kind::S_factor: // factor
      case symbol_kind::S_zenvar: // zenvar
      case symbol_kind::S_82_func_content: // func-content
      case symbol_kind::S_term: // term
        value.YY_MOVE_OR_COPY< std::shared_ptr<ZfxASTNode> > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_arrcontents: // arrcontents
      case symbol_kind::S_75_foreach_step: // foreach-step
      case symbol_kind::S_funcargs: // funcargs
        value.YY_MOVE_OR_COPY< std::vector<std::shared_ptr<ZfxASTNode>> > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_RPAREN: // RPAREN
      case symbol_kind::S_IDENTIFIER: // IDENTIFIER
      case symbol_kind::S_LITERAL: // LITERAL
      case symbol_kind::S_UNCOMPSTR: // UNCOMPSTR
      case symbol_kind::S_DOLLAR: // DOLLAR
      case symbol_kind::S_DOLLARVARNAME: // DOLLARVARNAME
      case symbol_kind::S_COMPARE: // COMPARE
      case symbol_kind::S_QUESTION: // QUESTION
      case symbol_kind::S_COLON: // COLON
      case symbol_kind::S_ZFXVAR: // ZFXVAR
      case symbol_kind::S_LBRACKET: // LBRACKET
      case symbol_kind::S_RBRACKET: // RBRACKET
      case symbol_kind::S_DOT: // DOT
      case symbol_kind::S_VARNAME: // VARNAME
      case symbol_kind::S_SEMICOLON: // SEMICOLON
      case symbol_kind::S_EQUALTO: // EQUALTO
      case symbol_kind::S_IF: // IF
      case symbol_kind::S_FOR: // FOR
      case symbol_kind::S_WHILE: // WHILE
      case symbol_kind::S_AUTOINC: // AUTOINC
      case symbol_kind::S_AUTODEC: // AUTODEC
      case symbol_kind::S_LSQBRACKET: // LSQBRACKET
      case symbol_kind::S_RSQBRACKET: // RSQBRACKET
      case symbol_kind::S_ADDASSIGN: // ADDASSIGN
      case symbol_kind::S_MULASSIGN: // MULASSIGN
      case symbol_kind::S_SUBASSIGN: // SUBASSIGN
      case symbol_kind::S_DIVASSIGN: // DIVASSIGN
      case symbol_kind::S_RETURN: // RETURN
      case symbol_kind::S_CONTINUE: // CONTINUE
      case symbol_kind::S_BREAK: // BREAK
      case symbol_kind::S_TYPE: // TYPE
      case symbol_kind::S_ATTRAT: // ATTRAT
      case symbol_kind::S_FOREACH: // FOREACH
      case symbol_kind::S_DO: // DO
      case symbol_kind::S_LPAREN: // LPAREN
        value.YY_MOVE_OR_COPY< string > (YY_MOVE (that.value));
        break;

      default:
        break;
    }

#if 201103L <= YY_CPLUSPLUS
    // that is emptied.
    that.state = empty_state;
#endif
  }

   ZfxParser ::stack_symbol_type::stack_symbol_type (state_type s, YY_MOVE_REF (symbol_type) that)
    : super_type (s, YY_MOVE (that.location))
  {
    switch (that.kind ())
    {
      case symbol_kind::S_TRUE: // TRUE
      case symbol_kind::S_FALSE: // FALSE
      case symbol_kind::S_61_bool_stmt: // bool-stmt
      case symbol_kind::S_67_array_mark: // array-mark
        value.move< bool > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_NUMBER: // NUMBER
        value.move< float > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_60_assign_op: // assign-op
        value.move< operatorVals > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_56_zfx_program: // zfx-program
      case symbol_kind::S_57_multi_statements: // multi-statements
      case symbol_kind::S_58_general_statement: // general-statement
      case symbol_kind::S_59_array_or_exp: // array-or-exp
      case symbol_kind::S_62_assign_statement: // assign-statement
      case symbol_kind::S_63_jump_statement: // jump-statement
      case symbol_kind::S_arrcontent: // arrcontent
      case symbol_kind::S_66_array_stmt: // array-stmt
      case symbol_kind::S_68_only_declare: // only-declare
      case symbol_kind::S_69_declare_statement: // declare-statement
      case symbol_kind::S_70_code_block: // code-block
      case symbol_kind::S_71_if_statement: // if-statement
      case symbol_kind::S_72_for_begin: // for-begin
      case symbol_kind::S_73_for_condition: // for-condition
      case symbol_kind::S_74_for_step: // for-step
      case symbol_kind::S_76_loop_statement: // loop-statement
      case symbol_kind::S_77_exp_statement: // exp-statement
      case symbol_kind::S_compareexp: // compareexp
      case symbol_kind::S_factor: // factor
      case symbol_kind::S_zenvar: // zenvar
      case symbol_kind::S_82_func_content: // func-content
      case symbol_kind::S_term: // term
        value.move< std::shared_ptr<ZfxASTNode> > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_arrcontents: // arrcontents
      case symbol_kind::S_75_foreach_step: // foreach-step
      case symbol_kind::S_funcargs: // funcargs
        value.move< std::vector<std::shared_ptr<ZfxASTNode>> > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_RPAREN: // RPAREN
      case symbol_kind::S_IDENTIFIER: // IDENTIFIER
      case symbol_kind::S_LITERAL: // LITERAL
      case symbol_kind::S_UNCOMPSTR: // UNCOMPSTR
      case symbol_kind::S_DOLLAR: // DOLLAR
      case symbol_kind::S_DOLLARVARNAME: // DOLLARVARNAME
      case symbol_kind::S_COMPARE: // COMPARE
      case symbol_kind::S_QUESTION: // QUESTION
      case symbol_kind::S_COLON: // COLON
      case symbol_kind::S_ZFXVAR: // ZFXVAR
      case symbol_kind::S_LBRACKET: // LBRACKET
      case symbol_kind::S_RBRACKET: // RBRACKET
      case symbol_kind::S_DOT: // DOT
      case symbol_kind::S_VARNAME: // VARNAME
      case symbol_kind::S_SEMICOLON: // SEMICOLON
      case symbol_kind::S_EQUALTO: // EQUALTO
      case symbol_kind::S_IF: // IF
      case symbol_kind::S_FOR: // FOR
      case symbol_kind::S_WHILE: // WHILE
      case symbol_kind::S_AUTOINC: // AUTOINC
      case symbol_kind::S_AUTODEC: // AUTODEC
      case symbol_kind::S_LSQBRACKET: // LSQBRACKET
      case symbol_kind::S_RSQBRACKET: // RSQBRACKET
      case symbol_kind::S_ADDASSIGN: // ADDASSIGN
      case symbol_kind::S_MULASSIGN: // MULASSIGN
      case symbol_kind::S_SUBASSIGN: // SUBASSIGN
      case symbol_kind::S_DIVASSIGN: // DIVASSIGN
      case symbol_kind::S_RETURN: // RETURN
      case symbol_kind::S_CONTINUE: // CONTINUE
      case symbol_kind::S_BREAK: // BREAK
      case symbol_kind::S_TYPE: // TYPE
      case symbol_kind::S_ATTRAT: // ATTRAT
      case symbol_kind::S_FOREACH: // FOREACH
      case symbol_kind::S_DO: // DO
      case symbol_kind::S_LPAREN: // LPAREN
        value.move< string > (YY_MOVE (that.value));
        break;

      default:
        break;
    }

    // that is emptied.
    that.kind_ = symbol_kind::S_YYEMPTY;
  }

#if YY_CPLUSPLUS < 201103L
   ZfxParser ::stack_symbol_type&
   ZfxParser ::stack_symbol_type::operator= (const stack_symbol_type& that)
  {
    state = that.state;
    switch (that.kind ())
    {
      case symbol_kind::S_TRUE: // TRUE
      case symbol_kind::S_FALSE: // FALSE
      case symbol_kind::S_61_bool_stmt: // bool-stmt
      case symbol_kind::S_67_array_mark: // array-mark
        value.copy< bool > (that.value);
        break;

      case symbol_kind::S_NUMBER: // NUMBER
        value.copy< float > (that.value);
        break;

      case symbol_kind::S_60_assign_op: // assign-op
        value.copy< operatorVals > (that.value);
        break;

      case symbol_kind::S_56_zfx_program: // zfx-program
      case symbol_kind::S_57_multi_statements: // multi-statements
      case symbol_kind::S_58_general_statement: // general-statement
      case symbol_kind::S_59_array_or_exp: // array-or-exp
      case symbol_kind::S_62_assign_statement: // assign-statement
      case symbol_kind::S_63_jump_statement: // jump-statement
      case symbol_kind::S_arrcontent: // arrcontent
      case symbol_kind::S_66_array_stmt: // array-stmt
      case symbol_kind::S_68_only_declare: // only-declare
      case symbol_kind::S_69_declare_statement: // declare-statement
      case symbol_kind::S_70_code_block: // code-block
      case symbol_kind::S_71_if_statement: // if-statement
      case symbol_kind::S_72_for_begin: // for-begin
      case symbol_kind::S_73_for_condition: // for-condition
      case symbol_kind::S_74_for_step: // for-step
      case symbol_kind::S_76_loop_statement: // loop-statement
      case symbol_kind::S_77_exp_statement: // exp-statement
      case symbol_kind::S_compareexp: // compareexp
      case symbol_kind::S_factor: // factor
      case symbol_kind::S_zenvar: // zenvar
      case symbol_kind::S_82_func_content: // func-content
      case symbol_kind::S_term: // term
        value.copy< std::shared_ptr<ZfxASTNode> > (that.value);
        break;

      case symbol_kind::S_arrcontents: // arrcontents
      case symbol_kind::S_75_foreach_step: // foreach-step
      case symbol_kind::S_funcargs: // funcargs
        value.copy< std::vector<std::shared_ptr<ZfxASTNode>> > (that.value);
        break;

      case symbol_kind::S_RPAREN: // RPAREN
      case symbol_kind::S_IDENTIFIER: // IDENTIFIER
      case symbol_kind::S_LITERAL: // LITERAL
      case symbol_kind::S_UNCOMPSTR: // UNCOMPSTR
      case symbol_kind::S_DOLLAR: // DOLLAR
      case symbol_kind::S_DOLLARVARNAME: // DOLLARVARNAME
      case symbol_kind::S_COMPARE: // COMPARE
      case symbol_kind::S_QUESTION: // QUESTION
      case symbol_kind::S_COLON: // COLON
      case symbol_kind::S_ZFXVAR: // ZFXVAR
      case symbol_kind::S_LBRACKET: // LBRACKET
      case symbol_kind::S_RBRACKET: // RBRACKET
      case symbol_kind::S_DOT: // DOT
      case symbol_kind::S_VARNAME: // VARNAME
      case symbol_kind::S_SEMICOLON: // SEMICOLON
      case symbol_kind::S_EQUALTO: // EQUALTO
      case symbol_kind::S_IF: // IF
      case symbol_kind::S_FOR: // FOR
      case symbol_kind::S_WHILE: // WHILE
      case symbol_kind::S_AUTOINC: // AUTOINC
      case symbol_kind::S_AUTODEC: // AUTODEC
      case symbol_kind::S_LSQBRACKET: // LSQBRACKET
      case symbol_kind::S_RSQBRACKET: // RSQBRACKET
      case symbol_kind::S_ADDASSIGN: // ADDASSIGN
      case symbol_kind::S_MULASSIGN: // MULASSIGN
      case symbol_kind::S_SUBASSIGN: // SUBASSIGN
      case symbol_kind::S_DIVASSIGN: // DIVASSIGN
      case symbol_kind::S_RETURN: // RETURN
      case symbol_kind::S_CONTINUE: // CONTINUE
      case symbol_kind::S_BREAK: // BREAK
      case symbol_kind::S_TYPE: // TYPE
      case symbol_kind::S_ATTRAT: // ATTRAT
      case symbol_kind::S_FOREACH: // FOREACH
      case symbol_kind::S_DO: // DO
      case symbol_kind::S_LPAREN: // LPAREN
        value.copy< string > (that.value);
        break;

      default:
        break;
    }

    location = that.location;
    return *this;
  }

   ZfxParser ::stack_symbol_type&
   ZfxParser ::stack_symbol_type::operator= (stack_symbol_type& that)
  {
    state = that.state;
    switch (that.kind ())
    {
      case symbol_kind::S_TRUE: // TRUE
      case symbol_kind::S_FALSE: // FALSE
      case symbol_kind::S_61_bool_stmt: // bool-stmt
      case symbol_kind::S_67_array_mark: // array-mark
        value.move< bool > (that.value);
        break;

      case symbol_kind::S_NUMBER: // NUMBER
        value.move< float > (that.value);
        break;

      case symbol_kind::S_60_assign_op: // assign-op
        value.move< operatorVals > (that.value);
        break;

      case symbol_kind::S_56_zfx_program: // zfx-program
      case symbol_kind::S_57_multi_statements: // multi-statements
      case symbol_kind::S_58_general_statement: // general-statement
      case symbol_kind::S_59_array_or_exp: // array-or-exp
      case symbol_kind::S_62_assign_statement: // assign-statement
      case symbol_kind::S_63_jump_statement: // jump-statement
      case symbol_kind::S_arrcontent: // arrcontent
      case symbol_kind::S_66_array_stmt: // array-stmt
      case symbol_kind::S_68_only_declare: // only-declare
      case symbol_kind::S_69_declare_statement: // declare-statement
      case symbol_kind::S_70_code_block: // code-block
      case symbol_kind::S_71_if_statement: // if-statement
      case symbol_kind::S_72_for_begin: // for-begin
      case symbol_kind::S_73_for_condition: // for-condition
      case symbol_kind::S_74_for_step: // for-step
      case symbol_kind::S_76_loop_statement: // loop-statement
      case symbol_kind::S_77_exp_statement: // exp-statement
      case symbol_kind::S_compareexp: // compareexp
      case symbol_kind::S_factor: // factor
      case symbol_kind::S_zenvar: // zenvar
      case symbol_kind::S_82_func_content: // func-content
      case symbol_kind::S_term: // term
        value.move< std::shared_ptr<ZfxASTNode> > (that.value);
        break;

      case symbol_kind::S_arrcontents: // arrcontents
      case symbol_kind::S_75_foreach_step: // foreach-step
      case symbol_kind::S_funcargs: // funcargs
        value.move< std::vector<std::shared_ptr<ZfxASTNode>> > (that.value);
        break;

      case symbol_kind::S_RPAREN: // RPAREN
      case symbol_kind::S_IDENTIFIER: // IDENTIFIER
      case symbol_kind::S_LITERAL: // LITERAL
      case symbol_kind::S_UNCOMPSTR: // UNCOMPSTR
      case symbol_kind::S_DOLLAR: // DOLLAR
      case symbol_kind::S_DOLLARVARNAME: // DOLLARVARNAME
      case symbol_kind::S_COMPARE: // COMPARE
      case symbol_kind::S_QUESTION: // QUESTION
      case symbol_kind::S_COLON: // COLON
      case symbol_kind::S_ZFXVAR: // ZFXVAR
      case symbol_kind::S_LBRACKET: // LBRACKET
      case symbol_kind::S_RBRACKET: // RBRACKET
      case symbol_kind::S_DOT: // DOT
      case symbol_kind::S_VARNAME: // VARNAME
      case symbol_kind::S_SEMICOLON: // SEMICOLON
      case symbol_kind::S_EQUALTO: // EQUALTO
      case symbol_kind::S_IF: // IF
      case symbol_kind::S_FOR: // FOR
      case symbol_kind::S_WHILE: // WHILE
      case symbol_kind::S_AUTOINC: // AUTOINC
      case symbol_kind::S_AUTODEC: // AUTODEC
      case symbol_kind::S_LSQBRACKET: // LSQBRACKET
      case symbol_kind::S_RSQBRACKET: // RSQBRACKET
      case symbol_kind::S_ADDASSIGN: // ADDASSIGN
      case symbol_kind::S_MULASSIGN: // MULASSIGN
      case symbol_kind::S_SUBASSIGN: // SUBASSIGN
      case symbol_kind::S_DIVASSIGN: // DIVASSIGN
      case symbol_kind::S_RETURN: // RETURN
      case symbol_kind::S_CONTINUE: // CONTINUE
      case symbol_kind::S_BREAK: // BREAK
      case symbol_kind::S_TYPE: // TYPE
      case symbol_kind::S_ATTRAT: // ATTRAT
      case symbol_kind::S_FOREACH: // FOREACH
      case symbol_kind::S_DO: // DO
      case symbol_kind::S_LPAREN: // LPAREN
        value.move< string > (that.value);
        break;

      default:
        break;
    }

    location = that.location;
    // that is emptied.
    that.state = empty_state;
    return *this;
  }
#endif

  template <typename Base>
  void
   ZfxParser ::yy_destroy_ (const char* yymsg, basic_symbol<Base>& yysym) const
  {
    if (yymsg)
      YY_SYMBOL_PRINT (yymsg, yysym);
  }

#if YYDEBUG
  template <typename Base>
  void
   ZfxParser ::yy_print_ (std::ostream& yyo, const basic_symbol<Base>& yysym) const
  {
    std::ostream& yyoutput = yyo;
    YY_USE (yyoutput);
    if (yysym.empty ())
      yyo << "empty symbol";
    else
      {
        symbol_kind_type yykind = yysym.kind ();
        yyo << (yykind < YYNTOKENS ? "token" : "nterm")
            << ' ' << yysym.name () << " ("
            << yysym.location << ": ";
        YY_USE (yykind);
        yyo << ')';
      }
  }
#endif

  void
   ZfxParser ::yypush_ (const char* m, YY_MOVE_REF (stack_symbol_type) sym)
  {
    if (m)
      YY_SYMBOL_PRINT (m, sym);
    yystack_.push (YY_MOVE (sym));
  }

  void
   ZfxParser ::yypush_ (const char* m, state_type s, YY_MOVE_REF (symbol_type) sym)
  {
#if 201103L <= YY_CPLUSPLUS
    yypush_ (m, stack_symbol_type (s, std::move (sym)));
#else
    stack_symbol_type ss (s, sym);
    yypush_ (m, ss);
#endif
  }

  void
   ZfxParser ::yypop_ (int n) YY_NOEXCEPT
  {
    yystack_.pop (n);
  }

#if YYDEBUG
  std::ostream&
   ZfxParser ::debug_stream () const
  {
    return *yycdebug_;
  }

  void
   ZfxParser ::set_debug_stream (std::ostream& o)
  {
    yycdebug_ = &o;
  }


   ZfxParser ::debug_level_type
   ZfxParser ::debug_level () const
  {
    return yydebug_;
  }

  void
   ZfxParser ::set_debug_level (debug_level_type l)
  {
    yydebug_ = l;
  }
#endif // YYDEBUG

   ZfxParser ::state_type
   ZfxParser ::yy_lr_goto_state_ (state_type yystate, int yysym)
  {
    int yyr = yypgoto_[yysym - YYNTOKENS] + yystate;
    if (0 <= yyr && yyr <= yylast_ && yycheck_[yyr] == yystate)
      return yytable_[yyr];
    else
      return yydefgoto_[yysym - YYNTOKENS];
  }

  bool
   ZfxParser ::yy_pact_value_is_default_ (int yyvalue) YY_NOEXCEPT
  {
    return yyvalue == yypact_ninf_;
  }

  bool
   ZfxParser ::yy_table_value_is_error_ (int yyvalue) YY_NOEXCEPT
  {
    return yyvalue == yytable_ninf_;
  }

  int
   ZfxParser ::operator() ()
  {
    return parse ();
  }

  int
   ZfxParser ::parse ()
  {
    int yyn;
    /// Length of the RHS of the rule being reduced.
    int yylen = 0;

    // Error handling.
    int yynerrs_ = 0;
    int yyerrstatus_ = 0;

    /// The lookahead symbol.
    symbol_type yyla;

    /// The locations where the error started and ended.
    stack_symbol_type yyerror_range[3];

    /// The return value of parse ().
    int yyresult;

#if YY_EXCEPTIONS
    try
#endif // YY_EXCEPTIONS
      {
    YYCDEBUG << "Starting parse\n";


    /* Initialize the stack.  The initial state will be set in
       yynewstate, since the latter expects the semantical and the
       location values to have been already stored, initialize these
       stacks with a primary value.  */
    yystack_.clear ();
    yypush_ (YY_NULLPTR, 0, YY_MOVE (yyla));

  /*-----------------------------------------------.
  | yynewstate -- push a new symbol on the stack.  |
  `-----------------------------------------------*/
  yynewstate:
    YYCDEBUG << "Entering state " << int (yystack_[0].state) << '\n';
    YY_STACK_PRINT ();

    // Accept?
    if (yystack_[0].state == yyfinal_)
      YYACCEPT;

    goto yybackup;


  /*-----------.
  | yybackup.  |
  `-----------*/
  yybackup:
    // Try to take a decision without lookahead.
    yyn = yypact_[+yystack_[0].state];
    if (yy_pact_value_is_default_ (yyn))
      goto yydefault;

    // Read a lookahead token.
    if (yyla.empty ())
      {
        YYCDEBUG << "Reading a token\n";
#if YY_EXCEPTIONS
        try
#endif // YY_EXCEPTIONS
          {
            symbol_type yylookahead (yylex (scanner, driver));
            yyla.move (yylookahead);
          }
#if YY_EXCEPTIONS
        catch (const syntax_error& yyexc)
          {
            YYCDEBUG << "Caught exception: " << yyexc.what() << '\n';
            error (yyexc);
            goto yyerrlab1;
          }
#endif // YY_EXCEPTIONS
      }
    YY_SYMBOL_PRINT ("Next token is", yyla);

    if (yyla.kind () == symbol_kind::S_YYerror)
    {
      // The scanner already issued an error message, process directly
      // to error recovery.  But do not keep the error token as
      // lookahead, it is too special and may lead us to an endless
      // loop in error recovery. */
      yyla.kind_ = symbol_kind::S_YYUNDEF;
      goto yyerrlab1;
    }

    /* If the proper action on seeing token YYLA.TYPE is to reduce or
       to detect an error, take that action.  */
    yyn += yyla.kind ();
    if (yyn < 0 || yylast_ < yyn || yycheck_[yyn] != yyla.kind ())
      {
        goto yydefault;
      }

    // Reduce or error.
    yyn = yytable_[yyn];
    if (yyn <= 0)
      {
        if (yy_table_value_is_error_ (yyn))
          goto yyerrlab;
        yyn = -yyn;
        goto yyreduce;
      }

    // Count tokens shifted since error; after three, turn off error status.
    if (yyerrstatus_)
      --yyerrstatus_;

    // Shift the lookahead token.
    yypush_ ("Shifting", state_type (yyn), YY_MOVE (yyla));
    goto yynewstate;


  /*-----------------------------------------------------------.
  | yydefault -- do the default action for the current state.  |
  `-----------------------------------------------------------*/
  yydefault:
    yyn = yydefact_[+yystack_[0].state];
    if (yyn == 0)
      goto yyerrlab;
    goto yyreduce;


  /*-----------------------------.
  | yyreduce -- do a reduction.  |
  `-----------------------------*/
  yyreduce:
    yylen = yyr2_[yyn];
    {
      stack_symbol_type yylhs;
      yylhs.state = yy_lr_goto_state_ (yystack_[yylen].state, yyr1_[yyn]);
      /* Variants are always initialized to an empty instance of the
         correct type. The default '$$ = $1' action is NOT applied
         when using variants.  */
      switch (yyr1_[yyn])
    {
      case symbol_kind::S_TRUE: // TRUE
      case symbol_kind::S_FALSE: // FALSE
      case symbol_kind::S_61_bool_stmt: // bool-stmt
      case symbol_kind::S_67_array_mark: // array-mark
        yylhs.value.emplace< bool > ();
        break;

      case symbol_kind::S_NUMBER: // NUMBER
        yylhs.value.emplace< float > ();
        break;

      case symbol_kind::S_60_assign_op: // assign-op
        yylhs.value.emplace< operatorVals > ();
        break;

      case symbol_kind::S_56_zfx_program: // zfx-program
      case symbol_kind::S_57_multi_statements: // multi-statements
      case symbol_kind::S_58_general_statement: // general-statement
      case symbol_kind::S_59_array_or_exp: // array-or-exp
      case symbol_kind::S_62_assign_statement: // assign-statement
      case symbol_kind::S_63_jump_statement: // jump-statement
      case symbol_kind::S_arrcontent: // arrcontent
      case symbol_kind::S_66_array_stmt: // array-stmt
      case symbol_kind::S_68_only_declare: // only-declare
      case symbol_kind::S_69_declare_statement: // declare-statement
      case symbol_kind::S_70_code_block: // code-block
      case symbol_kind::S_71_if_statement: // if-statement
      case symbol_kind::S_72_for_begin: // for-begin
      case symbol_kind::S_73_for_condition: // for-condition
      case symbol_kind::S_74_for_step: // for-step
      case symbol_kind::S_76_loop_statement: // loop-statement
      case symbol_kind::S_77_exp_statement: // exp-statement
      case symbol_kind::S_compareexp: // compareexp
      case symbol_kind::S_factor: // factor
      case symbol_kind::S_zenvar: // zenvar
      case symbol_kind::S_82_func_content: // func-content
      case symbol_kind::S_term: // term
        yylhs.value.emplace< std::shared_ptr<ZfxASTNode> > ();
        break;

      case symbol_kind::S_arrcontents: // arrcontents
      case symbol_kind::S_75_foreach_step: // foreach-step
      case symbol_kind::S_funcargs: // funcargs
        yylhs.value.emplace< std::vector<std::shared_ptr<ZfxASTNode>> > ();
        break;

      case symbol_kind::S_RPAREN: // RPAREN
      case symbol_kind::S_IDENTIFIER: // IDENTIFIER
      case symbol_kind::S_LITERAL: // LITERAL
      case symbol_kind::S_UNCOMPSTR: // UNCOMPSTR
      case symbol_kind::S_DOLLAR: // DOLLAR
      case symbol_kind::S_DOLLARVARNAME: // DOLLARVARNAME
      case symbol_kind::S_COMPARE: // COMPARE
      case symbol_kind::S_QUESTION: // QUESTION
      case symbol_kind::S_COLON: // COLON
      case symbol_kind::S_ZFXVAR: // ZFXVAR
      case symbol_kind::S_LBRACKET: // LBRACKET
      case symbol_kind::S_RBRACKET: // RBRACKET
      case symbol_kind::S_DOT: // DOT
      case symbol_kind::S_VARNAME: // VARNAME
      case symbol_kind::S_SEMICOLON: // SEMICOLON
      case symbol_kind::S_EQUALTO: // EQUALTO
      case symbol_kind::S_IF: // IF
      case symbol_kind::S_FOR: // FOR
      case symbol_kind::S_WHILE: // WHILE
      case symbol_kind::S_AUTOINC: // AUTOINC
      case symbol_kind::S_AUTODEC: // AUTODEC
      case symbol_kind::S_LSQBRACKET: // LSQBRACKET
      case symbol_kind::S_RSQBRACKET: // RSQBRACKET
      case symbol_kind::S_ADDASSIGN: // ADDASSIGN
      case symbol_kind::S_MULASSIGN: // MULASSIGN
      case symbol_kind::S_SUBASSIGN: // SUBASSIGN
      case symbol_kind::S_DIVASSIGN: // DIVASSIGN
      case symbol_kind::S_RETURN: // RETURN
      case symbol_kind::S_CONTINUE: // CONTINUE
      case symbol_kind::S_BREAK: // BREAK
      case symbol_kind::S_TYPE: // TYPE
      case symbol_kind::S_ATTRAT: // ATTRAT
      case symbol_kind::S_FOREACH: // FOREACH
      case symbol_kind::S_DO: // DO
      case symbol_kind::S_LPAREN: // LPAREN
        yylhs.value.emplace< string > ();
        break;

      default:
        break;
    }


      // Default location.
      {
        stack_type::slice range (yystack_, yylen);
        YYLLOC_DEFAULT (yylhs.location, range, yylen);
        yyerror_range[1].location = yylhs.location;
      }

      // Perform the reduction.
      YY_REDUCE_PRINT (yyn);
#if YY_EXCEPTIONS
      try
#endif // YY_EXCEPTIONS
        {
          switch (yyn)
            {
  case 2: // zfx-program: END
#line 131 "zfxparser.y"
                 {
            std::cout << "END" << std::endl;
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(CODEBLOCK, DEFAULT_FUNCVAL, {});
            driver.setASTResult(yylhs.value.as < std::shared_ptr<ZfxASTNode> > ());
        }
#line 973 "zfxparser.cpp"
    break;

  case 3: // zfx-program: multi-statements zfx-program
#line 136 "zfxparser.y"
                                   {
            addChild(yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[1].value.as < std::shared_ptr<ZfxASTNode> > ());
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ();
        }
#line 982 "zfxparser.cpp"
    break;

  case 4: // multi-statements: %empty
#line 142 "zfxparser.y"
                         {
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(CODEBLOCK, DEFAULT_FUNCVAL, {});
        }
#line 990 "zfxparser.cpp"
    break;

  case 5: // multi-statements: general-statement multi-statements
#line 145 "zfxparser.y"
                                         {
            addChild(yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[1].value.as < std::shared_ptr<ZfxASTNode> > ());
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ();
        }
#line 999 "zfxparser.cpp"
    break;

  case 6: // general-statement: declare-statement SEMICOLON
#line 151 "zfxparser.y"
                                               { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[1].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1005 "zfxparser.cpp"
    break;

  case 7: // general-statement: assign-statement SEMICOLON
#line 152 "zfxparser.y"
                                 { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[1].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1011 "zfxparser.cpp"
    break;

  case 8: // general-statement: if-statement
#line 153 "zfxparser.y"
                   { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1017 "zfxparser.cpp"
    break;

  case 9: // general-statement: loop-statement
#line 154 "zfxparser.y"
                     { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1023 "zfxparser.cpp"
    break;

  case 10: // general-statement: jump-statement SEMICOLON
#line 155 "zfxparser.y"
                               { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[1].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1029 "zfxparser.cpp"
    break;

  case 11: // general-statement: exp-statement SEMICOLON
#line 156 "zfxparser.y"
                              { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[1].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1035 "zfxparser.cpp"
    break;

  case 12: // array-or-exp: exp-statement
#line 159 "zfxparser.y"
                            { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1041 "zfxparser.cpp"
    break;

  case 13: // array-or-exp: array-stmt
#line 160 "zfxparser.y"
                 { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1047 "zfxparser.cpp"
    break;

  case 14: // assign-op: EQUALTO
#line 163 "zfxparser.y"
                   { yylhs.value.as < operatorVals > () = AssignTo; }
#line 1053 "zfxparser.cpp"
    break;

  case 15: // assign-op: ADDASSIGN
#line 164 "zfxparser.y"
                { yylhs.value.as < operatorVals > () = AddAssign; }
#line 1059 "zfxparser.cpp"
    break;

  case 16: // assign-op: MULASSIGN
#line 165 "zfxparser.y"
                { yylhs.value.as < operatorVals > () = MulAssign; }
#line 1065 "zfxparser.cpp"
    break;

  case 17: // assign-op: SUBASSIGN
#line 166 "zfxparser.y"
                { yylhs.value.as < operatorVals > () = SubAssign; }
#line 1071 "zfxparser.cpp"
    break;

  case 18: // assign-op: DIVASSIGN
#line 167 "zfxparser.y"
                { yylhs.value.as < operatorVals > () = DivAssign; }
#line 1077 "zfxparser.cpp"
    break;

  case 19: // bool-stmt: TRUE
#line 170 "zfxparser.y"
                { yylhs.value.as < bool > () = true; }
#line 1083 "zfxparser.cpp"
    break;

  case 20: // bool-stmt: FALSE
#line 171 "zfxparser.y"
            { yylhs.value.as < bool > () = false; }
#line 1089 "zfxparser.cpp"
    break;

  case 21: // assign-statement: zenvar assign-op array-or-exp
#line 174 "zfxparser.y"
                                                {
            std::vector<std::shared_ptr<ZfxASTNode>> children({yystack_[2].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()});
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(ASSIGNMENT, yystack_[1].value.as < operatorVals > (), children);
        }
#line 1098 "zfxparser.cpp"
    break;

  case 22: // jump-statement: BREAK
#line 180 "zfxparser.y"
                      { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(JUMP, JUMP_BREAK, {}); }
#line 1104 "zfxparser.cpp"
    break;

  case 23: // jump-statement: RETURN
#line 181 "zfxparser.y"
               { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(JUMP, JUMP_RETURN, {}); }
#line 1110 "zfxparser.cpp"
    break;

  case 24: // jump-statement: CONTINUE
#line 182 "zfxparser.y"
               { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(JUMP, JUMP_CONTINUE, {}); }
#line 1116 "zfxparser.cpp"
    break;

  case 25: // arrcontent: exp-statement
#line 185 "zfxparser.y"
                          { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1122 "zfxparser.cpp"
    break;

  case 26: // arrcontent: array-stmt
#line 186 "zfxparser.y"
                 { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1128 "zfxparser.cpp"
    break;

  case 27: // arrcontents: arrcontent
#line 189 "zfxparser.y"
                                   { yylhs.value.as < std::vector<std::shared_ptr<ZfxASTNode>> > () = std::vector<std::shared_ptr<ZfxASTNode>>({yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()}); }
#line 1134 "zfxparser.cpp"
    break;

  case 28: // arrcontents: arrcontents COMMA arrcontent
#line 190 "zfxparser.y"
                                   { yystack_[2].value.as < std::vector<std::shared_ptr<ZfxASTNode>> > ().push_back(yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()); yylhs.value.as < std::vector<std::shared_ptr<ZfxASTNode>> > () = yystack_[2].value.as < std::vector<std::shared_ptr<ZfxASTNode>> > (); }
#line 1140 "zfxparser.cpp"
    break;

  case 29: // array-stmt: LBRACKET arrcontents RBRACKET
#line 193 "zfxparser.y"
                                          { 
        yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(ARRAY, DEFAULT_FUNCVAL, yystack_[1].value.as < std::vector<std::shared_ptr<ZfxASTNode>> > ());
    }
#line 1148 "zfxparser.cpp"
    break;

  case 30: // array-mark: %empty
#line 198 "zfxparser.y"
                   { yylhs.value.as < bool > () = false; }
#line 1154 "zfxparser.cpp"
    break;

  case 31: // array-mark: LSQBRACKET RSQBRACKET
#line 199 "zfxparser.y"
                            { yylhs.value.as < bool > () = true; }
#line 1160 "zfxparser.cpp"
    break;

  case 32: // only-declare: TYPE VARNAME array-mark
#line 202 "zfxparser.y"
                                      {
    auto typeNode = driver.makeTypeNode(yystack_[2].value.as < string > (), yystack_[0].value.as < bool > ());
    auto nameNode = driver.makeZfxVarNode(yystack_[1].value.as < string > ());
    std::vector<std::shared_ptr<ZfxASTNode>> children({typeNode, nameNode});
    yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(DECLARE, DEFAULT_FUNCVAL, children);
}
#line 1171 "zfxparser.cpp"
    break;

  case 33: // declare-statement: only-declare
#line 209 "zfxparser.y"
                                {
                yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ();
            }
#line 1179 "zfxparser.cpp"
    break;

  case 34: // declare-statement: TYPE VARNAME array-mark EQUALTO array-or-exp
#line 212 "zfxparser.y"
                                                   {
                auto typeNode = driver.makeTypeNode(yystack_[4].value.as < string > (), yystack_[2].value.as < bool > ());
                auto nameNode = driver.makeZfxVarNode(yystack_[3].value.as < string > ());
                std::vector<std::shared_ptr<ZfxASTNode>> children({typeNode, nameNode, yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()});
                yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(DECLARE, DEFAULT_FUNCVAL, children);
            }
#line 1190 "zfxparser.cpp"
    break;

  case 35: // code-block: LBRACKET multi-statements RBRACKET
#line 220 "zfxparser.y"
                                               { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[1].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1196 "zfxparser.cpp"
    break;

  case 36: // if-statement: IF LPAREN exp-statement RPAREN code-block
#line 223 "zfxparser.y"
                                                        {
            std::vector<std::shared_ptr<ZfxASTNode>> children({yystack_[2].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()});
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(IF, DEFAULT_FUNCVAL, children);
        }
#line 1205 "zfxparser.cpp"
    break;

  case 37: // if-statement: IF LPAREN exp-statement RPAREN general-statement
#line 227 "zfxparser.y"
                                                       {
            std::vector<std::shared_ptr<ZfxASTNode>> children({yystack_[2].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()});
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(IF, DEFAULT_FUNCVAL, children);
        }
#line 1214 "zfxparser.cpp"
    break;

  case 38: // for-begin: SEMICOLON
#line 233 "zfxparser.y"
                     { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeEmptyNode(); }
#line 1220 "zfxparser.cpp"
    break;

  case 39: // for-begin: declare-statement SEMICOLON
#line 234 "zfxparser.y"
                                  { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[1].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1226 "zfxparser.cpp"
    break;

  case 40: // for-begin: assign-statement SEMICOLON
#line 235 "zfxparser.y"
                                 { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[1].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1232 "zfxparser.cpp"
    break;

  case 41: // for-begin: exp-statement SEMICOLON
#line 236 "zfxparser.y"
                              { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[1].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1238 "zfxparser.cpp"
    break;

  case 42: // for-condition: SEMICOLON
#line 239 "zfxparser.y"
                          { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeEmptyNode(); }
#line 1244 "zfxparser.cpp"
    break;

  case 43: // for-condition: exp-statement SEMICOLON
#line 240 "zfxparser.y"
                              { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[1].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1250 "zfxparser.cpp"
    break;

  case 44: // for-step: %empty
#line 243 "zfxparser.y"
                 { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeEmptyNode(); }
#line 1256 "zfxparser.cpp"
    break;

  case 45: // for-step: exp-statement
#line 244 "zfxparser.y"
                    { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1262 "zfxparser.cpp"
    break;

  case 46: // for-step: assign-statement
#line 245 "zfxparser.y"
                       { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1268 "zfxparser.cpp"
    break;

  case 47: // foreach-step: VARNAME
#line 248 "zfxparser.y"
                      {
            auto varNode = driver.makeZfxVarNode(yystack_[0].value.as < string > ());
            yylhs.value.as < std::vector<std::shared_ptr<ZfxASTNode>> > () = std::vector<std::shared_ptr<ZfxASTNode>>({varNode});
        }
#line 1277 "zfxparser.cpp"
    break;

  case 48: // foreach-step: TYPE VARNAME
#line 252 "zfxparser.y"
                   {
            /* 类型不是必要的，只是为了兼容一些编程习惯，比如foreach(int a : arr)*/
            auto varNode = driver.makeZfxVarNode(yystack_[0].value.as < string > ());
            yylhs.value.as < std::vector<std::shared_ptr<ZfxASTNode>> > () = std::vector<std::shared_ptr<ZfxASTNode>>({varNode});
        }
#line 1287 "zfxparser.cpp"
    break;

  case 49: // foreach-step: LSQBRACKET VARNAME COMMA VARNAME RSQBRACKET
#line 257 "zfxparser.y"
                                                  {
            auto idxNode = driver.makeZfxVarNode(yystack_[3].value.as < string > ());
            auto varNode = driver.makeZfxVarNode(yystack_[1].value.as < string > ());
            yylhs.value.as < std::vector<std::shared_ptr<ZfxASTNode>> > () = std::vector<std::shared_ptr<ZfxASTNode>>({idxNode, varNode});
        }
#line 1297 "zfxparser.cpp"
    break;

  case 50: // loop-statement: FOR LPAREN for-begin for-condition for-step RPAREN code-block
#line 264 "zfxparser.y"
                                                                              {
            std::vector<std::shared_ptr<ZfxASTNode>> children({yystack_[4].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[3].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[2].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()});
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(FOR, DEFAULT_FUNCVAL, children);
        }
#line 1306 "zfxparser.cpp"
    break;

  case 51: // loop-statement: FOREACH LPAREN foreach-step COLON zenvar RPAREN code-block
#line 268 "zfxparser.y"
                                                                 {
            yystack_[4].value.as < std::vector<std::shared_ptr<ZfxASTNode>> > ().push_back(yystack_[2].value.as < std::shared_ptr<ZfxASTNode> > ());
            yystack_[4].value.as < std::vector<std::shared_ptr<ZfxASTNode>> > ().push_back(yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ());
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(FOREACH, DEFAULT_FUNCVAL, yystack_[4].value.as < std::vector<std::shared_ptr<ZfxASTNode>> > ());
        }
#line 1316 "zfxparser.cpp"
    break;

  case 52: // loop-statement: WHILE LPAREN exp-statement RPAREN code-block
#line 273 "zfxparser.y"
                                                   {
            std::vector<std::shared_ptr<ZfxASTNode>> children({yystack_[2].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()});
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(WHILE, DEFAULT_FUNCVAL, children);
        }
#line 1325 "zfxparser.cpp"
    break;

  case 53: // loop-statement: DO code-block WHILE LPAREN exp-statement RPAREN SEMICOLON
#line 277 "zfxparser.y"
                                                                {
            std::vector<std::shared_ptr<ZfxASTNode>> children({yystack_[5].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[2].value.as < std::shared_ptr<ZfxASTNode> > ()});
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(DOWHILE, DEFAULT_FUNCVAL, children);
        }
#line 1334 "zfxparser.cpp"
    break;

  case 54: // exp-statement: compareexp
#line 283 "zfxparser.y"
                                    { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1340 "zfxparser.cpp"
    break;

  case 55: // exp-statement: exp-statement COMPARE compareexp
#line 284 "zfxparser.y"
                                        {
                std::vector<std::shared_ptr<ZfxASTNode>>children({yystack_[2].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()});
                yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(COMPOP, DEFAULT_FUNCVAL, children);
                yylhs.value.as < std::shared_ptr<ZfxASTNode> > ()->value = yystack_[1].value.as < string > ();
            }
#line 1350 "zfxparser.cpp"
    break;

  case 56: // exp-statement: exp-statement COMPARE compareexp QUESTION exp-statement COLON exp-statement
#line 289 "zfxparser.y"
                                                                                  {
                std::vector<std::shared_ptr<ZfxASTNode>> children({yystack_[6].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[4].value.as < std::shared_ptr<ZfxASTNode> > ()});
                auto spCond = driver.makeNewNode(COMPOP, DEFAULT_FUNCVAL, children);
                spCond->value = yystack_[5].value.as < string > ();

                std::vector<std::shared_ptr<ZfxASTNode>> exps({spCond, yystack_[2].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()});
                yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(CONDEXP, DEFAULT_FUNCVAL, exps);
            }
#line 1363 "zfxparser.cpp"
    break;

  case 57: // compareexp: factor
#line 299 "zfxparser.y"
                                { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1369 "zfxparser.cpp"
    break;

  case 58: // compareexp: compareexp ADD factor
#line 300 "zfxparser.y"
                            {
                std::vector<std::shared_ptr<ZfxASTNode>> children({yystack_[2].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()});
                yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(FOUROPERATIONS, PLUS, children);
            }
#line 1378 "zfxparser.cpp"
    break;

  case 59: // compareexp: compareexp SUB factor
#line 304 "zfxparser.y"
                            {
                std::vector<std::shared_ptr<ZfxASTNode>> children({yystack_[2].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()});
                yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(FOUROPERATIONS, MINUS, children);
            }
#line 1387 "zfxparser.cpp"
    break;

  case 60: // factor: term
#line 310 "zfxparser.y"
                        { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1393 "zfxparser.cpp"
    break;

  case 61: // factor: factor MUL term
#line 311 "zfxparser.y"
                        {
                std::vector<std::shared_ptr<ZfxASTNode>>children({yystack_[2].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()});
                yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(FOUROPERATIONS, MUL, children);
            }
#line 1402 "zfxparser.cpp"
    break;

  case 62: // factor: factor DIV term
#line 315 "zfxparser.y"
                      {
            std::vector<std::shared_ptr<ZfxASTNode>>children({yystack_[2].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()});
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(FOUROPERATIONS, DIV, children);
        }
#line 1411 "zfxparser.cpp"
    break;

  case 63: // zenvar: DOLLARVARNAME
#line 321 "zfxparser.y"
                        { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeZfxVarNode(yystack_[0].value.as < string > (), BulitInVar); }
#line 1417 "zfxparser.cpp"
    break;

  case 64: // zenvar: VARNAME
#line 322 "zfxparser.y"
                        { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeZfxVarNode(yystack_[0].value.as < string > ()); }
#line 1423 "zfxparser.cpp"
    break;

  case 65: // zenvar: ATTRAT zenvar
#line 323 "zfxparser.y"
                    {
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ();
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > ()->opVal = AttrMark;
        }
#line 1432 "zfxparser.cpp"
    break;

  case 66: // zenvar: zenvar DOT VARNAME
#line 327 "zfxparser.y"
                         {
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeComponentVisit(yystack_[2].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[0].value.as < string > ());
        }
#line 1440 "zfxparser.cpp"
    break;

  case 67: // zenvar: zenvar LSQBRACKET exp-statement RSQBRACKET
#line 330 "zfxparser.y"
                                                 {
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[3].value.as < std::shared_ptr<ZfxASTNode> > ();
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > ()->opVal = Indexing;
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > ()->children.push_back(yystack_[1].value.as < std::shared_ptr<ZfxASTNode> > ());
        }
#line 1450 "zfxparser.cpp"
    break;

  case 68: // zenvar: AUTOINC zenvar
#line 335 "zfxparser.y"
                     {
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ();
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > ()->opVal = AutoIncreaseFirst;
        }
#line 1459 "zfxparser.cpp"
    break;

  case 69: // zenvar: zenvar AUTOINC
#line 339 "zfxparser.y"
                     {
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[1].value.as < std::shared_ptr<ZfxASTNode> > ();
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > ()->opVal = AutoIncreaseLast;
        }
#line 1468 "zfxparser.cpp"
    break;

  case 70: // zenvar: AUTODEC zenvar
#line 343 "zfxparser.y"
                     {
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ();
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > ()->opVal = AutoDecreaseFirst;
        }
#line 1477 "zfxparser.cpp"
    break;

  case 71: // zenvar: zenvar AUTODEC
#line 347 "zfxparser.y"
                     {
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[1].value.as < std::shared_ptr<ZfxASTNode> > ();
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > ()->opVal = AutoDecreaseLast;
        }
#line 1486 "zfxparser.cpp"
    break;

  case 72: // funcargs: %empty
#line 353 "zfxparser.y"
                 { yylhs.value.as < std::vector<std::shared_ptr<ZfxASTNode>> > () = std::vector<std::shared_ptr<ZfxASTNode>>(); }
#line 1492 "zfxparser.cpp"
    break;

  case 73: // funcargs: exp-statement
#line 354 "zfxparser.y"
                               { yylhs.value.as < std::vector<std::shared_ptr<ZfxASTNode>> > () = std::vector<std::shared_ptr<ZfxASTNode>>({yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()}); }
#line 1498 "zfxparser.cpp"
    break;

  case 74: // funcargs: funcargs COMMA exp-statement
#line 355 "zfxparser.y"
                                   { yystack_[2].value.as < std::vector<std::shared_ptr<ZfxASTNode>> > ().push_back(yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()); yylhs.value.as < std::vector<std::shared_ptr<ZfxASTNode>> > () = yystack_[2].value.as < std::vector<std::shared_ptr<ZfxASTNode>> > (); }
#line 1504 "zfxparser.cpp"
    break;

  case 75: // func-content: LPAREN funcargs RPAREN
#line 359 "zfxparser.y"
                                     { 
        yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(FUNC, DEFAULT_FUNCVAL, yystack_[1].value.as < std::vector<std::shared_ptr<ZfxASTNode>> > ());
        yylhs.value.as < std::shared_ptr<ZfxASTNode> > ()->isParenthesisNodeComplete = true;
        yylhs.value.as < std::shared_ptr<ZfxASTNode> > ()->func_match = Match_Exactly;
    }
#line 1514 "zfxparser.cpp"
    break;

  case 76: // term: NUMBER
#line 367 "zfxparser.y"
                        { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNumberNode(yystack_[0].value.as < float > ()); }
#line 1520 "zfxparser.cpp"
    break;

  case 77: // term: bool-stmt
#line 368 "zfxparser.y"
                        { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeBoolNode(yystack_[0].value.as < bool > ()); }
#line 1526 "zfxparser.cpp"
    break;

  case 78: // term: LITERAL
#line 369 "zfxparser.y"
                        { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeStringNode(yystack_[0].value.as < string > ()); }
#line 1532 "zfxparser.cpp"
    break;

  case 79: // term: UNCOMPSTR
#line 370 "zfxparser.y"
                        { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeQuoteStringNode(yystack_[0].value.as < string > ()); }
#line 1538 "zfxparser.cpp"
    break;

  case 80: // term: LPAREN exp-statement RPAREN
#line 371 "zfxparser.y"
                                  { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[1].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1544 "zfxparser.cpp"
    break;

  case 81: // term: SUB exp-statement
#line 372 "zfxparser.y"
                                  { yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()->value = -1 * std::get<float>(yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()->value); yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1550 "zfxparser.cpp"
    break;

  case 82: // term: zenvar
#line 373 "zfxparser.y"
                        { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1556 "zfxparser.cpp"
    break;

  case 83: // term: VARNAME func-content
#line 374 "zfxparser.y"
                            { 
        yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ();
        yylhs.value.as < std::shared_ptr<ZfxASTNode> > ()->opVal = DEFAULT_FUNCVAL;
        yylhs.value.as < std::shared_ptr<ZfxASTNode> > ()->type = FUNC;
        yylhs.value.as < std::shared_ptr<ZfxASTNode> > ()->value = yystack_[1].value.as < string > ();
        yylhs.value.as < std::shared_ptr<ZfxASTNode> > ()->isParenthesisNode = true;
    }
#line 1568 "zfxparser.cpp"
    break;


#line 1572 "zfxparser.cpp"

            default:
              break;
            }
        }
#if YY_EXCEPTIONS
      catch (const syntax_error& yyexc)
        {
          YYCDEBUG << "Caught exception: " << yyexc.what() << '\n';
          error (yyexc);
          YYERROR;
        }
#endif // YY_EXCEPTIONS
      YY_SYMBOL_PRINT ("-> $$ =", yylhs);
      yypop_ (yylen);
      yylen = 0;

      // Shift the result of the reduction.
      yypush_ (YY_NULLPTR, YY_MOVE (yylhs));
    }
    goto yynewstate;


  /*--------------------------------------.
  | yyerrlab -- here on detecting error.  |
  `--------------------------------------*/
  yyerrlab:
    // If not already recovering from an error, report this error.
    if (!yyerrstatus_)
      {
        ++yynerrs_;
        context yyctx (*this, yyla);
        std::string msg = yysyntax_error_ (yyctx);
        error (yyla.location, YY_MOVE (msg));
      }


    yyerror_range[1].location = yyla.location;
    if (yyerrstatus_ == 3)
      {
        /* If just tried and failed to reuse lookahead token after an
           error, discard it.  */

        // Return failure if at end of input.
        if (yyla.kind () == symbol_kind::S_YYEOF)
          YYABORT;
        else if (!yyla.empty ())
          {
            yy_destroy_ ("Error: discarding", yyla);
            yyla.clear ();
          }
      }

    // Else will try to reuse lookahead token after shifting the error token.
    goto yyerrlab1;


  /*---------------------------------------------------.
  | yyerrorlab -- error raised explicitly by YYERROR.  |
  `---------------------------------------------------*/
  yyerrorlab:
    /* Pacify compilers when the user code never invokes YYERROR and
       the label yyerrorlab therefore never appears in user code.  */
    if (false)
      YYERROR;

    /* Do not reclaim the symbols of the rule whose action triggered
       this YYERROR.  */
    yypop_ (yylen);
    yylen = 0;
    YY_STACK_PRINT ();
    goto yyerrlab1;


  /*-------------------------------------------------------------.
  | yyerrlab1 -- common code for both syntax error and YYERROR.  |
  `-------------------------------------------------------------*/
  yyerrlab1:
    yyerrstatus_ = 3;   // Each real token shifted decrements this.
    // Pop stack until we find a state that shifts the error token.
    for (;;)
      {
        yyn = yypact_[+yystack_[0].state];
        if (!yy_pact_value_is_default_ (yyn))
          {
            yyn += symbol_kind::S_YYerror;
            if (0 <= yyn && yyn <= yylast_
                && yycheck_[yyn] == symbol_kind::S_YYerror)
              {
                yyn = yytable_[yyn];
                if (0 < yyn)
                  break;
              }
          }

        // Pop the current state because it cannot handle the error token.
        if (yystack_.size () == 1)
          YYABORT;

        yyerror_range[1].location = yystack_[0].location;
        yy_destroy_ ("Error: popping", yystack_[0]);
        yypop_ ();
        YY_STACK_PRINT ();
      }
    {
      stack_symbol_type error_token;

      yyerror_range[2].location = yyla.location;
      YYLLOC_DEFAULT (error_token.location, yyerror_range, 2);

      // Shift the error token.
      error_token.state = state_type (yyn);
      yypush_ ("Shifting", YY_MOVE (error_token));
    }
    goto yynewstate;


  /*-------------------------------------.
  | yyacceptlab -- YYACCEPT comes here.  |
  `-------------------------------------*/
  yyacceptlab:
    yyresult = 0;
    goto yyreturn;


  /*-----------------------------------.
  | yyabortlab -- YYABORT comes here.  |
  `-----------------------------------*/
  yyabortlab:
    yyresult = 1;
    goto yyreturn;


  /*-----------------------------------------------------.
  | yyreturn -- parsing is finished, return the result.  |
  `-----------------------------------------------------*/
  yyreturn:
    if (!yyla.empty ())
      yy_destroy_ ("Cleanup: discarding lookahead", yyla);

    /* Do not reclaim the symbols of the rule whose action triggered
       this YYABORT or YYACCEPT.  */
    yypop_ (yylen);
    YY_STACK_PRINT ();
    while (1 < yystack_.size ())
      {
        yy_destroy_ ("Cleanup: popping", yystack_[0]);
        yypop_ ();
      }

    return yyresult;
  }
#if YY_EXCEPTIONS
    catch (...)
      {
        YYCDEBUG << "Exception caught: cleaning lookahead and stack\n";
        // Do not try to display the values of the reclaimed symbols,
        // as their printers might throw an exception.
        if (!yyla.empty ())
          yy_destroy_ (YY_NULLPTR, yyla);

        while (1 < yystack_.size ())
          {
            yy_destroy_ (YY_NULLPTR, yystack_[0]);
            yypop_ ();
          }
        throw;
      }
#endif // YY_EXCEPTIONS
  }

  void
   ZfxParser ::error (const syntax_error& yyexc)
  {
    error (yyexc.location, yyexc.what ());
  }

  /* Return YYSTR after stripping away unnecessary quotes and
     backslashes, so that it's suitable for yyerror.  The heuristic is
     that double-quoting is unnecessary unless the string contains an
     apostrophe, a comma, or backslash (other than backslash-backslash).
     YYSTR is taken from yytname.  */
  std::string
   ZfxParser ::yytnamerr_ (const char *yystr)
  {
    if (*yystr == '"')
      {
        std::string yyr;
        char const *yyp = yystr;

        for (;;)
          switch (*++yyp)
            {
            case '\'':
            case ',':
              goto do_not_strip_quotes;

            case '\\':
              if (*++yyp != '\\')
                goto do_not_strip_quotes;
              else
                goto append;

            append:
            default:
              yyr += *yyp;
              break;

            case '"':
              return yyr;
            }
      do_not_strip_quotes: ;
      }

    return yystr;
  }

  std::string
   ZfxParser ::symbol_name (symbol_kind_type yysymbol)
  {
    return yytnamerr_ (yytname_[yysymbol]);
  }



  //  ZfxParser ::context.
   ZfxParser ::context::context (const  ZfxParser & yyparser, const symbol_type& yyla)
    : yyparser_ (yyparser)
    , yyla_ (yyla)
  {}

  int
   ZfxParser ::context::expected_tokens (symbol_kind_type yyarg[], int yyargn) const
  {
    // Actual number of expected tokens
    int yycount = 0;

    const int yyn = yypact_[+yyparser_.yystack_[0].state];
    if (!yy_pact_value_is_default_ (yyn))
      {
        /* Start YYX at -YYN if negative to avoid negative indexes in
           YYCHECK.  In other words, skip the first -YYN actions for
           this state because they are default actions.  */
        const int yyxbegin = yyn < 0 ? -yyn : 0;
        // Stay within bounds of both yycheck and yytname.
        const int yychecklim = yylast_ - yyn + 1;
        const int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
        for (int yyx = yyxbegin; yyx < yyxend; ++yyx)
          if (yycheck_[yyx + yyn] == yyx && yyx != symbol_kind::S_YYerror
              && !yy_table_value_is_error_ (yytable_[yyx + yyn]))
            {
              if (!yyarg)
                ++yycount;
              else if (yycount == yyargn)
                return 0;
              else
                yyarg[yycount++] = YY_CAST (symbol_kind_type, yyx);
            }
      }

    if (yyarg && yycount == 0 && 0 < yyargn)
      yyarg[0] = symbol_kind::S_YYEMPTY;
    return yycount;
  }






  int
   ZfxParser ::yy_syntax_error_arguments_ (const context& yyctx,
                                                 symbol_kind_type yyarg[], int yyargn) const
  {
    /* There are many possibilities here to consider:
       - If this state is a consistent state with a default action, then
         the only way this function was invoked is if the default action
         is an error action.  In that case, don't check for expected
         tokens because there are none.
       - The only way there can be no lookahead present (in yyla) is
         if this state is a consistent state with a default action.
         Thus, detecting the absence of a lookahead is sufficient to
         determine that there is no unexpected or expected token to
         report.  In that case, just report a simple "syntax error".
       - Don't assume there isn't a lookahead just because this state is
         a consistent state with a default action.  There might have
         been a previous inconsistent state, consistent state with a
         non-default action, or user semantic action that manipulated
         yyla.  (However, yyla is currently not documented for users.)
       - Of course, the expected token list depends on states to have
         correct lookahead information, and it depends on the parser not
         to perform extra reductions after fetching a lookahead from the
         scanner and before detecting a syntax error.  Thus, state merging
         (from LALR or IELR) and default reductions corrupt the expected
         token list.  However, the list is correct for canonical LR with
         one exception: it will still contain any token that will not be
         accepted due to an error action in a later state.
    */

    if (!yyctx.lookahead ().empty ())
      {
        if (yyarg)
          yyarg[0] = yyctx.token ();
        int yyn = yyctx.expected_tokens (yyarg ? yyarg + 1 : yyarg, yyargn - 1);
        return yyn + 1;
      }
    return 0;
  }

  // Generate an error message.
  std::string
   ZfxParser ::yysyntax_error_ (const context& yyctx) const
  {
    // Its maximum.
    enum { YYARGS_MAX = 5 };
    // Arguments of yyformat.
    symbol_kind_type yyarg[YYARGS_MAX];
    int yycount = yy_syntax_error_arguments_ (yyctx, yyarg, YYARGS_MAX);

    char const* yyformat = YY_NULLPTR;
    switch (yycount)
      {
#define YYCASE_(N, S)                         \
        case N:                               \
          yyformat = S;                       \
        break
      default: // Avoid compiler warnings.
        YYCASE_ (0, YY_("syntax error"));
        YYCASE_ (1, YY_("syntax error, unexpected %s"));
        YYCASE_ (2, YY_("syntax error, unexpected %s, expecting %s"));
        YYCASE_ (3, YY_("syntax error, unexpected %s, expecting %s or %s"));
        YYCASE_ (4, YY_("syntax error, unexpected %s, expecting %s or %s or %s"));
        YYCASE_ (5, YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s"));
#undef YYCASE_
      }

    std::string yyres;
    // Argument number.
    std::ptrdiff_t yyi = 0;
    for (char const* yyp = yyformat; *yyp; ++yyp)
      if (yyp[0] == '%' && yyp[1] == 's' && yyi < yycount)
        {
          yyres += symbol_name (yyarg[yyi++]);
          ++yyp;
        }
      else
        yyres += *yyp;
    return yyres;
  }


  const signed char  ZfxParser ::yypact_ninf_ = -104;

  const signed char  ZfxParser ::yytable_ninf_ = -1;

  const short
   ZfxParser ::yypact_[] =
  {
     109,  -104,  -104,  -104,  -104,  -104,  -104,  -104,   -46,   -41,
     -24,    -7,     2,     2,  -104,  -104,  -104,    47,     2,    34,
      71,   283,   283,    94,   109,   195,  -104,    73,    93,  -104,
      99,  -104,  -104,    64,   -33,    19,    76,  -104,   283,  -104,
     283,    32,   283,  -104,    46,    46,    96,    46,   -17,   195,
     101,   114,    46,    24,  -104,  -104,  -104,  -104,  -104,  -104,
     283,  -104,   283,   283,   283,   283,   108,  -104,  -104,  -104,
     283,  -104,  -104,  -104,  -104,   238,   114,    84,    33,  -104,
     110,   116,   251,    80,    49,   111,   117,  -104,   118,   121,
     115,   124,   100,  -104,    13,    19,    19,  -104,  -104,  -104,
      -8,   238,  -104,  -104,   114,  -104,   283,   152,  -104,  -104,
    -104,   283,   102,  -104,    71,  -104,   238,   143,  -104,     2,
    -104,   283,   283,  -104,  -104,    -6,  -104,   114,   114,  -104,
    -104,  -104,   157,   114,  -104,  -104,  -104,   137,    52,    50,
      74,   238,  -104,    71,   129,    71,   139,   283,  -104,  -104,
    -104,  -104,  -104,   114
  };

  const signed char
   ZfxParser ::yydefact_[] =
  {
       0,     2,    76,    19,    20,    78,    79,    63,    64,     0,
       0,     0,     0,     0,    23,    24,    22,     0,     0,     0,
       0,     0,     0,     0,     0,     4,    77,     0,     0,    33,
       0,     8,     9,     0,    54,    57,    82,    60,    72,    83,
       0,     0,     0,    64,    68,    70,    30,    65,     0,     4,
       0,    81,    82,     0,     1,     3,     5,     7,    10,     6,
       0,    11,     0,     0,     0,     0,     0,    14,    69,    71,
       0,    15,    16,    17,    18,     0,    73,     0,     0,    38,
       0,     0,     0,     0,     0,     0,    32,    47,     0,     0,
       0,     0,     0,    80,    55,    58,    59,    61,    62,    66,
       0,     0,    21,    13,    12,    75,     0,     0,    40,    39,
      42,    44,     0,    41,     0,    31,     0,     0,    48,     0,
      35,     0,     0,    67,    27,     0,    26,    25,    74,    37,
      36,    46,     0,    45,    43,    52,    34,     0,     0,     0,
       0,     0,    29,     0,     0,     0,     0,     0,    28,    50,
      49,    51,    53,    56
  };

  const short
   ZfxParser ::yypgoto_[] =
  {
    -104,   145,   -20,    60,    54,  -104,  -104,   -39,  -104,    30,
    -104,   -98,  -104,  -104,   131,  -103,  -104,  -104,  -104,  -104,
    -104,  -104,   -21,   125,    41,    10,  -104,  -104,     0
  };

  const unsigned char
   ZfxParser ::yydefgoto_[] =
  {
       0,    23,    24,    25,   102,    75,    26,    27,    28,   124,
     125,   103,    86,    29,    30,    50,    31,    82,   111,   132,
      90,    32,    33,    34,    35,    52,    77,    39,    37
  };

  const unsigned char
   ZfxParser ::yytable_[] =
  {
      51,    53,    80,   126,   130,    56,   141,    87,    38,    60,
      36,   135,    62,    40,    63,    88,   142,    76,     7,    78,
      83,    84,    44,    45,    89,   123,    43,    93,    47,    91,
      41,   122,    12,    13,    36,    36,   107,     2,     3,     4,
     149,    60,   151,   126,    18,     5,     6,    42,     7,   100,
      60,    36,   114,   146,   104,   145,     8,    79,    62,    36,
      63,   112,    12,    13,    97,    98,    60,    60,    64,    66,
      65,    46,   131,    17,    18,    66,    68,    69,    70,    21,
     127,    60,    68,    69,    70,   128,    22,   105,    48,    61,
     133,    60,    49,   147,    54,   104,   106,    60,    57,    66,
     139,   140,    67,    95,    96,   113,    68,    69,    70,     1,
      71,    72,    73,    74,     2,     3,     4,    36,    58,    60,
     127,    36,     5,     6,    59,     7,   153,   134,    85,   138,
      92,    60,    99,     8,   119,   108,     9,    10,    11,    12,
      13,   109,   117,   116,   115,   118,   120,    14,    15,    16,
      17,    18,    19,    20,   121,   137,    21,     2,     3,     4,
     143,   144,   150,    22,   152,     5,     6,   129,     7,    55,
     136,   148,    81,    49,     0,     0,     8,     0,     0,     9,
      10,    11,    12,    13,     0,    94,     0,     0,     0,     0,
      14,    15,    16,    17,    18,    19,    20,     0,     0,    21,
       2,     3,     4,     0,     0,     0,    22,     0,     5,     6,
       0,     7,     0,     0,     0,     0,     0,     0,     0,     8,
       0,     0,     9,    10,    11,    12,    13,     0,     0,     0,
       0,     0,     0,    14,    15,    16,    17,    18,    19,    20,
       0,     0,    21,     2,     3,     4,     0,     0,     0,    22,
       0,     5,     6,     0,     7,     0,     2,     3,     4,   101,
       0,     0,     8,     0,     5,     6,     0,     7,    12,    13,
       0,     0,     0,     0,     0,     8,   110,     0,     0,     0,
      18,    12,    13,     0,     0,    21,     0,     0,     2,     3,
       4,     0,    22,    18,     0,     0,     5,     6,    21,     7,
       0,     0,     0,     0,     0,    22,     0,     8,     0,     0,
       0,     0,     0,    12,    13,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    18,     0,     0,     0,     0,
      21,     0,     0,     0,     0,     0,     0,    22
  };

  const short
   ZfxParser ::yycheck_[] =
  {
      21,    22,    41,   101,   107,    25,    12,    24,    54,    17,
       0,   114,    45,    54,    47,    32,    22,    38,    16,    40,
      41,    42,    12,    13,    41,    33,    24,     3,    18,    49,
      54,    18,    30,    31,    24,    25,     3,     5,     6,     7,
     143,    17,   145,   141,    42,    13,    14,    54,    16,    70,
      17,    41,     3,     3,    75,     3,    24,    25,    45,    49,
      47,    82,    30,    31,    64,    65,    17,    17,    49,    23,
      51,    24,   111,    41,    42,    23,    30,    31,    32,    47,
     101,    17,    30,    31,    32,   106,    54,     3,    54,    25,
     111,    17,    21,    19,     0,   116,    12,    17,    25,    23,
     121,   122,    26,    62,    63,    25,    30,    31,    32,     0,
      34,    35,    36,    37,     5,     6,     7,   107,    25,    17,
     141,   111,    13,    14,    25,    16,   147,    25,    32,   119,
      29,    17,    24,    24,    19,    25,    27,    28,    29,    30,
      31,    25,    24,    26,    33,    24,    22,    38,    39,    40,
      41,    42,    43,    44,    54,    12,    47,     5,     6,     7,
       3,    24,    33,    54,    25,    13,    14,   107,    16,    24,
     116,   141,    41,    21,    -1,    -1,    24,    -1,    -1,    27,
      28,    29,    30,    31,    -1,    60,    -1,    -1,    -1,    -1,
      38,    39,    40,    41,    42,    43,    44,    -1,    -1,    47,
       5,     6,     7,    -1,    -1,    -1,    54,    -1,    13,    14,
      -1,    16,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    24,
      -1,    -1,    27,    28,    29,    30,    31,    -1,    -1,    -1,
      -1,    -1,    -1,    38,    39,    40,    41,    42,    43,    44,
      -1,    -1,    47,     5,     6,     7,    -1,    -1,    -1,    54,
      -1,    13,    14,    -1,    16,    -1,     5,     6,     7,    21,
      -1,    -1,    24,    -1,    13,    14,    -1,    16,    30,    31,
      -1,    -1,    -1,    -1,    -1,    24,    25,    -1,    -1,    -1,
      42,    30,    31,    -1,    -1,    47,    -1,    -1,     5,     6,
       7,    -1,    54,    42,    -1,    -1,    13,    14,    47,    16,
      -1,    -1,    -1,    -1,    -1,    54,    -1,    24,    -1,    -1,
      -1,    -1,    -1,    30,    31,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    42,    -1,    -1,    -1,    -1,
      47,    -1,    -1,    -1,    -1,    -1,    -1,    54
  };

  const signed char
   ZfxParser ::yystos_[] =
  {
       0,     0,     5,     6,     7,    13,    14,    16,    24,    27,
      28,    29,    30,    31,    38,    39,    40,    41,    42,    43,
      44,    47,    54,    56,    57,    58,    61,    62,    63,    68,
      69,    71,    76,    77,    78,    79,    80,    83,    54,    82,
      54,    54,    54,    24,    80,    80,    24,    80,    54,    21,
      70,    77,    80,    77,     0,    56,    57,    25,    25,    25,
      17,    25,    45,    47,    49,    51,    23,    26,    30,    31,
      32,    34,    35,    36,    37,    60,    77,    81,    77,    25,
      62,    69,    72,    77,    77,    32,    67,    24,    32,    41,
      75,    57,    29,     3,    78,    79,    79,    83,    83,    24,
      77,    21,    59,    66,    77,     3,    12,     3,    25,    25,
      25,    73,    77,    25,     3,    33,    26,    24,    24,    19,
      22,    54,    18,    33,    64,    65,    66,    77,    77,    58,
      70,    62,    74,    77,    25,    70,    59,    12,    80,    77,
      77,    12,    22,     3,    24,     3,     3,    19,    64,    70,
      33,    70,    25,    77
  };

  const signed char
   ZfxParser ::yyr1_[] =
  {
       0,    55,    56,    56,    57,    57,    58,    58,    58,    58,
      58,    58,    59,    59,    60,    60,    60,    60,    60,    61,
      61,    62,    63,    63,    63,    64,    64,    65,    65,    66,
      67,    67,    68,    69,    69,    70,    71,    71,    72,    72,
      72,    72,    73,    73,    74,    74,    74,    75,    75,    75,
      76,    76,    76,    76,    77,    77,    77,    78,    78,    78,
      79,    79,    79,    80,    80,    80,    80,    80,    80,    80,
      80,    80,    81,    81,    81,    82,    83,    83,    83,    83,
      83,    83,    83,    83
  };

  const signed char
   ZfxParser ::yyr2_[] =
  {
       0,     2,     1,     2,     0,     2,     2,     2,     1,     1,
       2,     2,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     3,     1,     1,     1,     1,     1,     1,     3,     3,
       0,     2,     3,     1,     5,     3,     5,     5,     1,     2,
       2,     2,     1,     2,     0,     1,     1,     1,     2,     5,
       7,     7,     5,     7,     1,     3,     7,     1,     3,     3,
       1,     3,     3,     1,     1,     2,     3,     4,     2,     2,
       2,     2,     0,     1,     3,     3,     1,     1,     1,     1,
       3,     2,     1,     2
  };


#if YYDEBUG || 1
  // YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
  // First, the terminals, then, starting at \a YYNTOKENS, nonterminals.
  const char*
  const  ZfxParser ::yytname_[] =
  {
  "END", "error", "\"invalid token\"", "RPAREN", "IDENTIFIER", "NUMBER",
  "TRUE", "FALSE", "EOL", "FRAME", "FPS", "PI", "COMMA", "LITERAL",
  "UNCOMPSTR", "DOLLAR", "DOLLARVARNAME", "COMPARE", "QUESTION", "COLON",
  "ZFXVAR", "LBRACKET", "RBRACKET", "DOT", "VARNAME", "SEMICOLON",
  "EQUALTO", "IF", "FOR", "WHILE", "AUTOINC", "AUTODEC", "LSQBRACKET",
  "RSQBRACKET", "ADDASSIGN", "MULASSIGN", "SUBASSIGN", "DIVASSIGN",
  "RETURN", "CONTINUE", "BREAK", "TYPE", "ATTRAT", "FOREACH", "DO", "ADD",
  "\"+\"", "SUB", "\"-\"", "MUL", "\"*\"", "DIV", "\"/\"", "NEG", "LPAREN",
  "$accept", "zfx-program", "multi-statements", "general-statement",
  "array-or-exp", "assign-op", "bool-stmt", "assign-statement",
  "jump-statement", "arrcontent", "arrcontents", "array-stmt",
  "array-mark", "only-declare", "declare-statement", "code-block",
  "if-statement", "for-begin", "for-condition", "for-step", "foreach-step",
  "loop-statement", "exp-statement", "compareexp", "factor", "zenvar",
  "funcargs", "func-content", "term", YY_NULLPTR
  };
#endif


#if YYDEBUG
  const short
   ZfxParser ::yyrline_[] =
  {
       0,   131,   131,   136,   142,   145,   151,   152,   153,   154,
     155,   156,   159,   160,   163,   164,   165,   166,   167,   170,
     171,   174,   180,   181,   182,   185,   186,   189,   190,   193,
     198,   199,   202,   209,   212,   220,   223,   227,   233,   234,
     235,   236,   239,   240,   243,   244,   245,   248,   252,   257,
     264,   268,   273,   277,   283,   284,   289,   299,   300,   304,
     310,   311,   315,   321,   322,   323,   327,   330,   335,   339,
     343,   347,   353,   354,   355,   359,   367,   368,   369,   370,
     371,   372,   373,   374
  };

  void
   ZfxParser ::yy_stack_print_ () const
  {
    *yycdebug_ << "Stack now";
    for (stack_type::const_iterator
           i = yystack_.begin (),
           i_end = yystack_.end ();
         i != i_end; ++i)
      *yycdebug_ << ' ' << int (i->state);
    *yycdebug_ << '\n';
  }

  void
   ZfxParser ::yy_reduce_print_ (int yyrule) const
  {
    int yylno = yyrline_[yyrule];
    int yynrhs = yyr2_[yyrule];
    // Print the symbols being reduced, and their result.
    *yycdebug_ << "Reducing stack by rule " << yyrule - 1
               << " (line " << yylno << "):\n";
    // The symbols being reduced.
    for (int yyi = 0; yyi < yynrhs; yyi++)
      YY_SYMBOL_PRINT ("   $" << yyi + 1 << " =",
                       yystack_[(yynrhs) - (yyi + 1)]);
  }
#endif // YYDEBUG


#line 10 "zfxparser.y"
} //  zeno 
#line 2184 "zfxparser.cpp"

#line 385 "zfxparser.y"


// Bison expects us to provide implementation - otherwise linker complains
void zeno::ZfxParser::error(const location &loc , const std::string &message) {
    cout << "Error: " << message << endl << "Error location: " << driver.location() << endl;
}

