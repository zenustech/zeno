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
      case symbol_kind::S_70_bool_stmt: // bool-stmt
      case symbol_kind::S_76_array_mark: // array-mark
        value.YY_MOVE_OR_COPY< bool > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_NUMBER: // NUMBER
        value.YY_MOVE_OR_COPY< float > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_68_assign_op: // assign-op
      case symbol_kind::S_85_compare_op: // compare-op
        value.YY_MOVE_OR_COPY< operatorVals > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_64_zfx_program: // zfx-program
      case symbol_kind::S_65_multi_statements: // multi-statements
      case symbol_kind::S_66_general_statement: // general-statement
      case symbol_kind::S_67_array_or_exp: // array-or-exp
      case symbol_kind::S_69_code_block: // code-block
      case symbol_kind::S_71_assign_statement: // assign-statement
      case symbol_kind::S_72_jump_statement: // jump-statement
      case symbol_kind::S_arrcontent: // arrcontent
      case symbol_kind::S_75_array_stmt: // array-stmt
      case symbol_kind::S_77_only_declare: // only-declare
      case symbol_kind::S_78_declare_statement: // declare-statement
      case symbol_kind::S_79_if_statement: // if-statement
      case symbol_kind::S_80_for_begin: // for-begin
      case symbol_kind::S_81_for_condition: // for-condition
      case symbol_kind::S_82_for_step: // for-step
      case symbol_kind::S_84_loop_statement: // loop-statement
      case symbol_kind::S_86_exp_statement: // exp-statement
      case symbol_kind::S_orexp: // orexp
      case symbol_kind::S_andexp: // andexp
      case symbol_kind::S_compareexp: // compareexp
      case symbol_kind::S_addsubexp: // addsubexp
      case symbol_kind::S_factor: // factor
      case symbol_kind::S_zenvar: // zenvar
      case symbol_kind::S_94_func_content: // func-content
      case symbol_kind::S_term: // term
        value.YY_MOVE_OR_COPY< std::shared_ptr<ZfxASTNode> > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_arrcontents: // arrcontents
      case symbol_kind::S_83_foreach_step: // foreach-step
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
      case symbol_kind::S_ASSIGNTO: // ASSIGNTO
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
      case symbol_kind::S_LESSTHAN: // LESSTHAN
      case symbol_kind::S_LESSEQUAL: // LESSEQUAL
      case symbol_kind::S_GREATTHAN: // GREATTHAN
      case symbol_kind::S_GREATEQUAL: // GREATEQUAL
      case symbol_kind::S_RETURN: // RETURN
      case symbol_kind::S_CONTINUE: // CONTINUE
      case symbol_kind::S_BREAK: // BREAK
      case symbol_kind::S_TYPE: // TYPE
      case symbol_kind::S_ATTRAT: // ATTRAT
      case symbol_kind::S_FOREACH: // FOREACH
      case symbol_kind::S_DO: // DO
      case symbol_kind::S_EQUALTO: // EQUALTO
      case symbol_kind::S_NOTEQUAL: // NOTEQUAL
      case symbol_kind::S_OR: // OR
      case symbol_kind::S_AND: // AND
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
      case symbol_kind::S_70_bool_stmt: // bool-stmt
      case symbol_kind::S_76_array_mark: // array-mark
        value.move< bool > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_NUMBER: // NUMBER
        value.move< float > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_68_assign_op: // assign-op
      case symbol_kind::S_85_compare_op: // compare-op
        value.move< operatorVals > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_64_zfx_program: // zfx-program
      case symbol_kind::S_65_multi_statements: // multi-statements
      case symbol_kind::S_66_general_statement: // general-statement
      case symbol_kind::S_67_array_or_exp: // array-or-exp
      case symbol_kind::S_69_code_block: // code-block
      case symbol_kind::S_71_assign_statement: // assign-statement
      case symbol_kind::S_72_jump_statement: // jump-statement
      case symbol_kind::S_arrcontent: // arrcontent
      case symbol_kind::S_75_array_stmt: // array-stmt
      case symbol_kind::S_77_only_declare: // only-declare
      case symbol_kind::S_78_declare_statement: // declare-statement
      case symbol_kind::S_79_if_statement: // if-statement
      case symbol_kind::S_80_for_begin: // for-begin
      case symbol_kind::S_81_for_condition: // for-condition
      case symbol_kind::S_82_for_step: // for-step
      case symbol_kind::S_84_loop_statement: // loop-statement
      case symbol_kind::S_86_exp_statement: // exp-statement
      case symbol_kind::S_orexp: // orexp
      case symbol_kind::S_andexp: // andexp
      case symbol_kind::S_compareexp: // compareexp
      case symbol_kind::S_addsubexp: // addsubexp
      case symbol_kind::S_factor: // factor
      case symbol_kind::S_zenvar: // zenvar
      case symbol_kind::S_94_func_content: // func-content
      case symbol_kind::S_term: // term
        value.move< std::shared_ptr<ZfxASTNode> > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_arrcontents: // arrcontents
      case symbol_kind::S_83_foreach_step: // foreach-step
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
      case symbol_kind::S_ASSIGNTO: // ASSIGNTO
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
      case symbol_kind::S_LESSTHAN: // LESSTHAN
      case symbol_kind::S_LESSEQUAL: // LESSEQUAL
      case symbol_kind::S_GREATTHAN: // GREATTHAN
      case symbol_kind::S_GREATEQUAL: // GREATEQUAL
      case symbol_kind::S_RETURN: // RETURN
      case symbol_kind::S_CONTINUE: // CONTINUE
      case symbol_kind::S_BREAK: // BREAK
      case symbol_kind::S_TYPE: // TYPE
      case symbol_kind::S_ATTRAT: // ATTRAT
      case symbol_kind::S_FOREACH: // FOREACH
      case symbol_kind::S_DO: // DO
      case symbol_kind::S_EQUALTO: // EQUALTO
      case symbol_kind::S_NOTEQUAL: // NOTEQUAL
      case symbol_kind::S_OR: // OR
      case symbol_kind::S_AND: // AND
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
      case symbol_kind::S_70_bool_stmt: // bool-stmt
      case symbol_kind::S_76_array_mark: // array-mark
        value.copy< bool > (that.value);
        break;

      case symbol_kind::S_NUMBER: // NUMBER
        value.copy< float > (that.value);
        break;

      case symbol_kind::S_68_assign_op: // assign-op
      case symbol_kind::S_85_compare_op: // compare-op
        value.copy< operatorVals > (that.value);
        break;

      case symbol_kind::S_64_zfx_program: // zfx-program
      case symbol_kind::S_65_multi_statements: // multi-statements
      case symbol_kind::S_66_general_statement: // general-statement
      case symbol_kind::S_67_array_or_exp: // array-or-exp
      case symbol_kind::S_69_code_block: // code-block
      case symbol_kind::S_71_assign_statement: // assign-statement
      case symbol_kind::S_72_jump_statement: // jump-statement
      case symbol_kind::S_arrcontent: // arrcontent
      case symbol_kind::S_75_array_stmt: // array-stmt
      case symbol_kind::S_77_only_declare: // only-declare
      case symbol_kind::S_78_declare_statement: // declare-statement
      case symbol_kind::S_79_if_statement: // if-statement
      case symbol_kind::S_80_for_begin: // for-begin
      case symbol_kind::S_81_for_condition: // for-condition
      case symbol_kind::S_82_for_step: // for-step
      case symbol_kind::S_84_loop_statement: // loop-statement
      case symbol_kind::S_86_exp_statement: // exp-statement
      case symbol_kind::S_orexp: // orexp
      case symbol_kind::S_andexp: // andexp
      case symbol_kind::S_compareexp: // compareexp
      case symbol_kind::S_addsubexp: // addsubexp
      case symbol_kind::S_factor: // factor
      case symbol_kind::S_zenvar: // zenvar
      case symbol_kind::S_94_func_content: // func-content
      case symbol_kind::S_term: // term
        value.copy< std::shared_ptr<ZfxASTNode> > (that.value);
        break;

      case symbol_kind::S_arrcontents: // arrcontents
      case symbol_kind::S_83_foreach_step: // foreach-step
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
      case symbol_kind::S_ASSIGNTO: // ASSIGNTO
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
      case symbol_kind::S_LESSTHAN: // LESSTHAN
      case symbol_kind::S_LESSEQUAL: // LESSEQUAL
      case symbol_kind::S_GREATTHAN: // GREATTHAN
      case symbol_kind::S_GREATEQUAL: // GREATEQUAL
      case symbol_kind::S_RETURN: // RETURN
      case symbol_kind::S_CONTINUE: // CONTINUE
      case symbol_kind::S_BREAK: // BREAK
      case symbol_kind::S_TYPE: // TYPE
      case symbol_kind::S_ATTRAT: // ATTRAT
      case symbol_kind::S_FOREACH: // FOREACH
      case symbol_kind::S_DO: // DO
      case symbol_kind::S_EQUALTO: // EQUALTO
      case symbol_kind::S_NOTEQUAL: // NOTEQUAL
      case symbol_kind::S_OR: // OR
      case symbol_kind::S_AND: // AND
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
      case symbol_kind::S_70_bool_stmt: // bool-stmt
      case symbol_kind::S_76_array_mark: // array-mark
        value.move< bool > (that.value);
        break;

      case symbol_kind::S_NUMBER: // NUMBER
        value.move< float > (that.value);
        break;

      case symbol_kind::S_68_assign_op: // assign-op
      case symbol_kind::S_85_compare_op: // compare-op
        value.move< operatorVals > (that.value);
        break;

      case symbol_kind::S_64_zfx_program: // zfx-program
      case symbol_kind::S_65_multi_statements: // multi-statements
      case symbol_kind::S_66_general_statement: // general-statement
      case symbol_kind::S_67_array_or_exp: // array-or-exp
      case symbol_kind::S_69_code_block: // code-block
      case symbol_kind::S_71_assign_statement: // assign-statement
      case symbol_kind::S_72_jump_statement: // jump-statement
      case symbol_kind::S_arrcontent: // arrcontent
      case symbol_kind::S_75_array_stmt: // array-stmt
      case symbol_kind::S_77_only_declare: // only-declare
      case symbol_kind::S_78_declare_statement: // declare-statement
      case symbol_kind::S_79_if_statement: // if-statement
      case symbol_kind::S_80_for_begin: // for-begin
      case symbol_kind::S_81_for_condition: // for-condition
      case symbol_kind::S_82_for_step: // for-step
      case symbol_kind::S_84_loop_statement: // loop-statement
      case symbol_kind::S_86_exp_statement: // exp-statement
      case symbol_kind::S_orexp: // orexp
      case symbol_kind::S_andexp: // andexp
      case symbol_kind::S_compareexp: // compareexp
      case symbol_kind::S_addsubexp: // addsubexp
      case symbol_kind::S_factor: // factor
      case symbol_kind::S_zenvar: // zenvar
      case symbol_kind::S_94_func_content: // func-content
      case symbol_kind::S_term: // term
        value.move< std::shared_ptr<ZfxASTNode> > (that.value);
        break;

      case symbol_kind::S_arrcontents: // arrcontents
      case symbol_kind::S_83_foreach_step: // foreach-step
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
      case symbol_kind::S_ASSIGNTO: // ASSIGNTO
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
      case symbol_kind::S_LESSTHAN: // LESSTHAN
      case symbol_kind::S_LESSEQUAL: // LESSEQUAL
      case symbol_kind::S_GREATTHAN: // GREATTHAN
      case symbol_kind::S_GREATEQUAL: // GREATEQUAL
      case symbol_kind::S_RETURN: // RETURN
      case symbol_kind::S_CONTINUE: // CONTINUE
      case symbol_kind::S_BREAK: // BREAK
      case symbol_kind::S_TYPE: // TYPE
      case symbol_kind::S_ATTRAT: // ATTRAT
      case symbol_kind::S_FOREACH: // FOREACH
      case symbol_kind::S_DO: // DO
      case symbol_kind::S_EQUALTO: // EQUALTO
      case symbol_kind::S_NOTEQUAL: // NOTEQUAL
      case symbol_kind::S_OR: // OR
      case symbol_kind::S_AND: // AND
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
      case symbol_kind::S_70_bool_stmt: // bool-stmt
      case symbol_kind::S_76_array_mark: // array-mark
        yylhs.value.emplace< bool > ();
        break;

      case symbol_kind::S_NUMBER: // NUMBER
        yylhs.value.emplace< float > ();
        break;

      case symbol_kind::S_68_assign_op: // assign-op
      case symbol_kind::S_85_compare_op: // compare-op
        yylhs.value.emplace< operatorVals > ();
        break;

      case symbol_kind::S_64_zfx_program: // zfx-program
      case symbol_kind::S_65_multi_statements: // multi-statements
      case symbol_kind::S_66_general_statement: // general-statement
      case symbol_kind::S_67_array_or_exp: // array-or-exp
      case symbol_kind::S_69_code_block: // code-block
      case symbol_kind::S_71_assign_statement: // assign-statement
      case symbol_kind::S_72_jump_statement: // jump-statement
      case symbol_kind::S_arrcontent: // arrcontent
      case symbol_kind::S_75_array_stmt: // array-stmt
      case symbol_kind::S_77_only_declare: // only-declare
      case symbol_kind::S_78_declare_statement: // declare-statement
      case symbol_kind::S_79_if_statement: // if-statement
      case symbol_kind::S_80_for_begin: // for-begin
      case symbol_kind::S_81_for_condition: // for-condition
      case symbol_kind::S_82_for_step: // for-step
      case symbol_kind::S_84_loop_statement: // loop-statement
      case symbol_kind::S_86_exp_statement: // exp-statement
      case symbol_kind::S_orexp: // orexp
      case symbol_kind::S_andexp: // andexp
      case symbol_kind::S_compareexp: // compareexp
      case symbol_kind::S_addsubexp: // addsubexp
      case symbol_kind::S_factor: // factor
      case symbol_kind::S_zenvar: // zenvar
      case symbol_kind::S_94_func_content: // func-content
      case symbol_kind::S_term: // term
        yylhs.value.emplace< std::shared_ptr<ZfxASTNode> > ();
        break;

      case symbol_kind::S_arrcontents: // arrcontents
      case symbol_kind::S_83_foreach_step: // foreach-step
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
      case symbol_kind::S_ASSIGNTO: // ASSIGNTO
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
      case symbol_kind::S_LESSTHAN: // LESSTHAN
      case symbol_kind::S_LESSEQUAL: // LESSEQUAL
      case symbol_kind::S_GREATTHAN: // GREATTHAN
      case symbol_kind::S_GREATEQUAL: // GREATEQUAL
      case symbol_kind::S_RETURN: // RETURN
      case symbol_kind::S_CONTINUE: // CONTINUE
      case symbol_kind::S_BREAK: // BREAK
      case symbol_kind::S_TYPE: // TYPE
      case symbol_kind::S_ATTRAT: // ATTRAT
      case symbol_kind::S_FOREACH: // FOREACH
      case symbol_kind::S_DO: // DO
      case symbol_kind::S_EQUALTO: // EQUALTO
      case symbol_kind::S_NOTEQUAL: // NOTEQUAL
      case symbol_kind::S_OR: // OR
      case symbol_kind::S_AND: // AND
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
#line 139 "zfxparser.y"
                 {
            std::cout << "END" << std::endl;
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(CODEBLOCK, DEFAULT_FUNCVAL, {});
            driver.setASTResult(yylhs.value.as < std::shared_ptr<ZfxASTNode> > ());
        }
#line 1033 "zfxparser.cpp"
    break;

  case 3: // zfx-program: multi-statements zfx-program
#line 144 "zfxparser.y"
                                   {
            addChild(yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[1].value.as < std::shared_ptr<ZfxASTNode> > ());
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ();
        }
#line 1042 "zfxparser.cpp"
    break;

  case 4: // multi-statements: %empty
#line 150 "zfxparser.y"
                         {
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(CODEBLOCK, DEFAULT_FUNCVAL, {});
        }
#line 1050 "zfxparser.cpp"
    break;

  case 5: // multi-statements: general-statement multi-statements
#line 153 "zfxparser.y"
                                         {
            addChild(yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[1].value.as < std::shared_ptr<ZfxASTNode> > ());
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ();
        }
#line 1059 "zfxparser.cpp"
    break;

  case 6: // general-statement: declare-statement SEMICOLON
#line 159 "zfxparser.y"
                                               { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[1].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1065 "zfxparser.cpp"
    break;

  case 7: // general-statement: assign-statement SEMICOLON
#line 160 "zfxparser.y"
                                 { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[1].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1071 "zfxparser.cpp"
    break;

  case 8: // general-statement: if-statement
#line 161 "zfxparser.y"
                   { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1077 "zfxparser.cpp"
    break;

  case 9: // general-statement: loop-statement
#line 162 "zfxparser.y"
                     { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1083 "zfxparser.cpp"
    break;

  case 10: // general-statement: jump-statement SEMICOLON
#line 163 "zfxparser.y"
                               { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[1].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1089 "zfxparser.cpp"
    break;

  case 11: // general-statement: exp-statement SEMICOLON
#line 164 "zfxparser.y"
                              { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[1].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1095 "zfxparser.cpp"
    break;

  case 12: // general-statement: code-block
#line 165 "zfxparser.y"
                 { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1101 "zfxparser.cpp"
    break;

  case 13: // array-or-exp: exp-statement
#line 168 "zfxparser.y"
                            { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1107 "zfxparser.cpp"
    break;

  case 14: // array-or-exp: array-stmt
#line 169 "zfxparser.y"
                 { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1113 "zfxparser.cpp"
    break;

  case 15: // assign-op: ASSIGNTO
#line 172 "zfxparser.y"
                    { yylhs.value.as < operatorVals > () = AssignTo; }
#line 1119 "zfxparser.cpp"
    break;

  case 16: // assign-op: ADDASSIGN
#line 173 "zfxparser.y"
                { yylhs.value.as < operatorVals > () = AddAssign; }
#line 1125 "zfxparser.cpp"
    break;

  case 17: // assign-op: MULASSIGN
#line 174 "zfxparser.y"
                { yylhs.value.as < operatorVals > () = MulAssign; }
#line 1131 "zfxparser.cpp"
    break;

  case 18: // assign-op: SUBASSIGN
#line 175 "zfxparser.y"
                { yylhs.value.as < operatorVals > () = SubAssign; }
#line 1137 "zfxparser.cpp"
    break;

  case 19: // assign-op: DIVASSIGN
#line 176 "zfxparser.y"
                { yylhs.value.as < operatorVals > () = DivAssign; }
#line 1143 "zfxparser.cpp"
    break;

  case 20: // code-block: LBRACKET multi-statements RBRACKET
#line 179 "zfxparser.y"
                                               { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[1].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1149 "zfxparser.cpp"
    break;

  case 21: // bool-stmt: TRUE
#line 182 "zfxparser.y"
                { yylhs.value.as < bool > () = true; }
#line 1155 "zfxparser.cpp"
    break;

  case 22: // bool-stmt: FALSE
#line 183 "zfxparser.y"
            { yylhs.value.as < bool > () = false; }
#line 1161 "zfxparser.cpp"
    break;

  case 23: // assign-statement: zenvar assign-op array-or-exp
#line 186 "zfxparser.y"
                                                {
            std::vector<std::shared_ptr<ZfxASTNode>> children({yystack_[2].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()});
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(ASSIGNMENT, yystack_[1].value.as < operatorVals > (), children);
        }
#line 1170 "zfxparser.cpp"
    break;

  case 24: // jump-statement: BREAK
#line 192 "zfxparser.y"
                      { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(JUMP, JUMP_BREAK, {}); }
#line 1176 "zfxparser.cpp"
    break;

  case 25: // jump-statement: RETURN
#line 193 "zfxparser.y"
               { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(JUMP, JUMP_RETURN, {}); }
#line 1182 "zfxparser.cpp"
    break;

  case 26: // jump-statement: CONTINUE
#line 194 "zfxparser.y"
               { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(JUMP, JUMP_CONTINUE, {}); }
#line 1188 "zfxparser.cpp"
    break;

  case 27: // arrcontent: exp-statement
#line 197 "zfxparser.y"
                          { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1194 "zfxparser.cpp"
    break;

  case 28: // arrcontent: array-stmt
#line 198 "zfxparser.y"
                 { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1200 "zfxparser.cpp"
    break;

  case 29: // arrcontents: arrcontent
#line 201 "zfxparser.y"
                                   { yylhs.value.as < std::vector<std::shared_ptr<ZfxASTNode>> > () = std::vector<std::shared_ptr<ZfxASTNode>>({yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()}); }
#line 1206 "zfxparser.cpp"
    break;

  case 30: // arrcontents: arrcontents COMMA arrcontent
#line 202 "zfxparser.y"
                                   { yystack_[2].value.as < std::vector<std::shared_ptr<ZfxASTNode>> > ().push_back(yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()); yylhs.value.as < std::vector<std::shared_ptr<ZfxASTNode>> > () = yystack_[2].value.as < std::vector<std::shared_ptr<ZfxASTNode>> > (); }
#line 1212 "zfxparser.cpp"
    break;

  case 31: // array-stmt: LBRACKET arrcontents RBRACKET
#line 205 "zfxparser.y"
                                          { 
        yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(ARRAY, DEFAULT_FUNCVAL, yystack_[1].value.as < std::vector<std::shared_ptr<ZfxASTNode>> > ());
    }
#line 1220 "zfxparser.cpp"
    break;

  case 32: // array-mark: %empty
#line 210 "zfxparser.y"
                   { yylhs.value.as < bool > () = false; }
#line 1226 "zfxparser.cpp"
    break;

  case 33: // array-mark: LSQBRACKET RSQBRACKET
#line 211 "zfxparser.y"
                            { yylhs.value.as < bool > () = true; }
#line 1232 "zfxparser.cpp"
    break;

  case 34: // only-declare: TYPE VARNAME array-mark
#line 214 "zfxparser.y"
                                      {
    auto typeNode = driver.makeTypeNode(yystack_[2].value.as < string > (), yystack_[0].value.as < bool > ());
    auto nameNode = driver.makeZfxVarNode(yystack_[1].value.as < string > ());
    std::vector<std::shared_ptr<ZfxASTNode>> children({typeNode, nameNode});
    yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(DECLARE, DEFAULT_FUNCVAL, children);
}
#line 1243 "zfxparser.cpp"
    break;

  case 35: // declare-statement: only-declare
#line 221 "zfxparser.y"
                                {
                yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ();
            }
#line 1251 "zfxparser.cpp"
    break;

  case 36: // declare-statement: TYPE VARNAME array-mark ASSIGNTO array-or-exp
#line 224 "zfxparser.y"
                                                    {
                auto typeNode = driver.makeTypeNode(yystack_[4].value.as < string > (), yystack_[2].value.as < bool > ());
                auto nameNode = driver.makeZfxVarNode(yystack_[3].value.as < string > ());
                std::vector<std::shared_ptr<ZfxASTNode>> children({typeNode, nameNode, yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()});
                yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(DECLARE, DEFAULT_FUNCVAL, children);
            }
#line 1262 "zfxparser.cpp"
    break;

  case 37: // if-statement: IF LPAREN exp-statement RPAREN code-block
#line 234 "zfxparser.y"
                                                        {
            std::vector<std::shared_ptr<ZfxASTNode>> children({yystack_[2].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()});
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(IF, DEFAULT_FUNCVAL, children);
        }
#line 1271 "zfxparser.cpp"
    break;

  case 38: // for-begin: SEMICOLON
#line 246 "zfxparser.y"
                     { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeEmptyNode(); }
#line 1277 "zfxparser.cpp"
    break;

  case 39: // for-begin: declare-statement SEMICOLON
#line 247 "zfxparser.y"
                                  { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[1].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1283 "zfxparser.cpp"
    break;

  case 40: // for-begin: assign-statement SEMICOLON
#line 248 "zfxparser.y"
                                 { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[1].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1289 "zfxparser.cpp"
    break;

  case 41: // for-begin: exp-statement SEMICOLON
#line 249 "zfxparser.y"
                              { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[1].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1295 "zfxparser.cpp"
    break;

  case 42: // for-condition: SEMICOLON
#line 252 "zfxparser.y"
                          { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeEmptyNode(); }
#line 1301 "zfxparser.cpp"
    break;

  case 43: // for-condition: exp-statement SEMICOLON
#line 253 "zfxparser.y"
                              { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[1].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1307 "zfxparser.cpp"
    break;

  case 44: // for-step: %empty
#line 256 "zfxparser.y"
                 { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeEmptyNode(); }
#line 1313 "zfxparser.cpp"
    break;

  case 45: // for-step: exp-statement
#line 257 "zfxparser.y"
                    { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1319 "zfxparser.cpp"
    break;

  case 46: // for-step: assign-statement
#line 258 "zfxparser.y"
                       { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1325 "zfxparser.cpp"
    break;

  case 47: // foreach-step: VARNAME
#line 261 "zfxparser.y"
                      {
            auto varNode = driver.makeZfxVarNode(yystack_[0].value.as < string > ());
            yylhs.value.as < std::vector<std::shared_ptr<ZfxASTNode>> > () = std::vector<std::shared_ptr<ZfxASTNode>>({varNode});
        }
#line 1334 "zfxparser.cpp"
    break;

  case 48: // foreach-step: TYPE VARNAME
#line 265 "zfxparser.y"
                   {
            /* 类型不是必要的，只是为了兼容一些编程习惯，比如foreach(int a : arr)*/
            auto varNode = driver.makeZfxVarNode(yystack_[0].value.as < string > ());
            yylhs.value.as < std::vector<std::shared_ptr<ZfxASTNode>> > () = std::vector<std::shared_ptr<ZfxASTNode>>({varNode});
        }
#line 1344 "zfxparser.cpp"
    break;

  case 49: // foreach-step: LSQBRACKET VARNAME COMMA VARNAME RSQBRACKET
#line 270 "zfxparser.y"
                                                  {
            auto idxNode = driver.makeZfxVarNode(yystack_[3].value.as < string > ());
            auto varNode = driver.makeZfxVarNode(yystack_[1].value.as < string > ());
            yylhs.value.as < std::vector<std::shared_ptr<ZfxASTNode>> > () = std::vector<std::shared_ptr<ZfxASTNode>>({idxNode, varNode});
        }
#line 1354 "zfxparser.cpp"
    break;

  case 50: // loop-statement: FOR LPAREN for-begin for-condition for-step RPAREN code-block
#line 277 "zfxparser.y"
                                                                              {
            std::vector<std::shared_ptr<ZfxASTNode>> children({yystack_[4].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[3].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[2].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()});
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(FOR, DEFAULT_FUNCVAL, children);
        }
#line 1363 "zfxparser.cpp"
    break;

  case 51: // loop-statement: FOREACH LPAREN foreach-step COLON zenvar RPAREN code-block
#line 281 "zfxparser.y"
                                                                 {
            yystack_[4].value.as < std::vector<std::shared_ptr<ZfxASTNode>> > ().push_back(yystack_[2].value.as < std::shared_ptr<ZfxASTNode> > ());
            yystack_[4].value.as < std::vector<std::shared_ptr<ZfxASTNode>> > ().push_back(yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ());
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(FOREACH, DEFAULT_FUNCVAL, yystack_[4].value.as < std::vector<std::shared_ptr<ZfxASTNode>> > ());
        }
#line 1373 "zfxparser.cpp"
    break;

  case 52: // loop-statement: WHILE LPAREN exp-statement RPAREN code-block
#line 286 "zfxparser.y"
                                                   {
            std::vector<std::shared_ptr<ZfxASTNode>> children({yystack_[2].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()});
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(WHILE, DEFAULT_FUNCVAL, children);
        }
#line 1382 "zfxparser.cpp"
    break;

  case 53: // loop-statement: DO code-block WHILE LPAREN exp-statement RPAREN SEMICOLON
#line 290 "zfxparser.y"
                                                                {
            std::vector<std::shared_ptr<ZfxASTNode>> children({yystack_[5].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[2].value.as < std::shared_ptr<ZfxASTNode> > ()});
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(DOWHILE, DEFAULT_FUNCVAL, children);
        }
#line 1391 "zfxparser.cpp"
    break;

  case 54: // compare-op: LESSTHAN
#line 296 "zfxparser.y"
                     { yylhs.value.as < operatorVals > () = Less; }
#line 1397 "zfxparser.cpp"
    break;

  case 55: // compare-op: LESSEQUAL
#line 297 "zfxparser.y"
                { yylhs.value.as < operatorVals > () = LessEqual; }
#line 1403 "zfxparser.cpp"
    break;

  case 56: // compare-op: GREATTHAN
#line 298 "zfxparser.y"
                { yylhs.value.as < operatorVals > () = Greater; }
#line 1409 "zfxparser.cpp"
    break;

  case 57: // compare-op: GREATEQUAL
#line 299 "zfxparser.y"
                 { yylhs.value.as < operatorVals > () = GreaterEqual; }
#line 1415 "zfxparser.cpp"
    break;

  case 58: // compare-op: EQUALTO
#line 300 "zfxparser.y"
              { yylhs.value.as < operatorVals > () = Equal; }
#line 1421 "zfxparser.cpp"
    break;

  case 59: // compare-op: NOTEQUAL
#line 301 "zfxparser.y"
               { yylhs.value.as < operatorVals > () = NotEqual; }
#line 1427 "zfxparser.cpp"
    break;

  case 60: // exp-statement: orexp
#line 304 "zfxparser.y"
                               { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1433 "zfxparser.cpp"
    break;

  case 61: // orexp: andexp
#line 307 "zfxparser.y"
                        { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1439 "zfxparser.cpp"
    break;

  case 62: // orexp: orexp OR andexp
#line 308 "zfxparser.y"
                        {
        std::vector<std::shared_ptr<ZfxASTNode>>children({yystack_[2].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()});
        yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(FOUROPERATIONS, OR, children);
    }
#line 1448 "zfxparser.cpp"
    break;

  case 63: // andexp: compareexp
#line 314 "zfxparser.y"
                        { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1454 "zfxparser.cpp"
    break;

  case 64: // andexp: andexp AND compareexp
#line 315 "zfxparser.y"
                            {
        std::vector<std::shared_ptr<ZfxASTNode>>children({yystack_[2].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()});
        yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(FOUROPERATIONS, AND, children);    
    }
#line 1463 "zfxparser.cpp"
    break;

  case 65: // compareexp: addsubexp
#line 321 "zfxparser.y"
                        { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1469 "zfxparser.cpp"
    break;

  case 66: // compareexp: compareexp compare-op addsubexp
#line 322 "zfxparser.y"
                                      {
            std::vector<std::shared_ptr<ZfxASTNode>>children({yystack_[2].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()});
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(COMPOP, yystack_[1].value.as < operatorVals > (), children);
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > ()->value = yystack_[1].value.as < operatorVals > ();
        }
#line 1479 "zfxparser.cpp"
    break;

  case 67: // compareexp: compareexp compare-op addsubexp QUESTION exp-statement COLON exp-statement
#line 327 "zfxparser.y"
                                                                                 {
            std::vector<std::shared_ptr<ZfxASTNode>> children({yystack_[6].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[4].value.as < std::shared_ptr<ZfxASTNode> > ()});
            auto spCond = driver.makeNewNode(COMPOP, yystack_[5].value.as < operatorVals > (), children);
            spCond->value = yystack_[5].value.as < operatorVals > ();

            std::vector<std::shared_ptr<ZfxASTNode>> exps({spCond, yystack_[2].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()});
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(CONDEXP, DEFAULT_FUNCVAL, exps);
        }
#line 1492 "zfxparser.cpp"
    break;

  case 68: // addsubexp: factor
#line 337 "zfxparser.y"
                               { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1498 "zfxparser.cpp"
    break;

  case 69: // addsubexp: addsubexp ADD factor
#line 338 "zfxparser.y"
                           {
                std::vector<std::shared_ptr<ZfxASTNode>> children({yystack_[2].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()});
                yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(FOUROPERATIONS, PLUS, children);
            }
#line 1507 "zfxparser.cpp"
    break;

  case 70: // addsubexp: addsubexp SUB factor
#line 342 "zfxparser.y"
                           {
                std::vector<std::shared_ptr<ZfxASTNode>> children({yystack_[2].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()});
                yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(FOUROPERATIONS, MINUS, children);
            }
#line 1516 "zfxparser.cpp"
    break;

  case 71: // factor: term
#line 350 "zfxparser.y"
                        { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1522 "zfxparser.cpp"
    break;

  case 72: // factor: factor MUL term
#line 351 "zfxparser.y"
                        {
                std::vector<std::shared_ptr<ZfxASTNode>>children({yystack_[2].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()});
                yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(FOUROPERATIONS, MUL, children);
            }
#line 1531 "zfxparser.cpp"
    break;

  case 73: // factor: factor DIV term
#line 355 "zfxparser.y"
                      {
            std::vector<std::shared_ptr<ZfxASTNode>>children({yystack_[2].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()});
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(FOUROPERATIONS, DIV, children);
        }
#line 1540 "zfxparser.cpp"
    break;

  case 74: // zenvar: DOLLARVARNAME
#line 361 "zfxparser.y"
                        { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeZfxVarNode(yystack_[0].value.as < string > (), BulitInVar); }
#line 1546 "zfxparser.cpp"
    break;

  case 75: // zenvar: VARNAME
#line 362 "zfxparser.y"
                        { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeZfxVarNode(yystack_[0].value.as < string > ()); }
#line 1552 "zfxparser.cpp"
    break;

  case 76: // zenvar: ATTRAT zenvar
#line 363 "zfxparser.y"
                    {
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ();
            driver.markZfxAttr(yylhs.value.as < std::shared_ptr<ZfxASTNode> > ());
        }
#line 1561 "zfxparser.cpp"
    break;

  case 77: // zenvar: zenvar DOT VARNAME
#line 367 "zfxparser.y"
                         {
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeComponentVisit(yystack_[2].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[0].value.as < string > ());
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > ()->opVal = COMPVISIT;
        }
#line 1570 "zfxparser.cpp"
    break;

  case 78: // zenvar: zenvar LSQBRACKET exp-statement RSQBRACKET
#line 371 "zfxparser.y"
                                                 {
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[3].value.as < std::shared_ptr<ZfxASTNode> > ();
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > ()->opVal = Indexing;
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > ()->children.push_back(yystack_[1].value.as < std::shared_ptr<ZfxASTNode> > ());
        }
#line 1580 "zfxparser.cpp"
    break;

  case 79: // zenvar: AUTOINC zenvar
#line 376 "zfxparser.y"
                     {
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ();
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > ()->opVal = AutoIncreaseFirst;
        }
#line 1589 "zfxparser.cpp"
    break;

  case 80: // zenvar: zenvar AUTOINC
#line 380 "zfxparser.y"
                     {
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[1].value.as < std::shared_ptr<ZfxASTNode> > ();
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > ()->opVal = AutoIncreaseLast;
        }
#line 1598 "zfxparser.cpp"
    break;

  case 81: // zenvar: AUTODEC zenvar
#line 384 "zfxparser.y"
                     {
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ();
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > ()->opVal = AutoDecreaseFirst;
        }
#line 1607 "zfxparser.cpp"
    break;

  case 82: // zenvar: zenvar AUTODEC
#line 388 "zfxparser.y"
                     {
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[1].value.as < std::shared_ptr<ZfxASTNode> > ();
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > ()->opVal = AutoDecreaseLast;
        }
#line 1616 "zfxparser.cpp"
    break;

  case 83: // funcargs: %empty
#line 394 "zfxparser.y"
                 { yylhs.value.as < std::vector<std::shared_ptr<ZfxASTNode>> > () = std::vector<std::shared_ptr<ZfxASTNode>>(); }
#line 1622 "zfxparser.cpp"
    break;

  case 84: // funcargs: exp-statement
#line 395 "zfxparser.y"
                               { yylhs.value.as < std::vector<std::shared_ptr<ZfxASTNode>> > () = std::vector<std::shared_ptr<ZfxASTNode>>({yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()}); }
#line 1628 "zfxparser.cpp"
    break;

  case 85: // funcargs: funcargs COMMA exp-statement
#line 396 "zfxparser.y"
                                   { yystack_[2].value.as < std::vector<std::shared_ptr<ZfxASTNode>> > ().push_back(yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()); yylhs.value.as < std::vector<std::shared_ptr<ZfxASTNode>> > () = yystack_[2].value.as < std::vector<std::shared_ptr<ZfxASTNode>> > (); }
#line 1634 "zfxparser.cpp"
    break;

  case 86: // func-content: LPAREN funcargs RPAREN
#line 400 "zfxparser.y"
                                     { 
        yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(FUNC, DEFAULT_FUNCVAL, yystack_[1].value.as < std::vector<std::shared_ptr<ZfxASTNode>> > ());
        yylhs.value.as < std::shared_ptr<ZfxASTNode> > ()->isParenthesisNodeComplete = true;
        yylhs.value.as < std::shared_ptr<ZfxASTNode> > ()->func_match = Match_Exactly;
    }
#line 1644 "zfxparser.cpp"
    break;

  case 87: // term: NUMBER
#line 408 "zfxparser.y"
                        { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNumberNode(yystack_[0].value.as < float > ()); }
#line 1650 "zfxparser.cpp"
    break;

  case 88: // term: bool-stmt
#line 409 "zfxparser.y"
                        { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeBoolNode(yystack_[0].value.as < bool > ()); }
#line 1656 "zfxparser.cpp"
    break;

  case 89: // term: LITERAL
#line 410 "zfxparser.y"
                        { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeStringNode(yystack_[0].value.as < string > ()); }
#line 1662 "zfxparser.cpp"
    break;

  case 90: // term: UNCOMPSTR
#line 411 "zfxparser.y"
                        { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeQuoteStringNode(yystack_[0].value.as < string > ()); }
#line 1668 "zfxparser.cpp"
    break;

  case 91: // term: LPAREN exp-statement RPAREN
#line 412 "zfxparser.y"
                                  { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[1].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1674 "zfxparser.cpp"
    break;

  case 92: // term: SUB exp-statement
#line 413 "zfxparser.y"
                                  { yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()->value = -1 * std::get<float>(yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()->value); yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1680 "zfxparser.cpp"
    break;

  case 93: // term: zenvar
#line 414 "zfxparser.y"
                        { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1686 "zfxparser.cpp"
    break;

  case 94: // term: VARNAME func-content
#line 415 "zfxparser.y"
                            { 
        yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ();
        yylhs.value.as < std::shared_ptr<ZfxASTNode> > ()->opVal = DEFAULT_FUNCVAL;
        yylhs.value.as < std::shared_ptr<ZfxASTNode> > ()->type = FUNC;
        yylhs.value.as < std::shared_ptr<ZfxASTNode> > ()->value = yystack_[1].value.as < string > ();
        yylhs.value.as < std::shared_ptr<ZfxASTNode> > ()->isParenthesisNode = true;
    }
#line 1698 "zfxparser.cpp"
    break;


#line 1702 "zfxparser.cpp"

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


  const signed char  ZfxParser ::yypact_ninf_ = -113;

  const signed char  ZfxParser ::yytable_ninf_ = -1;

  const short
   ZfxParser ::yypact_[] =
  {
     129,  -113,  -113,  -113,  -113,  -113,  -113,  -113,   173,   -57,
     -50,   -45,   -31,   -10,   -10,  -113,  -113,  -113,    28,   -10,
      -7,    42,   264,   264,    67,   129,   173,  -113,  -113,    48,
      52,  -113,    53,  -113,  -113,    58,    33,    35,    19,   -20,
      -8,    85,  -113,    66,   264,  -113,   264,    40,   264,  -113,
      49,    49,    57,    49,   -13,    61,  -113,    49,    88,  -113,
    -113,  -113,  -113,  -113,  -113,  -113,   264,   264,  -113,  -113,
    -113,  -113,  -113,  -113,   264,   264,   264,   264,   264,    68,
    -113,  -113,  -113,   264,  -113,  -113,  -113,  -113,   217,  -113,
    -113,     4,    90,  -113,    71,    72,   237,    73,    97,    74,
      75,  -113,    81,    82,    93,    56,  -113,    35,    19,    -5,
      -8,    -8,  -113,  -113,  -113,    91,   217,  -113,  -113,  -113,
    -113,   264,    42,  -113,  -113,  -113,   264,    98,  -113,    42,
    -113,   217,   113,  -113,   -10,   264,   264,  -113,  -113,    -4,
    -113,  -113,  -113,  -113,  -113,   123,  -113,  -113,  -113,  -113,
     103,     7,   125,   111,   217,  -113,    42,   100,    42,   106,
     264,  -113,  -113,  -113,  -113,  -113,  -113
  };

  const signed char
   ZfxParser ::yydefact_[] =
  {
       0,     2,    87,    21,    22,    89,    90,    74,     4,    75,
       0,     0,     0,     0,     0,    25,    26,    24,     0,     0,
       0,     0,     0,     0,     0,     0,     4,    12,    88,     0,
       0,    35,     0,     8,     9,     0,    60,    61,    63,    65,
      68,    93,    71,     0,    83,    94,     0,     0,     0,    75,
      79,    81,    32,    76,     0,     0,    92,    93,     0,     1,
       3,     5,     7,    10,     6,    11,     0,     0,    54,    55,
      56,    57,    58,    59,     0,     0,     0,     0,     0,     0,
      15,    80,    82,     0,    16,    17,    18,    19,     0,    20,
      84,     0,     0,    38,     0,     0,     0,     0,     0,     0,
      34,    47,     0,     0,     0,     0,    91,    62,    64,    66,
      69,    70,    72,    73,    77,     0,     0,    23,    14,    13,
      86,     0,     0,    40,    39,    42,    44,     0,    41,     0,
      33,     0,     0,    48,     0,     0,     0,    78,    29,     0,
      28,    27,    85,    37,    46,     0,    45,    43,    52,    36,
       0,     0,     0,     0,     0,    31,     0,     0,     0,     0,
       0,    30,    50,    49,    51,    53,    67
  };

  const signed char
   ZfxParser ::yypgoto_[] =
  {
    -113,   115,     1,  -113,    13,  -113,   -19,  -113,   -44,  -113,
      -6,  -113,  -112,  -113,  -113,    99,  -113,  -113,  -113,  -113,
    -113,  -113,  -113,   -22,  -113,    86,    80,    77,   -32,    15,
    -113,  -113,    -2
  };

  const unsigned char
   ZfxParser ::yydefgoto_[] =
  {
       0,    24,    25,    26,   117,    88,    27,    28,    29,    30,
     138,   139,   118,   100,    31,    32,    33,    96,   126,   145,
     104,    34,    74,    35,    36,    37,    38,    39,    40,    57,
      91,    45,    42
  };

  const unsigned char
   ZfxParser ::yytable_[] =
  {
      56,    58,    55,    94,   140,    44,     7,   120,   154,    43,
     158,   101,    46,   136,    49,    41,   121,    47,   155,   102,
      13,    14,    90,    41,    92,    97,    98,    61,    50,    51,
      79,    48,   103,    75,    53,    76,    19,    81,    82,    83,
      41,    41,   140,   110,   111,     2,     3,     4,    75,    77,
      76,    78,    52,     5,     6,    54,     7,    68,    69,    70,
      71,   115,    41,     8,     9,    93,   119,    59,    72,    73,
      13,    14,    79,    62,   127,   112,   113,    63,    64,    81,
      82,    83,   144,    65,    66,    18,    19,    67,    89,    99,
     105,   106,   114,   122,   141,    22,   123,   124,   128,   142,
     129,   131,    23,   143,   146,   132,   133,   130,    79,   119,
     148,    80,   134,   152,   153,    81,    82,    83,   135,    84,
      85,    86,    87,   147,   137,   150,   156,   157,   159,     1,
     160,   165,   141,   163,     2,     3,     4,   162,   166,   164,
      60,    41,     5,     6,   149,     7,    95,   108,   161,   151,
       8,   109,   107,     9,     0,     0,    10,    11,    12,    13,
      14,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    15,    16,    17,    18,    19,    20,    21,     2,     3,
       4,     0,     0,     0,    22,     0,     5,     6,     0,     7,
       0,    23,     0,     0,     8,     0,     0,     9,     0,     0,
      10,    11,    12,    13,    14,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    15,    16,    17,    18,    19,
      20,    21,     2,     3,     4,     0,     0,     0,    22,     0,
       5,     6,     0,     7,     0,    23,     0,     0,   116,     0,
       0,     9,     2,     3,     4,     0,     0,    13,    14,     0,
       5,     6,     0,     7,     0,     0,     0,     0,     0,     0,
       0,     9,   125,    19,     0,     0,     0,    13,    14,     2,
       3,     4,    22,     0,     0,     0,     0,     5,     6,    23,
       7,     0,     0,    19,     0,     0,     0,     0,     9,     0,
       0,     0,    22,     0,    13,    14,     0,     0,     0,    23,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      19,     0,     0,     0,     0,     0,     0,     0,     0,    22,
       0,     0,     0,     0,     0,     0,    23
  };

  const short
   ZfxParser ::yycheck_[] =
  {
      22,    23,    21,    47,   116,    62,    16,     3,    12,     8,
       3,    24,    62,    18,    24,     0,    12,    62,    22,    32,
      30,    31,    44,     8,    46,    47,    48,    26,    13,    14,
      23,    62,    45,    53,    19,    55,    46,    30,    31,    32,
      25,    26,   154,    75,    76,     5,     6,     7,    53,    57,
      55,    59,    24,    13,    14,    62,    16,    38,    39,    40,
      41,    83,    47,    21,    24,    25,    88,     0,    49,    50,
      30,    31,    23,    25,    96,    77,    78,    25,    25,    30,
      31,    32,   126,    25,    51,    45,    46,    52,    22,    32,
      29,     3,    24,     3,   116,    55,    25,    25,    25,   121,
       3,    26,    62,   122,   126,    24,    24,    33,    23,   131,
     129,    26,    19,   135,   136,    30,    31,    32,    62,    34,
      35,    36,    37,    25,    33,    12,     3,    24,     3,     0,
      19,    25,   154,    33,     5,     6,     7,   156,   160,   158,
      25,   126,    13,    14,   131,    16,    47,    67,   154,   134,
      21,    74,    66,    24,    -1,    -1,    27,    28,    29,    30,
      31,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    42,    43,    44,    45,    46,    47,    48,     5,     6,
       7,    -1,    -1,    -1,    55,    -1,    13,    14,    -1,    16,
      -1,    62,    -1,    -1,    21,    -1,    -1,    24,    -1,    -1,
      27,    28,    29,    30,    31,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    42,    43,    44,    45,    46,
      47,    48,     5,     6,     7,    -1,    -1,    -1,    55,    -1,
      13,    14,    -1,    16,    -1,    62,    -1,    -1,    21,    -1,
      -1,    24,     5,     6,     7,    -1,    -1,    30,    31,    -1,
      13,    14,    -1,    16,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    24,    25,    46,    -1,    -1,    -1,    30,    31,     5,
       6,     7,    55,    -1,    -1,    -1,    -1,    13,    14,    62,
      16,    -1,    -1,    46,    -1,    -1,    -1,    -1,    24,    -1,
      -1,    -1,    55,    -1,    30,    31,    -1,    -1,    -1,    62,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      46,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    55,
      -1,    -1,    -1,    -1,    -1,    -1,    62
  };

  const signed char
   ZfxParser ::yystos_[] =
  {
       0,     0,     5,     6,     7,    13,    14,    16,    21,    24,
      27,    28,    29,    30,    31,    42,    43,    44,    45,    46,
      47,    48,    55,    62,    64,    65,    66,    69,    70,    71,
      72,    77,    78,    79,    84,    86,    87,    88,    89,    90,
      91,    92,    95,    65,    62,    94,    62,    62,    62,    24,
      92,    92,    24,    92,    62,    69,    86,    92,    86,     0,
      64,    65,    25,    25,    25,    25,    51,    52,    38,    39,
      40,    41,    49,    50,    85,    53,    55,    57,    59,    23,
      26,    30,    31,    32,    34,    35,    36,    37,    68,    22,
      86,    93,    86,    25,    71,    78,    80,    86,    86,    32,
      76,    24,    32,    45,    83,    29,     3,    88,    89,    90,
      91,    91,    95,    95,    24,    86,    21,    67,    75,    86,
       3,    12,     3,    25,    25,    25,    81,    86,    25,     3,
      33,    26,    24,    24,    19,    62,    18,    33,    73,    74,
      75,    86,    86,    69,    71,    82,    86,    25,    69,    67,
      12,    92,    86,    86,    12,    22,     3,    24,     3,     3,
      19,    73,    69,    33,    69,    25,    86
  };

  const signed char
   ZfxParser ::yyr1_[] =
  {
       0,    63,    64,    64,    65,    65,    66,    66,    66,    66,
      66,    66,    66,    67,    67,    68,    68,    68,    68,    68,
      69,    70,    70,    71,    72,    72,    72,    73,    73,    74,
      74,    75,    76,    76,    77,    78,    78,    79,    80,    80,
      80,    80,    81,    81,    82,    82,    82,    83,    83,    83,
      84,    84,    84,    84,    85,    85,    85,    85,    85,    85,
      86,    87,    87,    88,    88,    89,    89,    89,    90,    90,
      90,    91,    91,    91,    92,    92,    92,    92,    92,    92,
      92,    92,    92,    93,    93,    93,    94,    95,    95,    95,
      95,    95,    95,    95,    95
  };

  const signed char
   ZfxParser ::yyr2_[] =
  {
       0,     2,     1,     2,     0,     2,     2,     2,     1,     1,
       2,     2,     1,     1,     1,     1,     1,     1,     1,     1,
       3,     1,     1,     3,     1,     1,     1,     1,     1,     1,
       3,     3,     0,     2,     3,     1,     5,     5,     1,     2,
       2,     2,     1,     2,     0,     1,     1,     1,     2,     5,
       7,     7,     5,     7,     1,     1,     1,     1,     1,     1,
       1,     1,     3,     1,     3,     1,     3,     7,     1,     3,
       3,     1,     3,     3,     1,     1,     2,     3,     4,     2,
       2,     2,     2,     0,     1,     3,     3,     1,     1,     1,
       1,     3,     2,     1,     2
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
  "ASSIGNTO", "IF", "FOR", "WHILE", "AUTOINC", "AUTODEC", "LSQBRACKET",
  "RSQBRACKET", "ADDASSIGN", "MULASSIGN", "SUBASSIGN", "DIVASSIGN",
  "LESSTHAN", "LESSEQUAL", "GREATTHAN", "GREATEQUAL", "RETURN", "CONTINUE",
  "BREAK", "TYPE", "ATTRAT", "FOREACH", "DO", "EQUALTO", "NOTEQUAL", "OR",
  "AND", "ADD", "\"+\"", "SUB", "\"-\"", "MUL", "\"*\"", "DIV", "\"/\"",
  "NEG", "LPAREN", "$accept", "zfx-program", "multi-statements",
  "general-statement", "array-or-exp", "assign-op", "code-block",
  "bool-stmt", "assign-statement", "jump-statement", "arrcontent",
  "arrcontents", "array-stmt", "array-mark", "only-declare",
  "declare-statement", "if-statement", "for-begin", "for-condition",
  "for-step", "foreach-step", "loop-statement", "compare-op",
  "exp-statement", "orexp", "andexp", "compareexp", "addsubexp", "factor",
  "zenvar", "funcargs", "func-content", "term", YY_NULLPTR
  };
#endif


#if YYDEBUG
  const short
   ZfxParser ::yyrline_[] =
  {
       0,   139,   139,   144,   150,   153,   159,   160,   161,   162,
     163,   164,   165,   168,   169,   172,   173,   174,   175,   176,
     179,   182,   183,   186,   192,   193,   194,   197,   198,   201,
     202,   205,   210,   211,   214,   221,   224,   234,   246,   247,
     248,   249,   252,   253,   256,   257,   258,   261,   265,   270,
     277,   281,   286,   290,   296,   297,   298,   299,   300,   301,
     304,   307,   308,   314,   315,   321,   322,   327,   337,   338,
     342,   350,   351,   355,   361,   362,   363,   367,   371,   376,
     380,   384,   388,   394,   395,   396,   400,   408,   409,   410,
     411,   412,   413,   414,   415
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
#line 2322 "zfxparser.cpp"

#line 426 "zfxparser.y"


// Bison expects us to provide implementation - otherwise linker complains
void zeno::ZfxParser::error(const location &loc , const std::string &message) {
    cout << "Error: " << message << endl << "Error location: " << driver.location() << endl;
}

