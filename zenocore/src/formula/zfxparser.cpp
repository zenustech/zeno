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
      case symbol_kind::S_59_bool_stmt: // bool-stmt
      case symbol_kind::S_65_array_mark: // array-mark
        value.YY_MOVE_OR_COPY< bool > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_NUMBER: // NUMBER
        value.YY_MOVE_OR_COPY< float > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_58_assign_op: // assign-op
        value.YY_MOVE_OR_COPY< operatorVals > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_54_zfx_program: // zfx-program
      case symbol_kind::S_55_multi_statements: // multi-statements
      case symbol_kind::S_56_general_statement: // general-statement
      case symbol_kind::S_57_array_or_exp: // array-or-exp
      case symbol_kind::S_60_assign_statement: // assign-statement
      case symbol_kind::S_61_jump_statement: // jump-statement
      case symbol_kind::S_arrcontent: // arrcontent
      case symbol_kind::S_64_array_stmt: // array-stmt
      case symbol_kind::S_66_declare_statement: // declare-statement
      case symbol_kind::S_67_code_block: // code-block
      case symbol_kind::S_68_if_statement: // if-statement
      case symbol_kind::S_69_for_begin: // for-begin
      case symbol_kind::S_70_for_condition: // for-condition
      case symbol_kind::S_71_for_step: // for-step
      case symbol_kind::S_72_for_statement: // for-statement
      case symbol_kind::S_73_exp_statement: // exp-statement
      case symbol_kind::S_compareexp: // compareexp
      case symbol_kind::S_factor: // factor
      case symbol_kind::S_zenvar: // zenvar
      case symbol_kind::S_78_func_content: // func-content
      case symbol_kind::S_term: // term
        value.YY_MOVE_OR_COPY< std::shared_ptr<ZfxASTNode> > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_arrcontents: // arrcontents
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
      case symbol_kind::S_59_bool_stmt: // bool-stmt
      case symbol_kind::S_65_array_mark: // array-mark
        value.move< bool > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_NUMBER: // NUMBER
        value.move< float > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_58_assign_op: // assign-op
        value.move< operatorVals > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_54_zfx_program: // zfx-program
      case symbol_kind::S_55_multi_statements: // multi-statements
      case symbol_kind::S_56_general_statement: // general-statement
      case symbol_kind::S_57_array_or_exp: // array-or-exp
      case symbol_kind::S_60_assign_statement: // assign-statement
      case symbol_kind::S_61_jump_statement: // jump-statement
      case symbol_kind::S_arrcontent: // arrcontent
      case symbol_kind::S_64_array_stmt: // array-stmt
      case symbol_kind::S_66_declare_statement: // declare-statement
      case symbol_kind::S_67_code_block: // code-block
      case symbol_kind::S_68_if_statement: // if-statement
      case symbol_kind::S_69_for_begin: // for-begin
      case symbol_kind::S_70_for_condition: // for-condition
      case symbol_kind::S_71_for_step: // for-step
      case symbol_kind::S_72_for_statement: // for-statement
      case symbol_kind::S_73_exp_statement: // exp-statement
      case symbol_kind::S_compareexp: // compareexp
      case symbol_kind::S_factor: // factor
      case symbol_kind::S_zenvar: // zenvar
      case symbol_kind::S_78_func_content: // func-content
      case symbol_kind::S_term: // term
        value.move< std::shared_ptr<ZfxASTNode> > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_arrcontents: // arrcontents
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
      case symbol_kind::S_59_bool_stmt: // bool-stmt
      case symbol_kind::S_65_array_mark: // array-mark
        value.copy< bool > (that.value);
        break;

      case symbol_kind::S_NUMBER: // NUMBER
        value.copy< float > (that.value);
        break;

      case symbol_kind::S_58_assign_op: // assign-op
        value.copy< operatorVals > (that.value);
        break;

      case symbol_kind::S_54_zfx_program: // zfx-program
      case symbol_kind::S_55_multi_statements: // multi-statements
      case symbol_kind::S_56_general_statement: // general-statement
      case symbol_kind::S_57_array_or_exp: // array-or-exp
      case symbol_kind::S_60_assign_statement: // assign-statement
      case symbol_kind::S_61_jump_statement: // jump-statement
      case symbol_kind::S_arrcontent: // arrcontent
      case symbol_kind::S_64_array_stmt: // array-stmt
      case symbol_kind::S_66_declare_statement: // declare-statement
      case symbol_kind::S_67_code_block: // code-block
      case symbol_kind::S_68_if_statement: // if-statement
      case symbol_kind::S_69_for_begin: // for-begin
      case symbol_kind::S_70_for_condition: // for-condition
      case symbol_kind::S_71_for_step: // for-step
      case symbol_kind::S_72_for_statement: // for-statement
      case symbol_kind::S_73_exp_statement: // exp-statement
      case symbol_kind::S_compareexp: // compareexp
      case symbol_kind::S_factor: // factor
      case symbol_kind::S_zenvar: // zenvar
      case symbol_kind::S_78_func_content: // func-content
      case symbol_kind::S_term: // term
        value.copy< std::shared_ptr<ZfxASTNode> > (that.value);
        break;

      case symbol_kind::S_arrcontents: // arrcontents
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
      case symbol_kind::S_59_bool_stmt: // bool-stmt
      case symbol_kind::S_65_array_mark: // array-mark
        value.move< bool > (that.value);
        break;

      case symbol_kind::S_NUMBER: // NUMBER
        value.move< float > (that.value);
        break;

      case symbol_kind::S_58_assign_op: // assign-op
        value.move< operatorVals > (that.value);
        break;

      case symbol_kind::S_54_zfx_program: // zfx-program
      case symbol_kind::S_55_multi_statements: // multi-statements
      case symbol_kind::S_56_general_statement: // general-statement
      case symbol_kind::S_57_array_or_exp: // array-or-exp
      case symbol_kind::S_60_assign_statement: // assign-statement
      case symbol_kind::S_61_jump_statement: // jump-statement
      case symbol_kind::S_arrcontent: // arrcontent
      case symbol_kind::S_64_array_stmt: // array-stmt
      case symbol_kind::S_66_declare_statement: // declare-statement
      case symbol_kind::S_67_code_block: // code-block
      case symbol_kind::S_68_if_statement: // if-statement
      case symbol_kind::S_69_for_begin: // for-begin
      case symbol_kind::S_70_for_condition: // for-condition
      case symbol_kind::S_71_for_step: // for-step
      case symbol_kind::S_72_for_statement: // for-statement
      case symbol_kind::S_73_exp_statement: // exp-statement
      case symbol_kind::S_compareexp: // compareexp
      case symbol_kind::S_factor: // factor
      case symbol_kind::S_zenvar: // zenvar
      case symbol_kind::S_78_func_content: // func-content
      case symbol_kind::S_term: // term
        value.move< std::shared_ptr<ZfxASTNode> > (that.value);
        break;

      case symbol_kind::S_arrcontents: // arrcontents
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
      case symbol_kind::S_59_bool_stmt: // bool-stmt
      case symbol_kind::S_65_array_mark: // array-mark
        yylhs.value.emplace< bool > ();
        break;

      case symbol_kind::S_NUMBER: // NUMBER
        yylhs.value.emplace< float > ();
        break;

      case symbol_kind::S_58_assign_op: // assign-op
        yylhs.value.emplace< operatorVals > ();
        break;

      case symbol_kind::S_54_zfx_program: // zfx-program
      case symbol_kind::S_55_multi_statements: // multi-statements
      case symbol_kind::S_56_general_statement: // general-statement
      case symbol_kind::S_57_array_or_exp: // array-or-exp
      case symbol_kind::S_60_assign_statement: // assign-statement
      case symbol_kind::S_61_jump_statement: // jump-statement
      case symbol_kind::S_arrcontent: // arrcontent
      case symbol_kind::S_64_array_stmt: // array-stmt
      case symbol_kind::S_66_declare_statement: // declare-statement
      case symbol_kind::S_67_code_block: // code-block
      case symbol_kind::S_68_if_statement: // if-statement
      case symbol_kind::S_69_for_begin: // for-begin
      case symbol_kind::S_70_for_condition: // for-condition
      case symbol_kind::S_71_for_step: // for-step
      case symbol_kind::S_72_for_statement: // for-statement
      case symbol_kind::S_73_exp_statement: // exp-statement
      case symbol_kind::S_compareexp: // compareexp
      case symbol_kind::S_factor: // factor
      case symbol_kind::S_zenvar: // zenvar
      case symbol_kind::S_78_func_content: // func-content
      case symbol_kind::S_term: // term
        yylhs.value.emplace< std::shared_ptr<ZfxASTNode> > ();
        break;

      case symbol_kind::S_arrcontents: // arrcontents
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
#line 129 "zfxparser.y"
                 {
            std::cout << "END" << std::endl;
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(CODEBLOCK, DEFAULT_FUNCVAL, {});
            driver.setASTResult(yylhs.value.as < std::shared_ptr<ZfxASTNode> > ());
        }
#line 953 "zfxparser.cpp"
    break;

  case 3: // zfx-program: multi-statements zfx-program
#line 134 "zfxparser.y"
                                   {
            addChild(yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[1].value.as < std::shared_ptr<ZfxASTNode> > ());
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ();
        }
#line 962 "zfxparser.cpp"
    break;

  case 4: // multi-statements: %empty
#line 140 "zfxparser.y"
                         {
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(CODEBLOCK, DEFAULT_FUNCVAL, {});
        }
#line 970 "zfxparser.cpp"
    break;

  case 5: // multi-statements: general-statement multi-statements
#line 143 "zfxparser.y"
                                         {
            addChild(yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[1].value.as < std::shared_ptr<ZfxASTNode> > ());
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ();
        }
#line 979 "zfxparser.cpp"
    break;

  case 6: // general-statement: declare-statement SEMICOLON
#line 149 "zfxparser.y"
                                               { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[1].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 985 "zfxparser.cpp"
    break;

  case 7: // general-statement: assign-statement SEMICOLON
#line 150 "zfxparser.y"
                                 { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[1].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 991 "zfxparser.cpp"
    break;

  case 8: // general-statement: if-statement
#line 151 "zfxparser.y"
                   { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 997 "zfxparser.cpp"
    break;

  case 9: // general-statement: for-statement
#line 152 "zfxparser.y"
                    { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1003 "zfxparser.cpp"
    break;

  case 10: // general-statement: jump-statement SEMICOLON
#line 153 "zfxparser.y"
                               { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[1].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1009 "zfxparser.cpp"
    break;

  case 11: // general-statement: exp-statement SEMICOLON
#line 154 "zfxparser.y"
                              { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[1].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1015 "zfxparser.cpp"
    break;

  case 12: // array-or-exp: exp-statement
#line 157 "zfxparser.y"
                            { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1021 "zfxparser.cpp"
    break;

  case 13: // array-or-exp: array-stmt
#line 158 "zfxparser.y"
                 { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1027 "zfxparser.cpp"
    break;

  case 14: // assign-op: EQUALTO
#line 161 "zfxparser.y"
                   { yylhs.value.as < operatorVals > () = AssignTo; }
#line 1033 "zfxparser.cpp"
    break;

  case 15: // assign-op: ADDASSIGN
#line 162 "zfxparser.y"
                { yylhs.value.as < operatorVals > () = AddAssign; }
#line 1039 "zfxparser.cpp"
    break;

  case 16: // assign-op: MULASSIGN
#line 163 "zfxparser.y"
                { yylhs.value.as < operatorVals > () = MulAssign; }
#line 1045 "zfxparser.cpp"
    break;

  case 17: // assign-op: SUBASSIGN
#line 164 "zfxparser.y"
                { yylhs.value.as < operatorVals > () = SubAssign; }
#line 1051 "zfxparser.cpp"
    break;

  case 18: // assign-op: DIVASSIGN
#line 165 "zfxparser.y"
                { yylhs.value.as < operatorVals > () = DivAssign; }
#line 1057 "zfxparser.cpp"
    break;

  case 19: // bool-stmt: TRUE
#line 168 "zfxparser.y"
                { yylhs.value.as < bool > () = true; }
#line 1063 "zfxparser.cpp"
    break;

  case 20: // bool-stmt: FALSE
#line 169 "zfxparser.y"
            { yylhs.value.as < bool > () = false; }
#line 1069 "zfxparser.cpp"
    break;

  case 21: // assign-statement: zenvar assign-op array-or-exp
#line 172 "zfxparser.y"
                                                {
            std::vector<std::shared_ptr<ZfxASTNode>> children({yystack_[2].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()});
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(ASSIGNMENT, yystack_[1].value.as < operatorVals > (), children);
        }
#line 1078 "zfxparser.cpp"
    break;

  case 22: // jump-statement: BREAK
#line 178 "zfxparser.y"
                      { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(JUMP, JUMP_BREAK, {}); }
#line 1084 "zfxparser.cpp"
    break;

  case 23: // jump-statement: RETURN
#line 179 "zfxparser.y"
               { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(JUMP, JUMP_RETURN, {}); }
#line 1090 "zfxparser.cpp"
    break;

  case 24: // jump-statement: CONTINUE
#line 180 "zfxparser.y"
               { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(JUMP, JUMP_CONTINUE, {}); }
#line 1096 "zfxparser.cpp"
    break;

  case 25: // arrcontent: exp-statement
#line 183 "zfxparser.y"
                          { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1102 "zfxparser.cpp"
    break;

  case 26: // arrcontent: array-stmt
#line 184 "zfxparser.y"
                 { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1108 "zfxparser.cpp"
    break;

  case 27: // arrcontents: arrcontent
#line 187 "zfxparser.y"
                                   { yylhs.value.as < std::vector<std::shared_ptr<ZfxASTNode>> > () = std::vector<std::shared_ptr<ZfxASTNode>>({yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()}); }
#line 1114 "zfxparser.cpp"
    break;

  case 28: // arrcontents: arrcontents COMMA arrcontent
#line 188 "zfxparser.y"
                                   { yystack_[2].value.as < std::vector<std::shared_ptr<ZfxASTNode>> > ().push_back(yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()); yylhs.value.as < std::vector<std::shared_ptr<ZfxASTNode>> > () = yystack_[2].value.as < std::vector<std::shared_ptr<ZfxASTNode>> > (); }
#line 1120 "zfxparser.cpp"
    break;

  case 29: // array-stmt: LBRACKET arrcontents RBRACKET
#line 191 "zfxparser.y"
                                          { 
        yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(ARRAY, DEFAULT_FUNCVAL, yystack_[1].value.as < std::vector<std::shared_ptr<ZfxASTNode>> > ());
    }
#line 1128 "zfxparser.cpp"
    break;

  case 30: // array-mark: %empty
#line 196 "zfxparser.y"
                   { yylhs.value.as < bool > () = false; }
#line 1134 "zfxparser.cpp"
    break;

  case 31: // array-mark: LSQBRACKET RSQBRACKET
#line 197 "zfxparser.y"
                            { yylhs.value.as < bool > () = true; }
#line 1140 "zfxparser.cpp"
    break;

  case 32: // declare-statement: TYPE VARNAME array-mark
#line 200 "zfxparser.y"
                                           {
                auto typeNode = driver.makeTypeNode(yystack_[2].value.as < string > (), yystack_[0].value.as < bool > ());
                auto nameNode = driver.makeZfxVarNode(yystack_[1].value.as < string > ());
                std::vector<std::shared_ptr<ZfxASTNode>> children({typeNode, nameNode});
                yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(DECLARE, DEFAULT_FUNCVAL, children);
            }
#line 1151 "zfxparser.cpp"
    break;

  case 33: // declare-statement: TYPE VARNAME array-mark EQUALTO array-or-exp
#line 206 "zfxparser.y"
                                                   {
                auto typeNode = driver.makeTypeNode(yystack_[4].value.as < string > (), yystack_[2].value.as < bool > ());
                auto nameNode = driver.makeZfxVarNode(yystack_[3].value.as < string > ());
                std::vector<std::shared_ptr<ZfxASTNode>> children({typeNode, nameNode, yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()});
                yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(DECLARE, DEFAULT_FUNCVAL, children);
            }
#line 1162 "zfxparser.cpp"
    break;

  case 34: // code-block: LBRACKET multi-statements RBRACKET
#line 214 "zfxparser.y"
                                               { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[1].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1168 "zfxparser.cpp"
    break;

  case 35: // if-statement: IF LPAREN exp-statement RPAREN code-block
#line 217 "zfxparser.y"
                                                        {
            std::vector<std::shared_ptr<ZfxASTNode>> children({yystack_[2].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()});
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(IF, DEFAULT_FUNCVAL, children);
        }
#line 1177 "zfxparser.cpp"
    break;

  case 36: // for-begin: SEMICOLON
#line 223 "zfxparser.y"
                     { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeEmptyNode(); }
#line 1183 "zfxparser.cpp"
    break;

  case 37: // for-begin: declare-statement SEMICOLON
#line 224 "zfxparser.y"
                                  { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[1].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1189 "zfxparser.cpp"
    break;

  case 38: // for-begin: assign-statement SEMICOLON
#line 225 "zfxparser.y"
                                 { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[1].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1195 "zfxparser.cpp"
    break;

  case 39: // for-begin: exp-statement SEMICOLON
#line 226 "zfxparser.y"
                              { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[1].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1201 "zfxparser.cpp"
    break;

  case 40: // for-condition: SEMICOLON
#line 229 "zfxparser.y"
                          { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeEmptyNode(); }
#line 1207 "zfxparser.cpp"
    break;

  case 41: // for-condition: exp-statement SEMICOLON
#line 230 "zfxparser.y"
                              { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[1].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1213 "zfxparser.cpp"
    break;

  case 42: // for-step: %empty
#line 233 "zfxparser.y"
                 { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeEmptyNode(); }
#line 1219 "zfxparser.cpp"
    break;

  case 43: // for-step: exp-statement
#line 234 "zfxparser.y"
                    { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1225 "zfxparser.cpp"
    break;

  case 44: // for-step: assign-statement
#line 235 "zfxparser.y"
                       { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1231 "zfxparser.cpp"
    break;

  case 45: // for-statement: FOR LPAREN for-begin for-condition for-step RPAREN code-block
#line 238 "zfxparser.y"
                                                                             {
            std::vector<std::shared_ptr<ZfxASTNode>> children({yystack_[4].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[3].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[2].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()});
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(FOR, DEFAULT_FUNCVAL, children);
        }
#line 1240 "zfxparser.cpp"
    break;

  case 46: // exp-statement: compareexp
#line 244 "zfxparser.y"
                                    { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1246 "zfxparser.cpp"
    break;

  case 47: // exp-statement: exp-statement COMPARE compareexp
#line 245 "zfxparser.y"
                                        {
                std::vector<std::shared_ptr<ZfxASTNode>>children({yystack_[2].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()});
                yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(COMPOP, DEFAULT_FUNCVAL, children);
                yylhs.value.as < std::shared_ptr<ZfxASTNode> > ()->value = yystack_[1].value.as < string > ();
            }
#line 1256 "zfxparser.cpp"
    break;

  case 48: // exp-statement: exp-statement COMPARE compareexp QUESTION exp-statement COLON exp-statement
#line 250 "zfxparser.y"
                                                                                  {
                std::vector<std::shared_ptr<ZfxASTNode>> children({yystack_[6].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[4].value.as < std::shared_ptr<ZfxASTNode> > ()});
                auto spCond = driver.makeNewNode(COMPOP, DEFAULT_FUNCVAL, children);
                spCond->value = yystack_[5].value.as < string > ();

                std::vector<std::shared_ptr<ZfxASTNode>> exps({spCond, yystack_[2].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()});
                yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(CONDEXP, DEFAULT_FUNCVAL, exps);
            }
#line 1269 "zfxparser.cpp"
    break;

  case 49: // compareexp: factor
#line 260 "zfxparser.y"
                                { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1275 "zfxparser.cpp"
    break;

  case 50: // compareexp: compareexp ADD factor
#line 261 "zfxparser.y"
                            {
                std::vector<std::shared_ptr<ZfxASTNode>> children({yystack_[2].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()});
                yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(FOUROPERATIONS, PLUS, children);
            }
#line 1284 "zfxparser.cpp"
    break;

  case 51: // compareexp: compareexp SUB factor
#line 265 "zfxparser.y"
                            {
                std::vector<std::shared_ptr<ZfxASTNode>> children({yystack_[2].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()});
                yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(FOUROPERATIONS, MINUS, children);
            }
#line 1293 "zfxparser.cpp"
    break;

  case 52: // factor: term
#line 271 "zfxparser.y"
                        { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1299 "zfxparser.cpp"
    break;

  case 53: // factor: factor MUL term
#line 272 "zfxparser.y"
                        {
                std::vector<std::shared_ptr<ZfxASTNode>>children({yystack_[2].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()});
                yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(FOUROPERATIONS, MUL, children);
            }
#line 1308 "zfxparser.cpp"
    break;

  case 54: // factor: factor DIV term
#line 276 "zfxparser.y"
                      {
            std::vector<std::shared_ptr<ZfxASTNode>>children({yystack_[2].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()});
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(FOUROPERATIONS, DIV, children);
        }
#line 1317 "zfxparser.cpp"
    break;

  case 55: // zenvar: DOLLARVARNAME
#line 282 "zfxparser.y"
                        { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeZfxVarNode(yystack_[0].value.as < string > (), BulitInVar); }
#line 1323 "zfxparser.cpp"
    break;

  case 56: // zenvar: VARNAME
#line 283 "zfxparser.y"
                        { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeZfxVarNode(yystack_[0].value.as < string > ()); }
#line 1329 "zfxparser.cpp"
    break;

  case 57: // zenvar: ATTRAT zenvar
#line 284 "zfxparser.y"
                    {
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ();
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > ()->opVal = AttrMark;
        }
#line 1338 "zfxparser.cpp"
    break;

  case 58: // zenvar: zenvar DOT VARNAME
#line 288 "zfxparser.y"
                         {
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeComponentVisit(yystack_[2].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[0].value.as < string > ());
        }
#line 1346 "zfxparser.cpp"
    break;

  case 59: // zenvar: zenvar LSQBRACKET exp-statement RSQBRACKET
#line 291 "zfxparser.y"
                                                 {
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[3].value.as < std::shared_ptr<ZfxASTNode> > ();
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > ()->opVal = Indexing;
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > ()->children.push_back(yystack_[1].value.as < std::shared_ptr<ZfxASTNode> > ());
        }
#line 1356 "zfxparser.cpp"
    break;

  case 60: // zenvar: AUTOINC zenvar
#line 296 "zfxparser.y"
                     {
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ();
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > ()->opVal = AutoIncreaseFirst;
        }
#line 1365 "zfxparser.cpp"
    break;

  case 61: // zenvar: zenvar AUTOINC
#line 300 "zfxparser.y"
                     {
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[1].value.as < std::shared_ptr<ZfxASTNode> > ();
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > ()->opVal = AutoIncreaseLast;
        }
#line 1374 "zfxparser.cpp"
    break;

  case 62: // zenvar: AUTODEC zenvar
#line 304 "zfxparser.y"
                     {
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ();
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > ()->opVal = AutoDecreaseFirst;
        }
#line 1383 "zfxparser.cpp"
    break;

  case 63: // zenvar: zenvar AUTODEC
#line 308 "zfxparser.y"
                     {
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[1].value.as < std::shared_ptr<ZfxASTNode> > ();
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > ()->opVal = AutoDecreaseLast;
        }
#line 1392 "zfxparser.cpp"
    break;

  case 64: // funcargs: %empty
#line 314 "zfxparser.y"
                 { yylhs.value.as < std::vector<std::shared_ptr<ZfxASTNode>> > () = std::vector<std::shared_ptr<ZfxASTNode>>(); }
#line 1398 "zfxparser.cpp"
    break;

  case 65: // funcargs: exp-statement
#line 315 "zfxparser.y"
                               { yylhs.value.as < std::vector<std::shared_ptr<ZfxASTNode>> > () = std::vector<std::shared_ptr<ZfxASTNode>>({yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()}); }
#line 1404 "zfxparser.cpp"
    break;

  case 66: // funcargs: funcargs COMMA exp-statement
#line 316 "zfxparser.y"
                                   { yystack_[2].value.as < std::vector<std::shared_ptr<ZfxASTNode>> > ().push_back(yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()); yylhs.value.as < std::vector<std::shared_ptr<ZfxASTNode>> > () = yystack_[2].value.as < std::vector<std::shared_ptr<ZfxASTNode>> > (); }
#line 1410 "zfxparser.cpp"
    break;

  case 67: // func-content: LPAREN funcargs RPAREN
#line 320 "zfxparser.y"
                                     { 
        yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(FUNC, DEFAULT_FUNCVAL, yystack_[1].value.as < std::vector<std::shared_ptr<ZfxASTNode>> > ());
        yylhs.value.as < std::shared_ptr<ZfxASTNode> > ()->isParenthesisNodeComplete = true;
        yylhs.value.as < std::shared_ptr<ZfxASTNode> > ()->func_match = Match_Exactly;
    }
#line 1420 "zfxparser.cpp"
    break;

  case 68: // term: NUMBER
#line 328 "zfxparser.y"
                        { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNumberNode(yystack_[0].value.as < float > ()); }
#line 1426 "zfxparser.cpp"
    break;

  case 69: // term: bool-stmt
#line 329 "zfxparser.y"
                        { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeBoolNode(yystack_[0].value.as < bool > ()); }
#line 1432 "zfxparser.cpp"
    break;

  case 70: // term: LITERAL
#line 330 "zfxparser.y"
                        { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeStringNode(yystack_[0].value.as < string > ()); }
#line 1438 "zfxparser.cpp"
    break;

  case 71: // term: UNCOMPSTR
#line 331 "zfxparser.y"
                        { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeQuoteStringNode(yystack_[0].value.as < string > ()); }
#line 1444 "zfxparser.cpp"
    break;

  case 72: // term: LPAREN exp-statement RPAREN
#line 332 "zfxparser.y"
                                  { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[1].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1450 "zfxparser.cpp"
    break;

  case 73: // term: SUB exp-statement
#line 333 "zfxparser.y"
                                  { yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()->value = -1 * std::get<float>(yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()->value); yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1456 "zfxparser.cpp"
    break;

  case 74: // term: zenvar
#line 334 "zfxparser.y"
                        { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 1462 "zfxparser.cpp"
    break;

  case 75: // term: VARNAME func-content
#line 335 "zfxparser.y"
                            { 
        yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ();
        yylhs.value.as < std::shared_ptr<ZfxASTNode> > ()->opVal = DEFAULT_FUNCVAL;
        yylhs.value.as < std::shared_ptr<ZfxASTNode> > ()->type = FUNC;
        yylhs.value.as < std::shared_ptr<ZfxASTNode> > ()->value = yystack_[1].value.as < string > ();
        yylhs.value.as < std::shared_ptr<ZfxASTNode> > ()->isParenthesisNode = true;
    }
#line 1474 "zfxparser.cpp"
    break;


#line 1478 "zfxparser.cpp"

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


  const signed char  ZfxParser ::yypact_ninf_ = -84;

  const signed char  ZfxParser ::yytable_ninf_ = -1;

  const short
   ZfxParser ::yypact_[] =
  {
      86,   -84,   -84,   -84,   -84,   -84,   -84,   -84,   -36,   -25,
      24,     8,     8,   -84,   -84,   -84,    44,     8,   189,   189,
      77,    86,   116,   -84,    53,    54,    57,   -84,   -84,    27,
     -30,    15,   191,   -84,   189,   -84,   189,    29,   -84,    17,
      17,     9,    17,    66,    17,     2,   -84,   -84,   -84,   -84,
     -84,   -84,   189,   -84,   189,   189,   189,   189,    60,   -84,
     -84,   -84,   189,   -84,   -84,   -84,   -84,   146,    66,    43,
       3,   -84,    63,    69,   159,    40,    62,    71,   -84,   -14,
      15,    15,   -84,   -84,   -84,    -3,   146,   -84,   -84,    66,
     -84,   189,    80,   -84,   -84,   -84,   189,    41,   -84,   -84,
     146,   189,   -84,   -84,     6,   -84,    66,    66,   116,   -84,
     -84,    95,    66,   -84,   -84,    50,   146,   -84,    81,    80,
     189,   -84,   -84,   -84,    66
  };

  const signed char
   ZfxParser ::yydefact_[] =
  {
       0,     2,    68,    19,    20,    70,    71,    55,    56,     0,
       0,     0,     0,    23,    24,    22,     0,     0,     0,     0,
       0,     0,     4,    69,     0,     0,     0,     8,     9,     0,
      46,    49,    74,    52,    64,    75,     0,     0,    56,    60,
      62,    30,    57,    73,    74,     0,     1,     3,     5,     7,
      10,     6,     0,    11,     0,     0,     0,     0,     0,    14,
      61,    63,     0,    15,    16,    17,    18,     0,    65,     0,
       0,    36,     0,     0,     0,     0,     0,    32,    72,    47,
      50,    51,    53,    54,    58,     0,     0,    21,    13,    12,
      67,     0,     0,    38,    37,    40,    42,     0,    39,    31,
       0,     0,    59,    27,     0,    26,    25,    66,     4,    35,
      44,     0,    43,    41,    33,     0,     0,    29,     0,     0,
       0,    28,    34,    45,    48
  };

  const signed char
   ZfxParser ::yypgoto_[] =
  {
     -84,    83,   -21,   -84,     7,   -84,   -84,   -35,   -84,   -10,
     -84,   -83,   -84,    74,    -7,   -84,   -84,   -84,   -84,   -84,
     -11,    67,   -45,     0,   -84,   -84,    16
  };

  const signed char
   ZfxParser ::yydefgoto_[] =
  {
       0,    20,    21,    22,    87,    67,    23,    24,    25,   103,
     104,    88,    77,    26,   109,    27,    74,    96,   111,    28,
      29,    30,    31,    44,    69,    35,    33
  };

  const signed char
   ZfxParser ::yytable_[] =
  {
      32,    48,    72,   105,   101,    78,    92,    43,    45,    80,
      81,    39,    40,    54,    52,    55,    34,    42,   116,    52,
      52,    32,    32,    68,     7,    70,    75,    36,   117,    54,
     102,    55,    38,   105,     2,     3,     4,    32,    11,    12,
      58,    76,     5,     6,    52,     7,    90,    60,    61,    62,
      17,    85,    53,     8,    71,    91,    89,    52,    52,    11,
      12,   110,    56,    97,    57,    98,   113,    52,    41,   120,
      16,    17,    82,    83,    18,   106,    37,    46,    49,    50,
     107,    19,    51,    52,    84,   112,     1,   118,    93,    89,
     115,     2,     3,     4,    94,    99,    32,   100,   119,     5,
       6,   108,     7,   122,    47,   106,   121,   114,    32,   124,
       8,    73,   123,     9,    10,     0,    11,    12,     0,    79,
       0,     2,     3,     4,    13,    14,    15,    16,    17,     5,
       6,    18,     7,     0,     0,     0,     0,     0,    19,     0,
       8,     0,     0,     9,    10,     0,    11,    12,     0,     0,
       0,     2,     3,     4,    13,    14,    15,    16,    17,     5,
       6,    18,     7,     0,     2,     3,     4,    86,    19,     0,
       8,     0,     5,     6,     0,     7,    11,    12,     0,     0,
       0,     0,     0,     8,    95,     0,     0,     0,    17,    11,
      12,    18,     0,     0,     2,     3,     4,     0,    19,     0,
       0,    17,     5,     6,    18,     7,     0,     0,     0,     0,
       0,    19,     0,     8,    58,     0,     0,    59,     0,    11,
      12,    60,    61,    62,     0,    63,    64,    65,    66,     0,
       0,    17,     0,     0,    18,     0,     0,     0,     0,     0,
       0,    19
  };

  const signed char
   ZfxParser ::yycheck_[] =
  {
       0,    22,    37,    86,    18,     3,     3,    18,    19,    54,
      55,    11,    12,    43,    17,    45,    52,    17,    12,    17,
      17,    21,    22,    34,    16,    36,    37,    52,    22,    43,
      33,    45,    24,   116,     5,     6,     7,    37,    30,    31,
      23,    32,    13,    14,    17,    16,     3,    30,    31,    32,
      42,    62,    25,    24,    25,    12,    67,    17,    17,    30,
      31,    96,    47,    74,    49,    25,    25,    17,    24,    19,
      41,    42,    56,    57,    45,    86,    52,     0,    25,    25,
      91,    52,    25,    17,    24,    96,     0,   108,    25,   100,
     101,     5,     6,     7,    25,    33,    96,    26,     3,    13,
      14,    21,    16,    22,    21,   116,   116,   100,   108,   120,
      24,    37,   119,    27,    28,    -1,    30,    31,    -1,    52,
      -1,     5,     6,     7,    38,    39,    40,    41,    42,    13,
      14,    45,    16,    -1,    -1,    -1,    -1,    -1,    52,    -1,
      24,    -1,    -1,    27,    28,    -1,    30,    31,    -1,    -1,
      -1,     5,     6,     7,    38,    39,    40,    41,    42,    13,
      14,    45,    16,    -1,     5,     6,     7,    21,    52,    -1,
      24,    -1,    13,    14,    -1,    16,    30,    31,    -1,    -1,
      -1,    -1,    -1,    24,    25,    -1,    -1,    -1,    42,    30,
      31,    45,    -1,    -1,     5,     6,     7,    -1,    52,    -1,
      -1,    42,    13,    14,    45,    16,    -1,    -1,    -1,    -1,
      -1,    52,    -1,    24,    23,    -1,    -1,    26,    -1,    30,
      31,    30,    31,    32,    -1,    34,    35,    36,    37,    -1,
      -1,    42,    -1,    -1,    45,    -1,    -1,    -1,    -1,    -1,
      -1,    52
  };

  const signed char
   ZfxParser ::yystos_[] =
  {
       0,     0,     5,     6,     7,    13,    14,    16,    24,    27,
      28,    30,    31,    38,    39,    40,    41,    42,    45,    52,
      54,    55,    56,    59,    60,    61,    66,    68,    72,    73,
      74,    75,    76,    79,    52,    78,    52,    52,    24,    76,
      76,    24,    76,    73,    76,    73,     0,    54,    55,    25,
      25,    25,    17,    25,    43,    45,    47,    49,    23,    26,
      30,    31,    32,    34,    35,    36,    37,    58,    73,    77,
      73,    25,    60,    66,    69,    73,    32,    65,     3,    74,
      75,    75,    79,    79,    24,    73,    21,    57,    64,    73,
       3,    12,     3,    25,    25,    25,    70,    73,    25,    33,
      26,    18,    33,    62,    63,    64,    73,    73,    21,    67,
      60,    71,    73,    25,    57,    73,    12,    22,    55,     3,
      19,    62,    22,    67,    73
  };

  const signed char
   ZfxParser ::yyr1_[] =
  {
       0,    53,    54,    54,    55,    55,    56,    56,    56,    56,
      56,    56,    57,    57,    58,    58,    58,    58,    58,    59,
      59,    60,    61,    61,    61,    62,    62,    63,    63,    64,
      65,    65,    66,    66,    67,    68,    69,    69,    69,    69,
      70,    70,    71,    71,    71,    72,    73,    73,    73,    74,
      74,    74,    75,    75,    75,    76,    76,    76,    76,    76,
      76,    76,    76,    76,    77,    77,    77,    78,    79,    79,
      79,    79,    79,    79,    79,    79
  };

  const signed char
   ZfxParser ::yyr2_[] =
  {
       0,     2,     1,     2,     0,     2,     2,     2,     1,     1,
       2,     2,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     3,     1,     1,     1,     1,     1,     1,     3,     3,
       0,     2,     3,     5,     3,     5,     1,     2,     2,     2,
       1,     2,     0,     1,     1,     7,     1,     3,     7,     1,
       3,     3,     1,     3,     3,     1,     1,     2,     3,     4,
       2,     2,     2,     2,     0,     1,     3,     3,     1,     1,
       1,     1,     3,     2,     1,     2
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
  "RETURN", "CONTINUE", "BREAK", "TYPE", "ATTRAT", "ADD", "\"+\"", "SUB",
  "\"-\"", "MUL", "\"*\"", "DIV", "\"/\"", "NEG", "LPAREN", "$accept",
  "zfx-program", "multi-statements", "general-statement", "array-or-exp",
  "assign-op", "bool-stmt", "assign-statement", "jump-statement",
  "arrcontent", "arrcontents", "array-stmt", "array-mark",
  "declare-statement", "code-block", "if-statement", "for-begin",
  "for-condition", "for-step", "for-statement", "exp-statement",
  "compareexp", "factor", "zenvar", "funcargs", "func-content", "term", YY_NULLPTR
  };
#endif


#if YYDEBUG
  const short
   ZfxParser ::yyrline_[] =
  {
       0,   129,   129,   134,   140,   143,   149,   150,   151,   152,
     153,   154,   157,   158,   161,   162,   163,   164,   165,   168,
     169,   172,   178,   179,   180,   183,   184,   187,   188,   191,
     196,   197,   200,   206,   214,   217,   223,   224,   225,   226,
     229,   230,   233,   234,   235,   238,   244,   245,   250,   260,
     261,   265,   271,   272,   276,   282,   283,   284,   288,   291,
     296,   300,   304,   308,   314,   315,   316,   320,   328,   329,
     330,   331,   332,   333,   334,   335
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
#line 2059 "zfxparser.cpp"

#line 346 "zfxparser.y"


// Bison expects us to provide implementation - otherwise linker complains
void zeno::ZfxParser::error(const location &loc , const std::string &message) {
    cout << "Error: " << message << endl << "Error location: " << driver.location() << endl;
}

