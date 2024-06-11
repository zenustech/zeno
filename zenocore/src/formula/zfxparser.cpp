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
      case symbol_kind::S_NUMBER: // NUMBER
        value.YY_MOVE_OR_COPY< float > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_calclist: // calclist
      case symbol_kind::S_37_general_statement: // general-statement
      case symbol_kind::S_38_declare_statement: // declare-statement
      case symbol_kind::S_39_exp_statement: // exp-statement
      case symbol_kind::S_compareexp: // compareexp
      case symbol_kind::S_factor: // factor
      case symbol_kind::S_zenvar: // zenvar
      case symbol_kind::S_44_func_content: // func-content
      case symbol_kind::S_term: // term
        value.YY_MOVE_OR_COPY< std::shared_ptr<ZfxASTNode> > (YY_MOVE (that.value));
        break;

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
      case symbol_kind::S_NUMBER: // NUMBER
        value.move< float > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_calclist: // calclist
      case symbol_kind::S_37_general_statement: // general-statement
      case symbol_kind::S_38_declare_statement: // declare-statement
      case symbol_kind::S_39_exp_statement: // exp-statement
      case symbol_kind::S_compareexp: // compareexp
      case symbol_kind::S_factor: // factor
      case symbol_kind::S_zenvar: // zenvar
      case symbol_kind::S_44_func_content: // func-content
      case symbol_kind::S_term: // term
        value.move< std::shared_ptr<ZfxASTNode> > (YY_MOVE (that.value));
        break;

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
      case symbol_kind::S_NUMBER: // NUMBER
        value.copy< float > (that.value);
        break;

      case symbol_kind::S_calclist: // calclist
      case symbol_kind::S_37_general_statement: // general-statement
      case symbol_kind::S_38_declare_statement: // declare-statement
      case symbol_kind::S_39_exp_statement: // exp-statement
      case symbol_kind::S_compareexp: // compareexp
      case symbol_kind::S_factor: // factor
      case symbol_kind::S_zenvar: // zenvar
      case symbol_kind::S_44_func_content: // func-content
      case symbol_kind::S_term: // term
        value.copy< std::shared_ptr<ZfxASTNode> > (that.value);
        break;

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
      case symbol_kind::S_NUMBER: // NUMBER
        value.move< float > (that.value);
        break;

      case symbol_kind::S_calclist: // calclist
      case symbol_kind::S_37_general_statement: // general-statement
      case symbol_kind::S_38_declare_statement: // declare-statement
      case symbol_kind::S_39_exp_statement: // exp-statement
      case symbol_kind::S_compareexp: // compareexp
      case symbol_kind::S_factor: // factor
      case symbol_kind::S_zenvar: // zenvar
      case symbol_kind::S_44_func_content: // func-content
      case symbol_kind::S_term: // term
        value.move< std::shared_ptr<ZfxASTNode> > (that.value);
        break;

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
      case symbol_kind::S_NUMBER: // NUMBER
        yylhs.value.emplace< float > ();
        break;

      case symbol_kind::S_calclist: // calclist
      case symbol_kind::S_37_general_statement: // general-statement
      case symbol_kind::S_38_declare_statement: // declare-statement
      case symbol_kind::S_39_exp_statement: // exp-statement
      case symbol_kind::S_compareexp: // compareexp
      case symbol_kind::S_factor: // factor
      case symbol_kind::S_zenvar: // zenvar
      case symbol_kind::S_44_func_content: // func-content
      case symbol_kind::S_term: // term
        yylhs.value.emplace< std::shared_ptr<ZfxASTNode> > ();
        break;

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
  case 2: // calclist: END
#line 108 "zfxparser.y"
             {
            //std::cout << "END" << std::endl;
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(CODEBLOCK, DEFAULT_FUNCVAL, {});
            driver.setASTResult(yylhs.value.as < std::shared_ptr<ZfxASTNode> > ());
        }
#line 753 "zfxparser.cpp"
    break;

  case 3: // calclist: general-statement calclist
#line 113 "zfxparser.y"
                                 {
            addChild(yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[1].value.as < std::shared_ptr<ZfxASTNode> > ());
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ();
        }
#line 762 "zfxparser.cpp"
    break;

  case 4: // general-statement: declare-statement SEMICOLON
#line 119 "zfxparser.y"
                                               { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[1].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 768 "zfxparser.cpp"
    break;

  case 5: // general-statement: exp-statement SEMICOLON
#line 120 "zfxparser.y"
                              { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[1].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 774 "zfxparser.cpp"
    break;

  case 6: // declare-statement: VARNAME VARNAME
#line 122 "zfxparser.y"
                                   {
                auto typeNode = driver.makeZfxVarNode(yystack_[1].value.as < string > ());
                auto nameNode = driver.makeZfxVarNode(yystack_[0].value.as < string > ());
                std::vector<std::shared_ptr<ZfxASTNode>> children({typeNode, nameNode});
                yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(DECLARE, DEFAULT_FUNCVAL, children);
            }
#line 785 "zfxparser.cpp"
    break;

  case 7: // declare-statement: VARNAME VARNAME EQUALTO exp-statement
#line 128 "zfxparser.y"
                                            {
                auto typeNode = driver.makeZfxVarNode(yystack_[3].value.as < string > ());
                auto nameNode = driver.makeZfxVarNode(yystack_[2].value.as < string > ());
                std::vector<std::shared_ptr<ZfxASTNode>> children({typeNode, nameNode, yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()});
                yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(DECLARE, DEFAULT_FUNCVAL, children);
            }
#line 796 "zfxparser.cpp"
    break;

  case 8: // exp-statement: compareexp
#line 136 "zfxparser.y"
                                    { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 802 "zfxparser.cpp"
    break;

  case 9: // exp-statement: exp-statement COMPARE compareexp
#line 137 "zfxparser.y"
                                        {
                std::vector<std::shared_ptr<ZfxASTNode>>children({yystack_[2].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()});
                yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(COMPOP, DEFAULT_FUNCVAL, children);
                yylhs.value.as < std::shared_ptr<ZfxASTNode> > ()->value = yystack_[1].value.as < string > ();
            }
#line 812 "zfxparser.cpp"
    break;

  case 10: // exp-statement: exp-statement COMPARE compareexp QUESTION exp-statement COLON exp-statement
#line 142 "zfxparser.y"
                                                                                  {
                std::vector<std::shared_ptr<ZfxASTNode>> children({yystack_[6].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[4].value.as < std::shared_ptr<ZfxASTNode> > ()});
                auto spCond = driver.makeNewNode(COMPOP, DEFAULT_FUNCVAL, children);
                spCond->value = yystack_[5].value.as < string > ();

                std::vector<std::shared_ptr<ZfxASTNode>> exps({spCond, yystack_[2].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()});
                yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(CONDEXP, DEFAULT_FUNCVAL, exps);
            }
#line 825 "zfxparser.cpp"
    break;

  case 11: // compareexp: factor
#line 152 "zfxparser.y"
                                { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 831 "zfxparser.cpp"
    break;

  case 12: // compareexp: compareexp ADD factor
#line 153 "zfxparser.y"
                            {
                std::vector<std::shared_ptr<ZfxASTNode>>children({yystack_[2].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()});
                yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(FOUROPERATIONS, PLUS, children);
            }
#line 840 "zfxparser.cpp"
    break;

  case 13: // compareexp: compareexp SUB factor
#line 157 "zfxparser.y"
                            {
                std::vector<std::shared_ptr<ZfxASTNode>>children({yystack_[2].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()});
                yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(FOUROPERATIONS, MINUS, children);
            }
#line 849 "zfxparser.cpp"
    break;

  case 14: // factor: term
#line 163 "zfxparser.y"
                        { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 855 "zfxparser.cpp"
    break;

  case 15: // factor: factor MUL term
#line 164 "zfxparser.y"
                        {
                std::vector<std::shared_ptr<ZfxASTNode>>children({yystack_[2].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()});
                yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(FOUROPERATIONS, MUL, children);
            }
#line 864 "zfxparser.cpp"
    break;

  case 16: // factor: factor DIV term
#line 168 "zfxparser.y"
                      {
            std::vector<std::shared_ptr<ZfxASTNode>>children({yystack_[2].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()});
            yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(FOUROPERATIONS, DIV, children);
        }
#line 873 "zfxparser.cpp"
    break;

  case 17: // zenvar: DOLLARVARNAME
#line 174 "zfxparser.y"
                      { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeZenVarNode(yystack_[0].value.as < string > ()); }
#line 879 "zfxparser.cpp"
    break;

  case 18: // zenvar: VARNAME
#line 175 "zfxparser.y"
              { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeZfxVarNode(yystack_[0].value.as < string > ()); }
#line 885 "zfxparser.cpp"
    break;

  case 19: // zenvar: zenvar DOT VARNAME
#line 176 "zfxparser.y"
                         { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeComponentVisit(yystack_[2].value.as < std::shared_ptr<ZfxASTNode> > (), yystack_[0].value.as < string > ()); }
#line 891 "zfxparser.cpp"
    break;

  case 20: // funcargs: exp-statement
#line 179 "zfxparser.y"
                                   { yylhs.value.as < std::vector<std::shared_ptr<ZfxASTNode>> > () = std::vector<std::shared_ptr<ZfxASTNode>>({yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()}); }
#line 897 "zfxparser.cpp"
    break;

  case 21: // funcargs: funcargs COMMA exp-statement
#line 180 "zfxparser.y"
                                   { yystack_[2].value.as < std::vector<std::shared_ptr<ZfxASTNode>> > ().push_back(yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()); yylhs.value.as < std::vector<std::shared_ptr<ZfxASTNode>> > () = yystack_[2].value.as < std::vector<std::shared_ptr<ZfxASTNode>> > (); }
#line 903 "zfxparser.cpp"
    break;

  case 22: // func-content: LPAREN funcargs RPAREN
#line 183 "zfxparser.y"
                                     { 
        yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNode(FUNC, DEFAULT_FUNCVAL, yystack_[1].value.as < std::vector<std::shared_ptr<ZfxASTNode>> > ());
        yylhs.value.as < std::shared_ptr<ZfxASTNode> > ()->isParenthesisNodeComplete = true;
        yylhs.value.as < std::shared_ptr<ZfxASTNode> > ()->func_match = Match_Exactly;
    }
#line 913 "zfxparser.cpp"
    break;

  case 23: // term: NUMBER
#line 190 "zfxparser.y"
                        { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeNewNumberNode(yystack_[0].value.as < float > ()); }
#line 919 "zfxparser.cpp"
    break;

  case 24: // term: LITERAL
#line 191 "zfxparser.y"
                        { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeStringNode(yystack_[0].value.as < string > ()); }
#line 925 "zfxparser.cpp"
    break;

  case 25: // term: UNCOMPSTR
#line 192 "zfxparser.y"
                        { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = driver.makeQuoteStringNode(yystack_[0].value.as < string > ()); }
#line 931 "zfxparser.cpp"
    break;

  case 26: // term: LPAREN exp-statement RPAREN
#line 193 "zfxparser.y"
                                  { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[1].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 937 "zfxparser.cpp"
    break;

  case 27: // term: SUB exp-statement
#line 194 "zfxparser.y"
                                  { yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()->value = -1 * std::get<float>(yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ()->value); yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 943 "zfxparser.cpp"
    break;

  case 28: // term: zenvar
#line 195 "zfxparser.y"
                        { yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > (); }
#line 949 "zfxparser.cpp"
    break;

  case 29: // term: VARNAME func-content
#line 196 "zfxparser.y"
                            { 
        yylhs.value.as < std::shared_ptr<ZfxASTNode> > () = yystack_[0].value.as < std::shared_ptr<ZfxASTNode> > ();
        yylhs.value.as < std::shared_ptr<ZfxASTNode> > ()->opVal = DEFAULT_FUNCVAL;
        yylhs.value.as < std::shared_ptr<ZfxASTNode> > ()->type = FUNC;
        yylhs.value.as < std::shared_ptr<ZfxASTNode> > ()->value = yystack_[1].value.as < string > ();
        yylhs.value.as < std::shared_ptr<ZfxASTNode> > ()->isParenthesisNode = true;
    }
#line 961 "zfxparser.cpp"
    break;


#line 965 "zfxparser.cpp"

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


  const signed char  ZfxParser ::yypact_ninf_ = -18;

  const signed char  ZfxParser ::yytable_ninf_ = -1;

  const signed char
   ZfxParser ::yypact_[] =
  {
       2,   -18,   -18,   -18,   -18,   -18,   -17,    -2,    -2,     4,
       2,   -15,     0,    10,    12,     9,   -18,    -5,    -2,   -18,
      11,    25,     3,   -18,   -18,   -18,    -2,   -18,    -2,    -2,
      -2,    -2,     5,    -2,    25,    18,   -18,     6,    12,    12,
     -18,   -18,   -18,    25,   -18,    -2,    -2,    25,    29,    -2,
      25
  };

  const signed char
   ZfxParser ::yydefact_[] =
  {
       0,     2,    23,    24,    25,    17,    18,     0,     0,     0,
       0,     0,     0,     8,    11,    28,    14,     6,     0,    29,
      18,    27,     0,     1,     3,     4,     0,     5,     0,     0,
       0,     0,     0,     0,    20,     0,    26,     9,    12,    13,
      15,    16,    19,     7,    22,     0,     0,    21,     0,     0,
      10
  };

  const signed char
   ZfxParser ::yypgoto_[] =
  {
     -18,    24,   -18,   -18,    -7,    23,    19,   -18,   -18,   -18,
      20
  };

  const signed char
   ZfxParser ::yydefgoto_[] =
  {
       0,     9,    10,    11,    12,    13,    14,    15,    35,    19,
      16
  };

  const signed char
   ZfxParser ::yytable_[] =
  {
      21,    22,     1,     2,    23,    17,    36,     2,    25,     3,
       4,    34,     5,     3,     4,    26,     5,    18,    26,    33,
      20,    44,    46,    27,     6,     7,    43,    42,    45,     7,
      32,    28,     8,    29,    24,    28,     8,    29,    47,    48,
      26,    30,    50,    31,    26,    18,    49,    38,    39,    37,
      40,    41
  };

  const signed char
   ZfxParser ::yycheck_[] =
  {
       7,     8,     0,     5,     0,    22,     3,     5,    23,    11,
      12,    18,    14,    11,    12,    15,    14,    34,    15,    24,
      22,     3,    16,    23,    22,    27,    33,    22,    10,    27,
      21,    25,    34,    27,    10,    25,    34,    27,    45,    46,
      15,    29,    49,    31,    15,    34,    17,    28,    29,    26,
      30,    31
  };

  const signed char
   ZfxParser ::yystos_[] =
  {
       0,     0,     5,    11,    12,    14,    22,    27,    34,    36,
      37,    38,    39,    40,    41,    42,    45,    22,    34,    44,
      22,    39,    39,     0,    36,    23,    15,    23,    25,    27,
      29,    31,    21,    24,    39,    43,     3,    40,    41,    41,
      45,    45,    22,    39,     3,    10,    16,    39,    39,    17,
      39
  };

  const signed char
   ZfxParser ::yyr1_[] =
  {
       0,    35,    36,    36,    37,    37,    38,    38,    39,    39,
      39,    40,    40,    40,    41,    41,    41,    42,    42,    42,
      43,    43,    44,    45,    45,    45,    45,    45,    45,    45
  };

  const signed char
   ZfxParser ::yyr2_[] =
  {
       0,     2,     1,     2,     2,     2,     2,     4,     1,     3,
       7,     1,     3,     3,     1,     3,     3,     1,     1,     3,
       1,     3,     3,     1,     1,     1,     3,     2,     1,     2
  };


#if YYDEBUG || 1
  // YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
  // First, the terminals, then, starting at \a YYNTOKENS, nonterminals.
  const char*
  const  ZfxParser ::yytname_[] =
  {
  "END", "error", "\"invalid token\"", "RPAREN", "IDENTIFIER", "NUMBER",
  "EOL", "FRAME", "FPS", "PI", "COMMA", "LITERAL", "UNCOMPSTR", "DOLLAR",
  "DOLLARVARNAME", "COMPARE", "QUESTION", "COLON", "ZFXVAR", "LBRACKET",
  "RBRACKET", "DOT", "VARNAME", "SEMICOLON", "EQUALTO", "ADD", "\"+\"",
  "SUB", "\"-\"", "MUL", "\"*\"", "DIV", "\"/\"", "NEG", "LPAREN",
  "$accept", "calclist", "general-statement", "declare-statement",
  "exp-statement", "compareexp", "factor", "zenvar", "funcargs",
  "func-content", "term", YY_NULLPTR
  };
#endif


#if YYDEBUG
  const unsigned char
   ZfxParser ::yyrline_[] =
  {
       0,   108,   108,   113,   119,   120,   122,   128,   136,   137,
     142,   152,   153,   157,   163,   164,   168,   174,   175,   176,
     179,   180,   183,   190,   191,   192,   193,   194,   195,   196
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
#line 1464 "zfxparser.cpp"

#line 212 "zfxparser.y"


// Bison expects us to provide implementation - otherwise linker complains
void zeno::ZfxParser::error(const location &loc , const std::string &message) {
    cout << "Error: " << message << endl << "Error location: " << driver.location() << endl;
}

