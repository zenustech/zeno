// A Bison parser, made by GNU Bison 3.8.2.

// Skeleton interface for Bison LALR(1) parsers in C++

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


/**
 ** \file zfxparser.hpp
 ** Define the  zeno ::parser class.
 */

// C++ LALR(1) parser skeleton written by Akim Demaille.

// DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
// especially those whose name start with YY_ or yy_.  They are
// private implementation details that can be changed or removed.

#ifndef YY_YY_ZFXPARSER_HPP_INCLUDED
# define YY_YY_ZFXPARSER_HPP_INCLUDED
// "%code requires" blocks.
#line 13 "zfxparser.y"

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
        class ZfxScanner;
        class ZfxExecute;
    }

#line 68 "zfxparser.hpp"

# include <cassert>
# include <cstdlib> // std::abort
# include <iostream>
# include <stdexcept>
# include <string>
# include <vector>

#if defined __cplusplus
# define YY_CPLUSPLUS __cplusplus
#else
# define YY_CPLUSPLUS 199711L
#endif

// Support move semantics when possible.
#if 201103L <= YY_CPLUSPLUS
# define YY_MOVE           std::move
# define YY_MOVE_OR_COPY   move
# define YY_MOVE_REF(Type) Type&&
# define YY_RVREF(Type)    Type&&
# define YY_COPY(Type)     Type
#else
# define YY_MOVE
# define YY_MOVE_OR_COPY   copy
# define YY_MOVE_REF(Type) Type&
# define YY_RVREF(Type)    const Type&
# define YY_COPY(Type)     const Type&
#endif

// Support noexcept when possible.
#if 201103L <= YY_CPLUSPLUS
# define YY_NOEXCEPT noexcept
# define YY_NOTHROW
#else
# define YY_NOEXCEPT
# define YY_NOTHROW throw ()
#endif

// Support constexpr when possible.
#if 201703 <= YY_CPLUSPLUS
# define YY_CONSTEXPR constexpr
#else
# define YY_CONSTEXPR
#endif
# include "location.hh"
#include <typeinfo>
#ifndef YY_ASSERT
# include <cassert>
# define YY_ASSERT assert
#endif


#ifndef YY_ATTRIBUTE_PURE
# if defined __GNUC__ && 2 < __GNUC__ + (96 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_PURE __attribute__ ((__pure__))
# else
#  define YY_ATTRIBUTE_PURE
# endif
#endif

#ifndef YY_ATTRIBUTE_UNUSED
# if defined __GNUC__ && 2 < __GNUC__ + (7 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_UNUSED __attribute__ ((__unused__))
# else
#  define YY_ATTRIBUTE_UNUSED
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YY_USE(E) ((void) (E))
#else
# define YY_USE(E) /* empty */
#endif

/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
#if defined __GNUC__ && ! defined __ICC && 406 <= __GNUC__ * 100 + __GNUC_MINOR__
# if __GNUC__ * 100 + __GNUC_MINOR__ < 407
#  define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                           \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")
# else
#  define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                           \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")              \
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# endif
# define YY_IGNORE_MAYBE_UNINITIALIZED_END      \
    _Pragma ("GCC diagnostic pop")
#else
# define YY_INITIAL_VALUE(Value) Value
#endif
#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
# define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif

#if defined __cplusplus && defined __GNUC__ && ! defined __ICC && 6 <= __GNUC__
# define YY_IGNORE_USELESS_CAST_BEGIN                          \
    _Pragma ("GCC diagnostic push")                            \
    _Pragma ("GCC diagnostic ignored \"-Wuseless-cast\"")
# define YY_IGNORE_USELESS_CAST_END            \
    _Pragma ("GCC diagnostic pop")
#endif
#ifndef YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_END
#endif

# ifndef YY_CAST
#  ifdef __cplusplus
#   define YY_CAST(Type, Val) static_cast<Type> (Val)
#   define YY_REINTERPRET_CAST(Type, Val) reinterpret_cast<Type> (Val)
#  else
#   define YY_CAST(Type, Val) ((Type) (Val))
#   define YY_REINTERPRET_CAST(Type, Val) ((Type) (Val))
#  endif
# endif
# ifndef YY_NULLPTR
#  if defined __cplusplus
#   if 201103L <= __cplusplus
#    define YY_NULLPTR nullptr
#   else
#    define YY_NULLPTR 0
#   endif
#  else
#   define YY_NULLPTR ((void*)0)
#  endif
# endif

/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 1
#endif

#line 10 "zfxparser.y"
namespace  zeno  {
#line 209 "zfxparser.hpp"




  /// A Bison parser.
  class  ZfxParser 
  {
  public:
#ifdef YYSTYPE
# ifdef __GNUC__
#  pragma GCC message "bison: do not #define YYSTYPE in C++, use %define api.value.type"
# endif
    typedef YYSTYPE value_type;
#else
  /// A buffer to store and retrieve objects.
  ///
  /// Sort of a variant, but does not keep track of the nature
  /// of the stored data, since that knowledge is available
  /// via the current parser state.
  class value_type
  {
  public:
    /// Type of *this.
    typedef value_type self_type;

    /// Empty construction.
    value_type () YY_NOEXCEPT
      : yyraw_ ()
      , yytypeid_ (YY_NULLPTR)
    {}

    /// Construct and fill.
    template <typename T>
    value_type (YY_RVREF (T) t)
      : yytypeid_ (&typeid (T))
    {
      YY_ASSERT (sizeof (T) <= size);
      new (yyas_<T> ()) T (YY_MOVE (t));
    }

#if 201103L <= YY_CPLUSPLUS
    /// Non copyable.
    value_type (const self_type&) = delete;
    /// Non copyable.
    self_type& operator= (const self_type&) = delete;
#endif

    /// Destruction, allowed only if empty.
    ~value_type () YY_NOEXCEPT
    {
      YY_ASSERT (!yytypeid_);
    }

# if 201103L <= YY_CPLUSPLUS
    /// Instantiate a \a T in here from \a t.
    template <typename T, typename... U>
    T&
    emplace (U&&... u)
    {
      YY_ASSERT (!yytypeid_);
      YY_ASSERT (sizeof (T) <= size);
      yytypeid_ = & typeid (T);
      return *new (yyas_<T> ()) T (std::forward <U>(u)...);
    }
# else
    /// Instantiate an empty \a T in here.
    template <typename T>
    T&
    emplace ()
    {
      YY_ASSERT (!yytypeid_);
      YY_ASSERT (sizeof (T) <= size);
      yytypeid_ = & typeid (T);
      return *new (yyas_<T> ()) T ();
    }

    /// Instantiate a \a T in here from \a t.
    template <typename T>
    T&
    emplace (const T& t)
    {
      YY_ASSERT (!yytypeid_);
      YY_ASSERT (sizeof (T) <= size);
      yytypeid_ = & typeid (T);
      return *new (yyas_<T> ()) T (t);
    }
# endif

    /// Instantiate an empty \a T in here.
    /// Obsolete, use emplace.
    template <typename T>
    T&
    build ()
    {
      return emplace<T> ();
    }

    /// Instantiate a \a T in here from \a t.
    /// Obsolete, use emplace.
    template <typename T>
    T&
    build (const T& t)
    {
      return emplace<T> (t);
    }

    /// Accessor to a built \a T.
    template <typename T>
    T&
    as () YY_NOEXCEPT
    {
      YY_ASSERT (yytypeid_);
      YY_ASSERT (*yytypeid_ == typeid (T));
      YY_ASSERT (sizeof (T) <= size);
      return *yyas_<T> ();
    }

    /// Const accessor to a built \a T (for %printer).
    template <typename T>
    const T&
    as () const YY_NOEXCEPT
    {
      YY_ASSERT (yytypeid_);
      YY_ASSERT (*yytypeid_ == typeid (T));
      YY_ASSERT (sizeof (T) <= size);
      return *yyas_<T> ();
    }

    /// Swap the content with \a that, of same type.
    ///
    /// Both variants must be built beforehand, because swapping the actual
    /// data requires reading it (with as()), and this is not possible on
    /// unconstructed variants: it would require some dynamic testing, which
    /// should not be the variant's responsibility.
    /// Swapping between built and (possibly) non-built is done with
    /// self_type::move ().
    template <typename T>
    void
    swap (self_type& that) YY_NOEXCEPT
    {
      YY_ASSERT (yytypeid_);
      YY_ASSERT (*yytypeid_ == *that.yytypeid_);
      std::swap (as<T> (), that.as<T> ());
    }

    /// Move the content of \a that to this.
    ///
    /// Destroys \a that.
    template <typename T>
    void
    move (self_type& that)
    {
# if 201103L <= YY_CPLUSPLUS
      emplace<T> (std::move (that.as<T> ()));
# else
      emplace<T> ();
      swap<T> (that);
# endif
      that.destroy<T> ();
    }

# if 201103L <= YY_CPLUSPLUS
    /// Move the content of \a that to this.
    template <typename T>
    void
    move (self_type&& that)
    {
      emplace<T> (std::move (that.as<T> ()));
      that.destroy<T> ();
    }
#endif

    /// Copy the content of \a that to this.
    template <typename T>
    void
    copy (const self_type& that)
    {
      emplace<T> (that.as<T> ());
    }

    /// Destroy the stored \a T.
    template <typename T>
    void
    destroy ()
    {
      as<T> ().~T ();
      yytypeid_ = YY_NULLPTR;
    }

  private:
#if YY_CPLUSPLUS < 201103L
    /// Non copyable.
    value_type (const self_type&);
    /// Non copyable.
    self_type& operator= (const self_type&);
#endif

    /// Accessor to raw memory as \a T.
    template <typename T>
    T*
    yyas_ () YY_NOEXCEPT
    {
      void *yyp = yyraw_;
      return static_cast<T*> (yyp);
     }

    /// Const accessor to raw memory as \a T.
    template <typename T>
    const T*
    yyas_ () const YY_NOEXCEPT
    {
      const void *yyp = yyraw_;
      return static_cast<const T*> (yyp);
     }

    /// An auxiliary type to compute the largest semantic type.
    union union_type
    {
      // NUMBER
      char dummy1[sizeof (float)];

      // calclist
      // general-statement
      // declare-statement
      // exp-statement
      // compareexp
      // factor
      // zenvar
      // func-content
      // term
      char dummy2[sizeof (std::shared_ptr<ZfxASTNode>)];

      // funcargs
      char dummy3[sizeof (std::vector<std::shared_ptr<ZfxASTNode>>)];

      // RPAREN
      // IDENTIFIER
      // LITERAL
      // UNCOMPSTR
      // DOLLAR
      // DOLLARVARNAME
      // COMPARE
      // QUESTION
      // COLON
      // ZFXVAR
      // LBRACKET
      // RBRACKET
      // DOT
      // VARNAME
      // SEMICOLON
      // EQUALTO
      // LPAREN
      char dummy4[sizeof (string)];
    };

    /// The size of the largest semantic type.
    enum { size = sizeof (union_type) };

    /// A buffer to store semantic values.
    union
    {
      /// Strongest alignment constraints.
      long double yyalign_me_;
      /// A buffer large enough to store any of the semantic values.
      char yyraw_[size];
    };

    /// Whether the content is built: if defined, the name of the stored type.
    const std::type_info *yytypeid_;
  };

#endif
    /// Backward compatibility (Bison 3.8).
    typedef value_type semantic_type;

    /// Symbol locations.
    typedef location location_type;

    /// Syntax errors thrown from user actions.
    struct syntax_error : std::runtime_error
    {
      syntax_error (const location_type& l, const std::string& m)
        : std::runtime_error (m)
        , location (l)
      {}

      syntax_error (const syntax_error& s)
        : std::runtime_error (s.what ())
        , location (s.location)
      {}

      ~syntax_error () YY_NOEXCEPT YY_NOTHROW;

      location_type location;
    };

    /// Token kinds.
    struct token
    {
      enum token_kind_type
      {
        TOKEN_YYEMPTY = -2,
    TOKEN_END = 0,                 // END
    TOKEN_YYerror = 256,           // error
    TOKEN_YYUNDEF = 257,           // "invalid token"
    TOKEN_RPAREN = 258,            // RPAREN
    TOKEN_IDENTIFIER = 259,        // IDENTIFIER
    TOKEN_NUMBER = 260,            // NUMBER
    TOKEN_EOL = 261,               // EOL
    TOKEN_FRAME = 262,             // FRAME
    TOKEN_FPS = 263,               // FPS
    TOKEN_PI = 264,                // PI
    TOKEN_COMMA = 265,             // COMMA
    TOKEN_LITERAL = 266,           // LITERAL
    TOKEN_UNCOMPSTR = 267,         // UNCOMPSTR
    TOKEN_DOLLAR = 268,            // DOLLAR
    TOKEN_DOLLARVARNAME = 269,     // DOLLARVARNAME
    TOKEN_COMPARE = 270,           // COMPARE
    TOKEN_QUESTION = 271,          // QUESTION
    TOKEN_COLON = 272,             // COLON
    TOKEN_ZFXVAR = 273,            // ZFXVAR
    TOKEN_LBRACKET = 274,          // LBRACKET
    TOKEN_RBRACKET = 275,          // RBRACKET
    TOKEN_DOT = 276,               // DOT
    TOKEN_VARNAME = 277,           // VARNAME
    TOKEN_SEMICOLON = 278,         // SEMICOLON
    TOKEN_EQUALTO = 279,           // EQUALTO
    TOKEN_ADD = 280,               // ADD
    TOKEN_SUB = 282,               // SUB
    TOKEN_MUL = 284,               // MUL
    TOKEN_DIV = 286,               // DIV
    TOKEN_NEG = 288,               // NEG
    TOKEN_LPAREN = 289             // LPAREN
      };
      /// Backward compatibility alias (Bison 3.6).
      typedef token_kind_type yytokentype;
    };

    /// Token kind, as returned by yylex.
    typedef token::token_kind_type token_kind_type;

    /// Backward compatibility alias (Bison 3.6).
    typedef token_kind_type token_type;

    /// Symbol kinds.
    struct symbol_kind
    {
      enum symbol_kind_type
      {
        YYNTOKENS = 35, ///< Number of tokens.
        S_YYEMPTY = -2,
        S_YYEOF = 0,                             // END
        S_YYerror = 1,                           // error
        S_YYUNDEF = 2,                           // "invalid token"
        S_RPAREN = 3,                            // RPAREN
        S_IDENTIFIER = 4,                        // IDENTIFIER
        S_NUMBER = 5,                            // NUMBER
        S_EOL = 6,                               // EOL
        S_FRAME = 7,                             // FRAME
        S_FPS = 8,                               // FPS
        S_PI = 9,                                // PI
        S_COMMA = 10,                            // COMMA
        S_LITERAL = 11,                          // LITERAL
        S_UNCOMPSTR = 12,                        // UNCOMPSTR
        S_DOLLAR = 13,                           // DOLLAR
        S_DOLLARVARNAME = 14,                    // DOLLARVARNAME
        S_COMPARE = 15,                          // COMPARE
        S_QUESTION = 16,                         // QUESTION
        S_COLON = 17,                            // COLON
        S_ZFXVAR = 18,                           // ZFXVAR
        S_LBRACKET = 19,                         // LBRACKET
        S_RBRACKET = 20,                         // RBRACKET
        S_DOT = 21,                              // DOT
        S_VARNAME = 22,                          // VARNAME
        S_SEMICOLON = 23,                        // SEMICOLON
        S_EQUALTO = 24,                          // EQUALTO
        S_ADD = 25,                              // ADD
        S_26_ = 26,                              // "+"
        S_SUB = 27,                              // SUB
        S_28_ = 28,                              // "-"
        S_MUL = 29,                              // MUL
        S_30_ = 30,                              // "*"
        S_DIV = 31,                              // DIV
        S_32_ = 32,                              // "/"
        S_NEG = 33,                              // NEG
        S_LPAREN = 34,                           // LPAREN
        S_YYACCEPT = 35,                         // $accept
        S_calclist = 36,                         // calclist
        S_37_general_statement = 37,             // general-statement
        S_38_declare_statement = 38,             // declare-statement
        S_39_exp_statement = 39,                 // exp-statement
        S_compareexp = 40,                       // compareexp
        S_factor = 41,                           // factor
        S_zenvar = 42,                           // zenvar
        S_funcargs = 43,                         // funcargs
        S_44_func_content = 44,                  // func-content
        S_term = 45                              // term
      };
    };

    /// (Internal) symbol kind.
    typedef symbol_kind::symbol_kind_type symbol_kind_type;

    /// The number of tokens.
    static const symbol_kind_type YYNTOKENS = symbol_kind::YYNTOKENS;

    /// A complete symbol.
    ///
    /// Expects its Base type to provide access to the symbol kind
    /// via kind ().
    ///
    /// Provide access to semantic value and location.
    template <typename Base>
    struct basic_symbol : Base
    {
      /// Alias to Base.
      typedef Base super_type;

      /// Default constructor.
      basic_symbol () YY_NOEXCEPT
        : value ()
        , location ()
      {}

#if 201103L <= YY_CPLUSPLUS
      /// Move constructor.
      basic_symbol (basic_symbol&& that)
        : Base (std::move (that))
        , value ()
        , location (std::move (that.location))
      {
        switch (this->kind ())
    {
      case symbol_kind::S_NUMBER: // NUMBER
        value.move< float > (std::move (that.value));
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
        value.move< std::shared_ptr<ZfxASTNode> > (std::move (that.value));
        break;

      case symbol_kind::S_funcargs: // funcargs
        value.move< std::vector<std::shared_ptr<ZfxASTNode>> > (std::move (that.value));
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
        value.move< string > (std::move (that.value));
        break;

      default:
        break;
    }

      }
#endif

      /// Copy constructor.
      basic_symbol (const basic_symbol& that);

      /// Constructors for typed symbols.
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, location_type&& l)
        : Base (t)
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const location_type& l)
        : Base (t)
        , location (l)
      {}
#endif

#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, float&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const float& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif

#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, std::shared_ptr<ZfxASTNode>&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const std::shared_ptr<ZfxASTNode>& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif

#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, std::vector<std::shared_ptr<ZfxASTNode>>&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const std::vector<std::shared_ptr<ZfxASTNode>>& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif

#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, string&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const string& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif

      /// Destroy the symbol.
      ~basic_symbol ()
      {
        clear ();
      }



      /// Destroy contents, and record that is empty.
      void clear () YY_NOEXCEPT
      {
        // User destructor.
        symbol_kind_type yykind = this->kind ();
        basic_symbol<Base>& yysym = *this;
        (void) yysym;
        switch (yykind)
        {
       default:
          break;
        }

        // Value type destructor.
switch (yykind)
    {
      case symbol_kind::S_NUMBER: // NUMBER
        value.template destroy< float > ();
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
        value.template destroy< std::shared_ptr<ZfxASTNode> > ();
        break;

      case symbol_kind::S_funcargs: // funcargs
        value.template destroy< std::vector<std::shared_ptr<ZfxASTNode>> > ();
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
        value.template destroy< string > ();
        break;

      default:
        break;
    }

        Base::clear ();
      }

      /// The user-facing name of this symbol.
      std::string name () const YY_NOEXCEPT
      {
        return  ZfxParser ::symbol_name (this->kind ());
      }

      /// Backward compatibility (Bison 3.6).
      symbol_kind_type type_get () const YY_NOEXCEPT;

      /// Whether empty.
      bool empty () const YY_NOEXCEPT;

      /// Destructive move, \a s is emptied into this.
      void move (basic_symbol& s);

      /// The semantic value.
      value_type value;

      /// The location.
      location_type location;

    private:
#if YY_CPLUSPLUS < 201103L
      /// Assignment operator.
      basic_symbol& operator= (const basic_symbol& that);
#endif
    };

    /// Type access provider for token (enum) based symbols.
    struct by_kind
    {
      /// The symbol kind as needed by the constructor.
      typedef token_kind_type kind_type;

      /// Default constructor.
      by_kind () YY_NOEXCEPT;

#if 201103L <= YY_CPLUSPLUS
      /// Move constructor.
      by_kind (by_kind&& that) YY_NOEXCEPT;
#endif

      /// Copy constructor.
      by_kind (const by_kind& that) YY_NOEXCEPT;

      /// Constructor from (external) token numbers.
      by_kind (kind_type t) YY_NOEXCEPT;



      /// Record that this symbol is empty.
      void clear () YY_NOEXCEPT;

      /// Steal the symbol kind from \a that.
      void move (by_kind& that);

      /// The (internal) type number (corresponding to \a type).
      /// \a empty when empty.
      symbol_kind_type kind () const YY_NOEXCEPT;

      /// Backward compatibility (Bison 3.6).
      symbol_kind_type type_get () const YY_NOEXCEPT;

      /// The symbol kind.
      /// \a S_YYEMPTY when empty.
      symbol_kind_type kind_;
    };

    /// Backward compatibility for a private implementation detail (Bison 3.6).
    typedef by_kind by_type;

    /// "External" symbols: returned by the scanner.
    struct symbol_type : basic_symbol<by_kind>
    {
      /// Superclass.
      typedef basic_symbol<by_kind> super_type;

      /// Empty symbol.
      symbol_type () YY_NOEXCEPT {}

      /// Constructor for valueless symbols, and symbols from each type.
#if 201103L <= YY_CPLUSPLUS
      symbol_type (int tok, location_type l)
        : super_type (token_kind_type (tok), std::move (l))
#else
      symbol_type (int tok, const location_type& l)
        : super_type (token_kind_type (tok), l)
#endif
      {
#if !defined _MSC_VER || defined __clang__
        YY_ASSERT (tok == token::TOKEN_END
                   || (token::TOKEN_YYerror <= tok && tok <= token::TOKEN_YYUNDEF)
                   || (token::TOKEN_EOL <= tok && tok <= token::TOKEN_COMMA)
                   || (token::TOKEN_ADD <= tok && tok <= token::TOKEN_NEG));
#endif
      }
#if 201103L <= YY_CPLUSPLUS
      symbol_type (int tok, float v, location_type l)
        : super_type (token_kind_type (tok), std::move (v), std::move (l))
#else
      symbol_type (int tok, const float& v, const location_type& l)
        : super_type (token_kind_type (tok), v, l)
#endif
      {
#if !defined _MSC_VER || defined __clang__
        YY_ASSERT (tok == token::TOKEN_NUMBER);
#endif
      }
#if 201103L <= YY_CPLUSPLUS
      symbol_type (int tok, string v, location_type l)
        : super_type (token_kind_type (tok), std::move (v), std::move (l))
#else
      symbol_type (int tok, const string& v, const location_type& l)
        : super_type (token_kind_type (tok), v, l)
#endif
      {
#if !defined _MSC_VER || defined __clang__
        YY_ASSERT ((token::TOKEN_RPAREN <= tok && tok <= token::TOKEN_IDENTIFIER)
                   || (token::TOKEN_LITERAL <= tok && tok <= token::TOKEN_EQUALTO)
                   || tok == token::TOKEN_LPAREN);
#endif
      }
    };

    /// Build a parser object.
     ZfxParser  (zeno::ZfxScanner &scanner_yyarg, zeno::ZfxExecute &driver_yyarg);
    virtual ~ ZfxParser  ();

#if 201103L <= YY_CPLUSPLUS
    /// Non copyable.
     ZfxParser  (const  ZfxParser &) = delete;
    /// Non copyable.
     ZfxParser & operator= (const  ZfxParser &) = delete;
#endif

    /// Parse.  An alias for parse ().
    /// \returns  0 iff parsing succeeded.
    int operator() ();

    /// Parse.
    /// \returns  0 iff parsing succeeded.
    virtual int parse ();

#if YYDEBUG
    /// The current debugging stream.
    std::ostream& debug_stream () const YY_ATTRIBUTE_PURE;
    /// Set the current debugging stream.
    void set_debug_stream (std::ostream &);

    /// Type for debugging levels.
    typedef int debug_level_type;
    /// The current debugging level.
    debug_level_type debug_level () const YY_ATTRIBUTE_PURE;
    /// Set the current debugging level.
    void set_debug_level (debug_level_type l);
#endif

    /// Report a syntax error.
    /// \param loc    where the syntax error is found.
    /// \param msg    a description of the syntax error.
    virtual void error (const location_type& loc, const std::string& msg);

    /// Report a syntax error.
    void error (const syntax_error& err);

    /// The user-facing name of the symbol whose (internal) number is
    /// YYSYMBOL.  No bounds checking.
    static std::string symbol_name (symbol_kind_type yysymbol);

    // Implementation of make_symbol for each token kind.
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_END (location_type l)
      {
        return symbol_type (token::TOKEN_END, std::move (l));
      }
#else
      static
      symbol_type
      make_END (const location_type& l)
      {
        return symbol_type (token::TOKEN_END, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_YYerror (location_type l)
      {
        return symbol_type (token::TOKEN_YYerror, std::move (l));
      }
#else
      static
      symbol_type
      make_YYerror (const location_type& l)
      {
        return symbol_type (token::TOKEN_YYerror, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_YYUNDEF (location_type l)
      {
        return symbol_type (token::TOKEN_YYUNDEF, std::move (l));
      }
#else
      static
      symbol_type
      make_YYUNDEF (const location_type& l)
      {
        return symbol_type (token::TOKEN_YYUNDEF, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_RPAREN (string v, location_type l)
      {
        return symbol_type (token::TOKEN_RPAREN, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_RPAREN (const string& v, const location_type& l)
      {
        return symbol_type (token::TOKEN_RPAREN, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_IDENTIFIER (string v, location_type l)
      {
        return symbol_type (token::TOKEN_IDENTIFIER, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_IDENTIFIER (const string& v, const location_type& l)
      {
        return symbol_type (token::TOKEN_IDENTIFIER, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_NUMBER (float v, location_type l)
      {
        return symbol_type (token::TOKEN_NUMBER, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_NUMBER (const float& v, const location_type& l)
      {
        return symbol_type (token::TOKEN_NUMBER, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_EOL (location_type l)
      {
        return symbol_type (token::TOKEN_EOL, std::move (l));
      }
#else
      static
      symbol_type
      make_EOL (const location_type& l)
      {
        return symbol_type (token::TOKEN_EOL, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_FRAME (location_type l)
      {
        return symbol_type (token::TOKEN_FRAME, std::move (l));
      }
#else
      static
      symbol_type
      make_FRAME (const location_type& l)
      {
        return symbol_type (token::TOKEN_FRAME, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_FPS (location_type l)
      {
        return symbol_type (token::TOKEN_FPS, std::move (l));
      }
#else
      static
      symbol_type
      make_FPS (const location_type& l)
      {
        return symbol_type (token::TOKEN_FPS, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_PI (location_type l)
      {
        return symbol_type (token::TOKEN_PI, std::move (l));
      }
#else
      static
      symbol_type
      make_PI (const location_type& l)
      {
        return symbol_type (token::TOKEN_PI, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_COMMA (location_type l)
      {
        return symbol_type (token::TOKEN_COMMA, std::move (l));
      }
#else
      static
      symbol_type
      make_COMMA (const location_type& l)
      {
        return symbol_type (token::TOKEN_COMMA, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_LITERAL (string v, location_type l)
      {
        return symbol_type (token::TOKEN_LITERAL, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_LITERAL (const string& v, const location_type& l)
      {
        return symbol_type (token::TOKEN_LITERAL, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_UNCOMPSTR (string v, location_type l)
      {
        return symbol_type (token::TOKEN_UNCOMPSTR, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_UNCOMPSTR (const string& v, const location_type& l)
      {
        return symbol_type (token::TOKEN_UNCOMPSTR, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DOLLAR (string v, location_type l)
      {
        return symbol_type (token::TOKEN_DOLLAR, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_DOLLAR (const string& v, const location_type& l)
      {
        return symbol_type (token::TOKEN_DOLLAR, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DOLLARVARNAME (string v, location_type l)
      {
        return symbol_type (token::TOKEN_DOLLARVARNAME, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_DOLLARVARNAME (const string& v, const location_type& l)
      {
        return symbol_type (token::TOKEN_DOLLARVARNAME, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_COMPARE (string v, location_type l)
      {
        return symbol_type (token::TOKEN_COMPARE, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_COMPARE (const string& v, const location_type& l)
      {
        return symbol_type (token::TOKEN_COMPARE, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_QUESTION (string v, location_type l)
      {
        return symbol_type (token::TOKEN_QUESTION, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_QUESTION (const string& v, const location_type& l)
      {
        return symbol_type (token::TOKEN_QUESTION, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_COLON (string v, location_type l)
      {
        return symbol_type (token::TOKEN_COLON, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_COLON (const string& v, const location_type& l)
      {
        return symbol_type (token::TOKEN_COLON, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ZFXVAR (string v, location_type l)
      {
        return symbol_type (token::TOKEN_ZFXVAR, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_ZFXVAR (const string& v, const location_type& l)
      {
        return symbol_type (token::TOKEN_ZFXVAR, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_LBRACKET (string v, location_type l)
      {
        return symbol_type (token::TOKEN_LBRACKET, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_LBRACKET (const string& v, const location_type& l)
      {
        return symbol_type (token::TOKEN_LBRACKET, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_RBRACKET (string v, location_type l)
      {
        return symbol_type (token::TOKEN_RBRACKET, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_RBRACKET (const string& v, const location_type& l)
      {
        return symbol_type (token::TOKEN_RBRACKET, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DOT (string v, location_type l)
      {
        return symbol_type (token::TOKEN_DOT, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_DOT (const string& v, const location_type& l)
      {
        return symbol_type (token::TOKEN_DOT, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_VARNAME (string v, location_type l)
      {
        return symbol_type (token::TOKEN_VARNAME, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_VARNAME (const string& v, const location_type& l)
      {
        return symbol_type (token::TOKEN_VARNAME, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SEMICOLON (string v, location_type l)
      {
        return symbol_type (token::TOKEN_SEMICOLON, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_SEMICOLON (const string& v, const location_type& l)
      {
        return symbol_type (token::TOKEN_SEMICOLON, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_EQUALTO (string v, location_type l)
      {
        return symbol_type (token::TOKEN_EQUALTO, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_EQUALTO (const string& v, const location_type& l)
      {
        return symbol_type (token::TOKEN_EQUALTO, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ADD (location_type l)
      {
        return symbol_type (token::TOKEN_ADD, std::move (l));
      }
#else
      static
      symbol_type
      make_ADD (const location_type& l)
      {
        return symbol_type (token::TOKEN_ADD, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SUB (location_type l)
      {
        return symbol_type (token::TOKEN_SUB, std::move (l));
      }
#else
      static
      symbol_type
      make_SUB (const location_type& l)
      {
        return symbol_type (token::TOKEN_SUB, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MUL (location_type l)
      {
        return symbol_type (token::TOKEN_MUL, std::move (l));
      }
#else
      static
      symbol_type
      make_MUL (const location_type& l)
      {
        return symbol_type (token::TOKEN_MUL, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DIV (location_type l)
      {
        return symbol_type (token::TOKEN_DIV, std::move (l));
      }
#else
      static
      symbol_type
      make_DIV (const location_type& l)
      {
        return symbol_type (token::TOKEN_DIV, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_NEG (location_type l)
      {
        return symbol_type (token::TOKEN_NEG, std::move (l));
      }
#else
      static
      symbol_type
      make_NEG (const location_type& l)
      {
        return symbol_type (token::TOKEN_NEG, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_LPAREN (string v, location_type l)
      {
        return symbol_type (token::TOKEN_LPAREN, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_LPAREN (const string& v, const location_type& l)
      {
        return symbol_type (token::TOKEN_LPAREN, v, l);
      }
#endif


    class context
    {
    public:
      context (const  ZfxParser & yyparser, const symbol_type& yyla);
      const symbol_type& lookahead () const YY_NOEXCEPT { return yyla_; }
      symbol_kind_type token () const YY_NOEXCEPT { return yyla_.kind (); }
      const location_type& location () const YY_NOEXCEPT { return yyla_.location; }

      /// Put in YYARG at most YYARGN of the expected tokens, and return the
      /// number of tokens stored in YYARG.  If YYARG is null, return the
      /// number of expected tokens (guaranteed to be less than YYNTOKENS).
      int expected_tokens (symbol_kind_type yyarg[], int yyargn) const;

    private:
      const  ZfxParser & yyparser_;
      const symbol_type& yyla_;
    };

  private:
#if YY_CPLUSPLUS < 201103L
    /// Non copyable.
     ZfxParser  (const  ZfxParser &);
    /// Non copyable.
     ZfxParser & operator= (const  ZfxParser &);
#endif


    /// Stored state numbers (used for stacks).
    typedef signed char state_type;

    /// The arguments of the error message.
    int yy_syntax_error_arguments_ (const context& yyctx,
                                    symbol_kind_type yyarg[], int yyargn) const;

    /// Generate an error message.
    /// \param yyctx     the context in which the error occurred.
    virtual std::string yysyntax_error_ (const context& yyctx) const;
    /// Compute post-reduction state.
    /// \param yystate   the current state
    /// \param yysym     the nonterminal to push on the stack
    static state_type yy_lr_goto_state_ (state_type yystate, int yysym);

    /// Whether the given \c yypact_ value indicates a defaulted state.
    /// \param yyvalue   the value to check
    static bool yy_pact_value_is_default_ (int yyvalue) YY_NOEXCEPT;

    /// Whether the given \c yytable_ value indicates a syntax error.
    /// \param yyvalue   the value to check
    static bool yy_table_value_is_error_ (int yyvalue) YY_NOEXCEPT;

    static const signed char yypact_ninf_;
    static const signed char yytable_ninf_;

    /// Convert a scanner token kind \a t to a symbol kind.
    /// In theory \a t should be a token_kind_type, but character literals
    /// are valid, yet not members of the token_kind_type enum.
    static symbol_kind_type yytranslate_ (int t) YY_NOEXCEPT;

    /// Convert the symbol name \a n to a form suitable for a diagnostic.
    static std::string yytnamerr_ (const char *yystr);

    /// For a symbol, its name in clear.
    static const char* const yytname_[];


    // Tables.
    // YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
    // STATE-NUM.
    static const signed char yypact_[];

    // YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
    // Performed when YYTABLE does not specify something else to do.  Zero
    // means the default is an error.
    static const signed char yydefact_[];

    // YYPGOTO[NTERM-NUM].
    static const signed char yypgoto_[];

    // YYDEFGOTO[NTERM-NUM].
    static const signed char yydefgoto_[];

    // YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
    // positive, shift that token.  If negative, reduce the rule whose
    // number is the opposite.  If YYTABLE_NINF, syntax error.
    static const signed char yytable_[];

    static const signed char yycheck_[];

    // YYSTOS[STATE-NUM] -- The symbol kind of the accessing symbol of
    // state STATE-NUM.
    static const signed char yystos_[];

    // YYR1[RULE-NUM] -- Symbol kind of the left-hand side of rule RULE-NUM.
    static const signed char yyr1_[];

    // YYR2[RULE-NUM] -- Number of symbols on the right-hand side of rule RULE-NUM.
    static const signed char yyr2_[];


#if YYDEBUG
    // YYRLINE[YYN] -- Source line where rule number YYN was defined.
    static const unsigned char yyrline_[];
    /// Report on the debug stream that the rule \a r is going to be reduced.
    virtual void yy_reduce_print_ (int r) const;
    /// Print the state stack on the debug stream.
    virtual void yy_stack_print_ () const;

    /// Debugging level.
    int yydebug_;
    /// Debug stream.
    std::ostream* yycdebug_;

    /// \brief Display a symbol kind, value and location.
    /// \param yyo    The output stream.
    /// \param yysym  The symbol.
    template <typename Base>
    void yy_print_ (std::ostream& yyo, const basic_symbol<Base>& yysym) const;
#endif

    /// \brief Reclaim the memory associated to a symbol.
    /// \param yymsg     Why this token is reclaimed.
    ///                  If null, print nothing.
    /// \param yysym     The symbol.
    template <typename Base>
    void yy_destroy_ (const char* yymsg, basic_symbol<Base>& yysym) const;

  private:
    /// Type access provider for state based symbols.
    struct by_state
    {
      /// Default constructor.
      by_state () YY_NOEXCEPT;

      /// The symbol kind as needed by the constructor.
      typedef state_type kind_type;

      /// Constructor.
      by_state (kind_type s) YY_NOEXCEPT;

      /// Copy constructor.
      by_state (const by_state& that) YY_NOEXCEPT;

      /// Record that this symbol is empty.
      void clear () YY_NOEXCEPT;

      /// Steal the symbol kind from \a that.
      void move (by_state& that);

      /// The symbol kind (corresponding to \a state).
      /// \a symbol_kind::S_YYEMPTY when empty.
      symbol_kind_type kind () const YY_NOEXCEPT;

      /// The state number used to denote an empty symbol.
      /// We use the initial state, as it does not have a value.
      enum { empty_state = 0 };

      /// The state.
      /// \a empty when empty.
      state_type state;
    };

    /// "Internal" symbol: element of the stack.
    struct stack_symbol_type : basic_symbol<by_state>
    {
      /// Superclass.
      typedef basic_symbol<by_state> super_type;
      /// Construct an empty symbol.
      stack_symbol_type ();
      /// Move or copy construction.
      stack_symbol_type (YY_RVREF (stack_symbol_type) that);
      /// Steal the contents from \a sym to build this.
      stack_symbol_type (state_type s, YY_MOVE_REF (symbol_type) sym);
#if YY_CPLUSPLUS < 201103L
      /// Assignment, needed by push_back by some old implementations.
      /// Moves the contents of that.
      stack_symbol_type& operator= (stack_symbol_type& that);

      /// Assignment, needed by push_back by other implementations.
      /// Needed by some other old implementations.
      stack_symbol_type& operator= (const stack_symbol_type& that);
#endif
    };

    /// A stack with random access from its top.
    template <typename T, typename S = std::vector<T> >
    class stack
    {
    public:
      // Hide our reversed order.
      typedef typename S::iterator iterator;
      typedef typename S::const_iterator const_iterator;
      typedef typename S::size_type size_type;
      typedef typename std::ptrdiff_t index_type;

      stack (size_type n = 200) YY_NOEXCEPT
        : seq_ (n)
      {}

#if 201103L <= YY_CPLUSPLUS
      /// Non copyable.
      stack (const stack&) = delete;
      /// Non copyable.
      stack& operator= (const stack&) = delete;
#endif

      /// Random access.
      ///
      /// Index 0 returns the topmost element.
      const T&
      operator[] (index_type i) const
      {
        return seq_[size_type (size () - 1 - i)];
      }

      /// Random access.
      ///
      /// Index 0 returns the topmost element.
      T&
      operator[] (index_type i)
      {
        return seq_[size_type (size () - 1 - i)];
      }

      /// Steal the contents of \a t.
      ///
      /// Close to move-semantics.
      void
      push (YY_MOVE_REF (T) t)
      {
        seq_.push_back (T ());
        operator[] (0).move (t);
      }

      /// Pop elements from the stack.
      void
      pop (std::ptrdiff_t n = 1) YY_NOEXCEPT
      {
        for (; 0 < n; --n)
          seq_.pop_back ();
      }

      /// Pop all elements from the stack.
      void
      clear () YY_NOEXCEPT
      {
        seq_.clear ();
      }

      /// Number of elements on the stack.
      index_type
      size () const YY_NOEXCEPT
      {
        return index_type (seq_.size ());
      }

      /// Iterator on top of the stack (going downwards).
      const_iterator
      begin () const YY_NOEXCEPT
      {
        return seq_.begin ();
      }

      /// Bottom of the stack.
      const_iterator
      end () const YY_NOEXCEPT
      {
        return seq_.end ();
      }

      /// Present a slice of the top of a stack.
      class slice
      {
      public:
        slice (const stack& stack, index_type range) YY_NOEXCEPT
          : stack_ (stack)
          , range_ (range)
        {}

        const T&
        operator[] (index_type i) const
        {
          return stack_[range_ - i];
        }

      private:
        const stack& stack_;
        index_type range_;
      };

    private:
#if YY_CPLUSPLUS < 201103L
      /// Non copyable.
      stack (const stack&);
      /// Non copyable.
      stack& operator= (const stack&);
#endif
      /// The wrapped container.
      S seq_;
    };


    /// Stack type.
    typedef stack<stack_symbol_type> stack_type;

    /// The stack.
    stack_type yystack_;

    /// Push a new state on the stack.
    /// \param m    a debug message to display
    ///             if null, no trace is output.
    /// \param sym  the symbol
    /// \warning the contents of \a s.value is stolen.
    void yypush_ (const char* m, YY_MOVE_REF (stack_symbol_type) sym);

    /// Push a new look ahead token on the state on the stack.
    /// \param m    a debug message to display
    ///             if null, no trace is output.
    /// \param s    the state
    /// \param sym  the symbol (for its value and location).
    /// \warning the contents of \a sym.value is stolen.
    void yypush_ (const char* m, state_type s, YY_MOVE_REF (symbol_type) sym);

    /// Pop \a n symbols from the stack.
    void yypop_ (int n = 1) YY_NOEXCEPT;

    /// Constants.
    enum
    {
      yylast_ = 51,     ///< Last index in yytable_.
      yynnts_ = 11,  ///< Number of nonterminal symbols.
      yyfinal_ = 23 ///< Termination state number.
    };


    // User arguments.
    zeno::ZfxScanner &scanner;
    zeno::ZfxExecute &driver;

  };

  inline
   ZfxParser ::symbol_kind_type
   ZfxParser ::yytranslate_ (int t) YY_NOEXCEPT
  {
    // YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to
    // TOKEN-NUM as returned by yylex.
    static
    const signed char
    translate_table[] =
    {
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34
    };
    // Last valid token kind.
    const int code_max = 289;

    if (t <= 0)
      return symbol_kind::S_YYEOF;
    else if (t <= code_max)
      return static_cast <symbol_kind_type> (translate_table[t]);
    else
      return symbol_kind::S_YYUNDEF;
  }

  // basic_symbol.
  template <typename Base>
   ZfxParser ::basic_symbol<Base>::basic_symbol (const basic_symbol& that)
    : Base (that)
    , value ()
    , location (that.location)
  {
    switch (this->kind ())
    {
      case symbol_kind::S_NUMBER: // NUMBER
        value.copy< float > (YY_MOVE (that.value));
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
        value.copy< std::shared_ptr<ZfxASTNode> > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_funcargs: // funcargs
        value.copy< std::vector<std::shared_ptr<ZfxASTNode>> > (YY_MOVE (that.value));
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
        value.copy< string > (YY_MOVE (that.value));
        break;

      default:
        break;
    }

  }




  template <typename Base>
   ZfxParser ::symbol_kind_type
   ZfxParser ::basic_symbol<Base>::type_get () const YY_NOEXCEPT
  {
    return this->kind ();
  }


  template <typename Base>
  bool
   ZfxParser ::basic_symbol<Base>::empty () const YY_NOEXCEPT
  {
    return this->kind () == symbol_kind::S_YYEMPTY;
  }

  template <typename Base>
  void
   ZfxParser ::basic_symbol<Base>::move (basic_symbol& s)
  {
    super_type::move (s);
    switch (this->kind ())
    {
      case symbol_kind::S_NUMBER: // NUMBER
        value.move< float > (YY_MOVE (s.value));
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
        value.move< std::shared_ptr<ZfxASTNode> > (YY_MOVE (s.value));
        break;

      case symbol_kind::S_funcargs: // funcargs
        value.move< std::vector<std::shared_ptr<ZfxASTNode>> > (YY_MOVE (s.value));
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
        value.move< string > (YY_MOVE (s.value));
        break;

      default:
        break;
    }

    location = YY_MOVE (s.location);
  }

  // by_kind.
  inline
   ZfxParser ::by_kind::by_kind () YY_NOEXCEPT
    : kind_ (symbol_kind::S_YYEMPTY)
  {}

#if 201103L <= YY_CPLUSPLUS
  inline
   ZfxParser ::by_kind::by_kind (by_kind&& that) YY_NOEXCEPT
    : kind_ (that.kind_)
  {
    that.clear ();
  }
#endif

  inline
   ZfxParser ::by_kind::by_kind (const by_kind& that) YY_NOEXCEPT
    : kind_ (that.kind_)
  {}

  inline
   ZfxParser ::by_kind::by_kind (token_kind_type t) YY_NOEXCEPT
    : kind_ (yytranslate_ (t))
  {}



  inline
  void
   ZfxParser ::by_kind::clear () YY_NOEXCEPT
  {
    kind_ = symbol_kind::S_YYEMPTY;
  }

  inline
  void
   ZfxParser ::by_kind::move (by_kind& that)
  {
    kind_ = that.kind_;
    that.clear ();
  }

  inline
   ZfxParser ::symbol_kind_type
   ZfxParser ::by_kind::kind () const YY_NOEXCEPT
  {
    return kind_;
  }


  inline
   ZfxParser ::symbol_kind_type
   ZfxParser ::by_kind::type_get () const YY_NOEXCEPT
  {
    return this->kind ();
  }


#line 10 "zfxparser.y"
} //  zeno 
#line 2048 "zfxparser.hpp"




#endif // !YY_YY_ZFXPARSER_HPP_INCLUDED
