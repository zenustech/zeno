dnl Minimum version of autoconf required.  Should coincide with the
dnl setting in the autogen.sh script.
AC_PREREQ(2.58)

AC_INIT(FTGL, 2.3.0, [https://sourceforge.net/projects/ftgl/], ftgl)
AC_CONFIG_SRCDIR(src/FTPoint.cpp)
AC_CONFIG_AUX_DIR(.auto)
AC_CONFIG_MACRO_DIR([m4])
AM_INIT_AUTOMAKE([1.6 no-define dist-zip dist-bzip2])
AM_CONFIG_HEADER(config.h)

FTGL_PROG_CXX

LT_MAJOR="2"
LT_MINOR="3"
LT_MICRO="0"
AC_SUBST(LT_MAJOR)
AC_SUBST(LT_MINOR)
AC_SUBST(LT_MICRO)
LT_VERSION="$LT_MAJOR:$LT_MINOR:$LT_MICRO"
AC_SUBST(LT_VERSION)

# make sure ECHO and ECHO_N got defined and substituted
if test "x$ECHO" = "x" ; then
    ECHO=echo
    AC_MSG_NOTICE([ECHO was not defined by configure so defining manually])
fi
AC_SUBST(ECHO)
AC_SUBST(ECHO_N)

AC_PROG_LIBTOOL
AC_PROG_INSTALL

# Checks for typedefs, structures, and compiler characteristics.

dnl These don't mix with C++
dnl AC_C_CONST
dnl AC_C_INLINE
AM_PROG_CC_C_O

FTGL_CPP_FUNC

# Checks for header files.
AC_HEADER_STDC
AC_CHECK_HEADER([stdlib.h])

# Check for system functions
AC_CHECK_FUNCS(wcsdup)
AC_CHECK_FUNCS(strndup)

# Checks for libraries.

AC_PATH_X

AC_CHECK_LIBM
AC_SUBST(LIBM)

AC_CHECK_FT2([9.0.3],[],
             [AC_MSG_ERROR([FreeType2 is required to compile this library])])

AC_PATH_XTRA

FTGL_CHECK_GL
FTGL_CHECK_GLUT
FTGL_CHECK_FONT

PKG_CHECK_MODULES(CPPUNIT, cppunit, [CPPUNIT="yes"], [CPPUNIT="no"])
AC_MSG_RESULT($CPPUNIT)
AM_CONDITIONAL(HAVE_CPPUNIT, test "x$CPPUNIT" != "xno")

dnl search the include directory (required for non-srcdir builds).
dnl should come after the system services checks otherwise headers
dnl may conflict.
CPPFLAGS="$CPPFLAGS -I\${top_srcdir}/src"

# Warning flags
BC_COMPILER_AND_LINKER_RECOGNIZES([-W -Wall -Wundef -Wfloat-equal  -Wpointer-arith -Wcast-align -Wcast-qual -Wshadow -Wsign-compare -Winline -Waggregate-return], [warning])
BC_COMPILER_AND_LINKER_RECOGNIZES([-Wstrict-prototypes -Wmissing-prototypes -Wnested-externs], [warncflags], [no])
if test "x$bc_warncflags_works" = "xyes" ; then
    CFLAGS="$CFLAGS -Wstrict-prototypes -Wmissing-prototypes -Wnested-externs"
fi

AC_ARG_ENABLE(doc,
  [  --enable-doc            build documentation (needs doxygen and LaTeX)])

# Build HTML documentatin?
DOXYGEN=no
if test "${enable_doc}" != "no"; then
  AC_PATH_PROG(DOXYGEN, doxygen, no)
fi
AM_CONDITIONAL(HAVE_DOXYGEN, test "x$DOXYGEN" != "xno")

# Build PDF documentation?
LATEX=no
if test "${enable_doc}" != "no"; then
  AC_PATH_PROG(LATEX, pdflatex, no)
  AC_PATH_PROG(KPSEWHICH, kpsewhich, no)
  AC_PATH_PROG(DVIPS, dvips, no)
  AC_PATH_PROG(CONVERT, convert, no)
  AC_PATH_PROG(EPSTOPDF, epstopdf, no)
  AC_PATH_PROGS(SED, gsed sed, no)
  if test "${DVIPS}" = "no" -o "${KPSEWHICH}" = "no" -o "${EPSTOPDF}" = "no" \
       -o "${CONVERT}" = "no" -o "${SED}" = "no"; then
    LATEX="no"
  fi
fi
if test "x${LATEX}" != "xno"; then
  AC_MSG_CHECKING(for a4.sty and a4wide.sty)
  if "${KPSEWHICH}" a4.sty >/dev/null 2>&1; then
    if "${KPSEWHICH}" a4wide.sty >/dev/null 2>&1; then
      AC_MSG_RESULT(yes)
    else
      LATEX="no"
      AC_MSG_RESULT(no)
    fi
  else
    LATEX="no"
    AC_MSG_RESULT(no)
  fi
fi
AM_CONDITIONAL(HAVE_LATEX, test "x${LATEX}" != "xno")

# make sure ECHO and ECHO_N got defined and substituted
if test "x$ECHO" = "x" ; then
    ECHO=echo
    AC_MSG_NOTICE([ECHO was not defined by configure so defining manually])
fi
AC_SUBST(ECHO)
AC_SUBST(ECHO_N)


AC_CONFIG_FILES([
  ftgl.pc
])
AC_CONFIG_FILES([
  Makefile
  demo/Makefile
  docs/Makefile
  docs/doxygen.cfg
  msvc/Makefile
  src/Makefile
  test/Makefile
])
AC_OUTPUT

dnl
dnl Expand the variables for summary reporting
dnl
prefix=`eval "echo $prefix"`
prefix=`eval "echo $prefix"`
bindir=`eval "echo $bindir"`
bindir=`eval "echo $bindir"`
sysconfdir=`eval "echo $sysconfdir"`
sysconfdir=`eval "echo $sysconfdir"`
mandir=`eval "echo $mandir"`
mandir=`eval "echo $mandir"`
datadir=`eval "echo $datadir"`
datadir=`eval "echo $datadir"`

AC_MSG_RESULT([Done.])
AC_MSG_RESULT([])
AC_MSG_RESULT([FTGL configured with the following settings:])
AC_MSG_RESULT([])
AC_MSG_RESULT([             Prefix: ${prefix}])
AC_MSG_RESULT([           Binaries: ${bindir}])
AC_MSG_RESULT([Configuration files: ${sysconfdir}])
AC_MSG_RESULT([         Data files: ${datadir}])
AC_MSG_RESULT([])
AC_MSG_RESULT([CC       = ${CC}])
AC_MSG_RESULT([CXX      = ${CXX}])
if test "x$CFLAGS" != "x" ; then
AC_MSG_RESULT([CFLAGS   = ${CFLAGS}])
fi
if test "x$CXXFLAGS" != "x" ; then
AC_MSG_RESULT([CXXFLAGS = ${CXXFLAGS}])
fi
if test "x$CPPFLAGS" != "x" ; then
AC_MSG_RESULT([CPPFLAGS = ${CPPFLAGS}])
fi
if test "x$LDFLAGS" != "x" ; then
AC_MSG_RESULT([LDFLAGS  = ${LDFLAGS}])
fi
if test "x$LIBS" != "x" ; then
AC_MSG_RESULT([LIBS     = ${LIBS}])
fi
AC_MSG_RESULT([])
AC_MSG_RESULT([---])
AC_MSG_RESULT([$0 complete, type 'make' to begin building])
AC_MSG_RESULT([])

# Local Variables:
# tab-width: 8
# mode: autoconf
# sh-indentation: 2
# sh-basic-offset: 2
# indent-tabs-mode: t
# End:
# ex: shiftwidth=2 tabstop=8
