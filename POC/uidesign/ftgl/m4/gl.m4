dnl FTGL_CHECK_GL()
dnl Check for OpenGL development environment and GLU >= 1.2
dnl
AC_DEFUN([FTGL_CHECK_GL],
[dnl
AC_REQUIRE([AC_PROG_CC])
AC_REQUIRE([AC_PATH_X])
AC_REQUIRE([AC_PATH_XTRA])

AC_ARG_WITH([--with-gl-inc],
    AC_HELP_STRING([--with-gl-inc=DIR],[Directory where GL/gl.h is installed]))
AC_ARG_WITH([--with-gl-lib],
    AC_HELP_STRING([--with-gl-lib=DIR],[Directory where OpenGL libraries are installed]))
AC_ARG_WITH([--with-glu-lib],
    AC_HELP_STRING([--with-glu-lib=DIR],[Directory where OpenGL GLU library is installed]))

AC_LANG_SAVE
AC_LANG_C

GL_SAVE_CPPFLAGS="$CPPFLAGS"
GL_SAVE_LIBS="$LIBS"

if test "x$no_x" != xyes ; then
    GL_CFLAGS="$X_CFLAGS"
    GL_X_LIBS="$X_PRE_LIBS $X_LIBS -lX11 -lXext -lXmu $X_EXTRA_LIBS"
fi

if test "x$with_gl_inc" != "xnone" ; then
    if test -d "$with_gl_inc" ; then
        GL_CFLAGS="-I$with_gl_inc"
    else
        GL_CFLAGS="$with_gl_inc"
    fi
else
    GL_CFLAGS=
fi

CPPFLAGS="$GL_CFLAGS"
AC_CHECK_HEADER([GL/gl.h], [AC_DEFINE([HAVE_GL_GL_H], 1, [Define to 1 if you have the <GL/gl.h header])], [
	AC_CHECK_HEADER([OpenGL/gl.h], [AC_DEFINE([HAVE_OPENGL_GL_H], 1, [Define to 1 if you have the <OpenGL/gl.h header])], [
		AC_MSG_ERROR([GL/gl.h or OpenGL/gl.h is needed, please specify its location with --with-gl-inc.  If this still fails, please contact henryj@paradise.net.nz, include the string FTGL somewhere in the subject line and provide a copy of the config.log file that was left behind.])
	])
])

dnl check whether the OpenGL framework is available
AC_MSG_CHECKING([for OpenGL framework (Darwin-specific)])
FRAMEWORK_OPENGL=""
PRELIBS="$LIBS"
LIBS="$LIBS -Xlinker -framework -Xlinker OpenGL"
# -Xlinker is used because libtool is busted prior to 1.6 wrt frameworks
AC_TRY_LINK([#include <OpenGL/gl.h>], [glBegin(GL_POINTS)],
    [GL_DYLIB="/System/Library/Frameworks/OpenGL.framework/Versions/A/Libraries/libGL.dylib"
     FRAMEWORK_OPENGL="-Xlinker -framework -Xlinker OpenGL -dylib_file $GL_DYLIB: $GL_DYLIB"
     ac_cv_search_glBegin="$FRAMEWORK_OPENGL"
     AC_MSG_RESULT(yes)],
    [AC_MSG_RESULT(no)])
if test x"$FRAMEWORK_OPENGL" != "x"; then
  with_gl_lib="$FRAMEWORK_OPENGL"
fi
AC_SUBST(FRAMEWORK_OPENGL)
LIBS="$PRELIBS"

AC_MSG_CHECKING([for GL library])
echo host=$host
case "x${host}" in
  x*-mingw32) GL_GL_LIBS="-lopengl32"
              GL_GLU_LIBS="-lglu32"
              ;;
  x*) GL_GL_LIBS="-lGL"
      GL_GLU_LIBS="-lGLU"
      ;;
esac

if test "x$with_gl_lib" != "x" ; then
    if test -d "$with_gl_lib" ; then
        LIBS="-L$with_gl_lib $GL_GL_LIBS"
    else
        LIBS="$with_gl_lib"
    fi
else
    LIBS="$GL_GL_LIBS"
fi
AC_LINK_IFELSE([AC_LANG_CALL([],[glBegin])],[HAVE_GL=yes],[
dnl This is done here so that we can check for the Win32 version of the
dnl GL library, which may not use cdecl calling convention.
 AC_TRY_LINK([#include <GL/gl.h>],[glBegin(GL_POINTS)],[HAVE_GL=yes],[HAVE_GL=no])]
)

if test "x$HAVE_GL" = xno ; then
    if test "x$GL_X_LIBS" != x ; then
        LIBS="$GL_GL_LIBS $GL_X_LIBS"
        AC_LINK_IFELSE([AC_LANG_CALL([],[glBegin])],[HAVE_GL=yes], [HAVE_GL=no])
    fi
fi
if test "x$HAVE_GL" = xyes ; then
    AC_MSG_RESULT([yes])
    GL_LIBS=$LIBS
else
    AC_MSG_RESULT([no])
    AC_MSG_ERROR([GL library could not be found, please specify its location with --with-gl-lib.  If this still fails, please contact henryj@paradise.net.nz, include the string FTGL somewhere in the subject line and provide a copy of the config.log file that was left behind.])
fi

AC_CHECK_HEADER([GL/glu.h], [AC_DEFINE([HAVE_GL_GLU_H], 1, [Define to 1 if you have the <GL/glu.h header])], [
	AC_CHECK_HEADER([OpenGL/glu.h], [AC_DEFINE([HAVE_OPENGL_GLU_H], 1, [Define to 1 if you have the <OpenGL/glu.h header])], [
		AC_MSG_ERROR([GL/glu.h or OpenGL/glu.h is needed, please specify its location with --with-gl-inc.  If this still fails, please contact henryj@paradise.net.nz, include the string FTGL somewhere in the subject line and provide a copy of the config.log file that was left behind.])
	])
])
AC_MSG_CHECKING([for GLU version >= 1.2])
AC_TRY_COMPILE([
#ifdef HAVE_GL_GLU_H
#  include <GL/glu.h>
#endif
#ifdef HAVE_OPENGL_GLU_H
#  include <OpenGL/glu.h>
#endif
], [
#if !defined(GLU_VERSION_1_2)
#error GLU too old
#endif
               ],
               [AC_MSG_RESULT([yes])],
               [AC_MSG_RESULT([no])
                AC_MSG_ERROR([GLU >= 1.2 is needed to compile this library])
               ])

if test "x$FRAMEWORK_OPENGL" = "x" ; then

AC_MSG_CHECKING([for GLU library])
if test "x$with_glu_lib" != "x" ; then
    if test -d "$with_glu_lib" ; then
        LIBS="$GL_LIBS -L$with_glu_lib $GL_GLU_LIBS"
    else
        LIBS="$GL_LIBS $with_glu_lib"
    fi
else
    LIBS="$GL_LIBS $GL_GLU_LIBS"
fi
AC_LINK_IFELSE([AC_LANG_CALL([],[gluNewTess])],[HAVE_GLU=yes], [
dnl This is done here so that we can check for the Win32 version of the
dnl GL library, which may not use cdecl calling convention.
 AC_TRY_LINK([#include <GL/glu.h>],[gluNewTess()],[HAVE_GLU=yes],[HAVE_GLU=no])]
)
if test "x$HAVE_GLU" = xno ; then
    if test "x$GL_X_LIBS" != x ; then
        LIBS="$GL_GLU_LIBS $GL_LIBS $GL_X_LIBS"
        AC_LINK_IFELSE([AC_LANG_CALL([],[gluNewTess])],[HAVE_GLU=yes], [HAVE_GLU=no])
    fi
fi
if test "x$HAVE_GLU" = xyes ; then
    AC_MSG_RESULT([yes])
    GL_LIBS="$LIBS"
else
    AC_MSG_RESULT([no])
    AC_MSG_ERROR([GLU library could not be found, please specify its location with --with-gl-lib.  If this still fails, please contact henryj@paradise.net.nz, include the string FTGL somewhere in the subject line and provide a copy of the config.log file that was left behind.])
fi

fi

AC_SUBST(GL_CFLAGS)
AC_SUBST(GL_LIBS)

CPPFLAGS="$GL_SAVE_CPPFLAGS"
LIBS="$GL_SAVE_LIBS"
AC_LANG_RESTORE
GL_X_LIBS=""
])
