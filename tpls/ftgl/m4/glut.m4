dnl FTGL_CHECK_GLUT()
dnl Check for GLUT development environment
dnl
AC_DEFUN([FTGL_CHECK_GLUT],
[dnl
AC_REQUIRE([AC_PROG_CC])dnl
AC_REQUIRE([AC_PATH_X])dnl
AC_REQUIRE([AC_PATH_XTRA])dnl
AC_REQUIRE([FTGL_CHECK_GL])dnl

AC_ARG_WITH([--with-glut-inc],
    AC_HELP_STRING([--with-glut-inc=DIR],[Directory where GL/glut.h is installed (optional)]))
AC_ARG_WITH([--with-glut-lib],
    AC_HELP_STRING([--with-glut-lib=DIR],[Directory where GLUT libraries are installed (optional)]))

AC_LANG_SAVE
AC_LANG_C

GLUT_SAVE_CPPFLAGS="$CPPFLAGS"
GLUT_SAVE_LIBS="$LIBS"

if test "$no_x" != "yes"; then
    GLUT_CFLAGS="$X_CFLAGS"
    GLUT_X_LIBS="$X_PRE_LIBS $X_LIBS -lX11 -lXext -lXmu $X_EXTRA_LIBS"
fi

if test "$with_glut_inc" != "none"; then
    if test -d "$with_glut_inc"; then
        GLUT_CFLAGS="-I$with_glut_inc"
    else
        GLUT_CFLAGS="$with_glut_inc"
    fi
else
    GLUT_CFLAGS=""
fi

# Check for GLUT headers
CPPFLAGS="$GLUT_CFLAGS"
AC_CHECK_HEADERS([GL/glut.h], [ac_cv_have_glut=yes],
  [AC_CHECK_HEADERS([GLUT/glut.h], [ac_cv_have_glut=yes],
    [ac_cv_have_glut=no])])

# Check for GLUT libraries
if test "$ac_cv_have_glut" = "yes"; then
    AC_MSG_CHECKING([for GLUT library])
    if test "$with_glut_lib" != ""; then
        if test -d "$with_glut_lib"; then
            LIBS="-L$with_glut_lib -lglut"
        else
            LIBS="$with_glut_lib"
        fi
    else
        LIBS="-lglut"
    fi

    AC_LINK_IFELSE(
        [AC_LANG_CALL([],[glutInit])],
        [ac_cv_have_glut=yes],
        [ac_cv_have_glut=no])
    if test "$ac_cv_have_glut" = "no"; then
        # Try again with the GL libs
        LIBS="-lglut $GL_LIBS"
        AC_LINK_IFELSE(
            [AC_LANG_CALL([],[glutInit])],
            [ac_cv_have_glut=yes],
            [ac_cv_have_glut=no])
    fi

    if test "$ac_cv_have_glut" = "no" && test "$GLUT_X_LIBS" != ""; then
        # Try again with the GL and X11 libs
        LIBS="-lglut $GL_LIBS $GLUT_X_LIBS"
        AC_LINK_IFELSE(
            [AC_LANG_CALL([],[glutInit])],
            [ac_cv_have_glut=yes],
            [ac_cv_have_glut=no])
    fi

    if test "$ac_cv_have_glut" = "no"; then
	# Try again with GLUT framework
	LIBS="-Xlinker -framework -Xlinker OpenGL -Xlinker -framework -Xlinker GLUT"
        AC_LINK_IFELSE(
            [AC_LANG_CALL([],[glutInit])],
            [ac_cv_have_glut=yes],
            [ac_cv_have_glut=no])
    fi

    if test "$ac_cv_have_glut" = "yes"; then
        AC_MSG_RESULT([yes])
        GLUT_LIBS="$LIBS"
    else
        AC_MSG_RESULT([no])
    fi
fi

if test "$ac_cv_have_glut" = "no"; then
    AC_MSG_WARN([GLUT headers not available, example program won't be compiled.])
fi

AM_CONDITIONAL(HAVE_GLUT, [test "$ac_cv_have_glut" = "yes"])

AC_SUBST(GLUT_CFLAGS)
AC_SUBST(GLUT_LIBS)
AC_LANG_RESTORE

CPPFLAGS="$GLUT_SAVE_CPPFLAGS"
LIBS="$GLUT_SAVE_LIBS"
GLUT_X_CFLAGS=
GLUT_X_LIBS=
])

