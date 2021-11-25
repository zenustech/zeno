dnl FTGL_CPP_FUNC()
dnl Check for __func__ or __FUNCTION__ and set __FUNC__ accordingly
dnl

AC_DEFUN([FTGL_CPP_FUNC], [dnl

AC_REQUIRE([AC_PROG_CC_STDC])

AC_CACHE_CHECK([for __func__], ac_cv_cpp_func,
    [AC_COMPILE_IFELSE([AC_LANG_PROGRAM([],
        [[char *function = __func__;]])],
        [ac_cv_cpp_func=yes],
        [AC_COMPILE_IFELSE([AC_LANG_PROGRAM([],
            [[char *function = __FUNCTION__;]])],
            [ac_cv_cpp_func=__FUNCTION__],
            [ac_cv_cpp_func=no]
        )]
    )]
)
             
if test $ac_cv_cpp_func = yes; then
    AC_DEFINE(__FUNC__, __func__, [Define to __FUNCTION__ or  "" if __func__ is not available.])
elif test $ac_cv_cpp_func = __FUNCTION__; then
    AC_DEFINE(__FUNC__, __FUNCTION__, [Define to __FUNCTION__ or "" if __func__ is not available.])
elif test $ac_cv_cpp_func = no; then
    AC_DEFINE(__FUNC__, "", [Define to __FUNCTION__ or "" if __func__ is not available.])
fi
])
