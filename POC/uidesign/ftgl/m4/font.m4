dnl FTGL_CHECK_FONT()
dnl Look for a TrueType font somewhere on the system. If no font is found,
dnl no big deal, example programs will just require one in the command line.
dnl This finds DejaVu, Bitstream and Microsoft fonts on Debian, Ubuntu, Gentoo,
dnl Fedora, Mandriva, Slackware and OS X systems.
dnl Also, we prefer serif fonts because they have elegant curves that render
dnl well in OpenGL.
dnl
AC_DEFUN([FTGL_CHECK_FONT],
[dnl
AC_MSG_CHECKING(for a TrueType font on the system)

dnl  First try: fontconfig
FONT_FILE="`fc-match -sv serif 2>/dev/null| sed -ne 's/.*\file:@<:@^"@:>@*"\(@<:@^"@:>@*\)".*/\1/p' | sed q`"

dnl  Second try: look into known paths
if test "$FONT_FILE" = ""; then
    for font in \
      DejaVuSerif.ttf VeraSe.ttf DejaVuSans.ttf Vera.ttf \
      times.ttf Times.ttf arial.ttf Arial.ttf; do
        for dir in \
          /usr/share/fonts \
          /usr/share/fonts/truetype \
          /usr/share/fonts/truetype/ttf-dejavu \
          /usr/share/fonts/truetype/ttf-bitstream-vera \
          /usr/share/fonts/TTF \
          /usr/share/fonts/TTF/dejavu \
          /usr/share/fonts/dejavu \
          /usr/share/fonts/ttf-dejavu \
          /usr/share/fonts/ttf-bitstream-vera \
          /usr/X11R6/lib/X11/fonts \
          /usr/X11R6/lib/X11/fonts/TTF; do
            if test -f "$dir/$font"; then FONT_FILE="$dir/$font"; break; fi
        done
        if test "$FONT_FILE" != no; then break; fi
    done
fi

if test "$FONT_FILE" != ""; then
    AC_DEFINE_UNQUOTED(FONT_FILE, "$FONT_FILE", [Define to the path to a TrueType font])
fi
AC_MSG_RESULT($FONT_FILE)
])
