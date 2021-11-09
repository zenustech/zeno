SUBDIRS = doc src tests samples

EXTRA_DIST = CMakeLists.txt \
             cmake \
             config.h.cmake.in \
             xerces-c.spec \
             xerces-c.pc.in \
             CREDITS \
             INSTALL \
             KEYS \
             LICENSE \
             NOTICE \
             README \
             config/pretty-make \
             reconf \
             scripts/createBindingsFormatFromHPP.pl \
             scripts/makeStringDefinition.pl \
             tools

AUTOMAKE_OPTIONS = foreign dist-bzip2 dist-zip dist-xz
ACLOCAL_AMFLAGS = -I m4

pkgconfigdir = @pkgconfigdir@
pkgconfig_DATA= xerces-c.pc
