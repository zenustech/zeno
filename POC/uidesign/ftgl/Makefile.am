
ACLOCAL_AMFLAGS = -I m4

SUBDIRS = src demo # test (does not build) # docs (overflows pdflatex)
DIST_SUBDIRS = $(SUBDIRS) msvc

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = ftgl.pc

DISTCLEANFILES = ftgl.pc
MAINTAINERCLEANFILES = Makefile.in aclocal.m4 configure config.h.in config.h.in~

EXTRA_DIST = \
	AUTHORS \
	BUGS \
	COPYING \
	ChangeLog \
	INSTALL \
	NEWS \
	README \
	TODO \
	autogen.sh \
	configure.ac \
	ftgl.pc.in \
	m4/compiler.m4 \
	m4/cxx.m4 \
	m4/font.m4 \
	m4/freetype2.m4 \
	m4/func.m4 \
	m4/gl.m4 \
	m4/glut.m4 \
	m4/pkg.m4 \
	$(NULL)

# Print out an informative summary.
all-local:
	@$(ECHO) "Done."
	@$(ECHO)
	@if test "x$(MAKECMDGOALS)" = "xall-am" -o "x$(.TARGETS)" = "xall-am" -o "x$(MAKECMDGOALS)" = "x" -o "x$(.TARGETS)" = "x" ; then \
	  $(ECHO) "---" ;\
	  $(ECHO) "Run 'make install' to begin installation into $(prefix)" ;\
	fi
	@$(ECHO)

maintainer-clean-local:
	-rm -rf .auto

# Upload documentation
DOC = docs/html docs/latex/ftgl.pdf
HOST = ftgl.sf.net
DIR = /home/groups/f/ft/ftgl/htdocs/
upload-doc:
	tar cz $(DOC) | ssh $(HOST) "cd $(DIR) && rm -Rf $(DOC) && tar xvz"

NULL =
