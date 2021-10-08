
documentationdir = $(datadir)/doc/ftgl
documentation_DATA = projects_using_ftgl.txt

if HAVE_DOXYGEN
htmldocdir = $(documentationdir)/html
htmldoc_DATA = html/doxygen.css
if HAVE_LATEX
pdfdocdir = $(documentationdir)
pdfdoc_DATA = latex/ftgl.pdf
endif
endif

PNGS = \
    images/logo.png \
    images/metrics.png \
    images/rasterfont.png \
    images/vectorfont.png \
    images/texturefont.png \
    $(NULL)

stamp-eps: $(PNGS)
if HAVE_LATEX
	for i in $^; do convert $$i $$i.eps; done
endif
	touch $@

html/doxygen.css: stamp-doxygen
stamp-doxygen: doxygen.cfg stamp-eps
	$(DOXYGEN) $^
	for file in html/*html; do \
	  $(SED) -e 's/%FTGL/FTGL/' < $$file > $$file.tmp; \
	  mv $$file.tmp $$file; \
	done
	touch $@

latex/ftgl.pdf: stamp-latex
stamp-latex: stamp-doxygen
	rm -f latex/ftgl.tex latex/ftgl.pdf
	mv latex/refman.tex latex/ftgl.tex
	$(SED) -e 's/setlength{/renewcommand{/' < latex/ftgl.tex > latex/refman.tex
	cd latex && $(MAKE) $(AM_CFLAGS) refman.pdf || (cat refman.log; exit 1)
	mv latex/refman.pdf latex/ftgl.pdf
	touch stamp-latex

clean: clean-local
clean-local:
	$(RM) -rf html latex
	$(RM) -f images/*.eps
	$(RM) -f stamp-doxygen stamp-latex stamp-eps

if HAVE_DOXYGEN
install-data-local: html/doxygen.css
	$(mkinstalldirs) $(DESTDIR)$(htmldocdir)/
	$(INSTALL) -m 0644 \
	  `find html -name '*.html' -o -name '*.gif' -o -name '*.png' -o -name '*.jpg'` \
	  $(DESTDIR)$(htmldocdir)/
endif

EXTRA_DIST = \
	$(PNGS) \
	$(documentation_DATA) \
	FTGL_1_3.gif \
	doxygen.cfg.in \
	ftgl.dox \
	tutorial.dox \
	projects_using_ftgl.txt \
	faq.dox \
	images/metrics.svg \
	$(NULL)

MAINTAINERCLEANFILES = Makefile.in

NULL =
