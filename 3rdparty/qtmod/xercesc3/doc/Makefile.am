#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

DOCS_CLASSPATH=$(abs_top_srcdir)/tools/jars/stylebook-1.0-b2.jar:$(abs_top_srcdir)/tools/jars/xalan.jar:$(abs_top_srcdir)/tools/jars/xerces.jar
DOCS_SRCSTYLE=$(abs_top_srcdir)/doc/style
DOCS_BUILDSTYLE=$(abs_top_builddir)/doc/style
DOCS_SOURCE=$(abs_top_srcdir)/doc/xerces-c_book.xml
DOCS_TARGET=$(abs_top_builddir)/doc/html

createapidocs:
	doxygen Doxyfile

# stylebook doesn't like the same style directory being specified
# twice for in-source builds.
createdocs:
	$(MKDIR_P) "$(DOCS_TARGET)"; \
	if [ "$(abs_top_srcdir)" != "$(abs_top_builddir)" ]; then \
	  for file in $(STYLE_FILES); do \
            $(INSTALL) -d $(builddir)/$$(dirname $$file); \
	    $(INSTALL_DATA) $(srcdir)/$$file $(builddir)/$$file; \
	  done; \
	fi; \
	java -Djava.awt.headless=true -classpath "$(DOCS_CLASSPATH)" \
	  org.apache.stylebook.StyleBook \
	  "targetDirectory=$(DOCS_TARGET)" \
	  "$(DOCS_SOURCE)" \
	  "$(DOCS_BUILDSTYLE)"

STYLE_FILES = \
             style/dtd/blocks.ent \
             style/dtd/book.dtd \
             style/dtd/changes.dtd \
             style/dtd/characters.ent \
             style/dtd/document.dtd \
             style/dtd/entities.ent.in \
             style/dtd/faqs.dtd \
             style/dtd/links.ent \
             style/dtd/markup.ent \
             style/graphics/button-a.gif \
             style/graphics/button-b.gif \
             style/graphics/section.gif \
             style/loader.xml \
             style/resources/bottom.gif \
             style/resources/button-asf-hi.gif \
             style/resources/button-asf-lo.gif \
             style/resources/button-w3c-hi.gif \
             style/resources/button-w3c-lo.gif \
             style/resources/button-xml-hi.gif \
             style/resources/button-xml-lo.gif \
             style/resources/close.gif \
             style/resources/dot.gif \
             style/resources/join.gif \
             style/resources/line.gif \
             style/resources/logo.gif \
             style/resources/note.gif \
             style/resources/right.gif \
             style/resources/script.js \
             style/resources/separator-space.gif \
             style/resources/separator.gif \
             style/resources/void.gif \
             style/stylesheets/any2header.xsl \
             style/stylesheets/any2project.xsl \
             style/stylesheets/book2group.xsl \
             style/stylesheets/book2project.xsl \
             style/stylesheets/changes2document.xsl \
             style/stylesheets/context2footer.xsl \
             style/stylesheets/context2label.xsl \
             style/stylesheets/context2section.xsl \
             style/stylesheets/directory2project.xsl \
             style/stylesheets/document2html.xsl \
             style/stylesheets/faqs2document.xsl \
             style/stylesheets/group2document.xsl

EXTRA_DIST = CMakeLists.txt \
             Doxyfile.in \
             Makefile.am \
             apidocs.xml \
             applications.xml \
             bug-report.xml \
             build.xml \
             charter.xml \
             createdoc.xml \
             domcount.xml \
             domprint.xml \
             download.xml \
             enumval.xml \
             faq-build.xml \
             faq-contributing.xml \
             faq-distrib.xml \
             faq-other.xml \
             faq-parse.xml \
             feedback.xml \
             html/ApacheDOMC++Binding.html \
             html/secadv/CVE-2015-0252.txt \
             html/secadv/CVE-2016-0729.txt \
             html/secadv/CVE-2016-4463.txt \
             install.xml \
             mailing-lists.xml \
             memparse.xml \
             migration.xml \
             migration_archive.xml \
             pparse.xml \
             program-dom.xml \
             program-others.xml \
             program-sax.xml \
             program-sax2.xml \
             program.xml \
             psviwriter.xml \
             readme.xml \
             redirect.xml \
             releases.xml \
             releases_archive.xml \
             releases_plan.xml \
             samples.xml \
             sax2count.xml \
             sax2print.xml \
             saxcount.xml \
             saxprint.xml \
             schema.xml \
             scmprint.xml \
             secadv.xml \
             senumval.xml \
             source-repository.xml \
             stdinparse.xml \
             xerces-c_book.xml \
             xinclude.xml \
             $(STYLE_FILES)

dist-hook: createapidocs createdocs
	$(MKDIR_P) "$(distdir)/html"; \
	cp -r html/* "$(distdir)/html"

# Clean up copied style files and generated documentation
clean-local:
	if [ "$(abs_top_srcdir)" != "$(abs_top_builddir)" ]; then \
	  for file in $(STYLE_FILES); do rm -f $(builddir)/$$file; done; \
          rm -rf html; \
	fi

.PHONY: createapidocs createdocs
