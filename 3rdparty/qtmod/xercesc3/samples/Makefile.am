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

#
# $Id$
#

bin_PROGRAMS =                      ${sampleprogs}

LDADD =                             ${top_builddir}/src/libxerces-c.la
# the  -I${top_builddir}/src is needed to pick the xercesc/util/Xerces_autoconf_config.hpp 
# header file when doing out-of-tree builds
AM_CPPFLAGS =                       -I${top_builddir}/src -I${top_srcdir}/src

EXTRA_DIST =                        CMakeLists.txt \
                                    data/personal.dtd \
                                    data/personal.xml \
                                    data/personal.xsd \
                                    data/personal-schema.xml \
                                    data/redirect.dtd \
                                    data/long.xml \
                                    expected \
                                    scripts

clean-local:
	-rm -rf observed

sampleprogs =

sampleprogs +=                      CreateDOMDocument
CreateDOMDocument_SOURCES =	        src/CreateDOMDocument/CreateDOMDocument.cpp

sampleprogs +=                      DOMCount
DOMCount_SOURCES =                  src/DOMCount/DOMCount.cpp \
                                    src/DOMCount/DOMCount.hpp

sampleprogs +=                      DOMPrint
DOMPrint_SOURCES =                  src/DOMPrint/DOMPrint.cpp \
                                    src/DOMPrint/DOMPrintErrorHandler.cpp \
                                    src/DOMPrint/DOMPrintErrorHandler.hpp \
                                    src/DOMPrint/DOMPrintFilter.cpp \
                                    src/DOMPrint/DOMPrintFilter.hpp \
                                    src/DOMPrint/DOMTreeErrorReporter.cpp \
                                    src/DOMPrint/DOMTreeErrorReporter.hpp

sampleprogs +=                      EnumVal
EnumVal_SOURCES =                   src/EnumVal/EnumVal.cpp

sampleprogs +=                      MemParse
MemParse_SOURCES =                  src/MemParse/MemParse.cpp \
                                    src/MemParse/MemParse.hpp \
                                    src/MemParse/MemParseHandlers.cpp \
                                    src/MemParse/MemParseHandlers.hpp

sampleprogs +=                      PParse
PParse_SOURCES =                    src/PParse/PParse.cpp \
                                    src/PParse/PParse.hpp \
                                    src/PParse/PParseHandlers.cpp \
                                    src/PParse/PParseHandlers.hpp

sampleprogs +=                      PSVIWriter
PSVIWriter_SOURCES =                src/PSVIWriter/PSVIWriter.cpp \
                                    src/PSVIWriter/PSVIWriter.hpp \
                                    src/PSVIWriter/PSVIWriterHandlers.cpp \
                                    src/PSVIWriter/PSVIWriterHandlers.hpp

sampleprogs +=                      Redirect
Redirect_SOURCES =                  src/Redirect/Redirect.cpp \
                                    src/Redirect/Redirect.hpp \
                                    src/Redirect/RedirectHandlers.cpp \
                                    src/Redirect/RedirectHandlers.hpp

sampleprogs +=                      SAX2Count
SAX2Count_SOURCES =                 src/SAX2Count/SAX2Count.cpp \
                                    src/SAX2Count/SAX2Count.hpp \
                                    src/SAX2Count/SAX2CountHandlers.cpp \
                                    src/SAX2Count/SAX2CountHandlers.hpp

sampleprogs +=                      SAX2Print
SAX2Print_SOURCES =                 src/SAX2Print/SAX2FilterHandlers.cpp \
                                    src/SAX2Print/SAX2FilterHandlers.hpp \
                                    src/SAX2Print/SAX2Print.cpp \
                                    src/SAX2Print/SAX2Print.hpp \
                                    src/SAX2Print/SAX2PrintHandlers.cpp \
                                    src/SAX2Print/SAX2PrintHandlers.hpp

sampleprogs +=                      SAXCount
SAXCount_SOURCES =                  src/SAXCount/SAXCount.cpp \
                                    src/SAXCount/SAXCount.hpp \
                                    src/SAXCount/SAXCountHandlers.cpp \
                                    src/SAXCount/SAXCountHandlers.hpp

sampleprogs +=                      SAXPrint
SAXPrint_SOURCES =                  src/SAXPrint/SAXPrint.cpp \
                                    src/SAXPrint/SAXPrint.hpp \
                                    src/SAXPrint/SAXPrintHandlers.cpp \
                                    src/SAXPrint/SAXPrintHandlers.hpp

sampleprogs +=                      SCMPrint
SCMPrint_SOURCES =                  src/SCMPrint/SCMPrint.cpp

sampleprogs +=                      SEnumVal
SEnumVal_SOURCES =                  src/SEnumVal/SEnumVal.cpp

sampleprogs +=                      StdInParse
StdInParse_SOURCES =                src/StdInParse/StdInParse.cpp \
                                    src/StdInParse/StdInParse.hpp \
                                    src/StdInParse/StdInParseHandlers.cpp \
                                    src/StdInParse/StdInParseHandlers.hpp

sampleprogs +=                      XInclude
XInclude_SOURCES =                  src/XInclude/XInclude.cpp \
                                    src/XInclude/XInclude.hpp


TESTS =					scripts/SAXCount \
					scripts/SAXCount1 \
					scripts/SAXCount2 \
					scripts/SAXCount3 \
					scripts/SAXPrint \
					scripts/SAXPrint1 \
					scripts/SAXPrint2 \
					scripts/SAXPrint3 \
					scripts/SAX2Count \
					scripts/SAX2Count1 \
					scripts/SAX2Count2 \
					scripts/SAX2Count3 \
					scripts/SAX2Print \
					scripts/SAX2Print1 \
					scripts/SAX2Print2 \
					scripts/SAX2Print3 \
					scripts/SAX2Print4 \
					scripts/SAX2Print5 \
					scripts/MemParse \
					scripts/MemParse1 \
					scripts/Redirect \
					scripts/Redirect1 \
					scripts/DOMCount \
					scripts/DOMCount1 \
					scripts/DOMCount2 \
					scripts/DOMCount3 \
					scripts/DOMPrint \
					scripts/DOMPrint1 \
					scripts/DOMPrint2 \
					scripts/DOMPrint3 \
					scripts/DOMPrint4 \
					scripts/DOMPrint5 \
					scripts/StdInParse1 \
					scripts/StdInParse2 \
					scripts/StdInParse3 \
					scripts/PParse \
					scripts/PParse1 \
					scripts/PParse2 \
					scripts/EnumVal \
					scripts/EnumVal1 \
					scripts/SEnumVal \
					scripts/SEnumVal1 \
					scripts/CreateDOMDocument \
					scripts/SAXPrintLong

XFAIL_TESTS =				scripts/SAXCount \
					scripts/SAXPrint \
					scripts/SAX2Count \
					scripts/SAX2Print \
					scripts/Redirect \
					scripts/DOMCount \
					scripts/DOMPrint \
					scripts/PParse \
					scripts/EnumVal \
					scripts/SEnumVal
