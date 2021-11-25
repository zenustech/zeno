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
#
#
# REVISIT:  this should really be generalized to support platforms
# other than Windows/MSVC++ and Linux/gcc.  However, the portability of 
# wide character I/O is very non-portable; so this is a nontrivial 
# prospect.  That's why this Makefile is provided only for gcc
# at the moment.

XLAT_CPP_OBJECTS = \
    Xlat.o \
    Xlat_CppSrc.o \
    Xlat_ICUResourceBundle.o \
    Xlat_MsgCatalog.o \
    Xlat_Win32RC.o


PLATFORM = LINUX
COMPILER = g++
GCC = yes
GXX = yes
CXXFLAGS =   -w -O  -D_GNU_SOURCE
CFLAGS =   -w -O  -D_GNU_SOURCE
LDFLAGS =    
EXTRA_LIBS =  
LIBRARY_SEARCH_PATHS=-L../../../lib/
LIBRARY_NAMES=-lxerces-c
INCLUDES=-I../../../include
CMP= -c ${CXXFLAGS}
CC= ${COMPILER} -c -D_REENTRANT -fpic
LINK = ${COMPILER} -fpic ${LDFLAGS}
EXTRA_LINK_OPTIONS=-lc ${EXTRA_LIBS}
APP_NAME=NLSXlat

all:: ${APP_NAME}

${APP_NAME}:: ${XLAT_CPP_OBJECTS}
	${LINK} ${XLAT_CPP_OBJECTS} -o ${APP_NAME} ${LIBRARY_SEARCH_PATHS} ${LIBRARY_NAMES} ${EXTRA_LINK_OPTIONS}

Xlat.o:: Xlat.cpp Xlat_ErrHandler.hpp Xlat_Formatter.hpp Xlat_Types.hpp
	${CC} ${INCLUDES} ${CMP} -o Xlat.o Xlat.cpp 
Xlat_CppSrc.o:: Xlat_CppSrc.cpp Xlat_CppSrc.hpp 
	${CC} ${INCLUDES} ${CMP} -o Xlat_CppSrc.o Xlat_CppSrc.cpp 
Xlat_ICUResourceBundle.o:: Xlat_ICUResourceBundle.cpp Xlat_ICUResourceBundle.hpp 
	${CC} ${INCLUDES} ${CMP} -o Xlat_ICUResourceBundle.o Xlat_ICUResourceBundle.cpp
Xlat_MsgCatalog.o:: Xlat_MsgCatalog.cpp Xlat_MsgCatalog.hpp 
	${CC} ${INCLUDES} ${CMP} -o Xlat_MsgCatalog.o Xlat_MsgCatalog.cpp
Xlat_Win32RC.o:: Xlat_Win32RC.cpp Xlat_Win32RC.hpp
	${CC} ${INCLUDES} ${CMP} -o Xlat_Win32RC.o Xlat_Win32RC.cpp

clean::
	rm -f ${XLAT_CPP_OBJECTS} ${APP_NAME}

