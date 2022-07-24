# $Id$
# Locate SFML library
# This module defines
# SFML_FOUND, if false, do not try to link to SFML
# SFML_LIBRARY, the name of the librarys to link against
# SFML_INCLUDE_DIR, where to find SFML headers
#
# By default this script will link to the shared-nondebug version of SFML
# You can change this by define this variables befor calling FIND_PACKAGE
#
# SFML_DEBUG - If defined it will link to the debug version of SFML
# SFML_STATIC - If defined it will link to the static version of SFML
#
# For example:
# SET(SFML_STATIC true)
# FIND_PACKAGE(SFML REQUIRED COMPONENTS System Window)
#
# Created by Nils Hasenbanck. Based on the FindSDL_*.cmake modules,
# created by Eric Wing, which were influenced by the FindSDL.cmake
# module, but with modifications to recognize OS X frameworks and
# additional Unix paths (FreeBSD, etc).
#
# Changelog:
# 2010-04-04 - Add support for visual studio 2008 (9.0)
# 2010-04-09 - Add support for visual studio 2005 (8.0)
#            - Now the test for the requested components is also implemented.
#            - It also will only link to the requested components
#            - You can chose wich debug/nondebug static/shared versions of the librarys you want to link to

SET(SFML_LIBRARY "")
SET(SFML_INCLUDE_DIR "")

SET( SFMLDIR $ENV{SFMLDIR} )
IF(WIN32 AND NOT(CYGWIN))
   # Convert backslashes to slashes
    STRING(REGEX REPLACE "\\\\" "/" SFMLDIR "${SFMLDIR}")
ENDIF(WIN32 AND NOT(CYGWIN))

SET(SFML_COMPONENTS
    System
    Audio
    Graphics
    Network
    Window
)

SET(SFML_MODE
   _SHARED_NONDEBUG
   _SHARED_DEBUG
   _STATIC_NONDEBUG
   _STATIC_DEBUG
)

SET(SFML_INCLUDE_SEARCH_DIR
    ~/Library/Frameworks
    /Library/Frameworks
    /usr/local/include/SFML
    /usr/include/SFML
    /usr/local/include
    /usr/include
    /sw/include/SFML # Fink
    /sw/include
    /opt/local/include/SFML # DarwinPorts
    /opt/local/include
    /opt/csw/include/SFML # Blastwave
    /opt/csw/include
    /opt/include/SFML
    /opt/include
    ${SFMLDIR}
    ${SFMLDIR}/include
)

SET(SFML_LIBRARY_SEARCH_DIR
    ~/Library/Frameworks
    /Library/Frameworks
    /usr/local
    /usr
    /sw
    /opt/local
    /opt/csw
    /opt
    ${SFMLDIR}
    ${SFMLDIR}/lib/vc2008
    ${SFMLDIR}/lib/vc2005
)

#looking for the include files
FOREACH(COMPONENT ${SFML_COMPONENTS})
   STRING(TOUPPER ${COMPONENT} UPPERCOMPONENT)
   STRING(TOLOWER ${COMPONENT} LOWERCOMPONENT)
      
   FIND_PATH(SFML_${UPPERCOMPONENT}_INCLUDE_DIR
      ${COMPONENT}.hpp
      PATH_SUFFIXES include SFML
      PATHS ${SFML_INCLUDE_SEARCH_DIR}
   )
   
   IF(SFML_${UPPERCOMPONENT}_INCLUDE_DIR)
      IF(WIN32)
         # In wxWIN we need the root include directory without the "/SFML" at the end... so we have to remove it.
         # This is a oversized "remove 5 chars at the right end of the string" function:
         string(LENGTH ${SFML_${UPPERCOMPONENT}_INCLUDE_DIR} STRING_SIZE)
         math(EXPR STRING_SIZE ${STRING_SIZE}-5)
         string(SUBSTRING "${SFML_${UPPERCOMPONENT}_INCLUDE_DIR}" 0 ${STRING_SIZE} SFML_${UPPERCOMPONENT}_INCLUDE_DIR)   
      ENDIF(WIN32)
      
      LIST(APPEND SFML_INCLUDE_DIR ${SFML_${UPPERCOMPONENT}_INCLUDE_DIR})
      LIST(REMOVE_DUPLICATES SFML_INCLUDE_DIR)
   ENDIF(SFML_${UPPERCOMPONENT}_INCLUDE_DIR)
ENDFOREACH(COMPONENT)

#looking for the librarys
FOREACH(MODE ${SFML_MODE})
   string(COMPARE EQUAL ${MODE} "_SHARED_NONDEBUG" string_equal_result)
   IF(string_equal_result)
      SET(_STA "")
      SET(_DBG "")
   ENDIF(string_equal_result)
   
   string(COMPARE EQUAL ${MODE} "_SHARED_DEBUG" string_equal_result)
   IF(string_equal_result)
      SET(_STA "")
      SET(_DBG "-d")
   ENDIF(string_equal_result)

   string(COMPARE EQUAL ${MODE} "_STATIC_NONDEBUG" string_equal_result)
   IF(string_equal_result)
      SET(_STA "-s")
      SET(_DBG "")
   ENDIF(string_equal_result)
   
   string(COMPARE EQUAL ${MODE} "_STATIC_DEBUG" string_equal_result)
   IF(string_equal_result)
      SET(_STA "-s")
      SET(_DBG "-d")
   ENDIF(string_equal_result)
   
   FOREACH(COMPONENT ${SFML_COMPONENTS})      
      STRING(TOUPPER ${COMPONENT} UPPERCOMPONENT)
      STRING(TOLOWER ${COMPONENT} LOWERCOMPONENT)
      FIND_LIBRARY(SFML_${UPPERCOMPONENT}_LIBRARY${MODE}
         NAMES sfml-${LOWERCOMPONENT}${_STA}${_DBG}
         PATH_SUFFIXES lib64 lib
         PATHS ${SFML_LIBRARY_SEARCH_DIR}
      )
   ENDFOREACH(COMPONENT)

   IF(WIN32)
      #Now we are looking for "sfml-main.lib".
      #Because we need it if we give ADD_EXECUTABLE the WIN32 switch to creat a GUI application (that one without a cmd promt)
      FIND_LIBRARY( SFML_MAIN_LIBRARY${MODE}
         NAMES sfml-main${_DBG}
         PATH_SUFFIXES lib64 lib
         PATHS ${SFML_LIBRARY_SEARCH_DIR}
      )
   ENDIF(WIN32)
ENDFOREACH(MODE)


#Test if we have the include directory, the system lib and other needed components
#We also fill SFML_LIBRARY here with all the files we like to link to

IF(NOT(SFML_DEBUG) AND NOT(SFML_STATIC))
   SET(MODE_LABEL "_SHARED_NONDEBUG")
ENDIF(NOT(SFML_DEBUG) AND NOT(SFML_STATIC))

IF(SFML_DEBUG AND NOT(SFML_STATIC))
   SET(MODE_LABEL "_SHARED_DEBUG")
ENDIF(SFML_DEBUG AND NOT(SFML_STATIC))

IF(NOT(SFML_DEBUG) AND SFML_STATIC)
   SET(MODE_LABEL "_STATIC_NONDEBUG")
ENDIF(NOT(SFML_DEBUG) AND SFML_STATIC)

IF(SFML_DEBUG AND SFML_STATIC)
   SET(MODE_LABEL "_STATIC_DEBUG")
ENDIF(SFML_DEBUG AND SFML_STATIC)

LIST(APPEND SFML_LIBRARY ${SFML_MAIN_LIBRARY${MODE_LABEL}})

LIST(APPEND SFML_FIND_COMPONENTS "System") #We allways need at last the System component
LIST(REMOVE_DUPLICATES SFML_FIND_COMPONENTS)

SET(SFML_FOUND "YES")
FOREACH(COMPONENT ${SFML_FIND_COMPONENTS})
   SET( MODUL_NAME SFML_${COMPONENT}_LIBRARY${MODE_LABEL} )
   STRING(TOUPPER ${MODUL_NAME} MODUL_NAME)

   IF(NOT ${MODUL_NAME})
      SET(SFML_FOUND "NO")
      MESSAGE("-- SFML: Could not locate : ${MODUL_NAME}")
   ELSE(NOT ${MODUL_NAME})
      LIST(APPEND SFML_LIBRARY ${${MODUL_NAME}})
   ENDIF(NOT ${MODUL_NAME})
ENDFOREACH(COMPONENT)

LIST(REMOVE_DUPLICATES SFML_LIBRARY)

IF(NOT SFML_INCLUDE_DIR)
   SET(SFML_FOUND "NO")
   MESSAGE("-- SFML: Could not locate include directory")
ENDIF(NOT SFML_INCLUDE_DIR)

IF(NOT SFML_FOUND)
    MESSAGE("Components of SFML are missing!")
   #MESSAGE(FATAL_ERROR "Components of SFML are missing!")
ENDIF(NOT SFML_FOUND)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(SFML DEFAULT_MSG SFML_LIBRARY SFML_INCLUDE_DIR)