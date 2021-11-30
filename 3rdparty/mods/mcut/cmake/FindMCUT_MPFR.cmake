# Copyright (c) 2020-2021 CutDigital Ltd.
# All rights reserved.
# 
# NOTE: This file is licensed under GPL-3.0-or-later (default). 
# A commercial license can be purchased from CutDigital Ltd. 
#  
# License details:
# 
# (A)  GNU General Public License ("GPL"); a copy of which you should have 
#      recieved with this file.
# 	    - see also: <http://www.gnu.org/licenses/>
# (B)  Commercial license.
#      - email: contact@cut-digital.com
# 
# The commercial license options is for users that wish to use MCUT in 
# their products for comercial purposes but do not wish to release their 
# software products under the GPL license. 
# 
# Author(s)     : Floyd M. Chitalu


if(MCUT_MPFR_INCLUDE_DIR AND MCUT_MPFR_LIBRARY)
    # Already in cache, be silent
    set(mcut_MPFR_FIND_QUIETLY TRUE)
endif()

find_path(MCUT_MPFR_INCLUDE_DIR NAMES mpfr.h)
find_library(MCUT_MPFR_LIBRARY NAMES mpfr)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(MCUT_MPFR DEFAULT_MSG MCUT_MPFR_INCLUDE_DIR MCUT_MPFR_LIBRARY)

mark_as_advanced(MCUT_MPFR_INCLUDE_DIR MCUT_MPFR_LIBRARY)

# NOTE: this has been adapted from CMake's FindPNG.cmake.
if(mcut_MPFR_FOUND AND NOT TARGET mcut::MPFR)
    add_library(mcut::MPFR UNKNOWN IMPORTED)
    set_target_properties(mcut::MPFR PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${MCUT_MPFR_INCLUDE_DIR}"
        IMPORTED_LINK_INTERFACE_LANGUAGES "C" IMPORTED_LOCATION "${MCUT_MPFR_LIBRARY}")
endif()
