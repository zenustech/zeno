#
# This file is part of KDDockWidgets.
#
# SPDX-FileCopyrightText: 2019-2021 Klarälvdalens Datakonsult AB, a KDAB Group company <info@kdab.com>
# Author: Jean-Michaël Celerier <jean-michael.celerier@kdab.com>
#
# SPDX-License-Identifier: GPL-2.0-only OR GPL-3.0-only
#
# Contact KDAB at <info@kdab.com> for commercial licensing options.
#

include(CMakeFindDependencyMacro)

find_dependency(Qt5Widgets REQUIRED)
if (OFF)
    find_dependency(Qt5Quick REQUIRED)
endif()

if (NOT WIN32 AND NOT APPLE AND NOT EMSCRIPTEN AND NOT OFF)
    find_dependency(Qt5X11Extras REQUIRED)
endif()

# Add the targets file
include("${CMAKE_CURRENT_LIST_DIR}/KDDockWidgetsTargets.cmake")
