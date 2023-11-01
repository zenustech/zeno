
set(qan_source_files
    qanBehaviour.cpp
    qanBottomRightResizer.cpp
    qanRightResizer.cpp
    qanBottomResizer.cpp
    qanConnector.cpp
    qanDraggable.cpp
    qanDraggableCtrl.cpp
    qanEdge.cpp
    qanEdgeItem.cpp
    qanEdgeDraggableCtrl.cpp
    qanGraph.cpp
    qanGraphView.cpp
    qanGrid.cpp
    qanLineGrid.cpp
    qanGroup.cpp
    qanGroupItem.cpp
    qanNavigable.cpp
    qanNavigablePreview.cpp
    qanNode.cpp
    qanNodeItem.cpp
    qanPortItem.cpp
    qanSelectable.cpp
    qanStyle.cpp
    qanStyleManager.cpp
    qanAnalysisTimeHeatMap.cpp
    qanUtils.cpp
    qanTableGroup.cpp
    qanTableCell.cpp
    qanTableBorder.cpp
    qanTableGroupItem.cpp
    )

set (qan_header_files
    qanAbstractDraggableCtrl.h
    qanBehaviour.h
    qanBottomRightResizer.h
    qanRightResizer.h
    qanBottomResizer.h
    qanConnector.h
    qanDraggable.h
    qanDraggableCtrl.h
    qanEdge.h
    qanEdgeDraggableCtrl.h
    qanEdgeItem.h
    qanGraph.h
    qanGraphView.h
    qanGrid.h
    qanGroup.h
    qanGroupItem.h
    qanLineGrid.h
    qanNavigable.h
    qanNavigablePreview.h
    qanNode.h
    qanNodeItem.h
    qanPortItem.h
    qanSelectable.h
    qanStyle.h
    qanStyleManager.h
    qanAnalysisTimeHeatMap.cpp
    qanUtils.h
    qanTableGroup.h
    qanTableCell.h
    qanTableBorder.h
    qanTableGroupItem.h
    QuickQanava.h
    gtpo/container_adapter.h
    gtpo/edge.h
    gtpo/graph.h
    gtpo/graph.hpp
    gtpo/graph_property.h
    gtpo/node.h
    gtpo/node.hpp
    gtpo/observable.h
    gtpo/observer.h
    )

source_group("Headers" FILES ${qan_header_files})
source_group("Sources" FILES ${qan_source_files})

set (qan_plugin_files
    ${CMAKE_CURRENT_SOURCE_DIR}/qanPlugin.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/qanPlugin.h
)

# Configure Qt
set(CMAKE_AUTOMOC ON)
qt_wrap_cpp(qan_source_files, qan_header_files)   # Creates .moc files from sources
set(CMAKE_AUTORCC ON)
set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS $<$<CONFIG:Debug>:QT_QML_DEBUG>)

# Configure QuickQanava library ###############################################
if(QUICK_QANAVA_BUILD_STATIC_QRC)
    list(APPEND qan_resources QuickQanava_static.qrc)
else()
    list(APPEND qan_resources QuickQanava_plugin.qrc)
endif(QUICK_QANAVA_BUILD_STATIC_QRC)

list(APPEND qan_resources GraphicalEffects${QT_VERSION_MAJOR}/QuickQanavaGraphicalEffects.qrc)
add_library(QuickQanava STATIC ${qan_source_files} ${qan_header_files} ${qan_resources})

target_include_directories(QuickQanava
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
    )

target_compile_features(QuickQanava
    PUBLIC
        cxx_std_14
    )

install(FILES
    ${qan_header_files}
    DESTINATION include/quickqanava
    )

install(TARGETS QuickQanava
    EXPORT Targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include/quickqanava
    )

set(CMAKE_INCLUDE_CURRENT_DIR ON)
target_link_libraries(QuickQanava PUBLIC QuickContainers Qt${QT_VERSION_MAJOR}::Core Qt${QT_VERSION_MAJOR}::Gui Qt${QT_VERSION_MAJOR}::QuickControls2)
if(QUICK_QANAVA_BUILD_STATIC_QRC)
    target_compile_definitions(QuickQanava PUBLIC -DQUICKQANAVA_STATIC)
endif(QUICK_QANAVA_BUILD_STATIC_QRC)

# Configure QuickQanava QML module plugin #####################################
set(PLUGIN_TARGET "quickqanavaplugin")
add_library(${PLUGIN_TARGET} SHARED ${qan_plugin_files})
target_link_libraries(${PLUGIN_TARGET} QuickQanava)

# QML module deployment
set(URI QuickQanava)
string(REPLACE "." "/" TARGETPATH ${URI})
if (NOT DEFINED QT_QMAKE_EXECUTABLE)
    get_target_property (QT_QMAKE_EXECUTABLE Qt${QT_VERSION_MAJOR}::qmake IMPORTED_LOCATION)
    if (NOT QT_QMAKE_EXECUTABLE)
        message(FATAL_ERROR "Cannot find qmake")
    endif()
endif()
execute_process(COMMAND ${QT_QMAKE_EXECUTABLE} -query QT_INSTALL_QML OUTPUT_VARIABLE QT_INSTALL_QML_RAW)
string(STRIP ${QT_INSTALL_QML_RAW} QT_INSTALL_QML)
set(DESTDIR "${QT_INSTALL_QML}/${TARGETPATH}")
install(TARGETS ${PLUGIN_TARGET} DESTINATION ${DESTDIR})

# Install qmldir file
if(QUICK_QANAVA_BUILD_STATIC_QRC)
    install(
        FILES ${CMAKE_CURRENT_SOURCE_DIR}/qmldir_static
        DESTINATION ${DESTDIR}
        RENAME qmldir
        )
else()
    install(
        FILES ${CMAKE_CURRENT_SOURCE_DIR}/qmldir_plugin
        DESTINATION ${DESTDIR}
        RENAME qmldir
        )
endif(QUICK_QANAVA_BUILD_STATIC_QRC)


# Install source QML files to ease debugging
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/	# sic! Need leading slash, otherwise it will install into subdir src
    DESTINATION ${DESTDIR}
    FILES_MATCHING PATTERN "*.qml"
    )

