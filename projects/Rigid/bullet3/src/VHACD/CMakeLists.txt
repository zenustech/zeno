INCLUDE_DIRECTORIES(
        ${BULLET_PHYSICS_SOURCE_DIR}/src
)

SET(VHACD_SRCS
        src/VHACD.cpp
        src/vhacdICHull.cpp
        src/vhacdManifoldMesh.cpp
        src/vhacdMesh.cpp
        src/vhacdVolume.cpp
)

SET(VHACD_INC_HDRS
        inc/VHACD.h
        inc/vhacdCircularList.h
        inc/vhacdICHull.h
        inc/vhacdManifoldMesh.h
        inc/vhacdMesh.h
        inc/vhacdMutex.h
        inc/vhacdSArray.h
        inc/vhacdTimer.h
        inc/vhacdVector.h
        inc/vhacdVHACD.h
        inc/vhacdVolume.h
        inc/vhacdCircularList.inl
        inc/vhacdVector.inl
)

ADD_LIBRARY(VHACD ${VHACD_SRCS} ${VHACD_INC_HDRS})
TARGET_INCLUDE_DIRECTORIES(
 VHACD PUBLIC .
)

IF (BUILD_SHARED_LIBS)
  TARGET_LINK_LIBRARIES(VHACD BulletCollision LinearMath)
ENDIF (BUILD_SHARED_LIBS)

IF (NOT MSVC)
    IF (NOT INTERNAL_CREATE_DISTRIBUTABLE_MSVC_PROJECTFILES)
        #FILES_MATCHING requires CMake 2.6
        IF (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} GREATER 2.5)
            IF (APPLE AND BUILD_SHARED_LIBS AND FRAMEWORK)
                INSTALL(TARGETS VHACD DESTINATION .)
            ELSE (APPLE AND BUILD_SHARED_LIBS AND FRAMEWORK)
                INSTALL(TARGETS VHACD
                                        RUNTIME DESTINATION bin
                                        LIBRARY DESTINATION lib${LIB_SUFFIX}
                                        ARCHIVE DESTINATION lib${LIB_SUFFIX})

                #INSTALL(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                #    DESTINATION ${INCLUDE_INSTALL_DIR} FILES_MATCHING PATTERN "*.h" PATTERN "*.inl" PATTERN
                #    ".svn" EXCLUDE PATTERN "CMakeFiles" EXCLUDE)
            ENDIF (APPLE AND BUILD_SHARED_LIBS AND FRAMEWORK)
        ENDIF (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} GREATER 2.5)

        IF (APPLE AND BUILD_SHARED_LIBS AND FRAMEWORK)
            SET_TARGET_PROPERTIES(VHACD PROPERTIES FRAMEWORK true)
            SET_TARGET_PROPERTIES(VHACD PROPERTIES PUBLIC_HEADER "${VHACD_INC_HDRS}")
        ENDIF (APPLE AND BUILD_SHARED_LIBS AND FRAMEWORK)
    ENDIF (NOT INTERNAL_CREATE_DISTRIBUTABLE_MSVC_PROJECTFILES)
ENDIF()