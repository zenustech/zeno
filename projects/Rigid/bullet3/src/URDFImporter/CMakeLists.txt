INCLUDE_DIRECTORIES(
        ${BULLET_PHYSICS_SOURCE_DIR}/src
        ${BULLET_PHYSICS_SOURCE_DIR}/../../../zeno/include
)

SET(URDF_SRCS
    b3ResourcePath.cpp
    urdfStringSplit.cpp
    UrdfParser.cpp
    URDF2Bullet.cpp
    tinyxml2.cpp
    MyMultiBodyCreator.cpp
    BulletUrdfImporter.cpp
        ReadObjPrim.cpp b3ImportMeshUtility.cpp)

SET(URDF_HDRS
    CommonFileIOInterface.h
    b3BulletDefaultFileIO.h
    b3ResourcePath.h
    URDFJointTypes.h
    urdfLexicalCast.h
    UrdfParser.h
    urdfStringSplit.h
    URDF2Bullet.h
    MultiBodyCreationInterface.h
    tinyxml2.h
    MyMultiBodyCreator.h
    BulletUrdfImporter.h
    UrdfFindMeshFile.h
    ShapeData.h
        ReadObjPrim.h b3ImportMeshUtility.h URDFImporterInterface.h)

ADD_LIBRARY(URDFImporter ${URDF_SRCS} ${URDF_HDRS})
TARGET_INCLUDE_DIRECTORIES(
        URDFImporter PUBLIC .
)

IF (BUILD_SHARED_LIBS)
    TARGET_LINK_LIBRARIES(URDFImporter BulletCollision LinearMath Zeno)
ENDIF (BUILD_SHARED_LIBS)

IF (NOT MSVC)
    IF (NOT INTERNAL_CREATE_DISTRIBUTABLE_MSVC_PROJECTFILES)
        #FILES_MATCHING requires CMake 2.6
        IF (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} GREATER 2.5)
            IF (APPLE AND BUILD_SHARED_LIBS AND FRAMEWORK)
                INSTALL(TARGETS URDFImporter DESTINATION .)
            ELSE (APPLE AND BUILD_SHARED_LIBS AND FRAMEWORK)
                INSTALL(TARGETS URDFImporter
                        RUNTIME DESTINATION bin
                        LIBRARY DESTINATION lib${LIB_SUFFIX}
                        ARCHIVE DESTINATION lib${LIB_SUFFIX})

                #INSTALL(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                #    DESTINATION ${INCLUDE_INSTALL_DIR} FILES_MATCHING PATTERN "*.h" PATTERN "*.inl" PATTERN
                #    ".svn" EXCLUDE PATTERN "CMakeFiles" EXCLUDE)
            ENDIF (APPLE AND BUILD_SHARED_LIBS AND FRAMEWORK)
        ENDIF (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} GREATER 2.5)

        IF (APPLE AND BUILD_SHARED_LIBS AND FRAMEWORK)
            SET_TARGET_PROPERTIES(URDFImporter PROPERTIES FRAMEWORK true)
            SET_TARGET_PROPERTIES(URDFImporter PROPERTIES PUBLIC_HEADER "${URDF_HDRS}")
        ENDIF (APPLE AND BUILD_SHARED_LIBS AND FRAMEWORK)
    ENDIF (NOT INTERNAL_CREATE_DISTRIBUTABLE_MSVC_PROJECTFILES)
ENDIF()
