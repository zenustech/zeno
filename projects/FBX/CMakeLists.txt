file(GLOB FBX_SOURCE *.cpp *.h)

option(USE_OFFICIAL_ASSIMP "Use Official Assimp Library" OFF)

target_sources(zeno PRIVATE ${FBX_SOURCE})
target_include_directories(zeno PRIVATE .)
target_include_directories(zeno PRIVATE ${CMAKE_SOURCE_DIR}/zenovis/stbi/include)

if(USE_OFFICIAL_ASSIMP)
    message(STATUS "FBX Extension: USE_OFFICIAL_ASSIMP ON")
    target_compile_definitions(zeno PRIVATE "USE_OFFICIAL_ASSIMP=1")
    add_subdirectory(lib/assimp)
else()
    add_subdirectory(assimp)
endif()

target_link_libraries(zeno PRIVATE assimp)

set(DEM_BD "${PROJECT_BINARY_DIR}/bin/dem")
message(STATUS "Dem Bin Directory ${DEM_BD}")

if(WIN32)
    configure_file(${CMAKE_CURRENT_LIST_DIR}/bin/Windows/DemBones.exe
            ${DEM_BD}/DemBones.exe COPYONLY)
    configure_file(${CMAKE_CURRENT_LIST_DIR}/bin/Windows/Alembic.dll
            ${DEM_BD}/dem/Alembic.dll COPYONLY)
    configure_file(${CMAKE_CURRENT_LIST_DIR}/bin/Windows/hdf5.dll
            ${DEM_BD}/dem/hdf5.dll COPYONLY)
    configure_file(${CMAKE_CURRENT_LIST_DIR}/bin/Windows/Imath-3_1.dll
            ${DEM_BD}/dem/Imath-3_1.dll COPYONLY)
    configure_file(${CMAKE_CURRENT_LIST_DIR}/bin/Windows/zlib1.dll
            ${DEM_BD}/dem/zlib1.dll COPYONLY)
else()
    configure_file(${CMAKE_CURRENT_LIST_DIR}/bin/Linux/DemBones ${DEM_BD}/DemBones COPYONLY)
endif()