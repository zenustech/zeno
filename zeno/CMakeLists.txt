file(GLOB ZENO_SOURCE core/*.cpp utils/*.cpp extra/*.cpp include/*/*.h include/*/*/*.h)
file(GLOB ZENO_NODES_SOURCE nodes/*.cpp nodes/*/*.cpp)

add_library(zeno SHARED ${ZENO_SOURCE} ${ZENO_NODES_SOURCE})

if (ZENO_ENABLE_OPENMP)
    find_package(OpenMP)
    if (TARGET OpenMP::OpenMP_CXX)
        message("found package: OpenMP::OpenMP_CXX")
        target_link_libraries(zeno PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()
if (ZENO_VISUALIZATION)
    target_compile_definitions(zeno PUBLIC -DZENO_VISUALIZATION)
    if (UNIX OR APPLE)
		target_link_libraries(zeno PRIVATE stdc++fs)  # for <experimental/filesystem>
	endif()
endif()
if (ZENO_GLOBALSTATE)
    target_compile_definitions(zeno PUBLIC -DZENO_GLOBALSTATE)
endif()
if (ZENO_BENCHMARKING)
    target_compile_definitions(zeno PRIVATE -DZENO_BENCHMARKING)
endif()
if (ZENO_FAULTHANDLER)
    target_compile_definitions(zeno PRIVATE -DZENO_FAULTHANDLER)
endif()

target_compile_definitions(zeno PRIVATE -DDLL_ZENO)
target_include_directories(zeno PUBLIC include)

target_link_libraries(zeno PRIVATE spdlog)

set(OUTPUT_DIR ${PROJECT_SOURCE_DIR}/zenqt/lib)
set_target_properties(zeno PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR}
    ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_DIR}
    LIBRARY_OUTPUT_DIRECTORY_DEBUG ${OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${OUTPUT_DIR}
    ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${OUTPUT_DIR}
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_DIR}
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_DIR}
    )
