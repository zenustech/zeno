file(GLOB PROJECT_SOURCE include/*/*.h *.cpp)

add_library(zenvdb SHARED ${PROJECT_SOURCE})
target_include_directories(zenvdb PUBLIC include)
target_include_directories(zenvdb PRIVATE ../zenbase/include)

find_package(OpenMP)
if (TARGET OpenMP::OpenMP_CXX)
    message("found package: OpenMP::OpenMP_CXX")
    target_link_libraries(zenvdb PRIVATE OpenMP::OpenMP_CXX)
endif()

find_package(TBB CONFIG REQUIRED COMPONENTS tbb tbbmalloc)
find_package(IlmBase REQUIRED COMPONENTS Half REQUIRED)
target_link_libraries(zenvdb PRIVATE openvdb IlmBase::Half TBB::tbbmalloc TBB::tbb)

find_package(zen REQUIRED)
target_link_libraries(zenvdb PRIVATE zen)
set_target_properties(zenvdb PROPERTIES
	LIBRARY_OUTPUT_DIRECTORY ${zen_AUTOLOAD_DIR}
	RUNTIME_OUTPUT_DIRECTORY ${zen_AUTOLOAD_DIR}
	ARCHIVE_OUTPUT_DIRECTORY ${zen_AUTOLOAD_DIR}
	LIBRARY_OUTPUT_DIRECTORY_DEBUG ${zen_AUTOLOAD_DIR}
	RUNTIME_OUTPUT_DIRECTORY_DEBUG ${zen_AUTOLOAD_DIR}
	ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${zen_AUTOLOAD_DIR}
	LIBRARY_OUTPUT_DIRECTORY_RELEASE ${zen_AUTOLOAD_DIR}
	RUNTIME_OUTPUT_DIRECTORY_RELEASE ${zen_AUTOLOAD_DIR}
	ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${zen_AUTOLOAD_DIR}
	)
