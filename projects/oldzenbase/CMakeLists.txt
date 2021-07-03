file(GLOB PROJECT_SOURCE include/*/*.h *.cpp mesh/*.cpp pars/*.cpp prim/*.cpp)

add_library(oldzenbase SHARED ${PROJECT_SOURCE})
target_compile_definitions(oldzenbase PRIVATE -DDLL_ZENBASE)
target_include_directories(oldzenbase PUBLIC include)

find_package(OpenMP)
if (TARGET OpenMP::OpenMP_CXX)
    message("found package: OpenMP::OpenMP_CXX")
    target_link_libraries(oldzenbase PRIVATE OpenMP::OpenMP_CXX)
endif()
#find_package(TBB CONFIG REQUIRED COMPONENTS tbb tbbmalloc)
#target_link_libraries(oldzenbase PRIVATE TBB::tbb TBB::tbbmalloc)

find_package(zeno REQUIRED)
target_link_libraries(oldzenbase PRIVATE zeno)

set_target_properties(oldzenbase PROPERTIES
	LIBRARY_OUTPUT_DIRECTORY ${zeno_AUTOLOAD_DIR}
	RUNTIME_OUTPUT_DIRECTORY ${zeno_AUTOLOAD_DIR}
	ARCHIVE_OUTPUT_DIRECTORY ${zeno_AUTOLOAD_DIR}
	LIBRARY_OUTPUT_DIRECTORY_DEBUG ${zeno_AUTOLOAD_DIR}
	RUNTIME_OUTPUT_DIRECTORY_DEBUG ${zeno_AUTOLOAD_DIR}
	ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${zeno_AUTOLOAD_DIR}
	LIBRARY_OUTPUT_DIRECTORY_RELEASE ${zeno_AUTOLOAD_DIR}
	RUNTIME_OUTPUT_DIRECTORY_RELEASE ${zeno_AUTOLOAD_DIR}
	ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${zeno_AUTOLOAD_DIR}
	)
