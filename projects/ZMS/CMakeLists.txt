file(GLOB ZMSLIB_SOURCE *.cpp *.h)

add_library(ZMSlib SHARED ${ZMSLIB_SOURCE})
#target_compile_options(ZMSlib PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -march=native -Wno-ignored-attributes>)
target_include_directories(ZMSlib PRIVATE .)
target_include_directories(ZMSlib PRIVATE ../oldzenbase/include)

find_package(OpenMP)
if (TARGET OpenMP::OpenMP_CXX)
    message("found package: OpenMP::OpenMP_CXX")
    target_link_libraries(ZMSlib PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(ZMSlib PRIVATE zeno)
get_property(zeno_AUTOLOAD_DIR TARGET zeno PROPERTY LIBRARY_OUTPUT_DIRECTORY)
set_target_properties(ZMSlib PROPERTIES
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
