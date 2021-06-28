file(GLOB ZMSLIB_SOURCE *.cpp *.h)

add_library(ZMSlib SHARED ${ZMSLIB_SOURCE})
#target_compile_options(ZMSlib PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -march=native -Wno-ignored-attributes>)
target_include_directories(ZMSlib PRIVATE .)
target_include_directories(ZMSlib PRIVATE ../zenbase/include)

find_package(OpenMP)
if (TARGET OpenMP::OpenMP_CXX)
    message("found package: OpenMP::OpenMP_CXX")
    target_link_libraries(ZMSlib PRIVATE OpenMP::OpenMP_CXX)
endif()

find_package(zen REQUIRED)
target_link_libraries(ZMSlib PRIVATE zen)
set_target_properties(ZMSlib PROPERTIES
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
