file(GLOB ZMSLIB_SOURCE *.cpp *.h)

add_library(ZMSlib SHARED ${ZMSLIB_SOURCE})
target_compile_options(ZMSlib PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -march=native -Wno-ignored-attributes>)
target_include_directories(ZMSlib PRIVATE .)

find_package(OpenMP)
if (TARGET OpenMP::OpenMP_CXX)
    message("found package: OpenMP::OpenMP_CXX")
    target_link_libraries(ZMSlib PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(ZMSlib PRIVATE zenbase)

find_package(zen REQUIRED)
target_link_libraries(ZMSlib PRIVATE zen)
set_property(TARGET ZMSlib PROPERTY LIBRARY_OUTPUT_DIRECTORY ${zen_INSTALL_DIR}/autoload)
set_property(TARGET ZMSlib PROPERTY RUNTIME_OUTPUT_DIRECTORY ${zen_INSTALL_DIR}/autoload)
