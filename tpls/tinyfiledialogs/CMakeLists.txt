add_library(tinyfiledialogs OBJECT)
target_sources(tinyfiledialogs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/tinyfiledialogs.c ${CMAKE_CURRENT_SOURCE_DIR}/tinyfiledialogs.h)
target_include_directories(tinyfiledialogs PUBLIC .)
