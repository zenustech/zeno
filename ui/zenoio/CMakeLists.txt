file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp *.ui)

add_library(zenoio OBJECT ${source})

if (ZENO_DEBUG_MSVC)
    zeno_dbg_msvc(zenoio)
endif()

target_link_libraries(zenoio PUBLIC zenocore)
target_include_directories(zenoio PRIVATE .. ../.. ../include)
