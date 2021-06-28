# Additional targets to perform clang-format/clang-tidy
# Get all project files
file(GLOB_RECURSE
     ALL_SOURCE_FILES
     *.[chi]pp *.[chi]xx *.cc *.hh *.ii *.[CHI] *.cu *.cuh
     )
list(FILTER ALL_SOURCE_FILES EXCLUDE REGEX "${PROJECT_SOURCE_DIR}/include/zensim/tpls/.*" )

# Adding clang-format target if executable is found
find_program(CLANG_FORMAT "clang-format")
if(CLANG_FORMAT)
  add_custom_target(
    zs-clang-format
    COMMAND /usr/bin/clang-format
    -i
    -style=file
    ${ALL_SOURCE_FILES}
    )
endif()

# Adding clang-tidy target if executable is found
find_program(CLANG_TIDY "clang-tidy")
if(CLANG_TIDY)
  add_custom_target(
    zs-clang-tidy
    COMMAND /usr/bin/clang-tidy
    ${ALL_SOURCE_FILES}
    -config=''
    --
    -std=c++17
    ${INCLUDE_DIRECTORIES}
    )
endif()