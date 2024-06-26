
message(CHECK_START "[Reflection] Starting to locate dependencies")

find_package(Clang REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG COMPONENTS Headers)

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include_directories(${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})
link_directories(${CLANG_LIBRARY_DIR} ${LLVM_LIBRARY_DIR})

add_definitions(${CLANG_DEFINITIONS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST}) 

if (MSVC)
    set(LIBCLANG_LIBRARY clangAST clangTooling CACHE INTERNAL "Clang targets to be linked")
    llvm_map_components_to_libnames(llvm_libs all)
    set(LLVM_LIBRARY ${llvm_libs} CACHE INTERNAL "LLVM targets to be linked")
else()
    set(LIBCLANG_LIBRARY clang-cpp CACHE INTERNAL "Clang targets to be linked")
    set(LLVM_LIBRARY LLVM CACHE INTERNAL "LLVM targets to be linked")
endif()

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using libraries: ${LIBCLANG_LIBRARY}")

set(REFLECTION_ARGPARSE_INCLUDE_DIR ${REFLECTION_CMAKE_SOURCE_DIR}/thirdparty/argparse/include)
message(STATUS "Added argparse header, use target_include_directories(Target \${REFLECTION_ARGPARSE_INCLUDE_DIR}) to use it")
set(REFLECTION_INJA_INCLUDE_DIR ${REFLECTION_CMAKE_SOURCE_DIR}/thirdparty/inja/include)
message(STATUS "Added inja header, use target_include_directories(Target \${REFLECTION_INJA_INCLUDE_DIR}) to use it")

message(CHECK_PASS "[Reflection] Found all dependencies successfully")

# add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../thirdparty/bitsery)
