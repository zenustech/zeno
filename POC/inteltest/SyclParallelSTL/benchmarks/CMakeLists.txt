file(GLOB EXAMPLE_FILES ./*.cpp)

foreach(file ${EXAMPLE_FILES})

  get_filename_component(SOURCE_NAME ${file} NAME_WE)
  message(STATUS " Adding ${SOURCE_NAME} ")

  include_directories(${COMPUTECPP_INCLUDE_DIRECTORY})

  add_executable(${SOURCE_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_NAME}.cpp  )
  target_compile_options(${SOURCE_NAME} PUBLIC ${HOST_COMPILER_OPTIONS})

  target_link_libraries(${SOURCE_NAME} PUBLIC ${CMAKE_THREAD_LIBS_INIT})

  add_sycl_to_target(${SOURCE_NAME} ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_NAME}.cpp)


endforeach()
