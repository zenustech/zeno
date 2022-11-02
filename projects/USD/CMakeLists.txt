file(GLOB USD_SOURCE *.cpp *.h)

target_sources(zeno PRIVATE ${USD_SOURCE})
zeno_disable_warning(${USD_SOURCE})
target_include_directories(zeno PRIVATE .)

message(STATUS "projects/USD PXR_LIBRARIES: ${PXR_LIBRARIES}")
target_link_libraries(zeno PRIVATE ${PXR_LIBRARIES})