file(GLOB IMGCV_SOURCE *.cpp *.h)

target_sources(zeno PRIVATE ${IMGCV_SOURCE})
target_include_directories(zeno PRIVATE .)

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc highgui)

target_include_directories(zeno PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(zeno PRIVATE ${OpenCV_LIBS})
