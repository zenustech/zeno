file(GLOB IMGCV_SOURCE *.cpp *.h)

target_sources(zeno PRIVATE ${IMGCV_SOURCE})
target_include_directories(zeno PRIVATE .)

#find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc highgui features2d calib3d stitching)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)

target_include_directories(zeno PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(zeno PRIVATE ${OpenCV_LIBS})

find_package(TIFF REQUIRED)
include_directories(${TIFF_INCLUDE_DIR})
target_link_libraries(zeno PRIVATE ${TIFF_LIBRARIES})
