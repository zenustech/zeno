file(GLOB IMGCV_SOURCE *.cpp *.h)

add_library(zenimgcv SHARED ${IMGCV_SOURCE})
target_link_libraries(zenimgcv PRIVATE zenocore)

target_include_directories(zenimgcv PRIVATE .)

#find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc highgui features2d calib3d stitching)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)

target_include_directories(zenimgcv PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(zenimgcv PRIVATE ${OpenCV_LIBS})

find_package(TIFF REQUIRED)
include_directories(${TIFF_INCLUDE_DIR})
target_link_libraries(zenimgcv PRIVATE ${TIFF_LIBRARIES})
