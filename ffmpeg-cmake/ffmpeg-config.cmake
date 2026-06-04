find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED libavcodec libavformat libavutil libswscale libavdevice libavfilter)

# Create the specific target definitions that most open-source projects expect
if(NOT TARGET FFMPEG::FFMPEG)
    add_library(FFMPEG::FFMPEG INTERFACE IMPORTED)
    set_target_properties(FFMPEG::FFMPEG PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${FFMPEG_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${FFMPEG_LIBRARIES}"
    )
endif()

# Map the global PkgConfig variables to what the standard Config style uses
set(FFMPEG_FOUND TRUE)
set(FFMPEG_INCLUDE_DIRS ${FFMPEG_INCLUDE_DIRS})
set(FFMPEG_LIBRARIES ${FFMPEG_LIBRARIES})
