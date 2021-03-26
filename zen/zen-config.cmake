find_path(zen_ROOT_DIR NAMES libzen.so PATHS
    /home/bate/Develop/Mn/Hg/zen/zen
    /home/dilei/Codes/Mn/Hg/zen/zen
    /usr/lib/python3.6/site-packages/zen
    /usr/lib/python3.7/site-packages/zen
    /usr/lib/python3.8/site-packages/zen
    /usr/lib/python3.9/site-packages/zen
    )
message("zen_ROOT_DIR="${zen_ROOT_DIR})
find_path(zen_INCLUDE_DIR NAMES zen/zen.h PATHS ${zen_ROOT_DIR}/include)
find_path(zen_LIBRARY_DIR NAMES libzen.so PATHS ${zen_ROOT_DIR})
message("zen_INCLUDE_DIR="${zen_INCLUDE_DIR})
message("zen_LIBRARY_DIR="${zen_LIBRARY_DIR})
