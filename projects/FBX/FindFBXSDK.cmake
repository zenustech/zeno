find_path(FBXSDK_INCLUDE_DIR 
	NAMES "fbxsdk.h"
	PATHS "./FBXSDK/include"
)

if(WIN32)
	find_path(FBXSDK_LIBS_DIR
		NAMES "libfbxsdk-md.lib"
		PATHS
			"./FBXSDK/lib/vs2019/x64/debug"
			"./FBXSDK/lib/vs2017/x64/debug"
			"./FBXSDK/lib/vs2015/x64/debug"
	)
	file(GLOB FBXSDK_LIBS "${FBXSDK_LIBS_DIR}/*-md.lib")
else()
	find_path(FBXSDK_LIBS_DIR
		NAMES "libfbxsdk.a"
		PATHS 
			"./FBXSDK/lib/clang/release"
			"./FBXSDK/lib/gcc/x64/release"
	)
	file(GLOB FBXSDK_LIBS "${FBXSDK_LIBS_DIR}/*.a")
endif()

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(
  FBXSDK DEFAULT_MSG FBXSDK_INCLUDE_DIR FBXSDK_LIBS
)

message(STATUS "FBXSDK: ${FBXSDK_INCLUDE_DIR}")
message(STATUS "FBXSDK: ${FBXSDK_LIBS_DIR}")
