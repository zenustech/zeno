
if (EXISTS ${LIB})
	cmake_path(GET LIB STEM dllname)
	if (EXISTS "${DLL_DIR}/${dllname}.dll")
		execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DLL_DIR}/${dllname}.dll" ${OUTPUT})
		message("copied dll ["${DLL_DIR}/${dllname}.dll"] to [${OUTPUT}]")
	endif()
endif()