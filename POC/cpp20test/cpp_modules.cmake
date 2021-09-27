cmake_minimum_required(VERSION 3.13)

set(CPPM_MSVC_MINVER "19.23.28106.4")
set(CPPM_CLANG_MINVER "9")
set(CPPM_SUPPORTED_COMPILER_MSG 
	"cpp_modules only supports MSVC ${CPPM_MSVC_MINVER}+ and Clang ${CPPM_CLANG_MINVER}+")

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS CPPM_MSVC_MINVER)
		message(FATAL_ERROR "${CPPM_SUPPORTED_COMPILER_MSG}")
	else()
		set(CPPM_COMMAND_FORMAT "msvc")
	endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
	if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS CPPM_CLANG_MINVER)
		message(FATAL_ERROR "${CPPM_SUPPORTED_COMPILER_MSG}")
	else()
		if(MSVC)
			set(CPPM_COMMAND_FORMAT "clangcl")
		else()
			set(CPPM_COMMAND_FORMAT "clang")
		endif()
	endif()
else()
	message(FATAL_ERROR "${CPPM_SUPPORTED_COMPILER_MSG}")
endif()

macro(required_find_common var name)
	if(NOT ${var})
		message(FATAL_ERROR "${type} ${name} not found")
	endif()
endmacro()
macro(required_find_path var name)
	find_path(${var} ${name} ${ARGN})
	required_find_common(${var} ${name})
endmacro()
macro(required_find_library var name)
	find_library(${var} ${name} ${ARGN})
	required_find_common(${var} ${name})
endmacro()
macro(required_find_program var name)
	find_program(${var} ${name} ${ARGN})
	required_find_common(${var} ${name})
endmacro()

if(NOT DEFINED CPPM_SCANNER_PATH)
	required_find_program(CPPM_SCANNER_PATH clang-scan-deps
		HINTS "${CMAKE_CURRENT_LIST_DIR}/../bin"
		DOC "path to the patched clang-scan-deps executable")
endif()
message(STATUS "using the scanner at '${CPPM_SCANNER_PATH}'")

if(CMAKE_GENERATOR MATCHES "Visual Studio")
	required_find_path(CPPM_TARGETS_PATH cpp_modules.targets
		HINTS "${CMAKE_CURRENT_LIST_DIR}/../etc" DOC "path containing cpp_modules.targets and its dependencies")
endif()
if(CMAKE_GENERATOR MATCHES "Ninja")
	set(CMAKE_MAKE_PROGRAM "${CMAKE_CURRENT_LIST_DIR}/../bin/ninja" CACHE FILEPATH "" FORCE)
	message(STATUS "using the ninja at '${CMAKE_MAKE_PROGRAM}'")
	file(WRITE "${CMAKE_BINARY_DIR}/scanner_config.txt" "tool_path ${CPPM_SCANNER_PATH}\n")
	file(APPEND "${CMAKE_BINARY_DIR}/scanner_config.txt" "command_format ${CPPM_COMMAND_FORMAT}\n")
endif()

function(target_cpp_modules targets)
	foreach(target ${ARGV})
		set_property(TARGET ${target} PROPERTY CXX_STANDARD 20)
		set_property(TARGET ${target} PROPERTY CXX_STANDARD_REQUIRED ON)
	endforeach()
	
	if(CMAKE_GENERATOR MATCHES "Visual Studio")
		foreach(target ${ARGV})
			#the following doesn't set EnableModules and so /module:stdifcdir is also not set:
			#target_compile_options(${target} PRIVATE "/experimental:module") 
			#so use a property sheet instead to set EnableModules:
			set_property(TARGET ${target} PROPERTY VS_USER_PROPS "${CPPM_TARGETS_PATH}/cpp_modules.props")
			set_property(TARGET ${target} PROPERTY VS_GLOBAL_LLVMInstallDir "C:\\Program Files\\LLVM")
			set_property(TARGET ${target} PROPERTY VS_GLOBAL_CppM_ClangScanDepsPath "${CPPM_SCANNER_PATH}")

			target_link_libraries(${target}
				"${CPPM_TARGETS_PATH}/cpp_modules.targets"
			)
		endforeach()
	
		add_library(_CPPM_ALL_BUILD EXCLUDE_FROM_ALL "${CPPM_TARGETS_PATH}/dummy.cpp")
		set_property(TARGET _CPPM_ALL_BUILD PROPERTY EXCLUDE_FROM_DEFAULT_BUILD TRUE)
		add_dependencies(_CPPM_ALL_BUILD ${ARGV})
		target_link_libraries(_CPPM_ALL_BUILD "${CPPM_TARGETS_PATH}/cpp_modules.targets")
	endif()
endfunction()

# transform all the paths in 'files_var' to absolute paths
function(cppm_list_transform_to_absolute_path files_var)
	set(files_in "${${files_var}}")
	set(files_out "")
	foreach(maybe_rel_path ${files_in})
		get_filename_component(abs_path "${maybe_rel_path}"
			ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
		list(APPEND files_out "${abs_path}")
	endforeach()
	set(${files_var} "${files_out}" PARENT_SCOPE)
endfunction()

# make all the paths in 'files_var' relative to 'base_path' unless they're absolute paths
function(cppm_list_transform_rel_paths base_path files_var)
	set(files_in "${${files_var}}")
	set(files_out "")
	foreach(in_path ${files_in})
		if(IS_ABSOLUTE "${in_path}")
			# generated header paths should be passed as absolute paths starting with the binary dir
			# but their paths in the ninja manifest ends up being relative to the binary dir
			if(in_path MATCHES "^${base_path}")
				file(RELATIVE_PATH out_path "${base_path}" "${in_path}")
			else()
				set(out_path "${in_path}")
			endif()
		else()
			get_filename_component(abs_path "${in_path}"
				ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
			file(RELATIVE_PATH out_path "${base_path}" "${abs_path}")
		endif()
		list(APPEND files_out "${out_path}")
	endforeach()
	set(${files_var} "${files_out}" PARENT_SCOPE)
endfunction()

function(target_cpp_header_units target)
	set(headers "${ARGN}")
	target_sources(${target} PRIVATE "${headers}")
	
	if(CMAKE_GENERATOR MATCHES "Ninja")
		foreach(header ${headers})
			# the MSBuild customization already adds these to the sources
			set_source_files_properties("${header}" PROPERTIES LANGUAGE CXX)
			if(NOT MSVC)
				set_source_files_properties("${header}" COMPILE_FLAGS "-xc++")
			endif()
		endforeach()
		# ninja's path lookup expects relative paths to be relative to the binary dir
		# related cmake issue: https://gitlab.kitware.com/cmake/cmake/issues/13894
		cppm_list_transform_rel_paths("${CMAKE_BINARY_DIR}" headers)
		#message(STATUS "header units: ${headers}")
		file(APPEND "${CMAKE_BINARY_DIR}/scanner_config.txt" "header_units ${target} ${headers}\n")
		# todo: maybe encode whether it's a header unit as bindings on the edge instead ?
	endif()
	
	if(CMAKE_GENERATOR MATCHES "Visual Studio")
		cppm_list_transform_to_absolute_path(headers)
		set_property(TARGET ${target} PROPERTY VS_GLOBAL_CppM_Header_Units "${headers}")
	endif()
endfunction()
