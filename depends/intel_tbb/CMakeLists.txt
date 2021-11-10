CMAKE_MINIMUM_REQUIRED(VERSION 2.6.2)

PROJECT(pg_tbb LANGUAGES C CXX)

ADD_DEFINITIONS("-D__TBB_BUILD=1")
ADD_DEFINITIONS("-D__TBB_EXCEPTIONS=1")
ADD_DEFINITIONS("-D__TBB_NO_IMPLICIT_LINKAGE=1")
ADD_DEFINITIONS("-DTBB_NO_LEGACY=1")

SET(PLATFORM_FILES )

IF(WIN32)
    IF(CMAKE_CL_64)
		SET(PLATFORM_FILES
			${PLATFORM_FILES}
			"${CMAKE_CURRENT_BINARY_DIR}/atomic_support.obj"
			src/tbb/win64-tbb-export.def
		)
		SET_SOURCE_FILES_PROPERTIES("${CMAKE_CURRENT_BINARY_DIR}/atomic_support.obj"
			PROPERTIES EXTERNAL_OBJECT 1)
		ADD_CUSTOM_COMMAND(
			OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/atomic_support.obj"
			COMMAND ml64 /DEM64T=1 /c /Zi "${CMAKE_CURRENT_SOURCE_DIR}/src/tbb/intel64-masm/atomic_support.asm" /Fo"${CMAKE_CURRENT_BINARY_DIR}/atomic_support.obj"
			DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/tbb/intel64-masm/atomic_support.asm"
		)
    ELSE(CMAKE_CL_64)
		SET(PLATFORM_FILES
			${PLATFORM_FILES}
			src/tbb/ia32-masm/lock_byte.obj
			src/tbb/ia32-masm/atomic_support.obj
		)
		SET_SOURCE_FILES_PROPERTIES(
			src/tbb/ia32-masm/lock_byte.obj
			src/tbb/ia32-masm/atomic_support.obj
			PROPERTIES EXTERNAL_OBJECT 1
		)
		IF(NOT CMAKE_COMPILER_IS_GNUCC)
			SET(PLATFORM_FILES ${PLATFORM_FILES} src/tbb/win32-tbb-export.def)
		ENDIF()
	ENDIF(CMAKE_CL_64)
	ADD_DEFINITIONS(-DUSE_WINTHREAD -D_WIN32_WINNT=0x0400 -DDO_ITT_NOTIFY -DUNICODE)
ELSEIF(APPLE)
	ADD_DEFINITIONS(-DUSE_PTHREAD)
ELSEIF(UNIX)
	ADD_DEFINITIONS(-DUSE_PTHREAD -DDO_ITT_NOTIFY)
ENDIF(WIN32)


#IF(WIN32)
#SET(STATIC_OR_SHARED SHARED)
#ELSE()
#SET(STATIC_OR_SHARED STATIC)
#ENDIF()

ADD_LIBRARY(pg_tbb ${STATIC_OR_SHARED}
	include/tbb/compat/ppl.h
	include/tbb/machine/ibm_aix51.h
	include/tbb/machine/linux_common.h
	include/tbb/machine/linux_ia32.h
	include/tbb/machine/linux_ia64.h
	include/tbb/machine/linux_intel64.h
	include/tbb/machine/mac_ppc.h
	include/tbb/machine/windows_ia32.h
	include/tbb/machine/windows_intel64.h
	include/tbb/_concurrent_queue_internal.h
	include/tbb/_tbb_windef.h
	include/tbb/aligned_space.h
	include/tbb/atomic.h
	include/tbb/blocked_range.h
	include/tbb/blocked_range2d.h
	include/tbb/blocked_range3d.h
	include/tbb/cache_aligned_allocator.h
	include/tbb/combinable.h
	include/tbb/concurrent_hash_map.h
	include/tbb/concurrent_queue.h
	include/tbb/concurrent_vector.h
	include/tbb/current_thread_name.h
	include/tbb/enumerable_thread_specific.h
	include/tbb/mutex.h
	include/tbb/null_mutex.h
	include/tbb/null_rw_mutex.h
	include/tbb/parallel_do.h
	include/tbb/parallel_for.h
	include/tbb/parallel_for_each.h
	include/tbb/parallel_invoke.h
	include/tbb/parallel_reduce.h
	include/tbb/parallel_scan.h
	include/tbb/parallel_sort.h
	include/tbb/parallel_while.h
	include/tbb/partitioner.h
	include/tbb/pipeline.h
	include/tbb/queuing_mutex.h
	include/tbb/queuing_rw_mutex.h
	include/tbb/recursive_mutex.h
	include/tbb/scalable_allocator.h
	include/tbb/spin_mutex.h
	include/tbb/spin_rw_mutex.h
	include/tbb/task.h
	include/tbb/task_group.h
	include/tbb/task_scheduler_init.h
	include/tbb/task_scheduler_observer.h
	include/tbb/tbb.h
	include/tbb/tbb_allocator.h
	include/tbb/tbb_config.h
	include/tbb/tbb_exception.h
	include/tbb/tbb_machine.h
	include/tbb/tbb_profiling.h
	include/tbb/tbb_stddef.h
	include/tbb/tbb_thread.h
	include/tbb/tbbmalloc_proxy.h
	include/tbb/tick_count.h
	src/tbb/cache_aligned_allocator.cpp
	src/tbb/concurrent_hash_map.cpp
	src/tbb/concurrent_queue.cpp
	src/tbb/concurrent_vector.cpp
	src/tbb/current_thread_name.cpp
	src/tbb/dynamic_link.cpp
	src/tbb/dynamic_link.h
	src/tbb/gate.h
	src/tbb/itt_notify.cpp
	src/tbb/itt_notify.h
	src/tbb/itt_notify_proxy.c
	src/tbb/mutex.cpp
	src/tbb/pipeline.cpp
	src/tbb/queuing_mutex.cpp
	src/tbb/queuing_rw_mutex.cpp
	src/tbb/recursive_mutex.cpp
	src/tbb/spin_mutex.cpp
	src/tbb/spin_rw_mutex.cpp
	src/tbb/task.cpp
	src/tbb/tbb_assert_impl.h
	src/tbb/tbb_misc.cpp
	src/tbb/tbb_misc.h
	#src/tbb/tbb_resource.rc
	src/tbb/tbb_thread.cpp
	src/tbb/tbb_version.h
	src/tbb/tls.h
	${PLATFORM_FILES}
)

TARGET_INCLUDE_DIRECTORIES(pg_tbb PUBLIC src)
TARGET_INCLUDE_DIRECTORIES(pg_tbb PUBLIC include)

TARGET_LINK_LIBRARIES(pg_tbb PUBLIC ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS})

SET_TARGET_PROPERTIES(pg_tbb PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
	LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

IF(WIN32)
	SET_TARGET_PROPERTIES(pg_tbb PROPERTIES PREFIX "")
ENDIF()
