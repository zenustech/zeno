#include <tbb/spin_mutex.h>
#include <tbb/current_thread_name.h>
#include <map>
#include <string.h>
#include <stdio.h>
#include <errno.h>

#if (defined _WIN32) || (defined _WIN64)
 #define WIN32_LEAN_AND_MEAN
 #include <windows.h>
#endif

#ifdef __linux__
 #include <sys/prctl.h>
#endif


/* Constants */

#if defined(_MSC_VER) && ((defined _WIN32) || (defined _WIN64))
 #define MS_VC_EXCEPTION 0x406D1388
#endif


/* Types */

namespace tbb
{

#if (defined _WIN32) || (defined _WIN64)
typedef DWORD threadid_t;
#endif


/* Structures */

#if defined(_MSC_VER) && ((defined _WIN32) || (defined _WIN64))

#pragma pack(push, 8)
typedef struct tagTHREADNAME_INFO
{
   DWORD dwType; // Must be 0x1000.
   LPCSTR szName; // Pointer to name (in user addr space).
   DWORD dwThreadID; // Thread ID (-1=caller thread).
   DWORD dwFlags; // Reserved for future use, must be zero.
} THREADNAME_INFO;
#pragma pack(pop)

#endif // _WIN32 || _WIN64


/* Private variables */

#if !defined(__linux__) && !defined(__APPLE__)
static tbb::spin_mutex g_Mutex;						/**< Mutex for access to the map of threads. */
static std::map<threadid_t, std::string> g_Names;	/**< Names of the threads. */
#endif


/* Private functions */

#if defined(_MSC_VER) && ((defined _WIN32) || (defined _WIN64))

//
// Sets name of the thread in the Visual Studio
//
static
void msvc_set_thread_name(THREADNAME_INFO * info)
{
	__try
	{
		RaiseException(MS_VC_EXCEPTION, 0, sizeof(THREADNAME_INFO) / sizeof(ULONG_PTR), (ULONG_PTR *)info);
	}
	__except(EXCEPTION_EXECUTE_HANDLER)
	{
	}
}

#endif


/* Public functions */

//
// Sets name of the current thread (for debugging purposes)
//
void set_current_thread_name(const char * name)
{
#if defined(_MSC_VER) && ((defined _WIN32) || (defined _WIN64))
	{
		tbb::spin_mutex::scoped_lock lock(g_Mutex);
		g_Names[GetCurrentThreadId()] = name;
	}	

	THREADNAME_INFO info;
	info.dwType = 0x1000;
	info.szName = name;
	info.dwThreadID = (DWORD)-1;
	info.dwFlags = 0;
	msvc_set_thread_name(&info);
#endif

#ifdef __linux__
	char buf[16];

	size_t len = strlen(name);
	if (len > 15)
		len = 15;

	memcpy(buf, name, len);
	buf[len] = 0;

	if (prctl(PR_SET_NAME, (unsigned long)buf, 0, 0, 0) < 0)
	{
	#ifdef __TBB_BUILD
		fprintf(stderr, "prctl(PR_SET_NAME, \"%s\") failed: %s", name, strerror(errno));
	#else
		logger << LOG_ERROR << "prctl(PR_SET_NAME, \"%s\") failed: %s"
			<< QString::fromLatin1(name) << strerror(errno);
	#endif
	}
#endif
}

//
// Returns name of the current thread
//
std::string get_current_thread_name()
{
#if defined(_MSC_VER) && ((defined _WIN32) || (defined _WIN64))
	tbb::spin_mutex::scoped_lock lock(g_Mutex);
	return g_Names[GetCurrentThreadId()];
#endif

#ifdef __linux__
	return std::string();
#endif
}

/**
 * Returns name of the specified thread (for debugging purposes).
 * @param id ID of the thread.
 */
std::string get_thread_name(long id)
{
#if defined(_MSC_VER) && ((defined _WIN32) || (defined _WIN64))
	tbb::spin_mutex::scoped_lock lock(g_Mutex);
	return g_Names[(threadid_t)id];
#endif

#ifdef __linux__
	return std::string();
#endif
}

}
