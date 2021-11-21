#ifndef __TBB_current_thread_name_H
#define __TBB_current_thread_name_H

/* Includes */

#include <string>


/* Functions */

namespace tbb
{

extern "C"
{

#ifdef _MSC_VER
#pragma warning(disable:4190) // Function has C-linkage specified, but returns UDT which is incompatible with C
#endif

/**
 * Sets name of the current thread (for debugging purposes).
 * @param name New name of the thread.
 */
void set_current_thread_name(const char * name);

/** Returns name of the current thread (for debugging purposes). */
std::string get_current_thread_name();

/**
 * Returns name of the specified thread (for debugging purposes).
 * @param id ID of the thread.
 */
std::string get_thread_name(long id);

#ifdef _MSC_VER
#pragma warning(default:4190) // Function has C-linkage specified, but returns UDT which is incompatible with C
#endif

}

}

#endif // __TBB_current_thread_name_H
