#ifndef B3_FILE_UTILS_H
#define B3_FILE_UTILS_H

#include <stdio.h>
#include "b3Scalar.h"
#include <stddef.h>  //ptrdiff_h
#include <string.h>

struct b3FileUtils
{
	b3FileUtils()
	{
	}
	virtual ~b3FileUtils()
	{
	}

	static bool findFile(const char* orgFileName, char* relativeFileName, int maxRelativeFileNameMaxLen)
	{
		FILE* f = 0;
		f = fopen(orgFileName, "rb");
		if (f)
		{
			//printf("original file found: [%s]\n", orgFileName);
			sprintf(relativeFileName, "%s", orgFileName);
			fclose(f);
			return true;
		}

		//printf("Trying various directories, relative to current working directory\n");
		const char* prefix[] = {"./", "./data/", "../data/", "../../data/", "../../../data/", "../../../../data/"};
		int numPrefixes = sizeof(prefix) / sizeof(const char*);

		f = 0;
		bool fileFound = false;

		for (int i = 0; !f && i < numPrefixes; i++)
		{
#ifdef _MSC_VER
			sprintf_s(relativeFileName, maxRelativeFileNameMaxLen, "%s%s", prefix[i], orgFileName);
#else
			sprintf(relativeFileName, "%s%s", prefix[i], orgFileName);
#endif
			f = fopen(relativeFileName, "rb");
			if (f)
			{
				fileFound = true;
				break;
			}
		}
		if (f)
		{
			fclose(f);
		}

		return fileFound;
	}

	static const char* strip2(const char* name, const char* pattern)
	{
		size_t const patlen = strlen(pattern);
		size_t patcnt = 0;
		const char* oriptr;
		const char* patloc;
		// find how many times the pattern occurs in the original string
		for (oriptr = name; (patloc = strstr(oriptr, pattern)); oriptr = patloc + patlen)
		{
			patcnt++;
		}
		return oriptr;
	}

	static char toLowerChar(const char t)
	{
		if (t >= (char)'A' && t <= (char)'Z')
			return t + ((char)'a' - (char)'A');
		else
			return t;
	}

	static void toLower(char* str)
	{
		int len = strlen(str);
		for (int i = 0; i < len; i++)
		{
			str[i] = toLowerChar(str[i]);
		}
	}

	/*static const char* strip2(const char* name, const char* pattern)
	{
		size_t const patlen = strlen(pattern);
		size_t patcnt = 0;
		const char * oriptr;
		const char * patloc;
		// find how many times the pattern occurs in the original string
		for (oriptr = name; patloc = strstr(oriptr, pattern); oriptr = patloc + patlen)
		{
			patcnt++;
		}
		return oriptr;
	}
	*/
};
#endif  //B3_FILE_UTILS_H
