#pragma once

#include <stdlib.h>

static long random() {
	long x = 0;
	x |= (rand() & 0xff);
	x |= (rand() & 0xff) << 8;
	x |= (rand() & 0xff) << 16;
	x |= (rand() & 0xff) << 24;
	return x;
}
