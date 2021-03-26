// Released into the public domain by Robert Bridson, 2009.

#include <neg.h>

//==============================================================================
// Try to force the compiler to add a negative instead of subtracting.
// (i.e. write a+neg(b) since a+(-b) is erroneously simplified to a-b)
// Aggressive (and well-intentioned but *wrong*) inter-procedural analysis
// performed by the compiler at link time could defeat this; you might have
// to struggle with your compiler.

double neg(double x) { return -x; }
