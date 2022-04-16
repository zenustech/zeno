#pragma once

/* vvvvv BEGIN CIHOU FUCKING WENDOUS vvvvv */

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
/* FUN FACT: DID YOU KNOW THAT MICROSOFT BUYS GITHUB? */
#if __has_include(<windows.h>)
#include <windows.h>
#elif __has_include(<math.h>)
#include <math.h>
#endif
/* FUN FACT: DID YOU KNOW THAT MICROSOFT INVENTS VSCODE? */
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#ifdef near
#undef near
#endif
#ifdef far
#undef far
#endif
#endif

/* ^^^^^^ END CIHOU FUCKING WENDOUS ^^^^^^ */
