#!/usr/bin/env python
import zen,sys,ctypes
for x in sys.argv[1]:
 if x=="d":
  zen.dumpDescriptors()
 elif x=="l":
  ctypes.cdll.LoadLibrary("build/FastFLIP/libFLIPlib.so")
 elif x=="p":
  print("hell,wrld")
