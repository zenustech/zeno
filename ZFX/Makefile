x:
	#-rm kernel.ptx
	#nvcc -arch=sm_75 -ptx kernel.cu -o kernel.ptx
	make -Cbuild -j
	build/ZFXtest

d:
	make -Cbuild -j
	gdb build/ZFXtest -ex r
