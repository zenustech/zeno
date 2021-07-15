x: cuda/kernel.ptx
	make -Cbuild -j
	build/ZFXtest

cuda/kernel.ptx: cuda/kernel.cu
	-rm -rf $@
	nvcc -arch=sm_75 -ptx $< -o $@

d: cuda/kernel.ptx
	make -Cbuild -j
	gdb build/ZFXtest -ex r
