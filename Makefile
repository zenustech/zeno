x:
	cmake -B build
	make -C build -j `python -c 'from multiprocessing import cpu_count; print(cpu_count() * 2)'`
	USE_GDB=1 ZSG_OPEN=assets/octree2.zsg ./run.sh
