x:
	cmake -B build
	make -C build -j `python -c 'from multiprocessing import cpu_count; print(cpu_count() * 2)'`
	USE_GDB= ZSG_OPEN=assets/mutable.zsg ./run.sh

y:
	scripts/alldlls.sh
