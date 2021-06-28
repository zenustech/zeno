all:
	cmake -B build
	make -C build -j `python -c 'from multiprocessing import cpu_count; print(cpu_count() * 2)'`
