default: all

all: core
	make -C Projects

core:
	cmake -B build
	make -C build -j `python -c 'from multiprocessing import cpu_count; print(cpu_count() * 2)'`

run: all
	./run.sh

install:
	python/setup.py install

dist:
	python/setup.py bdist_wheel

debug: all
	USE_GDB= ZEN_OPEN=assets/mesh.zsg ./run.sh
