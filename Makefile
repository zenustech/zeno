O=assets/numeric2.zsg

default: all run

install: prepare
	python/setup.py install

dist: prepare
	python/setup.py bdist_wheel

prepare:
	make -C build install

all:
	cmake -B build
	make -C build -j `python -c 'from multiprocessing import cpu_count; print(cpu_count() * 2)'`

run: all
	ZEN_OPEN=$O ./run.sh

clean_run:
	ZEN_OPEN=$O ./run.sh

debug: all
	USE_GDB=1 ZEN_SPROC=1 ZEN_OPEN=$O ./run.sh
