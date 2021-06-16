#O=assets/rigid.zsg

default: all run

all:
	cmake -B build
	make -C build -j `python -c 'from multiprocessing import cpu_count; print(cpu_count() + 4)'`

run: all
	ZEN_OPEN=$O ./run.sh

debug: all
	USE_GDB=1 ZEN_SPROC=1 ZEN_OPEN=$O ./run.sh

install:
	python/setup.py install

dist:
	python/setup.py bdist_wheel
