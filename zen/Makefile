build:
	cmake -B build
	make -C build

wheel: build
	python3 build.py

install: wheel
	rm -rf /tmp/env
	virtualenv /tmp/env
	source /tmp/env/bin/activate && pip install build/*.whl

.PHONY: build wheel install
