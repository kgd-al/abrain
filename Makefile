.PHONY: cpp

SHELL := /bin/bash
WD := $(shell pwd)

BUILD=debug
ESHN_ARGS= \
	-DESHN_WITH_DISTANCE=ON -DESHN_SUBSTRATE_DIMENSION=3 \
	-DWITH_TESTS=OFF -DWITH_DEBUG_INFO=OFF

all: cpp functions test dot

cpp: $(wildcard **/*.cpp) $(wildcard **/*.h) $(wildcard **/*.hpp)
	@source venv.sh; \
	mkdir -p build/${BUILD}; \
	cd build/${BUILD}; \
	cmake $(WD) -DCMAKE_BUILD_TYPE=$(BUILD) $(ESHN_ARGS); \
	make -s install;

functions:
	@./ps/plotter.sh

test:	cpp
	@source venv.sh; \
	cd python; \
	PYTHONPATH=. pytest -v --ff

dot:	$(wildcard **/*.cpp) $(wildcard **/*.h) $(wildcard **/*.hpp) $(wildcard **/*.py)
	@source venv.sh; \
	./make_package_dot.sh

clean:
	@find . -type d -a \( -name build -o -name '__pycache__' \) \
	| xargs rm -rvf
	@rm -rvf package.{dot,pdf} ps/*{eps,png}
