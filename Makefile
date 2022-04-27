# Add to PHONY target list so cmds always run even when nothing has changed
.PHONY: install docs lint test clean

install:
	pip3 install .

lint:
	yapf --recursive --in-place .
	find -name '*.cpp' -o -name '*.hpp' | xargs clang-format -i
	python3 utils/test_copyright.py

test:
	pytest --forked -n 5

clean:
	find . -name '*.so' -delete
	find . -name '*.so.lock' -delete
	find . -name '.rendered.*.cpp' -delete
