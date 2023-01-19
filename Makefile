# Add to PHONY target list so cmds always run even when nothing has changed
.PHONY: install test clean

install:
	pip3 install .

test:
	pytest --forked -n 5

clean:
	find . -name '*.so' -delete
	find . -name '*.so.lock' -delete
	find . -name '.rendered.*.cpp' -delete
