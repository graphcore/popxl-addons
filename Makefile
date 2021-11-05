# Add to PHONY target list so cmds always run even when nothing has changed
.PHONY: install docs lint test

install:
	pip3 install .

docs:
	sphinx-apidoc -o docs .
	sphinx-build -a -E -b html docs docs_html

lint:
	yapf --recursive --in-place .
	python3 utils/test_copyright.py

test:
	pytest .