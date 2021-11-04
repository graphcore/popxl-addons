# Add to PHONY target list so cmds always run even when nothing has changed
.PHONY: install docs

install:
	pip install .

docs:
	sphinx-apidoc -o docs .
	sphinx-build -a -E -b html docs docs_html
