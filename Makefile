SOURCE_GLOB=$(wildcard backend/*.py frontend/*.py)

IGNORE_PEP=C0102,C0103,C0111,C0301,C0302,C0303,C0304,C0305,W0120,W0123,W0401,W0603,W0612,W0614,W0621,W0622,W0703,E1003,E1101

export PYTHONPATH=./

.PHONY: all
all: clean init

.PHONY: clean
clean:
	rm -rf dist/* .pytest_cache/

.PHONY: lint
lint: pylint

.PHONY: pylint
pylint:
	pylint --output-format=parseable --disable=R --disable=${IGNORE_PEP} ${SOURCE_GLOB}

.PHONY: test-unit
test-unit:
	pytest

.PHONY: test
test: test-unit

.PHONY: init-db
init-db:
	python3 orm.py

.PHONY: run-backend
run-backend:
	pip3 install -r backend/requirements.txt
	python3 backend/web_handler.py

.PHONY: run-frontend
run-frontend:
	pip3 install -r frontend/requirements.txt
	python3 frontend/main.py
