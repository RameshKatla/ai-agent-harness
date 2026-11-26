.PHONY: test check run

test:
	python -m pytest

check:
	python -m compileall -q app tests

run:
	python -m uvicorn app.main:create_app --factory --no-access-log
