.PHONY: lint test api ci-test

lint:
	uv run ruff check .

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest evaluation/test_guardrails.py evaluation/test_feature_helpers.py evaluation/test_api.py -q

api:
	uv run uvicorn api:app --host 0.0.0.0 --port 8000

ci-test: lint test
