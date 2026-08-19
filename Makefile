.PHONY: docs docs-api docs-api-check

docs: docs-api-check
	$(MAKE) -C docs html

docs-api:
	python tools/generate_api_md.py

docs-api-check:
	python tools/generate_api_md.py --check
