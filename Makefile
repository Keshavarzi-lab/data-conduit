DOCS_DIR  = docs

.PHONY: docs
docs:
	python tools/generate_api_md.py
