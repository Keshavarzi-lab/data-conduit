NOTEBOOK  = src/data_conduit/data_conduit_demos/API_Reference.ipynb
DOCS_DIR  = docs

.PHONY: docs
docs:
	jupyter nbconvert --to html --no-input --execute \
		$(NOTEBOOK) \
		--output-dir $(DOCS_DIR)
	@echo "API reference updated → $(DOCS_DIR)/API_Reference.html"
