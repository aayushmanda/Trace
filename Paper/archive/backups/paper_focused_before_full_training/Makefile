PYTHON ?= /home/hariguru/aayus/.venv/bin/python
TECTONIC ?= tectonic
.PHONY: all figures verify clean
all: main.pdf
main.pdf: main.tex body.tex appendix.tex main.bib $(wildcard sections/*.tex) figures/controlled.pdf figures/transfer_table.tex
	$(TECTONIC) -X compile --outdir . --keep-logs main.tex
figures:
	$(PYTHON) scripts/regenerate.py
verify:
	$(PYTHON) scripts/verify_executors.py
	$(PYTHON) verification/theorem4_checks.py
clean:
	rm -f main.aux main.log main.bbl main.blg main.out
.PHONY: check audit-routing
check: main.pdf
	$(PYTHON) scripts/check_manuscript.py
audit-routing:
	$(PYTHON) scripts/audit_routing.py
