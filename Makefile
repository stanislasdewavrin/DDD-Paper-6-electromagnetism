PAPER = paper
LATEX = pdflatex -interaction=nonstopmode -halt-on-error
BIBTEX = bibtex
PYTHON = python3

.PHONY: all paper tests clean

all: tests paper

paper: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex references.bib
	$(LATEX) $(PAPER)
	$(BIBTEX) $(PAPER)
	$(LATEX) $(PAPER)
	$(LATEX) $(PAPER)

tests:
	$(PYTHON) code/01_oriented_sector_tests.py

clean:
	rm -f $(PAPER).aux $(PAPER).log $(PAPER).bbl $(PAPER).blg $(PAPER).out $(PAPER).toc
