PAPER := main
BUILD := .build
TEXMF := $(CURDIR)/third_party/texmf

export TEXINPUTS := $(TEXMF)//:
export BIBINPUTS := $(CURDIR):
export BSTINPUTS := $(CURDIR):

.PHONY: pdf clean

pdf:
	@mkdir -p $(BUILD)
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD) $(PAPER).tex
	cd $(BUILD) && bibtex $(PAPER)
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD) $(PAPER).tex
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD) $(PAPER).tex
	cp $(BUILD)/$(PAPER).pdf $(PAPER).pdf

clean:
	rm -rf $(BUILD) $(PAPER).pdf
