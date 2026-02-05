PDF=main.pdf

all: $(PDF)

$(PDF): main.tex refs.bib
	latexmk -pdf -interaction=nonstopmode -file-line-error -halt-on-error main.tex

clean:
	latexmk -C
