.PHONY: default
default: taming-strong-selection.pdf

%.pdf: %.tex
	mkdir -p tmp
	latexmk -pdf -outdir=tmp $^
	mv tmp/$(@F) .

extra/%.svg: extra/%.dvi
	dvisvgm --no-fonts=1 $^ -o $(@D)/%f.svg

extra/%.dvi: extra/%.tex
	latex -output-format=dvi -output-directory=$(@D) $^


clean:
	rm -f taming-strong-selection.pdf
	rm -rf tmp
