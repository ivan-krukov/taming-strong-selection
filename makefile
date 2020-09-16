.PHONY: default
default: disco.pdf

%.pdf: %.tex
	mkdir -p tmp
	latexmk -pdf -outdir=tmp $^
	mv tmp/$(@F) .

extra/%.svg: extra/%.dvi
	dvisvgm --no-fonts=1 $^ -o $(@D)/%f.svg

extra/%.dvi: extra/%.tex
	latex -output-format=dvi -output-directory=$(@D) $^


clean:
	rm -f disco.pdf
	rm -rf tmp
