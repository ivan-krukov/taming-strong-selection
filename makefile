disco.pdf: disco.tex
	mkdir -p tmp
	latexmk -pdf -outdir=tmp $^
	mv tmp/disco.pdf .
