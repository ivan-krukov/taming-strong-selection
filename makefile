paper.pdf: paper.tex
	mkdir -p tmp
	latexmk -pdf -outdir=tmp $^
	mv tmp/paper.pdf .
