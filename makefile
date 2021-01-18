.PHONY: default figures
default: taming-strong-selection.pdf
figures: fig/strong_selection_six_panel.pdf fig/combined.pdf fig/missing.pdf fig/critical_normal.pdf fig/skellam.pdf

data:
	mkdir -p data
	python3 src/generate_tables.py

fig/selection_six_panel.pdf: data
	python3 src/strong_selection.py

fig/combined.pdf:
	python3 src/combined.py

fig/missing.pdf:
	python3 src/plot_closures.py

fig/critical_normal.pdf:
	python3 src/critical_normal.py

fig/skellam.pdf:
	python3 src/skellam.py

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

transition_probability_explicit: src/transition_probability_explicit.c
	cc $^ -O3 -o $@
