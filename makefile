.PHONY: default figures
default: taming-strong-selection.pdf
figures: fig/strong_selection_six_panel.pdf fig/combined.pdf fig/missing.pdf fig/critical_normal.pdf fig/skellam.pdf fig/fixation_rate_N_100.pdf fig/fixation_rate_N_500.pdf fig/fixation_rate_N_1000.pdf
.PRECIOUS: data/fixation_rate_table_%.csv

data:
	mkdir -p data
	# Note that this can also be generated with batch.sh script
	python3 src/generate_tables.py

fig/afs_comp_small.pdf: data
	python3 src/afs_comp_panel.py --N-range 2000 200 --ns-range 0 10 --output $@

fig/afs_comp_big.pdf: data
	python3 src/afs_comp_panel.py --N-range 2000 1000 200 --ns-range 0 1 5 10 50 --output $@

fig/combined.pdf:
	python3 src/combined.py

fig/missing.pdf:
	python3 src/plot_closures.py

fig/critical_normal.pdf:
	python3 src/critical_normal.py

fig/skellam.pdf:
	python3 src/skellam.py

%.pdf: %.tex %.bib
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
	cc $^ -O3 -std=c99 -o $@

# FIXATION RATES

data/fixation_rate_table_%.csv: src/fixation_rate_comparison.py
	python src/fixation_rate_comparison.py $* > $@

fig/fixation_rate_N_%.pdf: data/fixation_rate_table_%.csv src/fixation_rate_plot.py
	python src/fixation_rate_plot.py $< $* $@

