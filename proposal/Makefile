# GNU Make automatic variables
# $< is the first prerequisite
# $* is the stem of target with which rule match (e.g., foo of foo.c) - what matches with % in a rulek
# $@ is the name of the target

targets = proposal

built_figtargets = #add images in
built_asytargets = 

base_figtargets = 

all: ${targets:%=%.pdf}

%.pdf:  %.tex
	pdflatex $<
	bibtex $*
	pdflatex $<
	pdflatex $<

# This makes eps files from fig files
%.eps:   %.fig
	fig2dev -L eps $< $@

# This makes eps files from dia diagram files
%.eps:   %.dia
	dia --filter=eps-builtin --export=$*.eps $<

# Make pdf files from eps files
%.pdf:   %.eps
	epstopdf $<

# This makes PDF files from fig files
%.pdf:   %.fig
	fig2dev -L pdf $< $@

# This makes PDF files from asy files
%.pdf:   %.asy
	asy -tex pdflatex -pdfreload $<

project.pdf: *.tex ${built_figtargets:%=%.pdf} ${base_figtargets} ${built_asytargets:%=%.pdf} project.bib

clean:
	/bin/rm -f ${targets:%=%.aux}
	/bin/rm -f ${targets:%=%.blg}
	/bin/rm -f ${targets:%=%.bbl}
	/bin/rm -f ${targets:%=%.log}
	/bin/rm -f ${targets:%=%.dvi}
	/bin/rm -f ${targets:%=%.pdfps}
	/bin/rm -f ${targets:%=%.nav}
	/bin/rm -f ${targets:%=%.out}
	/bin/rm -f ${targets:%=%.snm}
	/bin/rm -f ${targets:%=%.toc}
	/bin/rm -f ${built_figtargets:%=%.eps}

distclean: clean
	/bin/rm -f ${targets:%=%.ps}
	/bin/rm -f ${targets:%=%.pdf}
	/bin/rm -f ${built_figtargets:%=%.pdf}

fullclean: distclean
