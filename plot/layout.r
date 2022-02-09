WIDTH.PAPER <- 16.5
WIDTH.COL <- 7.82
INCH.PER.CM <- 0.394
BASE.SIZE <- 10
LINE.SIZE=0.6

theme_paper <- function() {
    return(theme_bw(base_size=BASE.SIZE) +
           theme(axis.title.x = element_text(size = BASE.SIZE),
                 axis.title.y = element_text(size = BASE.SIZE),
                 legend.title = element_text(size = BASE.SIZE),
                 legend.position = "none",
                ))
}

theme_paper_legend_top <- function() {
    return(theme_bw(base_size=BASE.SIZE) +
           theme(axis.title.x = element_text(size = BASE.SIZE),
                 axis.title.y = element_text(size = BASE.SIZE),
                 legend.title = element_text(size = BASE.SIZE),
                 legend.position = "top",
                ))
}

options(tikzDocumentDeclaration = "\\documentclass[final,3p,twocolumn]{elsarticle}",
        tikzLatexPackages = c(
            getOption( "tikzLatexPackages" ),
            "\\usepackage{amsmath}"
        ))
