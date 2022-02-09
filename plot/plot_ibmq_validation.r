#!/usr/bin/env Rscript
library(ggplot2)
library(tikzDevice)
library(stringr)
source("layout.r")
source("utils.r")

args <- validate_args()
file <- args[1]
tex <- args[3]
save_name <- get_savename(args[2], tex)

plot_data <- function(data) {
    plot <- ggplot(data) + 
        geom_line(mapping=aes(x=step, y=value), color="green4", size=LINE.SIZE) + 
        scale_x_continuous(name="Validation Step") + 
        scale_y_continuous(name="Validation return") +
        coord_fixed(ratio=0.325)

    return(plot)
}

data <- read.csv(file, stringsAsFactors= FALSE)

save_location = create_save_location(save_name)

if(tex) {
    tikz(save_location, width=WIDTH.COL*INCH.PER.CM, height=WIDTH.COL*INCH.PER.CM)
} else {
    pdf(save_location, width=WIDTH.COL*INCH.PER.CM, height=WIDTH.COL*INCH.PER.CM)
}
print(plot_data(data) + theme_paper())
dev.off()
