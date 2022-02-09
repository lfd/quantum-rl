#!/usr/bin/env Rscript
library(ggplot2)
library(tikzDevice)
library(stringr)
source("layout.r")
source("utils.r")

args <- validate_args()
root <- args[1]
tex <- args[3]
save_name <- get_savename(args[2], tex)

plot_data <- function(data) {
    plot <- ggplot(data) + 
        geom_line(mapping=aes(x=step, y=value), 
            alpha=0.3, color="green4", size=LINE.SIZE) + 
        stat_summary(mapping=aes(x=step, y=value), geom="line", 
            fun = mean, color="green4", size=LINE.SIZE) +
        scale_x_continuous(name="Step", limits=c(0,50000), 
            labels = scales::label_number_si()) + 
        scale_y_continuous(name="Validation return", limits=c(0,200)) +
        coord_fixed(ratio=200)

    return(plot)
}
data <- load_data_from_dir(root)
data$step <- data$step*100

save_location <- create_save_location(save_name)

if(tex) {
    tikz(save_location, width=WIDTH.COL*INCH.PER.CM, 
        height=WIDTH.COL*INCH.PER.CM)
} else {
    pdf(save_location, width=WIDTH.COL*INCH.PER.CM, 
        height=WIDTH.COL*INCH.PER.CM)
}
print(plot_data(data) + theme_paper())
dev.off()
