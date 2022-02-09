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
        geom_line(mapping=aes(x=step, y=value, group=run), alpha=0.1, 
            color="blue2", size=LINE.SIZE) + 
        scale_x_continuous(name="Episode") + 
        scale_y_continuous(name="Episode return") +
        stat_summary(mapping=aes(x=step, y=value), geom="line", fun = mean, 
            color="blue2", size=LINE.SIZE) +
        coord_fixed(ratio=0.4) +
        geom_line(mapping=aes(x=step, y=moving_avg, group=run), alpha=0.25, 
            color="red", size=LINE.SIZE) +
        stat_summary(geom="line", mapping=aes(x=step, y=moving_avg), fun = mean, 
            color="red", size=LINE.SIZE) 

    return(plot)
}

data <- load_data_from_dir(root)

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