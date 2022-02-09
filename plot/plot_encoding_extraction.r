#!/usr/bin/env Rscript
library(ggplot2)
library(scales)
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
        geom_line(mapping=aes(x=step, y=value, color=encoding, group=run), 
            alpha=0.1, size=LINE.SIZE) + 
        stat_summary(mapping=aes(x=step, y=value, color=encoding), 
            geom="line", fun = mean, size=LINE.SIZE) +
        scale_x_continuous(name="Step", limits=c(0,50000), 
            labels = scales::label_number_si()) + 
        scale_y_continuous(name="Validation return", limits=c(0,200)) +
        facet_grid(structure ~ extraction) +
        coord_fixed(ratio=180) +
        scale_color_manual(name="Encoding", 
            values=c("#D81B60", "#1E88E5", "#FFC107"))
            
    return(plot)
}

load_and_label_data <- function(files) {
    data <- label_data(files[1], 1)

    for(i in seq_along(files)) {
        if(!grepl("\\.csv$", files[i]) | i==1) {
            next
        }
        data_new <- label_data(files[i], i)

        data <- rbind(data, data_new)
    }
    return(data)
}

files <- list.files(path = root, full.names = TRUE, recursive = TRUE)

data <- load_and_label_data(files)

data$step <- data$step*100

save_location <- create_save_location(save_name)

if(tex) {
    tikz(save_location, width=WIDTH.PAPER*INCH.PER.CM, 
        height=WIDTH.PAPER*INCH.PER.CM*1.3)
} else {
    pdf(save_location, width=WIDTH.PAPER*INCH.PER.CM, 
        height=WIDTH.COL*INCH.PER.CM*1.3)
}
print(plot_data(data) + theme_paper_legend_top())
dev.off()
