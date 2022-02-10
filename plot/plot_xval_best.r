#!/usr/bin/env Rscript
library(ggplot2)
library(scales)
library(ggh4x)
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
        geom_line(mapping=aes(x=step, y=value, color=encoding), size=LINE.SIZE) + 
        scale_x_continuous(name="Step", limits=c(0,50000), labels = scales::label_number_si()) + 
        scale_y_continuous(name="Validation return", limits=c(0,200)) +
        facet_nested(encoding ~ structure + extraction) +
        coord_fixed(ratio=200) +
        scale_color_manual(values=c("#D81B60", "#1E88E5"))

    return(plot)
}

load_and_label_data <- function(files) {
    data <- label_data(files[1])

    for(i in seq_along(files)) {
        if(!grepl("\\.csv$", files[i]) | i==1) {
            next
        }
        data_new <- label_data(files[i])

        data <- rbind(data, data_new)
    }
    return(data)
}

files <- list.files(path = root, full.names = TRUE, recursive = TRUE)
data <- load_and_label_data(files)

data$step <- data$step*100

save_location <- create_save_location(save_name)
if (tex) {
    tikz(save_location, width=WIDTH.PAPER*INCH.PER.CM, height=WIDTH.COL*INCH.PER.CM)
} else {
    pdf(save_location, width=WIDTH.PAPER*INCH.PER.CM, height=WIDTH.COL*INCH.PER.CM)
}
print(plot_data(data) + theme_paper())
dev.off()
