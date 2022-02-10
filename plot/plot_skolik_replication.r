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
    data$structure = factor(data$structure, 
        levels=c("Pure Quantum Model", "Hybrid Model"))
    plot <- ggplot(data) + 
        geom_line(mapping=aes(x=step, y=value, group=run), 
            alpha=0.2, color="orangered2", size=LINE.SIZE) + 
        scale_x_continuous(name="Episode") + 
        scale_y_continuous(name="Episode return") +
        stat_summary(mapping=aes(x=step, y=value), geom="line", 
            fun = mean, color="orangered2",size=LINE.SIZE) +
        coord_fixed(ratio=4) +
        facet_grid(structure ~ .)

    return(plot)
}

label_data_repl <- function(file, i) {
    read.csv(file, stringsAsFactors= FALSE)

    data <- read.csv(file, stringsAsFactors= FALSE)

    if(grepl("pure", file, fixed=TRUE)) {
        data$structure <- "Pure Quantum Model"
    } else {
        data$structure <- "Hybrid Model"
    }

    data$run <- i

    return(data)
}

load_and_label_data <- function(files) {
    data <- label_data_repl(files[1], 1)

    for(i in seq_along(files)) {
        if(!grepl("\\.csv$", files[i]) | i==1) {
            next
        }
        data_new <- label_data_repl(files[i], i)

        data <- rbind(data, data_new)
    }
    return(data)
}

files <- list.files(path = root, full.names = TRUE, recursive = TRUE)

data <- load_and_label_data(files)
save_location <- create_save_location(save_name)

if(tex) {
    tikz(save_location, width=WIDTH.COL*INCH.PER.CM, height=WIDTH.COL*INCH.PER.CM*1.2)
} else {
    pdf(save_location, width=WIDTH.COL*INCH.PER.CM, height=WIDTH.COL*INCH.PER.CM*1.2)
}
print(plot_data(data) + theme_paper())
dev.off()