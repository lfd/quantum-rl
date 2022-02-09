#!/usr/bin/env Rscript
library(ggplot2)
library(scales)
library(tikzDevice)
library(stringr)
source("layout.r")
source("utils.r")

args <- validate_args_two_dirs()
root1 <- args[1]
root2 <- args[2]
tex <- args[4]
save_name <- get_savename(args[3], tex)

plot_data <- function(data, data_outliner) {
    plot <- ggplot() + 
        geom_line(data, mapping=aes(x=step, y=value, color=model, group=run), alpha=0.05, size=LINE.SIZE) + 
        stat_summary(data, mapping=aes(x=step, y=value, color=model), geom="line", fun = mean, size=LINE.SIZE) +
        geom_line(data_outliner, mapping=aes(x=step, y=value, color=model), alpha=0.2, size=LINE.SIZE) + 
        scale_x_continuous(name="Step", limits=c(0,50000), labels = scales::label_number_si()) + 
        scale_y_continuous(name="Validation return", limits=c(0,200)) +
        coord_fixed(ratio=200) +
        facet_grid(model ~ .) +
        scale_color_manual(values=c("purple2","green4"))

    return(plot)
}

label_data <- function(files, file, i, is_nn) {
    data <- read.csv(files[i], stringsAsFactors= FALSE)

    if(is_nn) {
        data$model <- "Classical NN" 
    } else{
        data$model <- "VQ-DQN"
    }

    data$run <- i

    return(data)
}

get_data_and_outliner <- function(files, is_nn) {
    data <- label_data(files, files[1], 1, FALSE)
    outliner <- data
    for(i in seq_along(files)) {
        if(!grepl("\\.csv$", files[i]) | i==1) {
            next
        }

        data_new <- label_data(files, files[i], i, is_nn)
        
        if(nrow(data_new) >= nrow(outliner)) {
            data <- rbind(data, outliner)
            outliner <- data_new
        } else {
            data <- rbind(data, data_new)
        }
    }

    return(list(data, outliner))
}


files_vqc <- list.files(path = root1, full.names = TRUE, recursive = TRUE)
data_and_outliner <- get_data_and_outliner(files_vqc, FALSE)
data_vqc <- data_and_outliner[[1]]
outliner_vqc <- data_and_outliner[[2]]

files_nn <- list.files(path = root2, full.names = TRUE, recursive = TRUE)
data_and_outliner <- get_data_and_outliner(files_nn, TRUE)
data_nn <- data_and_outliner[[1]]
outliner_nn <- data_and_outliner[[2]]

data <- rbind(data_vqc, data_nn)
outliner <- rbind(outliner_vqc, outliner_nn)
data$step <- data$step*100
outliner$step <- outliner$step*100

save_location <- create_save_location(save_name)

if(tex) {
    tikz(save_location, width=WIDTH.COL*INCH.PER.CM, 
        height=WIDTH.COL*INCH.PER.CM*1.2)
} else {
    pdf(save_location, width=WIDTH.COL*INCH.PER.CM, 
        height=WIDTH.COL*INCH.PER.CM*1.2)
}
print(plot_data(data, outliner) + theme_paper())
dev.off()