#!/usr/bin/env Rscript
library(ggplot2)
library(scales)
library(ggh4x)
source("layout.r")
source("utils.r")


args <- validate_args()
root <- args[1]
tex <- args[3]
save_name <- get_savename(args[2], tex)

plot_data <- function(data) {
    if(data$encoding[1] == "Continuous (C)") {
        color <- "#D81B60"
    } else if(data$encoding[1] == "Scaled and Continuous (SC)") {
        color <- "#1E88E5"
    }

    title <- paste(data$structure[1], data$extraction[1], data$encoding[1], sep=" - ")

    if(data$data_reup) {
        title <- paste("Data Re-Uploading", title, sep=" - ")
    }

    plot <- ggplot(data) + 
        ggtitle(title) +
        geom_line(mapping=aes(x=step, y=value),  color=color, size=LINE.SIZE) + 
        scale_x_continuous(name="Step", limits=c(0,50000), labels = scales::label_number_si()) + 
        scale_y_continuous(name="Validation return", limits=c(0,200)) +
        facet_nested(epsilon_duration + gamma ~ lr + lr_steps) +
        coord_fixed(ratio=200) 

    return(plot)
}

label_data_extended <- function(file) {
    data <- read.csv(file, stringsAsFactors= FALSE)
    if(grepl("lockwood", file, fixed=TRUE)) {
        data$structure <- "Lockwood and Si"
    } else {
        data$structure <- "Skolik et al."
    }
    if(grepl("_gsp", file, fixed=TRUE)) {
        data$extraction <- "Global Scaling with Pooling (GSP)"
    } else if(grepl("_ls", file, fixed=TRUE)) {
        data$extraction <- "Local Scaling (LS)"
    } else if(grepl("_gs", file, fixed=TRUE)) {
        data$extraction <- "Global Scaling (GS)"
    }

    if(grepl("_c", file, fixed=TRUE)) {
        data$encoding <- "Continuous (C)"
    } else if (grepl("_sc", file, fixed=TRUE)) {
        data$encoding <- "Scaled and Continuous (SC)"
    } else if(grepl("_sd", file, fixed=TRUE)){
        data$encoding <- "Scaled and Directional (SD)"
    }

    if(grepl("e10000", file, fixed=TRUE)) {
        data$epsilon_duration <- "10000"
    } else if(grepl("e20000", file, fixed=TRUE)){
        data$epsilon_duration <- "20000"
    } else {
        data$epsilon_duration <- "30000"
    }

    if(grepl("lr0.1", file, fixed=TRUE)) {
        data$lr <- "0.1"
    } else if(grepl("lr0.01", file, fixed=TRUE)){
        data$lr <- "0.01"
    } else {
        data$lr <- "0.001"
    }

    if(grepl("s2000", file, fixed=TRUE)){
        data$lr_steps <- "2000"
    } else {
        data$lr_steps <- "4000"
    }

    if(grepl("g0.999", file, fixed=TRUE)){
        data$gamma <- "0.999"
    } else {
        data$gamma <- "0.99"
    }

    if(grepl("data_reup", file, fixed=TRUE)){
        data$data_reup <- TRUE
    } else {
        data$data_reup <- FALSE
    }
    return(data)
}

load_and_label_data <- function(files) {
    data <- label_data_extended(files[1])

    for(i in seq_along(files)) {
        if(!grepl("\\.csv$", files[i]) | i==1) {
            next
        }

        data_new <- label_data_extended(files[i])

        data <- rbind(data, data_new)
    }
    return(data)
} 

files <- list.files(path = root, full.names = TRUE, recursive = TRUE)
data <- load_and_label_data(files)

save_location <- create_save_location(save_name)

data$step <- data$step*100

if(tex) {
    tikz(save_location, width=WIDTH.PAPER*INCH.PER.CM*2.5, height=WIDTH.COL*INCH.PER.CM*2.5)
} else {
    pdf(save_location, width=WIDTH.PAPER*INCH.PER.CM*2.5, height=WIDTH.COL*INCH.PER.CM*2.5)
}
print(plot_data(data) + theme_paper())
dev.off()
