library(ggplot2)
source("layout.r")

validate_args <- function() {
    args = commandArgs(trailingOnly=TRUE)
    if(length(args) == 3) {
        if (args[3] == "tex") args[3] <- TRUE
        else if (args[3] == "pdf") args[3] <- FALSE
        else stop("Usage: Rscript plot.R <path/to/dir> <savename> [tex|pdf]")
    }
    else if(length(args) != 2) {
        stop("Usage: Rscript plot.R <path/to/dir> <savename> [tex|pdf]")
    }
    else {
        args[3] <- FALSE
    }
    return(args)
}

validate_args_two_dirs <- function() {
    args = commandArgs(trailingOnly=TRUE)
    if(length(args) == 4) {
        if (args[4] == "tex") args[4] <- TRUE
        else if (args[4] == "pdf") args[4] <- FALSE
        else stop("Usage: Rscript plot.R <path/to/vqc_dir> <path/to/nn_dir> <savename> [tex|pdf]")
    }
    else if(length(args) != 3) {
        stop("Usage: Rscript plot.R <path/to/vqc_dir> <path/to/nn_dir> <savename> [tex|pdf]")
    }
    else {
        args[4] <- FALSE
    }
    return(args)
}

load_data_from_dir <- function(root) {
    files <- list.files(path = root, full.names = TRUE, recursive = TRUE)

    data <- read.csv(files[1], stringsAsFactors=FALSE)
    data$run <- 1

    for(i in seq_along(files)) {
        if(!grepl("\\.csv$", files[i]) | i==1) {
            next
        }
        data_new <- read.csv(files[i], stringsAsFactors= FALSE)

        data_new$run <- i
        data <- rbind(data, data_new)
    }
    return(data)
}

get_savename <- function(name, tex) {
    if(tex) suffix <- ".tex" else suffix <- ".pdf"
    return(paste(name, suffix, sep=""))
}

create_save_location <- function(save_name) {
    save_location <- file.path("plots", save_name)

    if(!dir.exists(dirname(save_location))) {
        dir.create(dirname(save_location), recursive=TRUE)
    }
    return(save_location)
}

label_data <- function(file, i=0) {
    data <- read.csv(file, stringsAsFactors= FALSE)

    if(grepl("lockwood", file, fixed=TRUE)) {
        data$structure <- "Lockwood and Si"
    } else {
        data$structure <- "Skolik et al."
    }

    if(grepl("_gsp", file, fixed=TRUE)) {
        data$extraction <- "GSP"
    } else if(grepl("_gs", file, fixed=TRUE)) {
        data$extraction <- "GS"
    } else if(grepl("_ls", file, fixed=TRUE)) {
        data$extraction <- "LS"
    }

    if(grepl("_sc", file, fixed=TRUE)) {
        data$encoding <- "SC"
    } else if (grepl("_c", file, fixed=TRUE)) {
        data$encoding <- "C"
    } else if(grepl("_sd", file, fixed=TRUE)){
        data$encoding <- "SD"
    }

    data$run <- i 

    return(data)
}