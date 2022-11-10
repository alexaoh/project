# We try to make some interesting plots concerning the data.
library(ggplot2)
library(dplyr)
library(tidyr)
library(GGally)
library(scales)

setwd("/home/ajo/gitRepos/project")

load("data/adult_data_categ.RData", verbose = T) # Categorical factors as they come originally. 
load("data/adult_data_binarized.RData", verbose = T)

cont <- c("age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week")
# List of categorical variables (used to reverse onehot encode later!)
categ <- setdiff(names(adult.data), cont)
categ <- categ[-length(categ)] # Remove the label "y"!

summary(adult.data[,cont])

# Make densities for each variable. 
make_dens <- function(variable){
  plt <- adult.data[,cont] %>% ggplot() +
    geom_density(aes(x = .data[[variable]])) +
    theme_minimal() 
  ggsave(paste0("plots/adult_data_dens_",variable,".pdf"), width = 9, height = 5)
  return(plt)
}

for (n in cont){
  make_dens(n) # We simply make the grid in latex instead!
}

# Get correlations of the continuous variables. 
c <- cor(adult.data[,cont])
c[lower.tri(c)] <- ""
knitr::kable(c,format = "latex", linesep = "", digits = 3, booktabs = T) %>% print()


# Not feasible because of the size of the plot (in Mb).
#pdf(file = "ggpairsplotAdultData.pdf", width = 9, height = 5)
#ggpairs(adult.data[,cont]) + 
#        theme_minimal() # Takes a very long time to produce result.
#dev.off()

# Make histograms for categorical features.

make_hist <- function(variable){
  plt <- adult.data[,categ] %>% ggplot() +
    geom_bar(aes(x = .data[[variable]], y = (..count..)/sum(..count..)), stat = "count") +
    theme_minimal() +
    ylab("Percentage") +
    scale_y_continuous(labels = percent, limits = c(0,1)) 
    ggsave(paste0("plots/adult_data_hist_",variable,".pdf"), width = 9, height = 5)
  return(plt)
}

for (n in categ){
  make_hist(n)
}

