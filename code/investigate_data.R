# We try to make some interesting plots concerning the data.
library(ggplot2)
library(ggmosaic)
library(dplyr)
library(tidyr)
library(GGally)
library(scales)

setwd("/home/ajo/gitRepos/project")

# Original data. 
load("data/adult_data_binarized.RData", verbose = T)
load("data/adult_data_categ.RData", verbose = T) 

# Data from experiment 1. 
load("unconditional_generated_trees_bin.RData", verbose = T)
load("unconditional_generated_trees_cat.RData", verbose = T)

cont <- c("age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week")
# List of categorical variables (used to reverse onehot encode later!)
categ <- setdiff(names(adult.data), cont)
categ <- categ[-length(categ)] # Remove the label "y"!

summary(adult.data[,cont])

# Make densities for each variable. 
make_dens <- function(variable, data){
  plt <- data[,cont] %>% ggplot() +
    geom_density(aes(x = .data[[variable]])) +
    theme_minimal() 
  ggsave(paste0("plots/exp1_dens_cat_",variable,".pdf"), width = 9, height = 5)
  return(plt)
}

for (n in cont){
  make_dens(n, D2) # We simply make the grid in latex instead!
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
make_hist <- function(variable, data){
  d <- data.frame(table(data[variable])) %>% mutate(ratio = round(Freq/(nrow(data)),3))
  plt <- d %>% ggplot(aes(x = Var1, y = ratio)) +
    geom_col() +
    theme_minimal() +
    ylab("Percentage") +
    xlab(variable) +
    scale_y_continuous(labels = percent, limits = c(0,1)) +
    geom_text(aes(label = ratio), vjust = -1)
    ggsave(paste0("plots/adult_data_hist_cat_",variable,".pdf"), width = 9, height = 5)
  return(plt)
}

for (n in categ){
  make_hist(n,adult.data)
}

##### Make cross-correlation tables between select categorical variables. 
make_cross_correlation <- function(data,first.feat, second.feat){
  # Make cross-correlation table between two features of the same data set.
  tab <- table(data[,first.feat], 
               data[,second.feat])
  tab <- addmargins(tab)
  return(tab)  
}
#(tab <- make_cross_correlation(adult.data, "sex","native_country"))
#knitr::kable(tab,format = "latex", linesep = "", digits = 3, booktabs = T) %>% print()
(tab <- make_cross_correlation(adult.data, "sex","workclass"))
knitr::kable(tab,format = "latex", linesep = "", digits = 3, booktabs = T) %>% print()
mosaicplot(tab, main = "")
(tab <- make_cross_correlation(adult.data, "sex","race"))
knitr::kable(tab,format = "latex", linesep = "", digits = 3, booktabs = T) %>% print()
(tab <- make_cross_correlation(adult.data, "sex","relationship"))
knitr::kable(tab,format = "latex", linesep = "", digits = 3, booktabs = T) %>% print()
#(tab <- make_cross_correlation(adult.data, "sex","marital_status"))
#knitr::kable(tab,format = "latex", linesep = "", digits = 3, booktabs = T) %>% print()

# Make mosaic plot instead of two-way tables, perhaps more informative.
make_mosaic_plot <- function(data, first.feat, second.feat, exp.num, vers.num){
  plt <- data[,categ] %>% ggplot() +
    geom_mosaic(aes(x = product(.data[[first.feat]]), fill = .data[[second.feat]])) +
    theme_minimal() +
    scale_fill_grey()
  ggsave(paste0("plots/mosaic/",exp.num,"_",first.feat,"_",second.feat,"_",vers.num,".pdf"), width = 9, height = 5)
  return(plt) # Not sure why this shit is not working!!?!?!??
}

print(make_mosaic_plot(adult.data,"sex","race","adult_data","bin"))
plt <- data %>% ggplot() +
  geom_mosaic(mapping = aes(x = product(sex), fill = race)) +
  theme_minimal() +
  scale_fill_grey()
print(plt)

#### Make ggplots for comparisons.
make_qqplot <- function(OG.data, gen.data, feature, exp.num){
  # Make qq-plots between generated data and original data. 
  # Always assume that the generated data set is larger (which is it in our experiments).

  adult <- (OG.data %>% select(feature))[[1]] # This is assumed smaller. 
  gen <- (gen.data %>% select(feature))[[1]] # This is assumed larger.
  
  # When the sizes differ, the function qqplot interpolates between the sorted values of the larger set.
  # That is, we estimate nrow(OG.data) quantiles from nrow(gen.data) by interpolating between the points. 
  # Essentially, the dimension of the largest data set is reduced using linear approximations between the points. 
  q <- qqplot((adult.data %>% select(feature))[[1]],(D2 %>% select(feature))[[1]], plot.it = F)
  
  # Gives roughly the sames as this:
  #A <- (D2 %>% select(feature))[[1]]
  #B <- (adult.data %>% select(feature))[[1]]
  #q2  <- data.frame("x" = sort(B), "y" = sort(quantile(A,probs=ppoints(B))))
  
  plt <- data.frame("x"=q$x, "y"= q$y)  %>% 
    ggplot(aes(x = x, y = y)) +
    geom_point(shape = 4) +
    ylab(paste0("Generated ",feature," quantiles")) +
    xlab(paste0("Adult ", feature, " quantiles")) +
    theme_minimal() +
    geom_abline(intercept = 0, slope = 1, colour = "black", linetype = "solid")
  ggsave(paste0("plots/qq/",exp.num,"_",feature,"_cat",".pdf"), width = 9, height = 5)
  return(plt)
}

for (n in cont){
  # Make qqplots for all features in exp1 generated data. 
  make_qqplot(adult.data, D2, n,"exp1") # Need to change the second data set and the last name in order 
                                        # to make the qqplots for different experiments. 
}