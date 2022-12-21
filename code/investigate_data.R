# We make some interesting plots concerning the generated data.
library(ggplot2)
library(ggmosaic)
library(dplyr)
library(tidyr)
library(GGally)
library(scales)
library(kableExtra)
options(knitr.kable.NA = '')

setwd("/home/ajo/gitRepos/project")

# Original data. 
load("data/adult_data_binarized.RData", verbose = T)
load("data/adult_data_categ.RData", verbose = T) 

# Data from experiment 1. 
load("results/unconditional_generated_trees_bin.RData", verbose = T)
load("results/unconditional_generated_trees_cat.RData", verbose = T)

# Data from experiment 2. 
# This data is simply loaded into R session after training the VAE. 

cont <- c("age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week")
# List of categorical variables (used to reverse onehot encode later!)
categ <- setdiff(names(adult.data), cont)
categ <- categ[-length(categ)] # Remove the label "y"!

summary(adult.data[,cont])

# Make density plots for each variable. 
make_dens <- function(variable, data, save, limits){
  plt <- data[,cont] %>% ggplot() +
    geom_density(aes(x = .data[[variable]])) +
    theme_minimal() + 
    scale_x_continuous(limits = limits)
    
  if (save){ggsave(paste0("plots/exp2_dens_cat_",variable,"_alg4.pdf"), width = 9, height = 5)}
  return(plt)
}

# Set the x-axis to the same values for each continuous variable in each experiment.
limits <- list(
  c(-1,100),
  c(-1,600000),
  c(-1,20),
  c(-1,10000),
  c(-1,1000),
  c(-1,100)
)

for (i in 1:length(cont)){
  plot(make_dens(cont[i], D2,F, limits[[i]])) # We simply make the grid in latex instead!
}

# Get correlations of the continuous variables. 
co <- cor(D2[,cont])
co[lower.tri(co)] <- NA
co <- data.frame(co)
colnames(co) <- c("age", "fnlwgt","ed_num","cap_gain","cap_loss","h_p_week")
co <- data.matrix(co)
kbl(co,format = "latex", linesep = "", digits = 3, booktabs = T) %>% 
  kable_styling(latex_options = c("scale_down")) %>% 
  column_spec(1, monospace = T) %>% 
  row_spec(0, monospace = T) %>% 
  print()


# Not feasible because of the size of the plot (in Mb).
#pdf(file = "ggpairsplotAdultData.pdf", width = 9, height = 5)
#ggpairs(adult.data[,cont]) + 
#        theme_minimal() # Takes a very long time to produce result.
#dev.off()

# Make histograms for categorical features.
make_hist <- function(variable, data, save){
  d <- data.frame(table(data[variable])) %>% mutate(ratio = round(Freq/(nrow(data)),3))
  plt <- d %>% ggplot(aes(x = Var1, y = ratio)) +
    geom_col() +
    theme_minimal() +
    ylab("Percentage") +
    xlab(variable) +
    scale_y_continuous(labels = scales::percent, limits = c(0,1)) +
    geom_text(aes(label = ratio), vjust = -1)
    if (save) {ggsave(paste0("plots/exp2_hist_cat_",variable,"_alg4.pdf"), width = 9, height = 5)}
  return(plt)
}

for (n in categ){
  plot(make_hist(n,D2, F))
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
#(tab <- make_cross_correlation(adult.data, "sex","workclass"))
#knitr::kable(tab,format = "latex", linesep = "", digits = 3, booktabs = T) %>% print()
#mosaicplot(tab, main = "")
#(tab <- make_cross_correlation(adult.data, "sex","race"))
#knitr::kable(tab,format = "latex", linesep = "", digits = 3, booktabs = T) %>% print()
#(tab <- make_cross_correlation(adult.data, "sex","relationship"))
#knitr::kable(tab,format = "latex", linesep = "", digits = 3, booktabs = T) %>% print()
#(tab <- make_cross_correlation(adult.data, "sex","marital_status"))
#knitr::kable(tab,format = "latex", linesep = "", digits = 3, booktabs = T) %>% print()

# Make mosaic plot instead of two-way tables, perhaps more informative.
make_mosaic_plot <- function(data, first.feat, second.feat, exp.num, vers.num){
  plt <- data[,categ] %>% ggplot() +
    geom_mosaic(aes(x = product(.data[[first.feat]]), fill = .data[[second.feat]])) +
    theme_minimal() +
    scale_fill_grey()
  ggsave(paste0("plots/mosaic/",exp.num,"_",first.feat,"_",second.feat,"_",vers.num,".pdf"), width = 9, height = 5)
  return(plt) # Not sure why not working at the moment. 
}
#print(make_mosaic_plot(adult.data,"sex","race","adult_data","bin"))

# I will do it manually instead, since the function did not work for the mosaic plot (a specific problem here).
plt <- D2 %>% ggplot() +
  geom_mosaic(mapping = aes(x = product(sex), fill = occupation)) +
  theme_minimal() +
  scale_fill_grey()
print(plt)
ggsave("plots/mosaic/exp2_race_relationship_cat_alg4.pdf", width = 9, height = 5)

#### Make Q-Q plots for comparisons.
make_qqplot <- function(OG.data, gen.data, feature, exp.num, save){
  # Make qq-plots between generated data and original data. 
  # Always assume that the generated data set is larger (which is it in our experiments).

  adult <- (OG.data %>% dplyr::select(feature))[[1]] # This is assumed smaller. 
  gen <- (gen.data %>% dplyr::select(feature))[[1]] # This is assumed larger.
  
  # When the sizes differ, the function qqplot interpolates between the sorted values of the larger set.
  # That is, we estimate nrow(OG.data) quantiles from nrow(gen.data) by interpolating between the points. 
  # Essentially, the dimension of the largest data set is reduced using linear approximations between the points. 
  q <- qqplot((adult.data %>% dplyr::select(feature))[[1]],(gen.data %>% dplyr::select(feature))[[1]], plot.it = F)
  
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
  if (save) {ggsave(paste0("plots/qq/",exp.num,"_",feature,"_cat_alg4",".pdf"), width = 9, height = 5)}
  return(plt)
}

for (n in cont){
  # Make qqplots for all features in exp1 generated data. 
  make_qqplot(adult.data, D2, n, "exp1", T)
                                        # Need to change the second data set and the last name in order 
                                        # to make the qqplots for different experiments. 
}
