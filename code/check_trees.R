# We check if the marginal distributions seem to be modeled sufficiently well.

# FIRST WE CHECK THE TREES THAT ARE BUILT FOR THE NORMALIZED DATA!

# Need to check the trees for both categorical and binarized data!
#load("data/adult_data_categ.RData", verbose = T) 
#load("data/adult_data_binarized.RData", verbose = T) 

# Tree 1. We check the root nodes. 
(tab1 <- table(adult.data %>% filter(age < 0.27) %>% select(workclass)))
tab1[1]/sum(tab1)
tab1[2]/sum(tab1)

(tab2 <- table(adult.data %>% filter(age >= 0.27 & age < 0.32) %>% select(workclass)))
tab2[1]/sum(tab2)
tab2[2]/sum(tab2)

(tab3 <- table(adult.data %>% filter(age >= 0.32 & sex == " Female") %>% select(workclass)))
tab3[1]/sum(tab3)
tab3[2]/sum(tab3)

(tab4 <- table(adult.data %>% filter(age >= 0.32 & age < 0.64 & sex == " Male") %>% select(workclass)))
tab4[1]/sum(tab4)
tab4[2]/sum(tab4)

(tab5 <- table(adult.data %>% filter(age >= 0.64 & sex == " Male") %>% select(workclass)))
tab5[1]/sum(tab5)
tab5[2]/sum(tab5)
# Looks good!

# Tree 2. We check the root nodes. 
mean((adult.data %>% filter(age >= 0.31) %>% select(fnlwgt))$fnlwgt)
mean((adult.data %>% filter(age < 0.31) %>% select(fnlwgt))$fnlwgt)
# The tree looks ok, but is does not use workclass or sex. 

# Tree 3. We check the root nodes. 
mean((adult.data %>% filter(age < 0.0068) %>% select(education_num))$education_num)
mean((adult.data %>% filter(age >= 0.0068 & age < 0.021) %>% select(education_num))$education_num)
mean((adult.data %>% filter(age >= 0.021 & age < 0.062) %>% select(education_num))$education_num)
mean((adult.data %>% filter(age >= 0.0068 & workclass == " Private" & age > 0.47) %>% select(education_num))$education_num)




# TREES BUILT FOR UN-NORMALIZED DATA! THESE TREES LOOK THE SAME, BUT WITH THE NORMALIZED DATA UN-NORMALIZED!
# Tree 1. We check the root nodes. 
(tab1 <- table(adult.data %>% filter(age < 37) %>% select(workclass)))
tab1[1]/sum(tab1)
tab1[2]/sum(tab1)

(tab2 <- table(adult.data %>% filter(age >= 37 & age < 41) %>% select(workclass)))
tab2[1]/sum(tab2)
tab2[2]/sum(tab2)

(tab3 <- table(adult.data %>% filter(age >= 41 & sex == " Female") %>% select(workclass)))
tab3[1]/sum(tab3)
tab3[2]/sum(tab3)

(tab4 <- table(adult.data %>% filter(age >= 41 & age < 64 & sex == " Male") %>% select(workclass)))
tab4[1]/sum(tab4)
tab4[2]/sum(tab4)

(tab5 <- table(adult.data %>% filter(age >= 89 & sex == " Male") %>% select(workclass)))
tab5[1]/sum(tab5)
tab5[2]/sum(tab5)

(tab5 <- table(adult.data %>% filter(age <75 & age < 67 & age >= 64 & sex == " Male") %>% select(workclass)))
tab5[1]/sum(tab5)
tab5[2]/sum(tab5)
# Looks good!

# Tree 2. We check the root nodes. 
mean((adult.data %>% filter(age >= 40) %>% select(fnlwgt))$fnlwgt)
mean((adult.data %>% filter(age < 40) %>% select(fnlwgt))$fnlwgt)
# The tree looks ok, but is does not use workclass or sex. 

# Tree 3. We check the root nodes. 
mean((adult.data %>% filter(age < 18) %>% select(education_num))$education_num)
mean((adult.data %>% filter(age >= 18 & age < 19) %>% select(education_num))$education_num)
mean((adult.data %>% filter(age >= 19 & age < 22) %>% select(education_num))$education_num)
mean((adult.data %>% filter(age >= 22 & workclass == " Private" & age >= 52) %>% select(education_num))$education_num)
mean((adult.data %>% filter(age >= 22 & workclass == " Private" & age < 52 & fnlwgt >= 209e+3) %>% select(education_num))$education_num)
mean((adult.data %>% filter(age >= 22 & workclass == " Private" & age < 52 & fnlwgt < 209e+3) %>% select(education_num))$education_num)
mean((adult.data %>% filter(age >= 22 & workclass != " Private" & age >= 58) %>% select(education_num))$education_num)
mean((adult.data %>% filter(age >= 22 & workclass != " Private" & age < 39) %>% select(education_num))$education_num)
mean((adult.data %>% filter(age >= 22 & workclass != " Private" & age >= 51) %>% select(education_num))$education_num)
mean((adult.data %>% filter(age >= 22 & workclass != " Private" & age < 51) %>% select(education_num))$education_num)
# Looks good!!

# Tree 4. We check the root nodes.
(tab1 <- table(adult.data %>% filter(age >= 28 & sex == " Male") %>% select(marital_status)))
tab1[1]/sum(tab1)
tab1[2]/sum(tab1)

(tab2 <- table(adult.data %>% filter(age < 28 & sex == " Male") %>% select(marital_status)))
tab2[1]/sum(tab2)
tab2[2]/sum(tab2)

(tab3 <- table(adult.data %>% filter(sex == " Female") %>% select(marital_status)))
tab3[1]/sum(tab3)
tab3[2]/sum(tab3)

# Tree 5. We check the root nodes. 
(tab <- table(adult.data %>% select(occupation)))
tab[1]/sum(tab)
tab[2]/sum(tab)

# Tree 6. We check the root nodes. 
(tab1 <- table(adult.data %>% filter(marital_status != " Married-civ-spouse") %>% select(relationship)))
tab1[1]/sum(tab1)
tab1[2]/sum(tab1)

(tab2 <- table(adult.data %>% filter(marital_status == " Married-civ-spouse" & sex == " Male") %>% select(relationship)))
tab2[1]/sum(tab2)
tab2[2]/sum(tab2)

(tab3 <- table(adult.data %>% filter(marital_status == " Married-civ-spouse" & sex != " Male") %>% select(relationship)))
tab3[1]/sum(tab3)
tab3[2]/sum(tab3)

# Tree 7.
(tab <- table(adult.data %>% select(race)))
tab[1]/sum(tab)
tab[2]/sum(tab)

# Tree 8. This tree is large!

# Tree 9. This tree is also quite large!
mean((adult.data %>% filter(education_num < 13 & marital_status == " Married-civ-spouse" & capital_gain >= 587) %>% select(capital_loss))$capital_loss)
mean((adult.data %>% filter(education_num < 13 & marital_status == " Married-civ-spouse" & capital_gain < 586.5) %>% select(capital_loss))$capital_loss)
# Looks ok!

# Tree 10. This tree is also quite large!

# Tree 11. This tree is also quite large!



###### Categorical data.
# Tree 1. We check the root nodes. 
(tab1 <- table(adult.data %>% select(workclass)))
tab1[1]/sum(tab1)
tab1[2]/sum(tab1)
tab1[3]/sum(tab1)
tab1[4]/sum(tab1)
tab1[5]/sum(tab1)
tab1[6]/sum(tab1)
tab1[7]/sum(tab1)

# Tree 2. We check the root nodes. 
mean((adult.data %>% filter(age >= 40) %>% select(fnlwgt))$fnlwgt)
mean((adult.data %>% filter(age < 40) %>% select(fnlwgt))$fnlwgt)
# The tree looks ok, but is does not use workclass or sex. 

# Tree 3. We check the root nodes. 
mean((adult.data %>% filter(age < 18 & workclass %in% c(" Private"," Without_pay"," Self-emp-inc")) %>% select(education_num))$education_num)
mean((adult.data %>% filter(age >= 18 & age < 19 & workclass %in% c(" Private"," Without_pay"," Self-emp-inc")) %>% select(education_num))$education_num)
mean((adult.data %>% filter(age >= 19 & age >= 52 & workclass %in% c(" Private"," Without_pay"," Self-emp-inc")) %>% select(education_num))$education_num)
mean((adult.data %>% filter(age >= 19 & age < 52 & age < 23 & workclass %in% c(" Private"," Without_pay"," Self-emp-inc")) %>% select(education_num))$education_num)
mean((adult.data %>% filter(age >= 19 & age < 52 & age >= 23 & workclass %in% c(" Private"," Without_pay"," Self-emp-inc")) %>% select(education_num))$education_num)
# Looks ok!

# Tree 7
(tab1 <- table(adult.data %>% select(race)))
tab1[1]/sum(tab1)
tab1[2]/sum(tab1)
tab1[3]/sum(tab1)
tab1[4]/sum(tab1)
tab1[5]/sum(tab1)
# Looks fine!
