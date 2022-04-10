library(arules)
library(arulesViz)
library(RColorBrewer)

File_Base_Path <- "E:/Data"
File_Path <- "lab-basket-analysis"
File_Name <- "Market_Basket_Optimisation.csv"

File_Location_Data <- paste(c(File_Base_Path,File_Path,File_Name), collapse = "/")
transaction_data <- read.transactions(File_Location_Data, sep = ",")

#Review of data
View(transaction_data)

## Step-2:
#  You can review summary of the transactions
summary(transaction_data)

#Absolute Frequency Plot
itemFrequencyPlot(transaction_data, topN = 25, col = brewer.pal(8, 'Pastel2'),
                  main = 'Absolute Item Frequency Plot', type = 'absolute',
                  ylab = 'Item Frequency (Absolute)', xlab = 'Items')

#Relative Frequency Plot
itemFrequencyPlot(transaction_data, topN = 25, col = brewer.pal(8, 'Pastel2'),
                  main = 'Relative Item Frequency Plot', type = 'relative',
                  ylab = 'Item Frequency (Relative)', xlab = 'Items')

##Step-3.
#Prepare and clean data
#Create a csv file contains transaction ID and items 
set.seed(345)
orders <- data.frame(transactionID = sample(789:90768, 7501, replace = T), 
                     items = paste("item", sample(1:3456, 7501, replace = T), sep = " "))   

#Write into the csv file
write.csv(orders,"prepare_transactions_data.csv")

#read it as transaction 
store_transaction <- read.transactions(file = "prepare_transactions_data.csv", 
                                       format = "single",
                                       sep = ",", cols = c("transactionID","items"), 
                                       rm.duplicates = T, header = TRUE)

#Create unique value with TRUE 
orders$const = TRUE
View(orders)
orders <- unique(orders)

#Shape the data as items names spread into columns
orders_mat_prep <- reshape(data = orders, idvar = "transactionID", timevar = "items", 
                           direction = "wide")

#Drop transaction column
orders_matrix_prep <- as.matrix(orders_mat_prep[,-1])

#Fill the random value for empty cell
orders_matrix_prep[is.na(orders_matrix_prep)] <- FALSE

#Correct column names
colnames(orders_matrix_prep) <- gsub(x = colnames(orders_matrix_prep), 
                                     pattern = "const\\.", replacement = "")

#Convert into transaction
order_trans2 <- as(orders_matrix_prep, "transactions")
View(order_trans2)

## Step-4
#--------------------------------------- Rule 1---------------------------------------------------------------
## Create Model and run apriori analysis for Association Rules mining
set.seed = 220
rules = apriori(transaction_data, parameter = list(support = 0.004, confidence = 0.2 , minlen = 3 ))


### Step-5 : Analyze the results
inspect_item_rules = inspect(sort(rules[1:10], by = 'lift')[1:9])


### Step-6: Visualize the results
plot(rules, method='grouped' , measure='confidence', shading = 'lift')

#---------------------------------------- Rule 2---------------------------------------------------------------

## Let's manipulate apriori usage
#1. Change Confidence and Support in rules modeling
rule1 <- apriori(transaction_data, parameter = list(support = 0.01, conf = 0.2))
summary(rule1)

#Inspect Top 25 rules
inspect_item_rule1 <- inspect(rule1[1:25])

#Filter rule as per confidence condition
filter_rule <- rule1[quality(rule1)$confidence>0.2]

#Visualization
plot(rule1[1:10], method = 'graph', engine = 'htmlwidget')
plot(filter_rule, method = 'two-key plot')

#----------------------------------------------- Rule 3 ------------------------------------------------

# 2. We can control LHS or RHS of the rules
rule2 <- apriori(transaction_data, parameter = list(supp = 0.001, conf = 0.8, maxlen = 7),
                 appearance = list(default = 'lhs', rhs = 'mineral water'))

inspect_rule2 <- inspect(head(rule2))

#Visualization
plot(rule2[1:20], method = "paracoord" )

#------------------------------------------ Rule 4 --------------------------------------------------

#3.1 Add arem parameter in association rule
rule3 <- apriori(transaction_data, parameter = list(supp = 0.0001, conf = 0.5, arem = "chi2"))
subrule3 <- head(rule3, n=10, by = 'lift')
inspect_rule3 <- inspect(rule3[1:20])

#Visualization
plot(subrule3, method = 'graph', engine = 'htmlwidget')

#-------------------------------------- Rule 5 -------------------------------------------------

# Remove Redundant rules
sub_rules <- which(colSums(is.subset(rules,rules))>1) 
print(sub_rules[1:20])
sub_select_rules <- inspect(rules[-sub_rules])

#---------------------------------------------------------------------------------------------------

#Conclusion : Mineral Water, eggs, spaghetti, French Fries, Chocolate are bought maximum with different combination of items 
# Also these 3 give major counties gives major association even on applying different rules.
#Therefore, it is profitable to put these things on shelf for display for maximum sale.
