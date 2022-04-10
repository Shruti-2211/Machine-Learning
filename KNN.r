library(e1071)        #SVM 
library(caret)      #Classification and Reession Package
library(class)

#Load iris data
data(iris)
View(iris)
library(Hmisc)
describe(iris)

#Split data
split.data <- sample.split(iris, SplitRatio = 0.70)

#train data
train.set <- subset(iris, split.data = TRUE)

#Test data
test.set <- subset(iris, split.data = FALSE)

#Scaling
train.scale <- scale(train.set[,1:4])
test.scale <- scale(test.set[,1:4])

#Create knn model for different k values
#k = 1
knn.model <- knn(train = train.scale, test = test.scale, cl = train.set$Species, k = 1)
confusionMatrix(test.set$Species, knn.model)
plot(knn.model)

#k = 3
knn.model1 <- knn(train = train.scale, test = test.scale, cl = train.set$Species, k = 3)
confusionMatrix(test.set$Species, knn.model1)
plot(knn.model1)

#k = 5
knn.model2 <- knn(train = train.scale, test = test.scale, cl = train.set$Species, k = 5)
confusionMatrix(test.set$Species, knn.model2)
plot(knn.model2)

#k = 7
knn.model3 <- knn(train = train.scale, test = test.scale, cl = train.set$Species, k = 7)
confusionMatrix(test.set$Species, knn.model3)
plot(knn.model3)

#k = 10
knn.model4 <- knn(train = train.scale, test = test.scale, cl = train.set$Species, k = 10)
confusionMatrix(test.set$Species, knn.model4)
plot(knn.model3)

#tune test data
#tune.knn is do for multiple values of k and find best one and display performance result
tune.data <- tune.knn(iris[,1:4], iris[,5], k = 1:5, tunecontrol = tune.control(sampling = "boot"))
summary(tune.data)
plot(tune.data)
