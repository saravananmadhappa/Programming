#loading libraries
library(knitr) library(caret) library(rpart) library(rpart.plot) library(rattle) library(randomForest) library(corrplot) set.seed(12345)
# set the URL for the download UrlTraining <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv" UrlTesting  <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"  # download the datasets training <- read.csv(url(UrlTraining)) testing  <- read.csv(url(UrlTesting))  # create a partition with the training dataset  inTrain  <- createDataPartition(training$classe, p=0.7, list=FALSE) TrainSet <- training[inTrain, ] TestSet  <- training[-inTrain, ] dim(TrainSet)
## [1] 13737   160
dim(TestSet)
## [1] 5885  160
# remove variables with Nearly Zero Variance NZV <- nearZeroVar(TrainSet) TrainSet <- TrainSet[, -NZV] TestSet  <- TestSet[, -NZV] dim(TrainSet)
## [1] 13737   106
dim(TestSet)
## [1] 5885  106
# remove variables that are mostly NA NoNA    <- sapply(TrainSet, function(x) mean(is.na(x))) > 0.95 TrainSet <- TrainSet[, NoNA==FALSE] TestSet  <- TestSet[, NoNA==FALSE] dim(TrainSet)
## [1] 13737    59
dim(TestSet)
## [1] 5885   59
# remove identification only variables (columns 1 to 5) TrainSet <- TrainSet[, -(1:5)] TestSet  <- TestSet[, -(1:5)] dim(TrainSet)
## [1] 13737    54
dim(TestSet)
## [1] 5885   54
#A correlation among variables is analysed before proceeding to the modeling procedures.
corMatrix <- cor(TrainSet[, -54]) corrplot(corMatrix, order = "FPC", method = "color", type = "lower",           tl.cex = 0.8, tl.col = rgb(0, 0, 0))

#The highly correlated variables are shown in dark colors in the graph above. To make an evem more compact analysis, a PCA (Principal Components Analysis) could be performed as pre-processing step to the datasets. Nevertheless, as the correlations are quite few, this step will not be applied for this assignment.

#Three methods will be applied to model the regressions (in the Train dataset) and the best one (with higher accuracy when applied to the Test dataset) will be used for the quiz predictions. The methods are: Random Forests, Decision Tree and Generalized Boosted Model, as described below.
#A Confusion Matrix is plotted at the end of each analysis to better visualize the accuracy of the models.
#Method: Random Forest
# model fit set.seed(12345) controlRF <- trainControl(method="cv", number=3, verboseIter=FALSE) modFitRandForest <- train(classe ~ ., data=TrainSet, method="rf",                           trControl=controlRF) modFitRandForest$finalModel
##  ## Call: ##  randomForest(x = x, y = y, mtry = param$mtry)  ##                Type of random forest: classification ##                      Number of trees: 500 ## No. of variables tried at each split: 27 ##  ##         OOB estimate of  error rate: 0.2% ## Confusion matrix: ##      A    B    C    D    E  class.error ## A 3904    1    0    0    1 0.0005120328 ## B    7 2650    1    0    0 0.0030097818 ## C    0    5 2391    0    0 0.0020868114 ## D    0    0    7 2244    1 0.0035523979 ## E    0    0    0    5 2520 0.0019801980
# prediction on Test dataset predictRandForest <- predict(modFitRandForest, newdata=TestSet) confMatRandForest <- confusionMatrix(predictRandForest, TestSet$classe) confMatRandForest
## Confusion Matrix and Statistics ##  ##           Reference ## Prediction    A    B    C    D    E ##          A 1674    5    0    0    0 ##          B    0 1133    3    0    0 ##          C    0    1 1023    9    0 ##          D    0    0    0  955    4 ##          E    0    0    0    0 1078 ##  ## Overall Statistics ##                                            ##                Accuracy : 0.9963           ##                  95% CI : (0.9943, 0.9977) ##     No Information Rate : 0.2845           ##     P-Value [Acc > NIR] : < 2.2e-16        ##                                            ##                   Kappa : 0.9953           ##  Mcnemar's Test P-Value : NA               ##  ## Statistics by Class: ##  ##                      Class: A Class: B Class: C Class: D Class: E ## Sensitivity            1.0000   0.9947   0.9971   0.9907   0.9963 ## Specificity            0.9988   0.9994   0.9979   0.9992   1.0000 ## Pos Pred Value         0.9970   0.9974   0.9903   0.9958   1.0000 ## Neg Pred Value         1.0000   0.9987   0.9994   0.9982   0.9992 ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839 ## Detection Rate         0.2845   0.1925   0.1738   0.1623   0.1832 ## Detection Prevalence   0.2853   0.1930   0.1755   0.1630   0.1832 ## Balanced Accuracy      0.9994   0.9971   0.9975   0.9949   0.9982
# plot matrix results plot(confMatRandForest$table, col = confMatRandForest$byClass,       main = paste("Random Forest - Accuracy =",                   round(confMatRandForest$overall['Accuracy'], 4)))

#Method: Decision Trees
# model fit set.seed(12345) modFitDecTree <- rpart(classe ~ ., data=TrainSet, method="class") fancyRpartPlot(modFitDecTree)

# prediction on Test dataset predictDecTree <- predict(modFitDecTree, newdata=TestSet, type="class") confMatDecTree <- confusionMatrix(predictDecTree, TestSet$classe) confMatDecTree
## Confusion Matrix and Statistics ##  ##           Reference ## Prediction    A    B    C    D    E ##          A 1530  269   51   79   16 ##          B   35  575   31   25   68 ##          C   17   73  743   68   84 ##          D   39  146  130  702  128 ##          E   53   76   71   90  786 ##  ## Overall Statistics ##                                           ##                Accuracy : 0.7368          ##                  95% CI : (0.7253, 0.748) ##     No Information Rate : 0.2845          ##     P-Value [Acc > NIR] : < 2.2e-16       ##                                           ##                   Kappa : 0.6656          ##  Mcnemar's Test P-Value : < 2.2e-16       ##  ## Statistics by Class: ##  ##                      Class: A Class: B Class: C Class: D Class: E ## Sensitivity            0.9140  0.50483   0.7242   0.7282   0.7264 ## Specificity            0.9014  0.96650   0.9502   0.9100   0.9396 ## Pos Pred Value         0.7866  0.78338   0.7543   0.6131   0.7305 ## Neg Pred Value         0.9635  0.89051   0.9422   0.9447   0.9384 ## Prevalence             0.2845  0.19354   0.1743   0.1638   0.1839 ## Detection Rate         0.2600  0.09771   0.1263   0.1193   0.1336 ## Detection Prevalence   0.3305  0.12472   0.1674   0.1946   0.1828 ## Balanced Accuracy      0.9077  0.73566   0.8372   0.8191   0.8330
# plot matrix results plot(confMatDecTree$table, col = confMatDecTree$byClass,       main = paste("Decision Tree - Accuracy =",                   round(confMatDecTree$overall['Accuracy'], 4)))

# Method: Generalized Boosted Model
# model fit set.seed(12345) controlGBM <- trainControl(method = "repeatedcv", number = 5, repeats = 1) modFitGBM  <- train(classe ~ ., data=TrainSet, method = "gbm",                     trControl = controlGBM, verbose = FALSE) modFitGBM$finalModel
## A gradient boosted model with multinomial loss function. ## 150 iterations were performed. ## There were 53 predictors of which 41 had non-zero influence.
# prediction on Test dataset predictGBM <- predict(modFitGBM, newdata=TestSet) confMatGBM <- confusionMatrix(predictGBM, TestSet$classe) confMatGBM
## Confusion Matrix and Statistics ##  ##           Reference ## Prediction    A    B    C    D    E ##          A 1669   13    0    2    0 ##          B    4 1113   23    5    3 ##          C    0   13  998   16    2 ##          D    1    0    5  941    8 ##          E    0    0    0    0 1069 ##  ## Overall Statistics ##                                            ##                Accuracy : 0.9839           ##                  95% CI : (0.9803, 0.9869) ##     No Information Rate : 0.2845           ##     P-Value [Acc > NIR] : < 2.2e-16        ##                                            ##                   Kappa : 0.9796           ##  Mcnemar's Test P-Value : NA               ##  ## Statistics by Class: ##  ##                      Class: A Class: B Class: C Class: D Class: E ## Sensitivity            0.9970   0.9772   0.9727   0.9761   0.9880 ## Specificity            0.9964   0.9926   0.9936   0.9972   1.0000 ## Pos Pred Value         0.9911   0.9695   0.9699   0.9853   1.0000 ## Neg Pred Value         0.9988   0.9945   0.9942   0.9953   0.9973 ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839 ## Detection Rate         0.2836   0.1891   0.1696   0.1599   0.1816 ## Detection Prevalence   0.2862   0.1951   0.1749   0.1623   0.1816 ## Balanced Accuracy      0.9967   0.9849   0.9832   0.9866   0.9940
# plot matrix results plot(confMatGBM$table, col = confMatGBM$byClass,       main = paste("GBM - Accuracy =", round(confMatGBM$overall['Accuracy'], 4)))


#Applying the Selected Model to the Test Data
#The accuracy of the 3 regression modeling methods above are:
#Random Forest : 0.9963
#Decision Tree : 0.7368
#GBM : 0.9839
#In that case, the Random Forest model will be applied to predict the 20 quiz results (testing dataset) as shown below.
predictTEST <- predict(modFitRandForest, newdata=testing) predictTEST
##  [1] B A B A A E D B A A B C B A E E A B B B ## Levels: A B C D E


