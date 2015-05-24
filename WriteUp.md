# PML-Course Project
Adrian CG  
22 Feb 2015  


```
## Loading required package: lattice
## Loading required package: ggplot2
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

## Data Preprocessing
On the first glance into the data, we observe a lot of wrongly concluded data types for the scalar measurement columns (ranging from 8 to 159). Thus, we coerce them to the appropriate numeric classes:


```r
for (i in 8:159) {
    pmlTraining[,i] <- as.numeric(pmlTraining[,i])
    pmlTesting[,i] <- as.numeric(pmlTesting[,i])
}
str(pmlTraining[,c(1:8,159,160)])
```

```
## 'data.frame':	19622 obs. of  10 variables:
##  $ X                   : int  1 2 3 4 5 6 7 8 9 10 ...
##  $ user_name           : Factor w/ 6 levels "adelmo","carlitos",..: 2 2 2 2 2 2 2 2 2 2 ...
##  $ raw_timestamp_part_1: int  1323084231 1323084231 1323084231 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 ...
##  $ raw_timestamp_part_2: int  788290 808298 820366 120339 196328 304277 368296 440390 484323 484434 ...
##  $ cvtd_timestamp      : Factor w/ 20 levels "02/12/2011 13:32",..: 9 9 9 9 9 9 9 9 9 9 ...
##  $ new_window          : Factor w/ 2 levels "no","yes": 1 1 1 1 1 1 1 1 1 1 ...
##  $ num_window          : int  11 11 11 12 12 12 12 12 12 12 ...
##  $ roll_belt           : num  1.41 1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.43 1.45 ...
##  $ magnet_forearm_z    : num  476 473 469 469 473 478 470 474 476 473 ...
##  $ classe              : Factor w/ 5 levels "A","B","C","D",..: 1 1 1 1 1 1 1 1 1 1 ...
```


## Prediction
### Data-Splitting
To asses our model we split our original training data in to 60% real training and 40% into a validation set, which we we later use for model assessment.

```r
inValidation <- createDataPartition(pmlTraining$classe, p=0.4, list=FALSE)
```

### Model Building
#### Feature Selection
After manual inspection of the entries in the training set, we found some samples at the end of each execrise window containing a full set of the derived variables, while others (beginning and within windows) only contain the base measurements. For classification purposes of a whole movement as mentioned in the original paper (Velloso, Bulling, Gellersen, Ugulino, and Fuks, 2013), the derived variables seem to be a better fit. However, sine they are rather sparse and can't be recomputed for single points within the exercise repetition, we constrain ourselves to the variables that don't contain `NA`s.


```r
measuredColumns <- !apply(is.na(pmlTesting), 2, any)
```

To check the existing data for some zero variance abnomalities, that could also negatively influence the prediction, we use the zero variance checks of the `caret` package and will subsequently remove them.

```r
nzvPml <- nearZeroVar(pmlTraining, saveMetrics= TRUE)
```

Likewise, for certain prediction algorithms, which are not model bound, it would be unwise to leave predictors in the dataset, that have unique values for most entries like the `X`, `new_window`, `num_window`, `raw_timestamp_part_1` that could lead to overfitting. Also `cvtd_timestamp` is somewhat synonymous to `user_name`. Thus we retain as simplified dataset:


```r
removeOverfitters <- rep(TRUE,160); removeOverfitters[c(1,3,5,7)] = FALSE;
pmlTrain <- pmlTraining[-inValidation, !(nzvPml$zeroVar | nzvPml$nzv) & measuredColumns & removeOverfitters]
pmlEval <- pmlTraining[inValidation, !(nzvPml$zeroVar | nzvPml$nzv) & measuredColumns & removeOverfitters]
pmlTest <- pmlTesting[, !(nzvPml$zeroVar | nzvPml$nzv) & measuredColumns & removeOverfitters]
```

### Algorithm Selection
We have been trying to get a successful computation run using the `caret` package and any of the many models available therein, however all tried algorithms (using the parallel computation routines available) ended up consuming all available ressources (some CPU-bound, some Memory-Bound) of our Late 2013 MacBook Pro (4 cores, 16GB). Thus, we resolved to a "simpler" model using random forests from the `randomForest` package.

### Computing Model & Assessing Prediction Accuracy
Since `randomForest` allows for computation of the model combined with an assessment of the prediction accuracy for a "test set" (here it will be our `pmlEval` set) in one pass, we compute that jointly:

```r
regressors <- !(colnames(pmlTrain)== "classe");
halRF <- randomForest(
    x=pmlTrain[,regressors], y=pmlTrain[,"classe"],
    xtest=pmlEval[,regressors], ytest=pmlEval[,"classe"],
    importance=TRUE, proximity=TRUE, keep.forest=TRUE
);
print(halRF)
```

```
## 
## Call:
##  randomForest(x = pmlTrain[, regressors], y = pmlTrain[, "classe"],      xtest = pmlEval[, regressors], ytest = pmlEval[, "classe"],      importance = TRUE, proximity = TRUE, keep.forest = TRUE) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.64%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3345    2    0    0    1   0.0008961
## B   16 2255    7    0    0   0.0100966
## C    0   12 2037    4    0   0.0077935
## D    0    0   26 1902    1   0.0139969
## E    0    0    2    4 2158   0.0027726
##                 Test set error rate: 0.61%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 2231    1    0    0    0    0.000448
## B   10 1507    2    0    0    0.007900
## C    0    6 1363    0    0    0.004383
## D    0    0   20 1266    1    0.016317
## E    0    0    1    7 1435    0.005544
```

Thus the compelling results for this test run using 60% of the training set as trainign, 40% as evaluation leads to a prediction accuracy estimate of the *out of bag error estimate from 0.61%* and an *error rate estimation on our evaluation set of 0.65%*.

### Application on Original Test-Dataset
Hence, we are now ready to to apply the generated model towards the "real" test set

```r
pmlTestPrediction <- predict(halRF, newdata=pmlTest, proximity=TRUE)
print(pmlTestPrediction$predicted)
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```
and generate the submission output

```r
pml_write_files(pmlTestPrediction$predicted)
```

## References
[1] E. Velloso, a. Bulling, H. Gellersen, et al. "Qualitative
activity recognition of weight lifting exercises". In: _Proceeding
AH '13 Proceedings of the 4th Augmented Human International
Conference_ (2013), pp. 116-123. DOI: 10.1145/2459236.2459256.
[1] E. Velloso, a. Bulling, H. Gellersen, et al. "Qualitative
activity recognition of weight lifting exercises". In: _Proceeding
AH '13 Proceedings of the 4th Augmented Human International
Conference_ (2013), pp. 116-123. DOI: 10.1145/2459236.2459256.

## Appendix
### Raw Usable Parameter List

```
##  chr [1:55] "user_name" "raw_timestamp_part_2" "roll_belt" "pitch_belt" "yaw_belt" "total_accel_belt" "gyros_belt_x" "gyros_belt_y" "gyros_belt_z" "accel_belt_x" "accel_belt_y" "accel_belt_z" "magnet_belt_x" "magnet_belt_y" "magnet_belt_z" "roll_arm" "pitch_arm" "yaw_arm" "total_accel_arm" "gyros_arm_x" "gyros_arm_y" "gyros_arm_z" "accel_arm_x" "accel_arm_y" "accel_arm_z" "magnet_arm_x" "magnet_arm_y" "magnet_arm_z" "roll_dumbbell" "pitch_dumbbell" "yaw_dumbbell" "total_accel_dumbbell" "gyros_dumbbell_x" "gyros_dumbbell_y" "gyros_dumbbell_z" "accel_dumbbell_x" "accel_dumbbell_y" "accel_dumbbell_z" "magnet_dumbbell_x" "magnet_dumbbell_y" "magnet_dumbbell_z" "roll_forearm" "pitch_forearm" "yaw_forearm" "total_accel_forearm" "gyros_forearm_x" "gyros_forearm_y" "gyros_forearm_z" "accel_forearm_x" "accel_forearm_y" "accel_forearm_z" "magnet_forearm_x" "magnet_forearm_y" "magnet_forearm_z" "classe"
```

