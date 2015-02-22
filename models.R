plmTrainControl <- trainControl(method="LOOCV", p=0.9, number=10, classProbs=TRUE)

svmFit <- train(classe ~., data=pmlTrain, method="svmPoly",preProc=c("center","scale"),tuneLength=8, metric="ROC")