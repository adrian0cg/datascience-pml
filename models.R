plmTrainControl <- trainControl(
    method="LOOCV", p=0.9,
    number=4, classProbs=TRUE,
    verboseIter=TRUE
)

adaFit <- train(
    classe ~., data=pmlTrain[inValidation[1],],
    preProc=c("center","scale"),
    method="ada",
    trControl=plmTrainControl,
    metric="ROC"
)

svmFit <- train(classe ~., data=pmlTrain, method="svmPoly",preProc=c("center","scale"),tuneLength=8, metric="ROC")