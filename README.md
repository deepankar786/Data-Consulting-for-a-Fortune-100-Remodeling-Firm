Improving Model Accuracy with Probability Scoring
Machine Learning Models

Binary classification problems are exceedingly
common across corporations, regardless of their industry. Popular
examples include classifying a patient as high-risk or low-risk or
predicting if a client will convert or not. The motivation for this
research is determining techniques to improve prediction
accuracy for operationalized models. Collaborating with a
national partner, we conducted feature importance tests and
engineering experiments to isolate industry-agnostic factors with
the most significant impact on the conversion rate. We also use
probability scoring to highlight incremental changes in accuracy
as we applied several improvement techniques to determine which
would significantly increases a model’s predictive power. We
compare five algorithms: XGBoost, LGBoost, CatBoost, and
MLP, and an ensemble of all four, while our results highlight the
superior accuracy of the ensemble, with a final log loss value of
0.5784. We also note that the highest levels of improvement in log
loss occurs at the beginning of the process, after downsampling
and using engineered custom metrics as inputs to the models. 
