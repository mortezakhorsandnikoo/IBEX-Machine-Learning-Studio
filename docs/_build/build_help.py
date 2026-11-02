"""Build the offline guide from reviewed, app-specific instructions."""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'Help'
(OUT / 'Images').mkdir(parents=True, exist_ok=True)
topics = []
def p(text): return {'Kind':'paragraph', 'Text':text}
def h(text): return {'Kind':'heading', 'Text':text}
def note(text): return {'Kind':'note', 'Text':text}
def steps(*items): return {'Kind':'steps', 'Items':items}
def bullets(*items): return {'Kind':'bullets', 'Items':items}
def image(name, caption): return {'Kind':'image', 'Image':name, 'Caption':caption}
def topic(id, category, title, summary, blocks, related=(), keywords=''):
    topics.append(dict(Id=id, Category=category, Title=title, Summary=summary,
                       Keywords=keywords, Blocks=blocks, Related=related))

topic('start','Getting started','Welcome to IBEX Machine Learning Studio',
      'A practical guide from your first CSV to a trained model, predictions and exported results.', [
    image('home.png','The Home page introduces the studio and its workflow.'),
    h('Your first project'),
    steps('Open Dataset and import a CSV with a header row and numeric data.',
          'Inspect the preview, distributions and correlation matrix. Check what each column means and its units.',
          'Use Prepare data to inspect missing values, scale features or apply PCA when appropriate.',
          'Choose Regression for a continuous target, Classification for categories, or Clustering to find groups without a target.',
          'Select your settings, train, and inspect held-out results. Use K-fold evaluation when you need to see performance across several splits.',
          'Enter original feature values in Predict. Save the model where that option is available, and export the figures and tables you need.'),
    h('Find an answer'),
    p('Open Help and user guide in the sidebar to browse all topics. Type words into Search the user guide: for example PCA prediction, K-fold, missing values, or epsilon. All entered words must occur in a topic. Use the category filter to narrow results, or Clear to see everything again. Matching text is highlighted in the article. Ctrl+F focuses search. Press F1 from a workspace page to open its instructions. Click an illustration to open it with a zoom slider; press Esc to close it. Back to workspace returns to your previous page without clearing its results.'),
    note('This guide and its illustrations are included in the application and work offline. Screenshots show example datasets; your numbers and plots will depend on your own data.'),
    p('Designed by Morteza Khorsand Nikoo. Contact: morteza.khorsandnikoo@uibk.ac.at. Optional support: https://ko-fi.com/mortezakhorsand.'),
    p('The workspace is held in memory during the session. Export results and save trained models explicitly before closing the application; there is no automatic save of an entire project.')
], ('dataset','choose-task','save','reset'))

topic('choose-task','Getting started','Choose the right task',
      'Decide what you want to predict before choosing an algorithm.',[
    bullets('Regression predicts a number, such as a measured response, cost or temperature. Select one target column; the other columns become input features.',
            'Classification predicts a category, such as class 0, 1 or 2. Store categories as numeric codes. A class code is a label, not a continuous quantity.',
            'Clustering groups similar rows without a target. Every numeric column is used, so remove identifiers and any outcome column that should not influence the groups before importing the clustering dataset.'),
    h('Start with a baseline'),
    p('Try Linear Regression for a numeric target or Logistic Regression for class labels. Compare a more flexible model only after you understand the baseline. Use the same rows, target and evaluation procedure for a fair comparison.'),
    note('Remove columns that reveal the answer, such as an output calculated directly from the target or measurements only available after the event you want to predict. High scores from this kind of leakage do not demonstrate useful prediction.')
],('regression','classification','clustering','evaluation'))

topic('dataset','Dataset','Import and inspect a dataset',
      'Load a CSV on the Dataset page and check that the imported rows and columns are correct.',[
    steps('Open Dataset, choose Import CSV, and select your file.',
          'Check the column names and numeric row count. The preview displays up to the first 500 numeric rows; exported dataset tables can contain the full numeric dataset.',
          'Select columns to inspect distributions and descriptive statistics. Look for unexpected ranges, constant columns and unusual values.',
          'Read the correlation matrix and its color scale. Continue to Prepare data when the values are ready for cleaning or transformation.'),
    image('dataset.png','Dataset preview and diagnostic charts from an example CSV.'),
    h('CSV format'),
    bullets('Use a comma delimiter, one header row, the same number of columns on each row, and a decimal point for decimal values.',
            'Use clear, unique column names and finite numeric values. Record units separately; the application does not infer them.',
            'The current loader splits on commas directly. Do not use commas inside quoted cells or thousands separators. Rows with a different column count are skipped.',
            'The initial numeric view includes rows only when every cell parses as a number. Missing or text-containing rows may be absent from this view even when available to the preparation tools.',
            'Import requires at least two numeric rows and two columns. A file containing only text categories or missing values cannot be imported directly; prepare numeric data first.'),
    note('Importing a different dataset replaces the shared workspace data. Save important results first. Keep an unchanged copy of the source CSV so you can restart preparation.'),
    h('A small synthetic example'),
    p('A CSV can begin with headers Length,Width,Response and rows such as 1.0,2.0,5.1 and 2.0,1.5,6.4. Use substantially more rows for meaningful model evaluation; two rows only illustrate the file structure.')
],('correlation','missing','prepare','troubleshooting'), 'csv import delimiter comma header numeric rows file')

topic('correlation','Dataset','Read the correlation matrix',
      'Use the table and its color ribbon to compare linear relationships between numeric columns.',[
    image('correlation.png','The ribbon maps negative through zero to positive Pearson correlation.'),
    bullets('Values near +1 indicate a strong positive linear relationship: the two columns tend to increase together.',
            'Values near −1 indicate a strong negative linear relationship: one tends to decrease as the other increases.',
            'Values near 0 indicate little linear relationship. A curved or other nonlinear relationship may still exist.',
            'The diagonal compares each column with itself. Constant columns do not provide meaningful correlation information.'),
    p('Use the numeric labels as well as color. The adjacent ribbon explains the minimum and maximum colors. Inspect highly correlated input columns before interpreting regression weights or using covariance-based anomaly detection.'),
    note('Correlation does not establish cause and effect. A strong target correlation may also reflect duplicated information or leakage. Check the meaning and timing of each measurement.')
],('dataset','pca','anomalies'))

topic('prepare','Prepare data','Prepare data: recommended workflow',
      'Inspect, clean and transform the current dataset before training.',[
    steps('Run Describe to inspect count, range, mean and spread for every column.',
          'Detect missing values and choose a handling method. Check how many rows remain.',
          'Encode categories if necessary and retain the mapping from original labels to codes.',
          'Inspect anomalies. Remove rows only when there is a justified reason.',
          'Choose normalization if feature scales differ substantially. Apply PCA only if reducing feature dimensions helps your task.',
          'Review the resulting columns before selecting a target and training.'),
    p('Preparation changes the dataset used for subsequent training. Several tools write a derived CSV beside the source file, so use a writable folder and keep the original CSV. A Reset page action clears preparation reports only; it does not undo transformations or change the data and models retained by other pages.'),
    note('Preparation is fitted to the current dataset before model evaluation. The app does not refit these steps separately inside each train/test split or K-fold training fold. Treat those scores as exploratory when preprocessing learned information from the evaluation rows. For a strict benchmark, fit preprocessing on training data only in a workflow that supports it.')
],('missing','encoding','anomalies','normalization','pca','reset'))

topic('describe','Prepare data','Describe: understand each column',
      'Check the scale and distribution of the data before choosing a model.',[
    steps('Open Prepare data and run Describe.', 'Compare counts, minimum, maximum, mean and standard deviation across columns.',
          'Investigate impossible ranges, nearly constant features, duplicated columns and extreme values before training.'),
    p('A large standard deviation does not automatically mean a useful feature. It may simply reflect different units. A mean can be misleading for skewed data; also inspect a histogram on Dataset.'),
    note('Statistics describe the current numeric dataset, which may already have been transformed. Reimport the original CSV to inspect the original measurements again.')
],('dataset','normalization','anomalies'))

topic('missing','Prepare data','Detect and handle missing values',
      'Choose whether to fill missing cells or remove incomplete rows.',[
    steps('Open Prepare data and select Detect Missing. This examines the CSV values used by the preparation tool.',
          'Choose Mean, Max, Min, Mode or RemoveRow, then select Handle Missing.',
          'Review the resulting data and row count before continuing.'),
    bullets('Mean fills from the available average and is sensitive to extreme values.',
            'Mode uses a frequent value and may be suitable for coded categories; inspect the result.',
            'Min or Max uses an extreme observed value. Use these only when that assumption is meaningful.',
            'RemoveRow discards incomplete observations and can reduce your sample size or introduce selection bias.'),
    note('Do not fill a missing target merely to manufacture a known answer. Keep a separate prediction dataset for rows with unknown outcomes. If an entire column lacks usable values, fix or remove it before modeling.'),
    p('Because the numeric preview excludes rows containing nonnumeric cells, a preview that appears complete does not prove the original file had no missing or text values.')
],('dataset','prepare','evaluation'))

topic('encoding','Prepare data','Encode categorical values',
      'Convert text categories into numeric codes and keep their meaning available.',[
    steps('Make sure the file can first be imported: at least two complete numeric rows are required by the initial loader.',
          'Run Label Encoder from Prepare data and inspect the displayed mapping.',
          'Save the encoded data and record the mapping so future inputs use exactly the same codes.'),
    p('The encoder assigns codes to categories in sorted label order. These codes do not necessarily describe an ordered quantity. A distance-based model can interpret a difference between codes as a numeric distance, which may be inappropriate for unordered categories.'),
    note('The prediction preprocessing recipe preserves normalization and PCA, but it is not a general text-category encoder. Enter the expected numeric category codes when predicting. If your CSV is entirely categorical, encode it before importing.')
],('dataset','classification','prediction'))

topic('anomalies','Prepare data','Detect anomalies with Gaussian density',
      'Inspect unusual combinations of numeric values before deciding whether to remove rows.',[
    steps('Choose Gaussian (Mahalanobis) in Anomaly Detection.',
          'Set Threshold; the default is 0.001. Initially leave Remove anomalies from data unchecked.',
          'Run Detect Anomalies and review flagged observations.',
          'If justified, enable removal and run again. Check the retained data before training.'),
    image('anomalies.png','Example anomaly diagnostic. A flagged point needs investigation; it is not automatically an error.'),
    p('The detector standardizes the numeric columns and evaluates a multivariate Gaussian density using covariance and Mahalanobis distance. A row is flagged when its density falls below the threshold. Increasing the threshold generally flags more rows.'),
    note('The threshold is a density cutoff, not a p-value or the probability that a row is wrong. All current numeric columns, including a target if present, influence the detector. Constant or dependent columns can cause covariance problems; inspect those columns before running it.'),
    p('Rare but valid operating conditions may be essential examples. Removing them can make prediction less reliable exactly where you most need it.')
],('correlation','normalization','prepare'))

topic('normalization','Prepare data','Normalize numeric values',
      'Make feature scales more comparable and understand the effect on the target.',[
    steps('Choose ZScore, MinMax or MaxAbs in Prepare data.', 'Apply normalization and inspect the transformed values.',
          'Train your model on the resulting dataset. For supported models, Predict accepts original feature values and applies the stored transformation.'),
    bullets('ZScore subtracts the mean and divides by the standard deviation. It is useful when columns have different units or magnitudes.',
            'MinMax maps the observed range to a common interval. New inputs outside the original range can produce values outside that interval.',
            'MaxAbs divides by the largest absolute value, preserving zero and the sign of values.'),
    note('The preparation normalization tool acts on all numeric columns, including the target. The stored recipe restores a supported prediction to the original target scale. Charts and training metrics may still describe the transformed dataset; check the axes and data used for that run.'),
    p('Applying transformations repeatedly changes the data repeatedly. Use the original CSV to start over rather than treating Reset page as undo. Scaling does not remove outliers or make a nonlinear relationship linear.')
],('prediction','pca','evaluation','reset'))

topic('pca','Prepare data','PCA: reduce dimensions and keep original prediction inputs',
      'Train with principal components while entering familiar original features at prediction time.',[
    steps('Open Prepare data and locate PCA.',
          'For supervised training, place the target in the last column and enable Last column is target (Y). This preserves the outcome instead of mixing it into components.',
          'Choose Variance Threshold, such as 0.95, to retain enough components to explain that fraction of feature variance, or choose Fixed Components and enter a positive count.',
          'Run PCA. Review the component columns, scree plot, cumulative variance and PCA scatter.',
          'Train using the preserved target column. When you open Predict, enter the original feature values; the model applies the saved PCA transformation internally.'),
    image('pca-variance.png','Cumulative explained variance helps decide how many components to retain.'),
    image('pca-prediction.png','Prediction inputs retain original feature names even when training uses PCA components.'),
    p('PCA standardizes input features and forms new axes ordered by explained variance. PC1 explains the most feature variance, PC2 the next most, and so on. A component combines several original features; it is not a new quantity the user must measure.'),
    note('A 0.95 variance threshold means 95% of feature variance, not 95% predictive accuracy. PCA can discard information useful for a target. Compare performance against a model using the original features.'),
    p('Do not choose a component as your target when the goal is to predict the original outcome. If the target was included in PCA, reimport the source file and repeat with the target preserved. Loading only a previously exported PC1/PC2 CSV does not recover the original transformation recipe.'),
    p('Save the newly trained model to retain its preprocessing recipe. Older models saved without that information may require transformed inputs or retraining from the original data.')
],('prediction','normalization','save','evaluation'), 'principal components explained variance scree PCA1 PCA2 original features')

topic('regression','Regression','Train a regression model',
      'Predict a continuous target and inspect error on data not used to fit the model.',[
    steps('Import and prepare your dataset. Select Regression.',
          'Choose an algorithm and select the target column. All remaining current columns are inputs.',
          'Set the training percentage and algorithm settings. Start with defaults, then change one setting at a time.',
          'Optionally enable K-fold and set the number of folds. Train the model and wait for completion.',
          'Review train and test R², MAE, RMSE and the available diagnostic plots. When using K-fold, inspect every fold and the Combined row.',
          'Open Predict and enter feature values in the fields shown. Save the model and export your results.'),
    image('regression.png','Example polynomial regression workflow with synthetic data.'),
    h('Read the charts'),
    bullets('Actual versus predicted: values close to the diagonal indicate close agreement. Check both the range and outlying errors.',
            'Residuals: inspect systematic patterns and changes in spread. A curved pattern can signal a missing nonlinear relationship.',
            'Cost or loss: shows the training objective during optimization where supported. A decreasing loss does not by itself prove good test performance.',
            'Importance or weights: interpretation depends on the model, scaling and any PCA or polynomial expansion. A large weight is not evidence of causation.'),
    note('Regression uses R² rather than classification accuracy. R² can be negative. Metrics on a normalized target are not automatically in the original physical units.')
],('linear','polynomial','reg-metrics','cross-validation','prediction'))

regressors = [
 ('linear','Linear Regression','A readable baseline for approximately linear relationships.',
  ['Iterations controls optimization steps; Learning rate controls step size. If loss rises or becomes invalid, reduce the learning rate and check feature scales.',
   'Regularization can be None, L1 (LASSO) or L2 (Ridge). L1 encourages sparse weights; L2 discourages large weights.'],
  'Compare residuals against actual values. Structured curvature suggests a linear model may be too simple; poor train and test scores can also indicate weak features.',('normalization','reg-metrics')),
 ('polynomial','Polynomial Regression','Add powers of features to capture smooth curvature.',
  ['Degree controls the highest power; supported degrees are 1 through 8. Start with degree 2 and compare test performance before increasing it.',
   'Iterations and Learning rate control optimization. Higher degrees can be numerically difficult, especially for large unscaled inputs.',
   'The implementation adds powers of each individual input. It does not automatically create all cross-feature interaction terms.'],
  'A high degree can fit noise and behave badly outside the training range. Check held-out error, residuals and the stability of predictions near the range limits.',('linear','normalization')),
 ('normal-equation','Normal Equation','Solve a linear regression problem directly rather than iterating gradient steps.',
  ['ridge adds regularization to stabilize the linear solution. Its default is 0; increasing it penalizes larger coefficients.'],
  'Useful for a compact linear baseline. Highly dependent columns or too many features relative to rows can make an unregularized solution unstable.',('linear','correlation')),
 ('lwr','Locally Weighted Regression','Fit a local relationship around each prediction point.',
  ['tau is the bandwidth. Smaller values emphasize nearby observations; larger values create smoother, less local fits.',
   'lambda regularizes local fits to improve numerical stability.'],
  'Local methods need nearby examples. Inputs far outside the observed region may be unreliable, and prediction can become slower as the dataset grows.',('normalization','prediction')),
 ('reg-forest','Random Forest regression','Average many trees to model nonlinear relationships and interactions.',
  ['trees sets the number of trees. More trees generally cost more time and memory.',
   'maxDepth limits tree complexity; minLeaf controls the minimum leaf size. Shallower trees and larger leaves tend to smooth predictions.'],
  'Compare training and test error before increasing depth. The optional tree diagram illustrates one tree, not the entire forest. Forest predictions are not a reliable way to extrapolate trends beyond the training domain.',('curves','evaluation')),
 ('reg-mlp','Neural Net (MLP) regression','Use a multilayer neural network for flexible nonlinear regression.',
  ['hiddenSize controls units per hidden layer; hiddenLayers controls depth.',
   'epochs controls training passes; learningRate controls update size. Scaling inputs is particularly useful.'],
  'Begin with a small network. If training error improves while test error worsens, reduce capacity or training duration and reconsider the data. Larger networks require more data and time.',('normalization','curves')),
 ('rbf','RBF Surrogate','Approximate a smooth response with Gaussian radial basis functions.',
  ['epsilon controls the Gaussian basis width: the kernel is exp(−(epsilon × distance)²), so larger epsilon makes each basis function narrower.',
   'smoothing regularizes the fit. Zero favors interpolation; positive smoothing can be useful for noisy measurements.'],
  'Feature scale strongly affects distance. The method uses a dense matrix tied to the number of samples, so large datasets can be expensive. Predictions far from observations need particular care.',('normalization','prediction')),
 ('reg-fasttree','FastTree (boosted) regression','Build an additive ensemble of regression trees.',
  ['numberOfTrees sets the boosting length; numberOfLeaves controls the complexity of each tree.',
   'learningRate controls the contribution of each new tree; minExamplesPerLeaf controls how small a leaf may be.'],
  'Smaller learning rates may need more trees. More leaves or very small leaves can overfit. Inspect held-out scores rather than optimizing training fit alone.',('curves','evaluation')),
 ('reg-lightgbm','LightGBM regression','Use gradient-boosted trees for nonlinear tabular data.',
  ['numberOfIterations sets the number of boosting rounds; numberOfLeaves limits tree complexity.',
   'learningRate controls update size; minExamplesPerLeaf limits leaf size.'],
  'Increase model complexity gradually. Compare results using the same target, data and validation method as other regressors.',('curves','compare')),
 ('ogd','Online Gradient Descent regression','Fit a linear model through iterative updates.',
  ['iterations sets the number of training passes; learningRate controls the update size.'],
  'Scale inputs and inspect held-out error. The algorithm name describes its optimization method; this app trains from the imported dataset and does not provide a live streaming data service.',('linear','normalization')),
 ('reg-xgboost','XGBoost regression','Use boosted trees to capture nonlinear tabular relationships.',
  ['numRounds controls boosting rounds; maxDepth controls the depth of trees.',
   'eta is the learning rate; subsample is the fraction of training rows sampled for each boosting round.'],
  'Deep trees and many rounds can overfit. Adjust depth and learning rate together thoughtfully and compare held-out error after each experiment.',('curves','evaluation'))
]
for id,title,summary,settings,advice,related in regressors:
    topic(id,'Regression algorithms',title,summary,[
        steps('Open Regression and select '+title+'.','Choose the target, training percentage and optional K-fold settings.','Set the controls described below, then train and inspect the results.'),
        h('Settings'),bullets(*settings),h('Interpret and refine'),p(advice),
        p('Use Predict with the feature names shown. For a model trained with a stored PCA or normalization recipe, enter original feature values. Save the model to retain that recipe.')],
        ('regression',)+related+('cross-validation','prediction'))

topic('reg-metrics','Evaluation','Understand regression scores',
      'Read R², MAE and RMSE together, with the target scale in mind.',[
    bullets('R² measures improvement over predicting the mean. A value of 1 is perfect, 0 matches that baseline, and a negative value is worse. Some summary cards display R² as a percentage; the fold table uses raw R².',
            'MAE is mean absolute error. Lower values mean smaller typical absolute errors, in the units of the target used in that evaluation.',
            'RMSE is root mean squared error. It penalizes larger errors more strongly than MAE and is also in the evaluated target units.'),
    image('regression-scatter.png','Actual versus predicted values: inspect spread and outliers as well as the overall score.'),
    note('R² is not meaningful for a constant target or a fold with fewer than two evaluated observations. N/A is more informative than treating such a score as zero. A high R² does not mean every prediction has a small error.'),
    p('If training scores are much better than test scores, investigate overfitting, leakage, small samples or distribution differences. If both are weak, examine data quality, features, optimization settings and model suitability.')
],('evaluation','cross-validation','normalization'))

topic('curves','Evaluation','Learning and validation curves',
      'Use optional advanced-regression curves to explore sample size and a parameter setting.',[
    steps('Select an advanced regression algorithm and enable Generate learning/validation curves.',
          'Train and allow extra time: these curves require multiple additional fits.',
          'Read both training and test curves rather than selecting a setting from training performance alone.'),
    p('The learning curve fits increasing fractions of the training rows (20%, 40%, 60%, 80% and 100%) and evaluates against the same held-out test rows. It helps show whether additional training examples might be useful.'),
    p('The validation curve varies the first listed hyperparameter across seven values while keeping other parameters fixed. It does not tune every setting and does not automatically select or retrain the best model.'),
    note('Repeatedly choosing settings based on the same test data makes that test less independent. Keep a separate final evaluation dataset when an unbiased final performance estimate matters.')
],('regression','evaluation','compare'))

topic('classification','Classification','Train a classification model',
      'Predict numeric class labels and check which classes the model confuses.',[
    steps('Import data with numeric class codes and select Classification.',
          'Choose the algorithm and target class column. Set the training percentage and the settings visible for that algorithm.',
          'Optionally enable K-fold and choose the fold count. Train the classifier.',
          'Review accuracy, precision, recall, F1 and the confusion matrix. With K-fold, inspect the fold table and Combined test accuracy.',
          'Use Predict with the expected feature values. Save the classifier as .aiclf and export results as needed.'),
    image('classification.png','Classification example with fold-by-fold training and test results.'),
    note('Class labels are categories even when represented by numbers. Class 2 is not necessarily twice class 1. Features are standardized internally by the classifier; avoid interpreting standardized feature weights as raw-unit effects.'),
    p('Random splitting is not stratified or grouped. With few examples in a class, a training fold may not represent that class adequately. Inspect class counts before trusting a high overall accuracy.')
],('cls-logistic','cls-metrics','cross-validation','prediction'))

classifiers = [
 ('cls-logistic','Logistic Regression','A baseline classifier using linear decision scores.','Iterations and Learning rate control optimization. The multiclass implementation uses one-vs-rest models.','Use it when a simple, interpretable baseline is valuable. Nonlinear class boundaries may require richer features or a different classifier.'),
 ('cls-knn','K-Nearest Neighbors','Assign a class using nearby training examples.','K is the number of neighbors. Small K is sensitive to individual observations; larger K smooths the decision but can obscure small classes.','Distances depend on scaling and on which features you include. Training accuracy need not be 100% when K is greater than one.'),
 ('cls-naive','Naive Bayes','Use a Gaussian probability model within each class.','The Gaussian model estimates a mean and variance for features within each class. It does not use a neighbor count or a tree depth.','This is a useful fast baseline. Strongly dependent features or non-Gaussian class distributions can limit its assumptions.'),
 ('cls-svm','SVM (linear)','Separate classes using linear margin-based decision rules.','C controls the tradeoff between regularization and training classification errors. The algorithm shown is a linear SVM, not an RBF-kernel SVM.','Scaling matters. A curved class boundary may remain difficult even after changing C.'),
 ('cls-forest','Random Forest','Combine decision trees for nonlinear class boundaries.','Trees, Max depth and Min leaf control ensemble size and tree complexity. More depth and smaller leaves allow a closer fit.','Inspect per-class scores and held-out accuracy. A very accurate training fit alone does not establish generalization.'),
 ('cls-fasttree','FastTree (boosted)','Use boosted trees for classification.','Trees controls the number of trees; Learning rate controls each boosting step; Min leaf controls the minimum examples per leaf. In this classifier, Max depth is converted to a leaf limit of twice its value, rather than a direct depth limit. The displayed Iterations field is not used. Multiclass training uses one-versus-all models.','Change settings one at a time and compare class-specific errors, not only overall accuracy.'),
 ('cls-lightgbm','LightGBM','Fit boosted decision trees to numeric tabular features.','The current classifier uses the underlying LightGBM trainer defaults. The shared Iterations, Learning rate and tree controls are visible but are not passed to this trainer. Changing those fields does not tune this classifier in this version.','Compare against Logistic Regression and Random Forest using the same data and validation procedure.'),
 ('cls-xgboost','XGBoost','Fit a gradient-boosted tree classifier.','Trees sets the number of boosting rounds; Max depth controls tree depth; Learning rate controls update size. The shared Iterations and Min leaf fields are not used by this classifier.','Check minority-class recall. Improvements in overall accuracy can conceal a worse result for a rare class.'),
 ('cls-ann','Neural Net (ANN)','Learn nonlinear boundaries with a feed-forward neural network.','Hidden size and Hidden layers set capacity; Iterations and Learning rate control optimization.','Start small and inspect training versus test performance. More units or layers do not guarantee better predictions.')
]
for id,title,summary,settings,advice in classifiers:
    topic(id,'Classification algorithms',title+' classifier',summary,[
        steps('Select Classification and choose '+title+'.','Choose the numeric class target and evaluation settings.','Train, then review the confusion matrix and per-class results.'),
        h('Settings and behavior'),p(settings),h('What to check'),p(advice),
        note('The classifier standardizes features using the current data before splitting. This and any prior preparation must be considered when interpreting validation results.')],
        ('classification','cls-metrics','evaluation','cross-validation'))

topic('cls-metrics','Evaluation','Accuracy, precision, recall, F1 and confusion matrix',
      'Look beyond the headline accuracy to understand class-specific mistakes.',[
    bullets('Accuracy is the fraction of correctly predicted labels. A percentage near 100 is better, but a dominant majority class can make this misleading.',
            'Precision for a class asks: among predictions of this class, how many were correct?',
            'Recall for a class asks: among actual examples of this class, how many did the model find?',
            'F1 combines precision and recall. Macro F1 gives each class equal weight rather than letting a large class dominate.',
            'In the confusion matrix, rows represent actual classes and columns represent predicted classes. Diagonal cells are correct predictions; off-diagonal cells identify the confusions.'),
    note('In K-fold mode, the fold table is the place to read held-out fold accuracy. The current classification result plots and confusion matrix describe the final model fit and should not be treated as a pooled out-of-fold confusion matrix.'),
    p('For example, if a rare class contains 10 observations and the model finds only 2, its recall is poor even when overall accuracy is high. Decide which errors matter for your use case before selecting a model.')
],('classification','cross-validation','evaluation'))

topic('cross-validation','Evaluation','K-fold: per-fold and combined results',
      'See training and test scores for each fold and understand the final Combined row.',[
    steps('Enable Use K-fold CV in the regression or classification training page.',
          'Choose the number of folds, commonly 5. Ensure each fold has enough observations and that classes are adequately represented.',
          'Train. The fold table records training and test sample counts and scores for each split.',
          'Read the Combined test score, then inspect variation between folds. Export the table to keep all fold results.'),
    image('folds.png','Example fold table. The Combined row summarizes predictions across folds.'),
    h('What the Combined row means'),
    p('Each observation is tested once across the held-out folds. Combined classification test accuracy is total correct held-out predictions divided by total held-out observations; it is weighted by sample count, not an unweighted average of percentages.'),
    p('Combined regression R² is calculated from pooled predictions and targets. It is not the arithmetic mean of the fold R² values. This matters especially when fold sizes or target distributions differ.'),
    p('Combined training results pool the training predictions made in every fold. Each observation is used for training repeatedly, typically k−1 times, so this count is larger than the dataset row count. It is not an independent test result.'),
    h('Final model and plots'),
    p('After evaluation, a model can be fitted to all available rows for prediction. Its fit is distinct from held-out CV performance. In linear/polynomial regression and classification, final-fit diagnostics should not be interpreted as out-of-fold plots. Advanced regression also presents its separate holdout diagnostics alongside the CV table.'),
    note('Folds are randomly shuffled, not stratified, grouped or time-aware. Preparation is not refitted within each fold. For time series, repeated subjects, grouped measurements or rigorous preprocessing validation, use an evaluation workflow designed for those conditions.'),
    p('R² can be N/A for a fold with a constant target or too few test samples. Reduce the fold count or obtain more suitable data. Do not replace undefined scores with zero.')
],('evaluation','reg-metrics','cls-metrics','save'), 'k folds cross validation final combined accuracy train test CV')

topic('evaluation','Evaluation','Evaluate without misleading yourself',
      'Separate exploratory comparisons from an independent final performance estimate.',[
    bullets('Training performance measures fit to examples the model has already seen. Test performance estimates behavior on held-out rows from the selected split.',
            'Keep the target, features, data rows and evaluation procedure consistent when comparing algorithms.',
            'Do not include outcome-derived columns or future information among inputs.',
            'A random row split does not respect time order, subjects or groups. Related observations can make scores look too optimistic.',
            'Preprocessing fitted on all data and repeated tuning against the same test rows reduce evaluation independence.'),
    p('The studio is useful for exploring data and models, but its current preprocessing steps are performed before the evaluation split, and classifier standardization also uses the current dataset before splitting. K-fold does not automatically remove this limitation.'),
    note('For a final benchmark, reserve independent observations and use a pipeline that fits transformations only on training data, applying the fitted transforms unchanged to evaluation data. Report the exact procedure with your scores.'),
    p('Large differences between folds can reflect small samples, unusual observations, imbalanced classes or real variability. Investigate rather than reporting only the best fold.')
],('cross-validation','compare','prepare','reg-metrics','cls-metrics'))

topic('clustering','Clustering','Find groups without a target',
      'Cluster observations and inspect whether the resulting groups are useful.',[
    image('clustering.png','K-Means run on 90 synthetic observations in three groups. This example is generated for illustration, not a real-world benchmark.'),
    steps('Import a dataset containing only the numeric columns you want to influence similarity. Remove identifiers and unwanted target columns beforehand.',
          'Open Clustering, choose an algorithm and set its parameters.',
          'Run clustering and inspect labels, cluster sizes, silhouette and the scatter plot.',
          'Use Assign for an additional observation where available, and export the assignments and chart.'),
    p('Clustering uses all current numeric columns and standardizes them internally. The two-dimensional scatter uses a PCA projection for display. Groups that overlap in this projection may be better separated in the full feature space, and visually separated points do not prove useful clusters.'),
    p('Silhouette compares within-cluster closeness with separation from other clusters. Values closer to 1 generally indicate clearer separation; negative values can indicate a questionable assignment. The app may show zero when fewer than two clusters exist, where the score is not a useful quality measure.'),
    note('Cluster numbers are identifiers, not ordered quality scores or known ground-truth classes. Different settings or runs can change both the grouping and its numbering.')
],('kmeans','dbscan','hierarchical','fuzzy','save'))

for id,title,summary,settings,behavior in [
 ('kmeans','K-Means','Partition observations around a chosen number of centers.',
  'K sets the number of clusters; Iterations limits optimization. Start with a small K, inspect sizes and silhouette, and compare several plausible choices.',
  'Works best for compact groups under the selected distance representation. It assigns new observations to the nearest center and does not identify noise by itself.'),
 ('dbscan','DBSCAN','Find dense groups and identify noise without fixing the number of clusters.',
  'Eps sets the neighborhood radius in standardized feature space; MinPts sets the required neighborhood density. Increase Eps if almost everything is noise, or reduce it if separate groups merge too readily.',
  'Noise has label −1. Assign uses the nearest non-noise labeled training observation; it does not rerun a density test for the new point. Even a distant input may receive a cluster label. If every training point is noise, assignment remains −1.'),
 ('hierarchical','Hierarchical clustering','Merge nearby groups until the requested number remains.',
  'Choose K and a linkage rule. Single linkage uses the nearest pair and can form chains; Complete uses the farthest pair; Average uses average pairwise distance.',
  'This method can become expensive for large datasets. The app displays the resulting clusters rather than a dendrogram. Assignment uses the nearest labeled training observation, not a new hierarchical fit.'),
 ('fuzzy','Fuzzy C-Means','Fit cluster centers with soft memberships during training.',
  'K sets the number of centers; Iterations limits optimization; fuzziness M must be greater than 1 and controls the softness of memberships.',
  'The app displays a hard label from the strongest membership. New assignment uses the nearest center; it should not be read as a calibrated class probability.')]:
    topic(id,'Clustering algorithms',title,summary,[
        steps('Open Clustering and select '+title+'.','Set the parameters below and run clustering.','Inspect the scatter, group sizes and silhouette, then export assignments.'),
        h('Settings'),p(settings),h('Interpretation and new observations'),p(behavior),
        note('All current numeric columns influence clustering. Choose inputs based on meaningful similarity, not merely on which columns are available.')],('clustering','normalization','visualize'))

topic('prediction','Prediction and saving','Predict with original feature values',
      'Apply the trained model and its saved preparation recipe to new observations.',[
    steps('Train a regression or classification model, or load a compatible saved model in the appropriate algorithm page.',
          'Read the feature names shown in Predict. Enter one finite numeric value for every required field, using the same units and meaning as the source data.',
          'For a model prepared with the stored normalization/PCA pipeline, enter original feature values rather than inventing PC1 or PC2 values.',
          'Run prediction and read the numeric response or class label. For supported normalized targets, the prediction is restored to the original target scale.'),
    image('pca-prediction.png','Original-feature prediction after PCA: the stored transformation is applied internally.'),
    note('Do not normalize an input manually and then feed it to a model that already stores that normalization; that applies the transformation twice. For encoded categories, use the correct numeric code and keep the mapping.'),
    p('The recipe is available when preparation and training occur through the supported workflow and is retained by newly saved models. A standalone CSV of principal components or a legacy model without metadata cannot reconstruct the original feature mapping.'),
    p('A numerical prediction is not a guarantee. Inputs far outside the observed ranges can be unreliable. Check whether the new observation resembles the data on which you trained.')
],('pca','normalization','encoding','save','troubleshooting'))

topic('save','Prediction and saving','Save models, tables, charts and page images',
      'Choose the export that preserves the information you need.',[
    bullets('Save page image captures the current page view. It is useful for a quick record of visible controls and results.',
            'Save charts/images exports the available charts, including charts below the current viewport. Use export title, font size, number formatting and stroke settings when preparing figures.',
            'Save tables exports CSV data for the available page tables. Keep the K-fold table if you need every fold and the Combined score.',
            'Regression Save model writes an .aimodel file; classification uses .aiclf. Load it in the appropriate model family. Newly saved supported models include the preprocessing recipe.',
            'Clustering provides exported assignments and charts; the page does not offer the same saved-model workflow as regression and classification.'),
    steps('Run the analysis so its results are present.','Choose the relevant save action and a writable output location.','Open the exported file and verify that it contains the expected rows, labels, legend and values.'),
    note('A screenshot is not a substitute for a numeric results table, and a model file is not a complete workspace backup. Keep the source CSV, preparation choices, target, parameters and evaluation results together.'),
    p('The Home page is an introduction and has no cover-image export. Import CSV is on Dataset. Help content is embedded in the application; users do not need a separate help folder or an internet connection.')
],('cross-validation','prediction','reset'))

topic('compare','Insights','Compare models in this session',
      'Review recorded experiments while keeping their evaluation procedures in context.',[
    steps('Train the models you want to examine, keeping the dataset and target consistent.',
          'Open Compare models and refresh the history if needed.',
          'Compare the recorded metrics and review the original page for diagnostic plots and fold tables.',
          'Export the table before clearing history or closing the application.'),
    note('The history is a session record, not a uniform cross-validation leaderboard. Stored metrics depend on the training path: some CV runs record final-fit values, polynomial CV records its CV summary, and advanced regressors retain holdout metrics. Use the original Combined fold tables for a consistent held-out CV comparison.'),
    p('Do not compare R² with classification accuracy, or raw-target error with error after target normalization. Record dataset, target, transformations and split settings alongside each experiment.'),
    p('Clear history or Reset page removes comparison records without deleting models retained on other pages.')
],('evaluation','cross-validation','reg-metrics','cls-metrics'))

topic('visualize','Insights','Visualize data and sort a line chart',
      'Choose a chart, select its columns and optionally order values from minimum to maximum.',[
    steps('Open Visualize and choose Line, Bar, Scatter, Pie or Radar.',
          'Select X and Y, and optional Y2 when the chart supports it.',
          'For a line chart, enable Sort X: min → max to connect points in ascending numeric X order. Toggle it off to return to the source row order.',
          'Inspect axis labels and export the chart or its table.'),
    image('visualize.png','Sorting X changes the display order while keeping each row’s X and Y values paired.'),
    bullets('Line connects observations in their display order. Sorting is helpful for a response-versus-input curve, but preserve source order when it represents time or a meaningful sequence.',
            'Scatter shows pairs without implying a connecting order.',
            'Bar limits the displayed series to the first 40 rows after ordering. Check what is included before interpreting it.',
            'Pie summarizes up to ten large absolute Y values with an Other group. Avoid using a pie chart to communicate signed positive/negative changes.',
            'Radar compares feature axes for a selected row. X sorting does not apply to it; different feature scales can dominate the shape.'),
    note('Sorting affects the chart and its exported series order; it does not reorder the shared dataset or retrain a model. Duplicate X values remain paired with their original Y values and can create vertical line segments.')
],('dataset','save','normalization'), 'sort ascending minimum maximum line diagram chart')

topic('reset','Getting started','Reset one page',
      'Clear the current workspace page without clearing information from other pages.',[
    steps('Open the workspace page you want to clear.','Select the green Reset page button after Save tables.','Start a new analysis on that page when ready.'),
    bullets('Regression reset clears that page’s models, results, plots and prediction fields across its algorithms.',
            'Classification and Clustering clear their own page results and relevant state.',
            'Prepare data reset clears its visible reports only. It keeps the prepared dataset and transformation information available to other pages.',
            'Dataset reset clears the page display while preserving shared data used elsewhere; reimport to populate its view again.',
            'Compare models reset clears comparison history, and Visualize reset clears its chart setup/results.'),
    note('Reset page is not an undo for data preparation. To restore original values, reimport the original CSV, which begins a new shared dataset workflow. Export anything you want to retain before clearing a page.')
],('prepare','dataset','save'))

topic('troubleshooting','Reference','Troubleshooting common problems',
      'Check input format, evaluation settings and model state before changing algorithms.',[
    h('CSV cannot load or rows are missing'),
    p('Check comma delimiters, decimal points, header names and equal column counts. Embedded commas in quoted cells are unsupported. The numeric loader excludes rows containing text or missing cells and requires at least two complete numeric rows and two columns. Use Detect Missing for the raw data where available.'),
    h('Prediction asks for PC1 or PC2'),
    p('Check whether you imported an already transformed PCA CSV or loaded an older model without its preprocessing recipe. Reimport the original data, preserve the final target column during PCA, retrain, and save the new model.'),
    h('R² is negative or N/A'),
    p('Negative R² means the model is worse than the mean baseline on the evaluated observations. N/A can mean a constant target or too few test rows. Inspect the data and reduce the fold count if necessary.'),
    h('Training loss is invalid or increasing'),
    p('Check for non-finite values, extreme magnitudes, a learning rate that is too high, or an unnecessarily high polynomial degree. Try a smaller step size and appropriate scaling, starting again from the original CSV when needed.'),
    h('No plot or an empty export'),
    p('Confirm that training completed successfully and the selected algorithm has results. Scroll to its chart area. Reset clears charts, and changing datasets invalidates prior results. Save charts/images exports chart content; Save page image only captures the visible page.'),
    h('Nearly every point is DBSCAN noise'),
    p('Check the chosen features and their variation. Try a larger Eps or a smaller MinPts thoughtfully, and inspect resulting group sizes. A dataset need not contain natural dense clusters.'),
    h('Anomaly detection cannot fit covariance'),
    p('Look for constant, duplicate or perfectly dependent numeric columns, and too few observations relative to features. Remove unsuitable columns in a copy of the data and retry.'),
    h('Export or preparation cannot write a file'),
    p('Select a writable folder, close a file that another application may have locked, and retry. Some preparation tools write derived CSV files alongside the source.'),
    h('Help search has no results'),
    p('Choose All topics and use fewer words. Search requires every entered word to occur in a topic. Try the exact control or algorithm name. Clear restores the full contents list.')
],('dataset','pca','reg-metrics','save'))

topic('cheat-sheet','Reference','Printable machine-learning cheat sheet',
      'An illustrated two-page A4 quick reference to workflow, preparation, metrics and every algorithm.',[
    steps('Click ML cheat sheet directly below Help and user guide in the sidebar to open the illustrated PDF immediately. These buttons stay visible even when the workspace navigation scrolls.',
          'Use Save cheat sheet PDF at the top of Help to choose where to save your copy.',
          'Use Open / print cheat sheet to open the embedded PDF in your default PDF reader.',
          'Press Ctrl+P in the PDF reader. Select A4 paper and Actual size or 100%. For Letter paper, choose Fit to printable area. For one double-sided sheet, select duplex printing with long-edge binding.'),
    p('The PDF uses illustrated plots, selectable text, embedded fonts and vector graphics for sharp printing at any zoom. It is included in the app and works offline; no separate help download is needed.'),
    p('Page 1 illustrates regression, classification, clustering, scaling, PCA, K-fold evaluation and overfitting. Page 2 shows actual-versus-predicted values, residual patterns, a worked confusion matrix and clusters with noise. An algorithm map summarizes all 11 regression, 9 classification and 4 clustering algorithms. Diagrams are illustrative rather than measured app results.'),
    note('Open / print opens your PDF reader; it does not send anything to a printer automatically. If opening fails, save the PDF and open it manually.')
],('choose-task','evaluation','glossary'), 'download print printable PDF cheat sheet quick reference A4')

topic('glossary','Reference','Glossary',
      'Quick definitions of the terms used throughout the studio.',[
    bullets('Observation / sample / row: one example in the dataset.',
            'Feature / input / X: a value available to the model when making a prediction.',
            'Target / outcome / Y: the quantity or label the supervised model learns to predict.',
            'Parameter / hyperparameter: a learned model quantity, or a user-selected setting such as degree, depth or learning rate.',
            'Training set: examples used to fit a model. Test set: held-out examples used for evaluation.',
            'Overfitting: learning detail or noise that does not generalize to new observations.',
            'Leakage: evaluation information entering training or preparation in a way that makes scores overly optimistic.',
            'PCA: principal component analysis, a feature transformation ordered by explained variance.',
            'Cross-validation / CV: repeated fitting and evaluation over different held-out folds.',
            'Residual: the difference between observed and predicted values; check the displayed axis convention.',
            'Ensemble: a combination of multiple models, such as a forest or boosted trees.',
            'Extrapolation: predicting outside the range or conditions represented by training data.')
],('choose-task','evaluation','pca','cross-validation'))

images = {
    'home.png': 'Tests/help-regression-verification/home.png',
    'dataset.png': 'Tests/help-regression-verification/dataset.png',
    'correlation.png': 'image/regression/linear/chart_correlation_matrix.png',
    'regression.png': 'Tests/help-regression-verification/polynomial.png',
    'classification.png': 'Tests/help-regression-verification/cv-classification.png',
    'folds.png': 'Tests/help-regression-verification/cv-polynomial-small.png',
    'pca-prediction.png': 'Tests/help-regression-verification/pca-original-feature-prediction.png',
    'visualize.png': 'Tests/help-regression-verification/visualization-sorted.png',
    'pca-variance.png': 'image/regression/linear/chart_cumulative_variance.png',
    'anomalies.png': 'image/regression/linear/chart_anomaly.png',
    'regression-scatter.png': 'image/regression/linear/chart_test_scatter.png',
    'clustering.png': 'Tests/help-verification/clustering.png',
}
for dest, source in images.items():
    shutil.copyfile(ROOT / source, OUT / 'Images' / dest)
ids = {t['Id'] for t in topics}
assert len(ids) == len(topics)
for t in topics:
    assert set(t['Related']) <= ids, t['Id']
    for block in t['Blocks']:
        if block['Kind'] == 'image': assert (OUT/'Images'/block['Image']).is_file()
(OUT/'guide.json').write_text(json.dumps(topics, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'Built {len(topics)} topics with {len(images)} embedded illustrations.')
