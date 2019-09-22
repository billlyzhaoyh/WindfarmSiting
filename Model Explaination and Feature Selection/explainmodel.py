#Explain model using three different mehtods that work well with sklearn estimators (do not know if compatible with other python packages)

#show permutation importance plot with the function
def permutation_importance(X_test,y_test,estimator,imputer):
	import eli5
	from eli5.sklearn import PermutationImportance
	X_test_tran = pd.DataFrame(imputer.fit_transform(X_test), columns = X_test.columns)
	perm = PermutationImportance(estimator, random_state=1).fit(X_test_tran, y_test)
	eli5.show_weights(perm, feature_names = X_test_tran.columns.tolist(),top=len(X_test.columns))

#print out all the partial dependene plot for the feature varibles with respect to wind speed (make sure %matplotlib inline is active)
def partial_dependence(X_test,y_test,estimator,imputer):
	from matplotlib import pyplot as plt
	from pdpbox import pdp, get_dataset, info_plots
	X_test_tran = pd.DataFrame(imputer.fit_transform(X_test), columns = X_test.columns)
	for feat_name in X_test_tran.columns:
		pdp_dist = pdp.pdp_isolate(model=estimator, dataset=X_test_tran, model_features=X_test_tran.columns, feature=feat_name)
    	pdp.pdp_plot(pdp_dist, feat_name)
    	plt.show()

# Use SHAP values to show the effect of each feature of a given wind turnbine location. Rownumber controls that if SHAP value will be displayed for a single row or a stack of sub samples
def shap_linear(X_test,y_test,estimator,imputer,rownumber=None,random_number=42):
	import shap  # package used to calculate Shap values
	X_test_tran = pd.DataFrame(imputer.fit_transform(X_test), columns = X_test.columns)
	explainer = shap.LinearExplainer(estimator,X_test_tran,feature_dependence="independent")
	if rownumber is None:
		sample=X_test_tran.sample(n=200, random_state=random_number)
		shap_values = explainer.shap_values(sample)
		shap.initjs()
    	ind=0
    	return shap.force_plot(explainer.expected_value, shap_values, sample),shap.summary_plot(shap_values, X_test_tran)
    else: 
    	rowsample = X_test_tran.iloc[rownumber].astype(float) 
    	shap_values = explainer.shap_values(rowsample)
    	shap.initjs()
    	ind=0
    	return shap.force_plot(explainer.expected_value, shap_values, rowsample),shap.summary_plot(shap_values, X_test_tran)

