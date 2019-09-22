#Fill in the missing value in the data with specified method


def fill_nan():
	from sklearn.impute import SimpleImputer
	import numpy as np
	imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
	return imp_mean
