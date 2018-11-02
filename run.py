print(__doc__)

import numpy as np
import pandas as pd
import sys
import re
import itertools
import urllib.request, json 

import lightgbm as lgb
from sklearn.model_selection import (train_test_split, GridSearchCV)
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss

import matplotlib.pyplot as plt

print("python_version:",sys.version)
print("lightgbm_version:",lgb.__version__)
print("numpy_version:",np.version.version)
print("pandas_version:",pd.__version__)



def read_data(weblink=None,infile=None,dump=False):
	"""
	Prepare Input Data
	"""
	class AppURLopener(urllib.request.FancyURLopener):
		version = "Mozilla/5.0"
	opener = AppURLopener()

	if weblink is not None:
		with opener.open(weblink) as url:
			jsdata = json.loads(url.read().decode())
			if dump == True and infile is not None:
				with open(infile, 'w') as outfile:
					json.dump(jsdata, outfile)
	elif infile is not None:
		with open(infile) as sample_json:
			jsdata = json.load(sample_json)
	else:
		print("Error: Both weblink and infile are Null!")
		sys.exit()

	df = pd.read_json(json.dumps(jsdata['acList']))
	
	return df


def data_cleanse(df):
	"""
	Data Cleansing
	"""
	# uncorrelated features
	uncorr_cols=['Id','From','To', 'Reg','Icao','CNum','ResetTrail','HasPic','Call','CallSus','Bad']

	# duplicated features
	dup_cols =['OpIcao']

	# feature columns with over 95% missing rate
	remove_cols=['Sat','Tag','PicX','PicY']

	# feature columns with unclear meaning, such as Cos
	unclear_cols=['Cos']

	# drop previous mentioned feature columns
	df.drop(uncorr_cols+dup_cols+remove_cols+unclear_cols,axis=1,inplace=True)

	# extract numeric value from FSeen
	df['FSeen'+'_tmp']=df['FSeen']
	df['FSeen']=df['FSeen_tmp'].apply(lambda epochtime: re.split('(\d+)', epochtime)[1]).astype(np.float64)
	df.drop(['FSeen_tmp'],axis=1,inplace=True)

	return df

def data_impute(df,num_cols=None,cat_cols=None):
	"""
	Data Imputation
	"""
	# fillna with mean value of the column (numeric feature with missing values)
	if num_cols is not None:
		for col in num_cols:
			df[col].fillna(df[col].mean(),inplace=True)	
	return df


def data_transform(df,cat_cols=None):
	"""
	Data Transformation
	"""
	# label encoding
	lbl = LabelEncoder()
	for col in cat_cols:
		df[col] = lbl.fit_transform(df[col].astype(str))
	return df


def feature_engineer(df):
	"""
	Feature Engineering (TBD)
	"""
	# Print/Plot histogram of all columns
	df.hist()
	plt.show()

	# Show collinearity plot between all columns
	#plt.matshow(df.corr())


def gridtune(clf,xtrain,ytrain,gridparams):
	"""
	HyperParameter Tuning
	"""
	grid = GridSearchCV(clf, gridparams, verbose=1, cv=5, n_jobs=-1)
	grid.fit(xtrain,ytrain)
	print("Best params after GridCV:\n", grid.best_params_)
	print("Best score after GridCV:\n",   grid.best_score_)

	return grid.best_params_


def get_xtrain(df,target=None):
	"""
	Generate Feature, Target for LightGBM
	"""
	y=df[target]
	X=df.drop(target,axis=1)
	return (X,y)

def get_dtrain(df,target=None, cat_cols=None):
	"""
	Generate Dataset for Training in LightGBM
	"""
	y=df[target]
	x_train=df.drop(target,axis=1)
	feature_name=x_train.columns.tolist()

	dtrain = lgb.Dataset(x_train,y,feature_name=feature_name,categorical_feature=cat_cols,free_raw_data=False)
	return dtrain

def plot_confusion_matrix(ytrue,ypred,classes,normalize=False,
	title='Confusion_matrix_plot',cmap=plt.cm.Blues):
	"""
	Plot Confusion Matrix
	"""
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	cm = confusion_matrix(ytrue, ypred)
	f, ax = plt.subplots()
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",
			color="white" if cm[i, j] > thresh else "black")
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.tight_layout()
	plt.savefig('confusion_matrix.png')


def main():
	"""
	Main Routine
	"""
	# Data input
	weblink="https://public-api.adsbexchange.com/VirtualRadar/AircraftList.json?trFmt=sa"
	infile ="input.data.json"
	tsfile ="test.data.json"

	#df=read_data(weblink=weblink,infile=infile,dump=True) #generate a dump json file
	training=read_data(weblink=None,infile=infile) #read training data from infile
	testing =read_data(weblink=None,infile=tsfile) #read testing data from tsfile

	# Check if unbalanced in target
	#print("[training data]:\n", training.groupby('WTC')['Id'].nunique())
	#print("[testing data]:\n", testing.groupby('WTC')['Id'].nunique())
	ntrain = training.shape[0]

	# Concat training and testing into one dataframe
	df = pd.concat([training,testing],axis=0)

	# Data Manipulation
	df=data_cleanse(df)

	# target column
	target = ['WTC']
	# numeric feature columns
	num_cols=['Alt','GAlt','Lat','Long','InHg','Spd','Trak','TAlt','TTrk','FSeen','PosTime','Year','Vsi']
	# categorical feature columns
	cat_cols=[ col for col in df.columns if col not in num_cols and col not in target ]
	#print("categorical columns: ", cat_cols)

	df = data_impute(df,num_cols=num_cols,cat_cols=None)
	df = data_transform(df,cat_cols=cat_cols)

	# Feature Engineering
	#feature_engineer(df)


	# Convert Data to LGB dataset
	dt = df[ntrain:]
	df = df[:ntrain]

	d_train=get_dtrain(df,target=target, cat_cols=cat_cols)
	x_train,y_train = get_xtrain(df,target=target)
	x_test, y_test  = get_xtrain(dt,target=target)

	# LightGBM Modeling
	params = {
		'task': 'train',
		'boosting_type': 'gbdt',
		'objective': 'multiclass',
		'num_class':4,
		'metric': 'multi_logloss',
		'learning_rate': 0.01,
		'max_depth': -1,
		'num_leaves': 256,
		'subsample': 0.9,
		'colsample_bytree': 0.8,
		'feature_fraction': 0.8,
		'bagging_fraction': 0.6,
		'bagging_freq': 10, 

		'min_split_gain': 0.5,
		'min_child_weight': 1, 
		'min_child_samples': 5 
		}

	# Hyperparameter tuning (Grid Search)
	clf = lgb.LGBMClassifier(
		boosting_type= 'gbdt', 
		objective = 'multiclass', 
		num_class = 4,  
		silent = True,
		max_depth = params['max_depth'],
		colsample_bytree = params['colsample_bytree'],
		subsample = params['subsample'], 
		min_split_gain = params['min_split_gain'], 
		min_child_weight = params['min_child_weight'], 
		min_child_samples = params['min_child_samples']
		)

	gridparams = {
		'learning_rate': [0.01],
		'n_estimators': [100],
		'num_leaves': [256], #best=256
		'boosting_type' : ['gbdt'], 
		'objective' : ['multiclass'],
		'max_depth':[16],#best=16
		'max_bin':[255], # large max_bin helps improve accuracy but might slow down training progress
		'colsample_bytree' : [0.6],#best=0.6
		'subsample' : [0.8], #best=0.8
		'bagging_freq':[10],  #need to tune for rf-boosting
		'bagging_fraction':[0.8], #best=0.8
		'feature_fraction': [0.6] #best=0.6
		}

	bst_params=gridtune(clf,x_train,y_train,gridparams)
	bst_params.pop('n_estimators')
	params.update(bst_params)
	print("params after optimization:\n ", params)

	# Stratified K-Fold Cross-Validation (on training data)
	lgb_cv = lgb.cv(params, d_train, num_boost_round=10000, nfold=5, shuffle=True, stratified=True, 
		verbose_eval=20, early_stopping_rounds=100)

	nround = lgb_cv['multi_logloss-mean'].index(np.min(lgb_cv['multi_logloss-mean']))
	print("Optimal Round from Cross-validation:\t", nround)

	# Retrain on full training data
	bst = lgb.train(params, d_train, num_boost_round=nround)
	bst.save_model('model.txt', num_iteration=nround)

	# Feature Importance Plot
	f, ax = plt.subplots(figsize=[10,10])
	lgb.plot_importance(bst, max_num_features=50, ax=ax)
	plt.title("Light GBM Feature Importance")
	plt.savefig('feature_importance.png')

	# Predict for testing data
	ypred = bst.predict(x_test, num_iteration=nround)
	print("LogLoss on Testing Data: ", log_loss(y_test.values,ypred))

	# Confusion Matrix for Testing Data
	yclaspred=[np.argmax(line) for line in ypred]
	ytrue = [i for item in y_test.values.tolist() for i in item ]
	class_names=['None','Light','Medium','Heavy']
	plot_confusion_matrix(ytrue,yclaspred,classes=class_names)
	
if __name__ == '__main__':
	main()

