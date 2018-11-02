Predicting Wave Turbulence Category
==========================
---------------
Input Files:
---------------
input.data.json: one snapshot at Nov 1st, 2018
test.data.json : one snapshot at Nov 2nd, 2018

Both JSON files are from Weblink http://public-api.adsbexchange.com/VirtualRadar/AircraftList.json?trFmt=sa

---------------
Output File:
---------------
- model.txt: The machine learning model by LightGBM
- feature_importance.png: Top 50 important features in best model (model.txt)
- confusion_matrix.png: Confusion Matrix Plot for Testing Data by best model (model.txt)
- log: a log file for screen output

---------------
How-to-Run:
---------------
python3 run.py > log

---------------
Requirements (Recommended):
---------------
- Python 3.6.3
- Scikit-Learn 0.19.1
- Pandas 0.21.0
- Numpy 1.13.3
- LightGBM 2.2.1