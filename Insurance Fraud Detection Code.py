


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import warnings    
warnings.filterwarnings('ignore')
plt.style.use('ggplot')   

dff=pd.read_csv('/content/insurance_claims.csv')

dff.head()

dff.replace('?',np.nan,inplace=True)

dff.info()

dff.describe()

dff.isna().sum()

dff['authorities_contacted'].fillna(0,inplace=True)

dff.info()

dff.describe()

dff.isna().sum()

import missingno as msno
msno.bar(dff)
plt.show()



dff['collision_type']=dff['collision_type'].fillna(dff['collision_type'].mode()[0])

dff['property_damage']=dff['property_damage'].fillna(dff['property_damage'].mode()[0])

dff['police_report_available']=dff['police_report_available'].fillna(dff['police_report_available'].mode()[0])

dff.isna().sum()

dff.drop(columns = ['age', 'total_claim_amount'], inplace = True, axis = 1)



to_drop = ['policy_number','policy_bind_date','policy_state','insured_zip','incident_location','incident_date',
           'incident_state','incident_city','insured_hobbies','auto_make','auto_model','auto_year', '_c39']

dff.drop(to_drop, inplace = True, axis = 1)

dff.head()

dff.nunique()

x = dff.drop('fraud_reported', axis = 1)
y = dff['fraud_reported']

dff.info()

cat_dff=x.select_dtypes(include=['object'])

cat_dff.head()

for col in cat_dff.columns:
  print(f"{col}:\n{cat_dff[col].unique()}\n")

cat_dff=pd.get_dummies(cat_dff,drop_first=True)

cat_dff.head()

num_dff=x.select_dtypes(include=['int64'])

num_dff.head()

x=pd.concat([num_dff,cat_dff],axis=1)

x.head()

plt.figure(figsize=(25,20))

plotnum=1

plt.figure(figsize = (25, 20))
plotnumber = 1

for col in x.columns:
    if plotnumber <= 24:
        ax = plt.subplot(5, 5, plotnumber)
        sns.distplot(x[col])
        plt.xlabel(col, fontsize = 15)

    plotnumber += 1

plt.tight_layout()
plt.show()

# Outliers Detection
plt.figure(figsize=(20,15))
plotnumber=1
for col in x.columns:
  if plotnumber <= 24:
    ax = plt.subplot(5, 5, plotnumber)
    sns.boxplot(x[col])
    plt.xlabel(col, fontsize = 15)
  plotnumber += 1
plt.tight_layout()
plt.show()



from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

x_train.head()

num_dff = x_train[['months_as_customer', 'policy_deductable', 'umbrella_limit',
       'capital-gains', 'capital-loss', 'incident_hour_of_the_day',
       'number_of_vehicles_involved', 'bodily_injuries', 'witnesses', 'injury_claim', 'property_claim',
       'vehicle_claim']]


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

scaled_data=scaler.fit_transform(num_dff)

scaled_num_dff = pd.DataFrame(data = scaled_data, columns = num_dff.columns, index = x_train.index)
scaled_num_dff.head()

x_train.drop(columns=scaled_num_dff.columns,inplace=True)

x_train=pd.concat([scaled_num_dff,x_train],axis=1)

x_train.head()

# -----------MODELS---------------

# importing ----------SVM (SUPPORT VECTOR MACHINES)----------------------

from sklearn.svm import SVC
svm=SVC()

svm.fit(x_train,y_train)

y_pred=svm.predict(x_test)

#  ----------Accuracy Score---------------

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

svm_train_acc = accuracy_score(y_train, svm.predict(x_train))
svm_test_acc = accuracy_score(y_test, y_pred)

print(f"Training accuracy of Support Vector Classifier is : {svm_train_acc}")
print(f"Test accuracy of Support Vector Classifier is : {svm_test_acc}")

print("\n",confusion_matrix(y_test, y_pred))
print("\nclassification report is....\n",classification_report(y_test, y_pred))

#=====================KNN MODEL===================

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(x_train, y_train)

y1_pred = knn.predict(x_test)

knn_train_acc = accuracy_score(y_train, knn.predict(x_train))
knn_test_acc = accuracy_score(y_test, y1_pred)

print(f"Training accuracy of KNN is : {knn_train_acc}")
print(f"\nTest accuracy of KNN is : {knn_test_acc}")

print("\n",confusion_matrix(y_test, y1_pred))
print("\nClassification Report is------\n",classification_report(y_test, y1_pred))

from sklearn.model_selection import train_test_split, GridSearchCV



param_grid = {
    'n_neighbors': [113, 2225, 557, 779],  # Test different values of K
    'weights': ['uniform', 'distance'],  # Test different weighting schemes
    'metric': ['euclidean', 'manhattan']  # Test different distance metrics
}


grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(x_train, y_train)


print("Best hyperparameters:", grid_search.best_params_)

# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
yy_pred = best_model.predict(x_test)
accuracy_knn = accuracy_score(y_test, yy_pred)
print("Accuracy on test set:", accuracy_knn)

#    DECISION TREEE
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)

y2_pred = dtc.predict(x_test)

dtc_train_acc = accuracy_score(y_train, dtc.predict(x_train))
dtc_test_acc = accuracy_score(y_test, y2_pred)

print(f"Training accuracy of Decision Tree is : {dtc_train_acc}")
print(f"Test accuracy of Decision Tree is : {dtc_test_acc}")

print(confusion_matrix(y_test, y2_pred))
print(classification_report(y_test, y2_pred))

# hyper parameter Tuning

from sklearn.model_selection import GridSearchCV
grid_params = {
    'criterion' : ['gini', 'entropy'],
    'max_depth' : [3, 5, 7, 10],
    'min_samples_split' : range(2, 10, 1),
    'min_samples_leaf' : range(2, 10, 1)
}

grid_search = GridSearchCV(dtc, grid_params, cv = 5, n_jobs = -1, verbose = 1)
grid_search.fit(x_train, y_train)

# best parameters and best score

print(grid_search.best_params_)
print(grid_search.best_score_)

# best estimator

dtc = grid_search.best_estimator_

y2_pred = dtc.predict(x_test)

dtc_train_acc = accuracy_score(y_train, dtc.predict(x_train))
dtc_test_acc = accuracy_score(y_test, y2_pred)

print(f"Training accuracy of Decision Tree is : {dtc_train_acc}")
print(f"Test accuracy of Decision Tree is : {dtc_test_acc}")

print("\n",confusion_matrix(y_test, y2_pred))
print("\nClassification Report is-----\n",classification_report(y_test, y2_pred))

from sklearn.tree import plot_tree
plot_tree(dtc)

# =========== RANDOM FOREST=================

from sklearn.ensemble import RandomForestClassifier

rand_clff = RandomForestClassifier(criterion= 'entropy', max_depth= 10, max_features= 'sqrt', min_samples_leaf= 1, min_samples_split= 3, n_estimators= 140)
rand_clff.fit(x_train, y_train)

y3_pred = rand_clff.predict(x_test)

rand_clff_train_acc = accuracy_score(y_train, rand_clff.predict(x_train))
rand_clff_test_acc = accuracy_score(y_test, y3_pred)

print(f"Training accuracy of Random Forest is : {rand_clff_train_acc}")
print(f"Test accuracy of Random Forest is : {rand_clff_test_acc}")

print(confusion_matrix(y_test, y3_pred))
print(classification_report(y_test, y3_pred))

#=============== GRADIENT BOOSTING=====================

from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier()
gb.fit(x_train, y_train)

# accuracy score, Classification Report and Confusiion matrix

gb_acc = accuracy_score(y_test, gb.predict(x_test))
gb_accu=accuracy_score(y_train,gb.predict(x_train))
print(f"Gradient Boosting classifier Accuracy is {accuracy_score(y_train, gb.predict(x_train))}\n")
#print(f"Test Accuracy of Gradient Boosting Classifier is {gb_acc} \n")

print(f"\nConfusion Matrix :- \n{confusion_matrix(y_test, gb.predict(x_test))}\n")
print(f"\nClassification Report :- \n {classification_report(y_test, gb.predict(x_test))}")

models = pd.DataFrame({
    'Model' : ['SVC', 'KNN', 'Decision Tree', 'Random Forest', 'Gradient Boost'],
    'Score' : [svm_test_acc, accuracy_knn, dtc_test_acc, rand_clff_test_acc, gb_accu]
})


models.sort_values(by = 'Score', ascending = False)

