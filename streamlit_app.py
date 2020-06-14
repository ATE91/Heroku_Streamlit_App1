import streamlit as st
import numpy as np
#Import Dataset
from sklearn import datasets
#Import Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Um daten von 3d in 2d zu bringen (für Plot)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#https://www.youtube.com/watch?v=Klqn--Mu2pE&feature=emb_logo
st.title("Test Streamlit App")

st.write("""
## Explore different classifier
Which one is the best?
""")

dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris","Breast Cancer", "Wine dataset"))

classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN","SVM", "Random Forest"))

def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris() # Dataset direkt aus SKLEARN laden
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    
    X = data.data
    y = data.target

    return X,y

X, y = get_dataset(dataset_name)

st.write("Shape of dataset:", X.shape)
st.write("Classes:", np.unique(y))
st.write("Number of classes:", len(np.unique(y)))


def add_parameter_UI(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15, 3) #Slider Widget um Parameter K auszuwählen
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0, 5.0)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15, 5) # Number of depth of each tree
        n_estimators = st.sidebar.slider("n_estimators", 1, 100, 10) # Number of Trees
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators

    return params

params = add_parameter_UI(classifier_name)
#add_parameter_UI(classifier_name) # Classifier Name aus Sidebar Widget

def get_classifier(clf_name, params):

    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"])
    else:
        clf = RandomForestClassifier(n_estimators=params["n_estimators"],
                                                max_depth=params["max_depth"],
                                                random_state=1234)
    return clf  

clf = get_classifier(classifier_name, params)

# Classification
st.write(clf)

X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1234)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

st.write(f"classifier = {classifier_name}")
st.write(f"accuracy = {acc}")

# Plot Dataset (erst PCA für 2d)
pca = PCA(2) # 2dimensionen behalten
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0] # alle samples, aber nur dimension 0
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.7, cmap="viridis")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.colorbar()

st.pyplot() #Streamlit plottet die daten statt plt.show()