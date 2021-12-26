from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import pickle
from joblib import dump, load

# Load the iris data
X, y = datasets.load_iris(return_X_y=True)

# Create a matrix, X, of features and a vector, y.


clf = LogisticRegression(random_state=42)
clf.fit(X, y)  

saved_model = pickle.dumps(clf)
dump(clf, 'sample_model.pkl')