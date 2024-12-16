# Step 1: Import Statements
# import libraries
import re
import heapq
import numpy as np
import pandas as pd
from urllib.request import urlopen
from tqdm import tqdm  # Import tqdm for progress bar

# Stopword dictionary
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk import download, wordnet
nltk.download('omw-1.4')
download('punkt')
nltk.download('punkt_tab')
download('stopwords')
download("wordnet")

# import gensim
from gensim.parsing.preprocessing import remove_stopwords
# For stemming
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer()
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split as tts, KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score as ac, confusion_matrix as cm, ConfusionMatrixDisplay as CMD
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

print("IMPORTS DONE")

# Step 1: Preprocessing
# Instructions: "For this question, you use the same data and pre-processing as in Question 3."
#wine_data = pd.read_csv('https://github.com/dvasiliu/AML/blob/main/Data%20Sets/winemagdata130kv2.csv?raw=true',quoting=2)

wine_data = pd.read_csv('winemagdata130kv2.csv',quoting=2)

wines = wine_data[["description","points"]]
wines_subset = wines.sample(10000,random_state=1693).reset_index(drop=True)
def text_preprocess(original_documents):
  reviews = original_documents.description.values + ' '+'fakeword'+' '
  joined_reviews = ' '
  for i in range(len(reviews)):
    joined_reviews = joined_reviews+reviews[i]
  # here we do the text pre-processing very fast
  # remove punctuation
  descriptions = re.sub('[^a-zA-Z0-9 ]','',joined_reviews)
  # remove stopwords
  descriptions = remove_stopwords(descriptions.lower())
  # we can use Porter Stemmer or we can Lemmatize
  # for Porter Stemmer
  #descriptions = [stemmer.stem(word) for word in descriptions.split()]
  # next we can stem or lemmatize
  descriptions = [lemmatizer.lemmatize(word) for word in descriptions.split()]
  descriptions = " ".join(descriptions)
  documents = descriptions.split('fakeword')
  documents = documents[:-1]
  return documents

documents = text_preprocess(wines_subset)

print("PREPROCESSING DONE")

vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(documents)

print("TFIDF VECTORIZER DONE")

y = wines_subset["points"]
y = y.where(y>90,other="Inferior").where(y<=90,other="Superior").values

print("TARGET VARIABLE DONE")

# Step 2: SVM with Radial Kernel
# Instructions: "Using the support vector machine model with a radial kernel, we want to find the values of the gamma and C hyperparameters 
#                such that correct predictions of wines being Superior is maximized. For this, you will use a 5-fold stratified cross-validated 
#                area (with random state remaining 1693) under the receiver operating characteristic curve ('auc') as a metric for evaluation 
#                and the algorithm of simulated annealing presented in class. Use a cooling rate of 0.98, 30 iterations, an initial temperature 
#                of 1, and bounds of (0.01, 0.1), (1, 10) (C, gamma)."

# Objective function to evaluate AUC for a given (C, gamma)
def objective_function_svm(params, X_train, y_train):
    C, gamma = params
    model = SVC(kernel='rbf', C=C, gamma=gamma, probability=True, random_state=1693)

    print("MADE IT HERE 2.1")

    # 5-fold stratified cross-validation using cross_val_score
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1693)

    print("MADE IT HERE 2.2")
    
    auc_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')

    print("MADE IT HERE 2.3")
    
    # Return the negative AUC score since we are minimizing in simulated annealing
    return -np.mean(auc_scores)

# Generate a random candidate solution for (C, gamma)
def get_candidate_solution_svm(current_solution, bounds, temp):
    C_new = current_solution[0] + np.random.normal() * (bounds[0][1] - bounds[0][0]) * temp
    gamma_new = current_solution[1] + np.random.normal() * (bounds[1][1] - bounds[1][0]) * temp
    
    #make sure C is within bounds (positive)
    C_new = np.clip(C_new, bounds[0][0], bounds[0][1])

    # Make sure gamma is within bounds (positive)
    gamma_new = np.clip(gamma_new, bounds[1][0], bounds[1][1])
    
    return [C_new, gamma_new]

# Simulated Annealing Algorithm for SVM Hyperparameter Tuning with tqdm
def simulated_annealing_svm(objective, bounds, max_iterations, initial_temp, cooling_rate, X_train, y_train):
    print("MADE IT HERE 1")
    # Initialize with random values for C and gamma within the bounds
    current_solution = [np.random.uniform(bounds[i][0], bounds[i][1]) for i in range(len(bounds))]
    print("MADE IT HERE 2")
    current_solution_cost = objective(current_solution, X_train, y_train)
    print("MADE IT HERE 3")
    best_solution = current_solution[:]
    print("MADE IT HERE 4")
    best_solution_cost = current_solution_cost
    print("MADE IT HERE 5")
    temp = initial_temp

    # Wrap the iteration loop with tqdm to show a progress bar
    for iteration in tqdm(range(max_iterations), desc="Simulated Annealing Progress", ncols=100):
        print("MADE IT HERE ?")
        # Get a candidate solution
        candidate_solution = get_candidate_solution_svm(current_solution, bounds, temp)
        candidate_solution_cost = objective(candidate_solution, X_train, y_train)

        # Check if the candidate solution is better
        if candidate_solution_cost < current_solution_cost:
            current_solution, current_solution_cost = candidate_solution, candidate_solution_cost
            if candidate_solution_cost < best_solution_cost:
                best_solution, best_solution_cost = candidate_solution, candidate_solution_cost
        else:
            # Accept worse solutions with a certain probability
            if np.random.uniform(0, 1) < np.exp((current_solution_cost - candidate_solution_cost) / temp):
                current_solution, current_solution_cost = candidate_solution, candidate_solution_cost

        # Decrease the temperature
        temp *= cooling_rate

        # Optional: Output information every 500 iterations
        if (iteration + 1) % 500 == 0:
            print(f"Iteration {iteration+1}, Temperature: {temp:.2f}, Current Solution: {current_solution}, Current Cost: {-current_solution_cost:.4f}")

    return best_solution, -best_solution_cost  # Return the best solution and its corresponding AUC

print("SIMULATED ANNEALING CLASS DEFINITIONS DONE")

# Define parameters
bounds = [(0.01, 0.1), (1, 10)]  #bounds for (C, gamma)
max_iterations = 30  #Number of iterations
initial_temp = 1  # initial temperature
cooling_rate = 0.98  # Cooling rate

print("BOUNDS SET")

# run Simulated Annealing to optimize C and gamma
best_solution_svm, best_auc_svm = simulated_annealing_svm(
    objective_function_svm, bounds, max_iterations, initial_temp, cooling_rate, x, y
)

print(f"Best C and gamma: {best_solution_svm}, Best AUC: {best_auc_svm:.4f}")

# To predict AUC on the test set
best_C, best_gamma = best_solution_svm
final_model = SVC(kernel='rbf', C=best_C, gamma=best_gamma, probability=True, random_state=1693)
final_model.fit(x, y)
y_pred_prob = final_model.predict_proba(x)[:, 1]
final_auc = roc_auc_score(y, y_pred_prob)

print(f"The best AUC on new/test data: {final_auc:.4f}")