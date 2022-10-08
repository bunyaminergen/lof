# importing libraries

import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors import NearestNeighbors

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

# using iris dataset

data = sns.load_dataset("iris")

data.head()

# writing a function for converting species column from string to numeric

def convert(species):
    if species == "setosa":
        return 0
    elif species == "versicolor":
        return 1
    elif species == "virginica":
        return 2
    else:
        raise ValueError("There is more than 3")
        
# applying to data the convert function

data["species"] = data["species"].apply(convert)

data['species'].value_counts()

data

lof = LocalOutlierFactor(n_neighbors = 17, contamination=0.1)

lof.fit_predict(data)

data_scores = lof.negative_outlier_factor_

# Sorting negative outlier factor scores 
np.sort(data_scores)[0:17]

# Choosing threshold value

tv = np.sort(data_scores)[7]

# Here I want to add the scores as a new column. it is optional.

data["LOF"] = data_scores.tolist()

data

# the sample corresponding to the threshold value

data[data_scores == tv]

# checking less than the threshold value and observing the outliers.

outliers = data[data_scores <= tv]
outliers

"""
Now we come to the topic. I need to talk a little bit about the Nearest Neighbors function before going into the codes.

The principle behind nearest neighbor methods is to find a predefined number of training samples closest in distance to the new point, and predict the label from these.
The number of samples can be a user-defined constant (k-nearest neighbor learning), or vary based on the local density of points (radius-based neighbor learning). 
The distance can, in general, be any metric measure: standard Euclidean distance is the most common choice. Neighbors-based methods are known as non-generalizing machine learning methods,
since they simply “remember” all of its training data (possibly transformed into a fast indexing structure such as a Ball Tree or KD Tree).

"""

def be(data: pd.DataFrame) -> pd.DataFrame:
  
  """
  Assing a variable to the Nearest Neighbors function and fit the data.

  """

  NN = NearestNeighbors(n_neighbors=5, radius=0.4, algorithm = 'kd_tree')
  NN.fit(data)
  
  abc = []

  cba = []

  for i in outliers.itertuples(index = False, name = 'Miuul'):
    
    """
    # kneighbors methodunu kullanarak en yakın komşuları buluyoruz
    Note: kneighbors methodunu kullanırken n_neighbors parametresini 2 olarak seçmelisiniz.
    
    """

    ab, ba = NN.kneighbors([list(i)], 2, return_distance=False)[0]

    abc.append(data.iloc[ab])
    cba.append(data.iloc[ba])

    # printing index number of outliers and nearest neighbor
    print(ab,ba)
    
  print("_" * 75)

  # showing OUTLIERS
  print("OUTLIERS")
  display(pd.DataFrame(abc))

  # showing NEAREST NEIGBORHS
  print("NEAREST NEIGBORHS")
  display(pd.DataFrame(cba))
  
be(data)

# Same function as above, except I will assign outliers to their nearest neighbors

def be_app(data: pd.DataFrame) -> pd.DataFrame:
  
  NN = NearestNeighbors(n_neighbors=2, radius=0.4, algorithm = 'kd_tree')
  NN.fit(data)
  
  abc = []

  cba = []

  for i in outliers.itertuples(index = False, name = 'Miuul'):

    ab, ba = NN.kneighbors([list(i)], 2, return_distance = False)[0]

    # basicly assigning outliers to nearest neighbors
    # Note: I named  as ab and ba, be careful not to confuse the two.

    data.iloc[ab] = data.iloc[ba]

    abc.append(data.iloc[ab])
    cba.append(data.iloc[ba])

    # printing index number of outliers and nearest neighbor
    print(ab,ba)
    
  print("_" * 75)

  # showing OUTLIERS
  print("OUTLIERS")
  display(pd.DataFrame(abc))

  # showing NEAREST NEIGBORHS
  print("NEAREST NEIGBORHS")
  display(pd.DataFrame(cba))
  
be_app(data)

data.iloc[[14]]

data.iloc[[131]]

# Bünyamin Ergen

"""
##### Resources

https://scikit-learn.org/stable/modules/neighbors.html

https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html

https://en.wikipedia.org/wiki/Local_outlier_factor

https://www.veribilimiokulu.com/local-outlier-factor-ile-anormallik-tespiti/
"""


https://towardsdatascience.com/local-outlier-factor-lof-algorithm-for-outlier-identification-8efb887d9843
