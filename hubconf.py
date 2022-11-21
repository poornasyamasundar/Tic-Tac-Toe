from sklearn import datasets
from torch import nn
import torch.optim as optim
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_blobs,make_circles
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_completeness_v_measure

# You can import whatever standard packages are required

# full sklearn, full pytorch, pandas, matplotlib, numpy are all available
# Ideally you do not need to pip install any other packages!
# Avoid pip install requirement on the evaluation program side, if you use above packages and sub-packages of them, then that is fine!

###### PART 1 ######

def get_data_blobs(n_points=100):
  pass
  # write your code here
  # Refer to sklearn data sets
  X, y = make_blobs(n_samples=n_points,random_state=0)
  # write your code ...
  return X,y

def get_data_circles(n_points=100):
  pass
  # write your code here
  # Refer to sklearn data sets
  X,y = make_circles(n_samples=n_points, random_state=0, noise = 0.4)
  # write your code ...
  return X,y

def get_data_mnist():
  pass
  # write your code here
  # Refer to sklearn data sets
  digits = datasets.load_digits()
  X = digits.images
  X = X.reshape(-1)
  y = digits.target
  # write your code ...
  return X,y

def build_kmeans(X=None,k=10):
  pass
  # k is a variable, calling function can give a different number
  # Refer to sklearn KMeans method
  km = KMeans(n_clusters=k,max_iter=1000,algorithm='auto',random_state=0)
  km.fit(X)
  # write your code ...
  return km

def assign_kmeans(km=None,X=None):
  pass
  # For each of the points in X, assign one of the means
  # refer to predict() function of the KMeans in sklearn
  # write your code ...
  ypred = km.predict(X)
  return ypred

def compare_clusterings(ypred_1=None,ypred_2=None):
  pass
  # refer to sklearn documentation for homogeneity, completeness and vscore
  h,c,v = 0,0,0 # you need to write your code to find proper values
  h,c,v = homogeneity_completeness_v_measure(ypred_1, ypred_2)
  return h,c,v

def build_lr_model(X=None, y=None):
  lr_model = None
  # write your code...
  # Build logistic regression, refer to sklearn
  lr_model = LogisticRegression(solver = 'liblinear',random_state=0).fit(X,y)
  return lr_model

def build_rf_model(X=None, y=None):
  pass
  rf_model = RandomForestClassifier(max_depth=5, random_state=0)
  rf_model = rf_model.fit(X,y)
  # write your code...
  # Build Random Forest classifier, refer to sklearn
  return rf_model

def get_metrics(model=None,X=None,y=None):
  # Obtain accuracy, precision, recall, f1score, auc score - refer to sklearn metrics
  acc, prec, rec, f1, auc = 0,0,0,0,0
  y_true = y
  y_pred = model.predict(X)
  dummy = 0
  prec, rec, f1, dummy = precision_recall_fscore_support(y_true, y_pred, average='macro')
  acc = accuracy_score(y_true,y_pred)
  auc = roc_auc_score(y_true,model.predict_proba(X), multi_class='ovr')
  # write your code here...
  return acc, prec, rec, f1, auc

def get_paramgrid_lr():
  # you need to return parameter grid dictionary for use in grid search cv
  # penalty: l1 or l2
  lr_param_grid = None
  lr_param_grid = {"penalty" : ["l1", "l2"]}
  # refer to sklearn documentation on grid search and logistic regression
  # write your code here...
  return lr_param_grid

def get_paramgrid_rf():
  # you need to return parameter grid dictionary for use in grid search cv
  # n_estimators: 1, 10, 100
  # criterion: gini, entropy
  # maximum depth: 1, 10, None  
  rf_param_grid = None
  rf_param_grid = {"n_estimators":[1,10,100], "criterion" : ["gini","entropy"], "max_depth":[1,10,None]}
  # refer to sklearn documentation on grid search and random forest classifier
  # write your code here...
  return rf_param_grid

def perform_gridsearch_cv_multimetric(model=None, param_grid=None, cv=5, X=None, y=None, metrics=['accuracy','roc_auc']):
  
  # you need to invoke sklearn grid search cv function
  # refer to sklearn documentation
  # the cv parameter can change, ie number of folds  
  
  # metrics = [] the evaluation program can change what metrics to choose
  
  top1_scores = []

  grid_search_cv = None
  
  grid_search_cv = GridSearchCV(model, param_grid, scoring = metrics[0])
  grid_search_cv.fit(X,y)
  
  top1_scores.append(grid_search_cv.best_score_)

  grid_search_cv = GridSearchCV(model, param_grid, scoring = metrics[1])
  grid_search_cv.fit(X,y)
  
  top1_scores.append(grid_search_cv.best_score_)
  # create a grid search cv object
  # fit the object on X and y input above
  # write your code here...
  
  # metric of choice will be asked here, refer to the-scoring-parameter-defining-model-evaluation-rules of sklearn documentation
  
  # refer to cv_results_ dictonary
  # return top 1 score for each of the metrics given, in the order given in metrics=... list
  
  return top1_scores

class MyNN(nn.Module):
  def __init__(self,inp_dim=64,hid_dim=13,num_classes=10):
    super(MyNN,self).__init__()
    
    # self.fc_encoder = None # write your code inp_dim to hid_dim mapper
    self.flatten = nn.Flatten()
    self.fc_encoder = nn.Linear(inp_dim,hid_dim)
    self.fc_decoder = nn.Linear(hid_dim,inp_dim)
    self.fc_classifier = nn.Linear(hid_dim,num_classes)
    # self.fc_decoder = None # write your code hid_dim to inp_dim mapper
    # self.fc_classifier = None # write your code to map hid_dim to num_classes
    
    # self.relu = None #write your code - relu object
    self.relu = nn.ReLU()
    self.softmax = nn.Softmax(dim=1)
    
  def forward(self,x):
    # print(x.shape)
    x = self.flatten(x)
    # print(x.shape)
    x_enc = self.fc_encoder(x)
    # print(x_enc.shape)
    x_enc = self.relu(x_enc)
    y_pred = self.fc_classifier(x_enc)
    # print(y_pred.shape)
    y_pred = self.softmax(y_pred)
    # print(y_pred.shape)
    x_dec = self.fc_decoder(x_enc)
    # print(x_dec.shape)
    return y_pred, x_dec
  
  # This a multi component loss function - lc1 for class prediction loss and lc2 for auto-encoding loss
  def loss_fn(self,x,yground,y_pred,xencdec):
    lc1 = 0 # write your code for cross entropy between yground and y_pred, advised to use torch.mean()
    # print(yground.shape)
    # print(y_pred.shape)
    lc1 = -(yground * torch.log(y_pred + 1e-4))
    lc1 = torch.sum(lc1)
    # auto encoding loss
    x = self.flatten(x)
    lc2 = torch.mean((x - xencdec)**2)
    
    lval = lc1 + lc2
    
    return lval
    
def get_mynn(inp_dim=64,hid_dim=13,num_classes=10):
  mynn = MyNN(inp_dim,hid_dim,num_classes)
  mynn.double()
  return mynn

def get_mnist_tensor():
  # download sklearn mnist
  # convert to tensor
  X, y = None, None
  digits = datasets.load_digits()
  X = digits.images
  y = digits.target
  X = torch.tensor(X)
  y = torch.tensor(y)
  # write your code
  return X,y

def get_loss_on_single_point(mynn=None,x0=None,y0= None):
  x0 = x0.reshape(1,x0.shape[0], x0.shape[1])
  y_pred, xencdec = mynn(x0)
  # print(x0.shape, y0.shape,y_pred.shape,xencdec.shape)
  lossval = mynn.loss_fn(x0,y0,y_pred,xencdec)
  # the lossval should have grad_fn attribute set
  return lossval

def train_combined_encdec_predictor(mynn=None,X=None,y=None, epochs=11):
  # X, y are provided as tensor
  # perform training on the entire data set (no batches etc.)
  # for each epoch, update weights
  
  optimizer = optim.SGD(mynn.parameters(), lr=0.01)
  y = nn.functional.one_hot(y, 10)
  for i in range(epochs):
    ypred, Xencdec = mynn(X)
    lval = mynn.loss_fn(X,y,ypred,Xencdec)

    optimizer.zero_grad()
    lval.backward()
    optimizer.step()
    
  return mynn
