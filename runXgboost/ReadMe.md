#xgboost regressor pipeline
## Data
Boston and California housing datasets  
Numeric features  
The targets to predict are the property prices
##scripts
1. **eda**  
Explore the dataset and modify where releavant 
   via the following stages:  
   Explore:  
   a. Descriptive statistics of features
   b. Nulls count   
      result: nulls count and percentiles per column  
      decision: Drop or Fill nulls, if Fill, how?  
   c. Outliers detection 
      method: zscore (stds from mean; default threshold=3)    
      result: outliers summary  
      decision: Drop outliers?
   
2. 

