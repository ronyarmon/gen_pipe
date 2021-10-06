# generic_pipeline
A generic format to run and store the results of an ml pipeline's tests  
data: California housing prices dataset; numeric columns only      

## Main Scripts  
*run.py*: A generic ml pipeline: Each run of the pipeline builds a new subdirectory under '/resluts/' to hold the tables and plots produced in exploration, preprocessing and modelling.   
The user can set a name for the run, establish general run metadats to be used in naming the directory, or allow for a default naming (run1, run2, etc).  
*paths.py* : The paths to the modules and results directories  

## Modules    
Functions to use for data exploration, preprocessing and modelling. Users add other steps to fit their models and features,  
for example, the identification and encoding of categorical features, other models, etc.  
*vizz modules*: Functions to support plots generation and writing   
*ml modules*:  Functions to support exploratory data analysis, pre processing and modelling  
