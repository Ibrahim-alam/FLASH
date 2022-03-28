# FLASH
Federated Learning for Combined Algorithm and Hyper parameter optimization

Function wise description
  'Clf_param_api' : it is the space of HPs for HyperOPT.
  'Clf_parameter_names': Contains the names of all the HPs to change in HPO (Although a bit redundant, only Clf_param_api can be used).
  'Clf_parameters_assign': Given classifiers and HPs, it assigns HP to classifiers.
  'Clf_param' : Random generation of HPs for different Classifiers.
  'clean_data' : given the raw dataset, make the dataset usable
  'fetch_data' : given data, splits data  in train, holdout, test
  'Top_Rand_HP_score' : Given the classifier and HP combinations, calculate the ensemble accuracy
  'main_func_FLASH' : DAUB + HPO in FL setting
  
File/version Specific functions:
  'HPO_combined_client': HyperOPT with centralized HPO
  
  
  
