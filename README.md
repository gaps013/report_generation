# Report Generation
## Steps to run the script:

### Create a new environment with the libraries in requirements file 

### Edit the following parameters in the config indiana file
1) _C.IMAGE_PATH - path to the image folder
2) _C.TRAIN_JSON_PATH - path to train json file
3) _C.VAL_JSON_PATH -  path to val json file
4) _C.TEST_JSON_PATH - path to test json file
5) _C.VOCAB_FILE_PATH - path to vocab file genetrated by the vocab builder
6) _C.VOCAB_MODEL_PATH - path to vocab modek file genetrated by the vocab builder
7) _C.SAVED_DATASET = False - if dataset is already saved
8) INIT_PATH - path where the output should be saved. It should have the following structure 
  
  INIT_PATH:
  
        |-> CheckPoints
        
        |-> DataSet
        
        |-> Graphs

#### After adjusting the necessary hyperparameters run train_val_Indiana.py file to start training

*** Visual model file is not necessary 
