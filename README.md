This repository contains LSTM model which classifies whether  a Bangla comment is sarcastic or not. I have not uploaded the dataset. Please, download the dataset into the project directory to run the code successfully. First, to run the code, we need to install specific version of torch and torchtext to avoid version conflicts.

`pip install -U torch==1.8.0 torchtext==0.9.0`

Next, xx_ent_wiki_sm is used to tokenize Bangla text so we need to run the following command to download it. 

`python -m spacy download xx_ent_wiki_sm`

**FILE Descriptions**
- **dataloader.py** : Contains code to split the dataset into train, validation, and test sets.
- **lstm_model.py** : LSTM model is defined here along with the explanation of parameters of network inside the \_\_init_\_\() function, output shape of each line in forward function using a dummy input, and the significance of shape of output of each line of forward function. To see the output shape, run lstm_model.py as:
  
  `python3 lstm_model.py`

- **training.py** : Contains code to train, validate, and test the LSTM. Saves checkpoints when loss decreases.
- **inference.py** : Contains the code to infer Bengali comment by loading saved model. 
- **main.py** : Main script to run. All the above mentioned module are imported in main.py. It has 2 modes:
  
    1. **Training mode**: To run in training mode, execute:

        `python3 main.py -t Y`
    2. **Inferencing mode**: To run in inferencing mode, execute:

         `python3 main.py -t N`

    Here, "-t" i.e. "--training" flag denotes if we want to activate training or inferencing mode. Default is training mode. During inferencing, the activate_inference_mode() function automatically reads from test data and prints prediction in console. User is not prompted to enter Bangla comment due to encoding issue of Bangla font in command line.

