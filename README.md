# GPLCMR
# **DPTCMR**

## **Project Overview**
DPTCMR is a **recommendation system training project** based on **Prompt learning** and **LightGCN**.  
It supports **multi-country datasets** and includes **model training, evaluation, and data processing** functionalities.


## **Directory Structure**
```
DPTCMR/ │── data/ # Contains user ratings, item information, and other dataset files │── model/ # Stores trained model weight files │── README.md # Project documentation, explaining usage and code structure │── config.py # Configuration file, including training parameters and data paths │── data.py # Data loading and preprocessing, including users, items, and interaction data │── evaluate.py # Model evaluation, including BPR loss calculation and metric computation │── model.py # Definition of LightGCN and Prompt learning models, including message passing and normalization │── prompt.py # Prompt-related code, including global embedding enhancement and attention mechanism │── train.py # Training script, covering model training, optimization, prompt tuning, and distillation learning
```


## **Running Instructions**
1. Ensure that **Python** and required dependencies are installed.
2. Run the following command to **train the model**:
    ```bash
    python train.py
    ```
3. Run the following command to **evaluate the model**:
    ```bash
    python evaluate.py
    ```

## **Dependencies**
Ensure the following Python dependencies are installed (**or refer to `requirements.txt`**):
```bash
pip install torch numpy pandas torch_geometric torch_sparse torch_scatter scikit-learn
