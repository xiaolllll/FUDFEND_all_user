## Quick Start

0. Prepare the environment
    ```
    conda create -n your_env python=3.9
    conda activate your_env
    pip install -r requirements.txt
    ```
1. Download Datasets
    You need Download the Dataset Twibot-20.
    Twibot-20:[link](https://github.com/BunsenFeng/TwiBot-20)
2. Pre-process dataset
    You need preprocess the dataset, the preprocess_1.py and preprocess_2.py shows the preprocess code.
    You should preprocess the dataset first.
    ```
    python preprocess.py
    ```
3. Start

    ```
    python read_data_form_encode_domain.py
    ```