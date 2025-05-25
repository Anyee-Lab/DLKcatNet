# DLKcatNet

Introduction
------------

The **DLKcatNet** is a dotnet program for running model DLKcat which 
supplies a deep-learning based prediction tool for kcat prediction.

Usage

- **For users who want to use DLKcatNet, please run these command lines at the terminal:**

    (1). Download the DLKcatNet package. 
    ```bash

         git clone https://github.com/Anyee-Lab/DLKcatNet.git
    ```

    (2). Download python_embedded Version 3.7.6. 
    ```http

         https://www.python.org/ftp/python/3.7.6/python-3.7.6-embed-amd64.zip
    ```

    (3). Install embedded python required package. 
    ```bash

         C:\Users\Anyee\AppData\Local\Programs\Python\Python37\python.exe -m pip install numpy==1.20.2 requests rdkit-pypi scikit-learn==0.23.2 -t C:\Users\Anyee\source\repos\FsTest\NetPython\python_embedded\Lib\site-packages
         
         C:\Users\Anyee\AppData\Local\Programs\Python\Python37\python.exe -m pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 typing-extensions==4.7.1 --extra-index-url https://download.pytorch.org/whl/cu116 --upgrade -t C:\Users\Anyee\source\repos\FsTest\NetPython\python_embedded\Lib\site-packages
    ```

    (4). Unzip the ``input.zip`` file under the ``Data`` directory.
    ```bash

         unzip DLKcat/DeeplearningApproach/Data/input.zip
    ```

    (5). Build it
    ```bash

         dotnet build
    ```
