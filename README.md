# CNN_Seq2Seq

-   An end-to-end convolutional implementation, suitable for document-level generative summarization (though it performs better in translation tasks).

# TODO
-   [x] CNN Encoder
-   [x] CNN Decoder
-   [x] Multi-step Attention
-   [x] Dilation
-   [x] Log all outputs to Visdom
-   [ ] Transform CNN to fully connected in Decoder during inference
-   [ ] Adaptive Softmax
-   [ ] Efficient memory usage with fp-16
-   [ ] Expand to other tasks

# Environment
-   Python 3.7
-   Ubuntu 16.04
-   PyTorch 1.0
-   Visdom
-   **Note:** Configure FastText for pretrained embeddings if needed.

# Data Acquisition
-   Create a `data` folder and download the [CNNDM dataset](https://drive.google.com/open?id=1buWz_W4slL2GPt4EPYQI7Lf0kkHfAtLT).

# Preprocessing
-   Run `preprocess.ipynb`

# Hyperparameter Tuning
-   See `parameters.py`

# Training
-   `python train.py`

# Testing
-   `python infer.py`

# Notebooks for Data Processing
-	`data_presentation.ipynb`: Dataset statistics
-	`make_pretrained_embedding.ipynb`: Builds embedding matrix from FastText pretrained embeddings
-	`preprocess.ipynb`: Preprocessing for the CNNDM dataset
-	`tensor_test.ipynb`: Additional tests

# Python Scripts for Model Training and Testing
-	`conv_seq2seq.py`: End-to-end convolutional model, including encoder and decoder classes
-	`deprecated_code.py`: Deprecated code
-	`infer.py`: Model inference
-	`layers.py`: Custom weight-initialized fully connected, convolutional, and masked temporal convolutional layers
-	`loss.py`: Cross-entropy loss calculation for each time step in the decoder sequence, with masking
-	`paramcount.py`: Model parameter count
-	`parameters.py`: Model hyperparameters
-	`train.py`: Model training
-	`visualization.py`: Model computation graph visualization

# Temporary Directories
-   `model_check`: Monitors training progress, logs, recorded losses, and training outputs
-   `model_graph`: Backpropagation computation graph visualizations
-   `save_model`: Saved model files
-   `model_output`, `system_output`: Outputs for ROUGE evaluation of summarization

# Performance
![figure](https://github.com/thinkwee/CNN_Seq2Seq/blob/master/sample.png)
