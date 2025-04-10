# Transformer

## Description
This project implements a Transformer model for language translation from English to Italian. It includes training scripts and configuration settings to customize the model's behavior.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd Transformer
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
To train the model, run the following command:
```bash
python train.py
```

## Configuration
The `config.py` file contains various settings that can be adjusted:
- `batch_size`: Number of samples per batch.
- `num_epochs`: Number of training epochs.
- `lr`: Learning rate for the optimizer.
- `seq_len`: Sequence length for input data.
- `d_model`: Dimensionality of the model.
- `lang_src`: Source language (e.g., 'en' for English).
- `lang_tgt`: Target language (e.g., 'it' for Italian).

## Model Details
The Transformer model is based on the architecture described in the "Attention is All You Need" paper. It uses self-attention mechanisms to process input sequences and generate translations.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
