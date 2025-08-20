# Human-Factors-Classifier-2.0

This repository provides a BERT-based classifier for automatically detecting Performance Shaping Factors (PSFs) in accident reports.
It builds on the MATA-D dataset (https://datacat.liverpool.ac.uk/1018/) where accidents have been manually assess for their PSFs using the CREAM.

Further discussion on the developed tool can be found in https://pureportal.strath.ac.uk/en/studentTheses/enhancing-safety-and-human-reliability-through-data-driven-and-nl and https://www.rpsonline.com.sg/proceedings/esrel2023/html/P294.html 

This project has two main components:
HF_Classifier_2.0_Training.py – Trains a separate BERT model for each PSF category.
HF_Classifier_2.0_NewReport.py – Loads the trained models and applies them to new accident reports, outputting predicted PSFs with confidence scores.

The ultimate goal is to expand MATA-D with new and modern accident reports, supporting research into human reliability and accident causation.

Setup
1. Clone the repo
  git clone https://github.com/yourusername/HF_Classifier_2.0.git
  cd HF_Classifier_2.0

2. Install dependencies
Recommend using Python 3.9+ with PyTorch and Hugging Face Transformers.
  pip install torch torchvision torchaudio
  pip install transformers
  pip install pandas scikit-learn tqdm nltk

3. Download NLTK stopwords
   import nltk
  nltk.download("stopwords")

4. Dataset
The training dataset is provided as MATA_D_VirtRaph_Complete.xlsx (derived from MATA-D).
It contains the Accident Description column and binary labels for each PSF.

Training
To train models for all PSF categories:
  python HF_Classifier_2.0_Training.py
  
A BERT model is trained per label in LABEL_COLUMNS.
Trained models are saved to Models/BERT_<LabelName>/.

Training uses:
  bert-base-uncased
  5 epochs, batch size 16
  Gradient accumulation for stability
  Mixed precision training (if CUDA is available)

Inference on New Reports
Prepare a plain text file (new_report.txt) with the description of the new accident/incident.
Then run:
  python HF_Classifier_2.0_NewReport.py
  
The script will:
  Clean and preprocess the text
  Chunk it into BERT-sized sequences
  Run predictions using each trained PSF model
  Output predicted PSFs with confidence scores

Notes
  Training can be GPU-intensive (one model per PSF).
  Models are binary classifiers (PSF present vs. absent).
  The inference script uses a confidence threshold (default = 0.6) to filter weak predictions.

Citation
If you use this code or models in your research, please cite (https://datacat.liverpool.ac.uk/1018/), (https://pureportal.strath.ac.uk/en/studentTheses/enhancing-safety-and-human-reliability-through-data-driven-and-nl) and (https://www.rpsonline.com.sg/proceedings/esrel2023/html/P294.html)
