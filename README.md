## **ACVPred: Enhanced Prediction of Anti-Coronavirus Peptides by data augmentation combined with Transfer Learning**

#### Datasets:

- ACVPs.csv: 154 ACVPs collected from ACovPepDB, PreAntiCoV  and iACVP.
- AVPs.csv: 1248 AVPs collected from PreAntiCoV and iACVP.
- nonAVPs.csv: 6296 non-AVPs collected from PreAntiCoV and iACVP.

#### Codes and model:

- augmentation.py: augmentation methods utilized in ACVPred
- model.py: prediction model structure and details in ACVPred
- predict.py: run this script to use ACVPred to apply prediction
- model.pth: model parameters

#### Usage:

run `python predict.py -i/--input [input_file] -o/--output [output_file]` in command

- **input_file:** route to your file containing peptides to be predicted in **FASTA** format.
- **output_file:** route to your result file containing prediction results in **CSV** format.