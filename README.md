# Bert for DA


# Installation
1. Create virtual environment with Python 3.7+
2. Run following commands:
```
git clone https://github.com/davidtw999/BertDA.git
cd learningFeedback
pip install -r requirements.txt
```

# Organization
The repository is organized as the following subfolders:

1. `src`: source code
2. `scripts`: scripts for running experiments
3. `data`: folder for datasets
4. `models`: saved models from running experiments

## Feature processing 
To generate tsv file for train and test dataset by parsing the csv file, run:

1. `python feature_embedding.py --wFlag train --train train --levelFlag firstlevel`
2. `python feature_embedding.py --wFlag test --test test --levelFlag firstlevel`
3. `python feature_embedding.py --wFlag train --train train --levelFlag secondlevel`
4. `python feature_embedding.py --wFlag test --test test --levelFlag secondlevel`

## Fine-tune model on full training dataset
To simply generate a model on the training dataset, run 

`bash scripts/train.sh`  

The model will be saved under a `base` in `models` directory.  
Results will be saved in `eval_results.txt`.

You can modify the parameters in `scripts/train.sh` for model development.

## Run test on the fine-tuned model 
`python -m src.test --models models/$SEED/$TASK_NAME/base`

You need to change the value of the seed and the task_name. The result will save into csv file
in your root directory

## Use the jupyter notebook Ensemble file to ensemble the multiple models

You need to manually move the result csv file into the test_fl or test_sl folders in PredOutcome directory.
Then run the jupyter notebook script with the valid diretory and file names.