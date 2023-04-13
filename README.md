## I. Abstract
The goal of this framework is to simplify the process you build and test a text classifier 
without worrying too much about the model implementation details.


## II. Set up environment
Running the following script, to make sure that all packages are installed in your project environment.
```bash
python -m pip install -r conf/requirements.txt
```

Before running the code, please append your project’s root directory to PYTHONPATH.
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/your/project/"
```

## III. Prepare Input
To run the framework, you just need to prepare:
- configuration file in yaml format
- a folder containing training and evaluation csv files

### Configuration File
The configuration file contains locations to your data, parameters to preprocess your raw text, 
and parameters to train a classification model.
You can find the basic example configuration file [here](https://bitbucket.es.ad.adp.com/projects/DSMAIN/repos/text-classifier/browse/config/example.yaml).

### A folder containing train.csv and test.csv
The CSV file contains two columns:
- Label: class labels encoded from 1
- Text: raw text

## IV. Preprocess

Preprocessing data with [torchtext](https://github.com/pytorch/text) on the fly will load all data in memory 
and it slow down the training speed, especially when loading large dataset.

To process sentences, in text_preprocessor.py,
```bash
from text_classifier.dataloader.preprocessor.sentence import run_prepro
```

To process documents, you can set sentence_limit in configuration file, and in text_preprocessor.py,
```bash
from text_classifier.dataloader.preprocessor.document import run_prepro
```

Now you can preprocess the data before training and store them locally.
```bash
python textclassifier/text_preprocessor.py --config ../config/cr_sents/textmlp.yaml 
```

## V. Training

To train a model, just run:

```bash
python textclassifier/text_trainer.py --config ../config/cr_sents/textmlp.yaml
```

## VI. Evaluation

To evaluate the trained model on your evaluation dataset, just run:

```bash
python textclassifier/text_tester.py --config ../config/cr_sents/textmlp.yaml
```

## VII. Interactive Test

You can also run the end-to-end test by inputing any text, just run:

```bash
python textclassifier/text_classify.py --config ../config/cr_sents/textmlp.yaml
```

## VIII. Analyze Incorrect Predictions

During evaluation, you can log the predictions that are not matched with the true label. 
This will helps to check which instances are wrongly predicted and analyze the reason, 
either because of the original data or because of the model.

```bash
python textclassifier/text_prediction_error_analyzer.py --config ../config/cr_sents/textmlp.yaml
```
