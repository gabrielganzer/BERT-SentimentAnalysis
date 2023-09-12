from time import time
import pandas as pd

import sparknlp
import torch
from loguru import logger

from bert import BERT_TEST_LABELS_PATH, BERT_TEST_PATH, BERT_TRAIN_PATH, BERTClassifier, train_bert
from spark import BERT_INPUT_COL, BERT_LABEL_COL, BERTSpark
from utils import text_preprocessing

text = "In a distant future, humanity has colonized other planets, \
    and advanced artificial intelligence governs our daily lives, \
    blurring the line between humans and machines."

device = "cuda" if torch.cuda.is_available() else "cpu"
spark_nlp = sparknlp.start(gpu=True if device == "cuda" else False)
bert_spark = BERTSpark(BERT_INPUT_COL, BERT_LABEL_COL)

bert_model = BERTClassifier(device, load_from_path=True)

logger.info("> Preparing datasets...")
df_train = pd.read_csv(BERT_TRAIN_PATH)
df_val = pd.merge(
    pd.read_csv(BERT_TEST_PATH),
    pd.read_csv(BERT_TEST_LABELS_PATH),
    on="id",
    how="inner",
)
train_dataset = spark_nlp.createDataFrame(df_train)
eval_dataset = spark_nlp.createDataFrame(df_val)

logger.info("> Train/Evaluate...")
bert_spark.train(train_dataset)
bert_spark.eval(BERT_INPUT_COL, BERT_LABEL_COL, eval_dataset)

logger.info("> Prediction...")
pred = bert_spark.predict(text)
start_time = time()
logger.info(f"> {pred}")
end_time = time()
logger.info(f"> Execution time: {end_time-start_time}s")

train_bert(device, bert_model)
logger.info("> Prediction...")
pred = bert_model.prediction(text_preprocessing(text))
start_time = time()
logger.info(f"> {pred}")
end_time = time()
logger.info(f"> Execution time: {end_time-start_time}s")
