from loguru import logger
from pyspark.ml import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sparknlp.annotator import BertSentenceEmbeddings, ClassifierDLApproach
from sparknlp.base import DocumentAssembler, LightPipeline

BERT_INPUT_COL = "description"
BERT_LABEL_COL = "category"
BERT_SPARK_EPOCHS = 20
BERT_SPARK_MODEL = "../models/sent_bert_base_uncased_agnews_en"
BERT_MODEL_UNCASED = "sent_bert_base_uncased"


class BERTSpark(Pipeline):
    def __init__(self, input_col: str, label_col: str) -> None:
        self.document_assembler = DocumentAssembler()
        self.document_assembler.setInputCol(input_col)
        self.document_assembler.setOutputCol("document")

        self.bert = BertSentenceEmbeddings.pretrained(BERT_MODEL_UNCASED)
        self.bert.setInputCols(["document"])
        self.bert.setOutputCol("sentence_embeddings")

        self.classifier_dl = ClassifierDLApproach()
        self.classifier_dl.setInputCols(["sentence_embeddings"])
        self.classifier_dl.setOutputCol("class")
        self.classifier_dl.setLabelColumn(label_col)
        self.classifier_dl.setMaxEpochs(BERT_SPARK_EPOCHS)
        self.classifier_dl.setEnableOutputLogs(True)

        self.bert_pipeline = Pipeline(
            stages=[self.document_assembler, self.bert, self.classifier_dl]
        )

    def train(self, train_dataset):
        self.bert_model = self.bert_pipeline.fit(train_dataset)
        self.light_model = LightPipeline(self.bert_model)

    def eval(self, input_col, label_col, eval_dataset):
        preds = self.bert_model.transform(eval_dataset)
        df = preds.select(label_col, input_col, "class.result").toPandas()
        df["result"] = df["result"].apply(lambda x: x[0])
        report = classification_report(df.category, df.result)

        logger.info(f"\n{report}")
        logger.info(accuracy_score(df.category, df.result))

    def predict(self, text):
        return self.light_model.annotate(text)["class"][0]
