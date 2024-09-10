import logging
import threading
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from model_service.tfserving_model_service import TfServingBaseService

#参考 https://support.huaweicloud.com/inference-modelarts/inference-modelarts-0080.html#inference-modelarts-0080

MAX_NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 1000
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class TextClassifier_service(TfServingBaseService):

    def __init__(self, model_name, model_path):
        self.model_name = model_name
        self.model_path = model_path
        self.model = None
        self.predict = None

        # 加载标签
        self.label = ["alt.atheism","comp.graphics","comp.os.ms-windows.misc",
        "comp.sys.ibm.pc.hardware","comp.sys.mac.hardware","comp.windows.x",
        "misc.forsale","rec.autos","rec.motorcycles","rec.sport.baseball",
        "rec.sport.hockey","sci.crypt","sci.electronics","sci.med","sci.space",
        "soc.religion.christian","talk.politics.guns","talk.politics.mideast",
        "talk.politics.misc","talk.religion.misc"]

        # 非阻塞方式加载saved_model模型，防止阻塞超时
        thread = threading.Thread(target=self.load_model)
        thread.start()

    def load_model(self):
        # load saved_model 格式的模型
        self.model = tf.saved_model.load(self.model_path)

        signature_defs = self.model.signatures.keys()

        signature = []
        # only one signature allowed
        for signature_def in signature_defs:
            signature.append(signature_def)

        if len(signature) == 1:
            model_signature = signature[0]
        else:
            logging.warning("signatures more than one, use serving_default signature from %s", signature)
            model_signature = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY

        self.predict = self.model.signatures[model_signature]
    
    def _preprocess(self, data):
        for k, v in data.items():
            texts = []
            for file_name, file_content in v.items():
                print("------------------------------------------------------------")
                print(file_name)
                print(file_content)
                print(type(file_content.getvalue()))
                print("------------------------------------------------------------")
                t=str(file_content.getvalue(),encoding="latin-1")
                i = t.find('\n\n')  # skip header
                if 0 < i:
                    t = t[i:]
                texts.append(t)
            tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
            tokenizer.fit_on_texts(texts)
            sequences = tokenizer.texts_to_sequences(texts)

            preprocessed_data =  tf.convert_to_tensor(pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH),dtype=tf.dtypes.int32)
            
        return preprocessed_data

    def _postprocess(self, data):
        results = []
        for k, v in data.items():
            print("~~~~~~~~~~~~~~~~~~~~~")
            print(v)
            print("~~~~~~~~~~~~~~~~~~~~~")
            result = tf.argmax(v[0])
            result=self.label[result]
            results.append(result)
        return results

    def _inference(self, data):
        return self.predict(data)


    
