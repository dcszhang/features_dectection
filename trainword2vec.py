import pickle
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

# 自定义回调类，用于记录每个 epoch 的损失
class LossLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
        self.previous_loss = 0

    def on_epoch_end(self, model):
        current_loss = model.get_latest_training_loss()
        epoch_loss = current_loss - self.previous_loss  # 计算当前 epoch 的损失
        self.previous_loss = current_loss
        print(f"Epoch {self.epoch + 1}, Loss: {epoch_loss:.4f}")
        self.epoch += 1

# 加载训练数据
def load_training_data(pkl_file):
    """
    从 .pkl 文件中加载训练数据。
    
    :param pkl_file: 包含指令序列的 .pkl 文件路径。
    :return: 加载后的训练数据列表。
    """
    with open(pkl_file, "rb") as f:
        training_data = pickle.load(f)
    print(f"Loaded {len(training_data)} instruction sequences from {pkl_file}")
    return training_data

# 训练 Word2Vec 模型
def train_word2vec_model(training_data, vector_size=64, window=5, min_count=1, epochs=100, model_path="word2vec.model"):
    """
    训练 Word2Vec 模型，并保存到指定路径。
    
    :param training_data: 用于训练的指令序列列表。
    :param vector_size: 嵌入向量的维度。
    :param window: Word2Vec 的上下文窗口大小。
    :param min_count: 忽略出现频率低于此值的词。
    :param epochs: 训练的总迭代次数。
    :param model_path: 保存训练好的 Word2Vec 模型的路径。
    """
    # 创建回调实例
    loss_logger = LossLogger()

    # 训练 Word2Vec 模型
    model = Word2Vec(
        sentences=training_data,
        vector_size=vector_size,  # 嵌入向量维度
        window=window,            # 上下文窗口大小
        min_count=min_count,      # 最小频率
        sg=1,                     # 使用 Skip-Gram 模型
        compute_loss=True,        # 启用损失计算
        epochs=epochs,            # 迭代次数
        callbacks=[loss_logger]   # 添加损失记录回调
    )
    # 保存模型
    model.save(model_path)
    print(f"Word2Vec model saved to {model_path}")

# 加载训练数据
training_data = load_training_data("dataset/training_data.pkl")

# 训练并保存模型
train_word2vec_model(
    training_data=training_data,
    vector_size=64,      # 嵌入向量维度
    window=3,             # 上下文窗口大小
    min_count=1,          # 忽略频率小于 1 的词
    epochs=200,           # 迭代次数
    model_path="word2vec.model"  # 保存路径
)
