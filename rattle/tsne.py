from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
def tsne_visualization(model):
    # 获取嵌入向量
    vocab = list(model.wv.index_to_key)
    vectors = model.wv[vocab]

    # 使用 t-SNE 降维
    tsne = TSNE(n_components=2, random_state=42)
    reduced_vectors = tsne.fit_transform(vectors)

    # 绘图
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], alpha=0.5)
    for i, word in enumerate(vocab):
        plt.annotate(word, (reduced_vectors[i, 0], reduced_vectors[i, 1]), fontsize=8)
    plt.title("t-SNE Visualization of Word Embeddings")
    plt.savefig("word2vec_tsne.png")