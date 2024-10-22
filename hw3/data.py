import os
import pandas as pd

def load_imdb_dataset(base_dir):
    """加载IMDB数据集，并返回训练和测试数据的DataFrame"""
    def load_reviews(data_type):
        reviews = []
        labels = []
        for label in ['pos', 'neg']:
            dir_path = os.path.join(base_dir, data_type, label)
            for filename in os.listdir(dir_path)[:2000]:
                if filename.endswith('.txt'):
                    with open(os.path.join(dir_path, filename), 'r', encoding='utf-8') as f:
                        reviews.append(f.read())
                        labels.append(1 if label == 'pos' else 0)
        return reviews, labels

    # 加载训练数据
    train_reviews, train_labels = load_reviews('train')
    # 加载测试数据
    test_reviews, test_labels = load_reviews('test')

    # 创建DataFrame
    train_df = pd.DataFrame({'review': train_reviews, 'label': train_labels})
    test_df = pd.DataFrame({'review': test_reviews, 'label': test_labels})

    return train_df, test_df

# 设置数据集路径
base_dir = 'aclImdb_v1/'  # 替换为你的数据集路径
train_df, test_df = load_imdb_dataset(base_dir)

# 查看数据集的前几行
print(train_df.head())
print(test_df.head())