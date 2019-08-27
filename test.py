
import BPNetwork

bpDir = BPNetwork.__dict__


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

train_text = news_content[0:1000]

tfidf_vec = TfidfVectorizer(max_features = 1000)

tfidf_fit = tfidf_vec.fit_transform(train_text)

vic_list = tfidf_vec.get_feature_names()
len(vic_list)
# vic_list

key_word_dict = {}
for i, w in enumerate(vic_list):
    key_word_dict[w] = i

count_vec = CountVectorizer(vocabulary = key_word_dict)

count_fit = count_vec.fit_transform(train_text)

count_feature = count_fit.toarray()
count_feature.shape

train_text[8]

count_feature[8]
