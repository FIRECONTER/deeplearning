import jieba
import re
import nltk
import numpy as np


# Open the book
with open('../data/公开信分词.txt', encoding='utf-8') as t:
    text = [l.strip() for l in t]

# PLEASE STAY LOW!
text = [t.lower() for t in text][:-10]

# Remove 'chapter i' strings
regexp = re.compile(r'chapter \d')
text = [t for t in text if not re.match(regexp, t)]

# combine all the text together
raw = ' '.join(text)
print('type of the raw text'+str(type(raw)))

# Here's the magic
tokens = [t for t in nltk.word_tokenize(raw) if t not in (',', '“', '”', '"')]
#tokens = [t for t in jieba.cut(raw) if t not in (',', '“', '”', '"')]
test_ndarr = np.array(tokens)

# a list of tokens
print('current tokens size is '+str(test_ndarr.shape))

distinct_tokens = set(tokens)
lexical_richness = len(distinct_tokens) / len(tokens)

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
ntext = nltk.Text(tokens)
# draw the picture of word/ offset
# 典型的词分布图像
ntext.dispersion_plot(['乐视', '资金','变革','生态','布局','硬件','用户',
                       '承诺', '责任','质疑', '窒息','歉意'])
