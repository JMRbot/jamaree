from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import stopwords

stop_words = stopwords.words('thai')
##print(stop_words)

##f1 = open('Post.txt',encoding="utf-8")
f1 = open('exam.txt')
for line in f1:
    words = word_tokenize(line, engine='newmm')
##    print(words)
    filtered_sentence = [w for w in words if not w in stop_words]
##    f = open('cutStopWordPost.txt','a')
    values = ''.join(str(filtered_sentence))
    print(filtered_sentence)
##    print(values)
##    f.writelines(values+'\n|')
##f.close()
f1.close()






