import nltk

txt="the cat ate the little mouse who was after the fresh cheese. "
new_token=nltk.word_tokenize(txt)
print(new_token)
new_tag=nltk.pos_tag(new_token)
print(new_tag)
grammer="NP:{<DD>?<JJ>*<NN>}"
chunkParser=nltk.RegexpParser(grammer)
chunked=chunkParser.parse(new_tag)
print(chunked)
