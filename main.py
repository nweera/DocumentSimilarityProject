from functions import*
#text1 = "Artificial intelligence is transforming industries by improving efficiency and enhancing customer experiences. With machine learning, companies can analyze large amounts of data to make informed decisions."
#text2 = "Machine learning allows companies to process extensive datasets to drive business decisions. By leveraging artificial intelligence, industries can enhance customer experiences and streamline operations."
text1=input("text1= ")
text2=input("text2= ")
tokens1 = preprocess(text1)
tokens2 = preprocess(text2)
combined_tokens = tokens1 + tokens2
vocab = word_frequency(combined_tokens)
word_to_int, int_to_word = create_mappings(vocab)
tf1 = compute_tf(text1, word_to_int)
tf2 = compute_tf(text2, word_to_int)
combined_text = text1 + " " + text2
idf = compute_idf(combined_text, word_to_int)
tfidf1 = compute_tfidf(tf1, idf)
tfidf2 = compute_tfidf(tf2, idf)
vec1 = [tfidf1.get(idx, 0) for idx in range(len(word_to_int))]
vec2 = [tfidf2.get(idx, 0) for idx in range(len(word_to_int))]
similarity = cosine_similarity(vec1, vec2)
print("Cosine Similarity between text1 and text2:", similarity)