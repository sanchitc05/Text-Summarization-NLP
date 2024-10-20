import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.summarization import summarize
import heapq

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Function to perform extractive summarization using Gensim
def extractive_summarization_gensim(text):
    summary = summarize(text, ratio=0.2)
    return summary

# Function for custom extractive summarization
def extractive_summarization_custom(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)

    freq_table = {}
    for word in words:
        word = word.lower()
        if word not in stop_words:
            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1

    sentences = sent_tokenize(text)
    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in freq_table:
                if sentence in sentence_scores:
                    sentence_scores[sentence] += freq_table[word]
                else:
                    sentence_scores[sentence] = freq_table[word]

    summary_sentences = heapq.nlargest(3, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summary_sentences)
    
    return summary

# Sample text
text = """
Artificial Intelligence (AI) is intelligence demonstrated by machines...
"""

# Generate summaries
print("Gensim Summary:")
print(extractive_summarization_gensim(text))

print("\nCustom Summary:")
print(extractive_summarization_custom(text))
