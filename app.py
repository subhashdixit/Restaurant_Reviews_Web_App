from flask import Flask, render_template, request
import pickle
import numpy as np
import re
"""library to clean data"""
import re
"""Natural Language Tool Kit"""
import nltk
nltk.download('stopwords')
"""to remove stopword"""
from nltk.corpus import stopwords
"""for Stemming propose"""
from nltk.stem.porter import PorterStemmer

model = pickle.load(open('model_random_classification.pkl', 'rb'))
cv = pickle.load(open('count_vectorizer.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def Predict_Sentiment(sample_review):
    word = []
    sample_review = re.sub('[^a-zA-Z]', ' ', sample_review)
    sample_review = sample_review.lower()
    sample_review = sample_review.split()
    """creating PorterStemmer object to take main stem of each word"""
    ps = PorterStemmer()
    """loop for stemming each word in string array at ith row"""   
    sample_review = [ps.stem(word) for word in sample_review if not word in set(stopwords.words('english'))]
    """rejoin all string array elements to create back into a string"""
    sample_review = ' '.join(sample_review) 
    """append each string to create array of clean text"""
    word.append(sample_review)
    temp = cv.transform(word).toarray()
    return model.predict(temp)[0]

@app.route('/predict', methods = ['POST'])
def predict_placement():
    review = request.form.get('reviews')
    # return render_template('index.html', result = review)
    # model.predict()
    if (Predict_Sentiment(str(review))):
        return render_template('index.html', result = "Good Review")
    else:
        return render_template('index.html', result = "Bad Review")

if __name__ == '__main__':
    app.run(debug=True, port = 3000)
    
    