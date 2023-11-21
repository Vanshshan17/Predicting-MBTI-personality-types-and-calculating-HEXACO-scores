import pandas as pd
import sklearn as sk
import numpy as np

import nltk
nltk.download('wordnet')
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem.porter import PorterStemmer

from bs4 import BeautifulSoup
import re,string,unicodedata

import gensim
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api

from flask import Flask, render_template, request

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from nltk.stem import WordNetLemmatizer 

# nltk.download('averaged_perceptron_tagger')


def backend_call(data):
    
    #pip install flask_ngrok
    dt=[]
    dt.append(data)
    #data = {"I am very shy with new people. I find it difficult to communicate with strangers."}

    data = pd.DataFrame(dt, columns=['posts'])


    def strip_html(text):
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()
    
    
    def remove_urls(text):
        # Regular expression to match URLs
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        # Replace URLs with an empty string
        clean_text = url_pattern.sub('', text)
        return clean_text

    #Removing the square brackets and notations
    def remove_between_square_brackets(text):
        return re.sub('[^A-Za-z0-9/. ]', '', text)

    #Lemmatizing the text
    def simple_lemmatizer(text):
        lemmatizer = WordNetLemmatizer()
        text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
        return text

    # Removing URL
    def remove_URL(sample):
        """Remove URLs from a sample string"""
        return re.sub(r"http\S+", "", sample)

    #set stopwords to english
    stop={'are', "shan't", "wasn't", 'before', 'yourself', "you'll", 'after', 'same', 'doesn', 'each', "you'd", 'd', 'more', "didn't", 'not', 'nor', 'what', 'both', 'can', 'up', 'themselves', 'it', 'these', 'only', 't', 'few', "mustn't", 'that', 'of', 'having', 's', 'shan', 'won', 'on', 'we', 'from', "you're", 'has', 'at', 'some', 'than', 'now', "weren't", 'have', 'an', 'other', "needn't", 'her', 'were', 'does', 'into', 'own', 'weren', 'off', 'his', "hadn't", 'aren', "isn't", 'had', 'been', 'again', 'wouldn', 'me', 'a', 'this', 'no', "doesn't", "aren't", 'too', 'i', 'as', 'was', 'haven', 'don', 'ours', 'over', 'your', 'if', 'against', 'them', 'be', "don't", 'where', 'its', 'itself', 'so', 'am', "couldn't", 'm', 'such', 'they', 'herself', 'theirs', 'our', 'he', 'do', 'mightn', 'mustn', 'here', 'you', 'o', 'when', 'who', 'further', 'y', 'very', 'then', 'their', 'couldn', 'those', 'hers', 'why', 'between', 'yours', "mightn't", 'wasn', 'because', 'my', "won't", 'down', 'ain', "wouldn't", 'just', 'most', 'for', 'himself', 'above', 'hasn', 'doing', 'any', 'being', 're', 'through', 'during', "shouldn't", 'the', "that'll", 'out', 'ourselves', "you've", 'there', 'isn', 'which', 'with', 'shouldn', 'is', "hasn't", 've', 'myself', 'needn', 'ma', 'about', 'should', 'until', "it's", 'or', 'did', 'him', 'but', 'how', 'hadn', 'by', 'in', "haven't", 'll', "she's", 'and', 'she', 'below', 'will', 'under', 'all', 'to', "should've", 'didn', 'once', 'yourselves', 'while', 'whom'}

    #Tokenization of text
    tokenizer=ToktokTokenizer()

    #Setting English stopwords
    stopword_list=list(stop)

    #removing the stopwords
    def remove_stopwords(text, is_lower_case=False):
        tokens = tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        if is_lower_case:
            filtered_tokens = [token for token in tokens if token not in stopword_list]
        else:
            filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
        filtered_text = ' '.join(filtered_tokens)    
        return filtered_text

    #Apply function on review column
    data['posts']=data['posts'].apply(strip_html)
    data['posts']=data['posts'].apply(remove_between_square_brackets)
    #data['posts']=data['posts'].apply(remove_notations)
    data['posts']=data['posts'].apply(remove_URL)
    data['posts']=data['posts'].apply(remove_stopwords)
    data['posts']=data['posts'].apply(remove_urls)

    data['posts']=data['posts'].apply(simple_lemmatizer)

    my_model = joblib.load('model_2.pkl')
    sentences =["honesty", "emotionality", "extraversion", "agreeableness", "conscientiousness","openness to experience"]
    x = data["posts"]
    vectors1 = [my_model.infer_vector([word for word in sent]).reshape(1,-1) for sent in sentences]
    vectors2 = [my_model.infer_vector([word for word in x]).reshape(1,-1)]


    sim_values=[]
    for i in range(1):
        x=data["posts"]
        vectors2 = [my_model.infer_vector([word for word in x]).reshape(1,-1)]
        array=[]
        for j in range(0,6):
            similarity = cosine_similarity(vectors1[j],vectors2[0])
            array.append(similarity[0][0])  
        #array.append(data["posts"])
        sim_values.append(array)
    df=pd.DataFrame(data=sim_values,columns=['h','e','x','a','c','o'])
    
    
    data3 = pd.read_csv("scores_new.csv")
    X = data3[['h', 'e', 'x', 'a', 'c' , 'o']]
    y = data3['type']
    
 
    rf = RandomForestClassifier(max_depth=5, n_estimators = 200, criterion = 'gini')
    rf.fit(X,y)
    
    y_pred = str(rf.predict(df[['h', 'e', 'x', 'a', 'c' , 'o']]))
    descr = {
        "['ISTJ']": "Quiet, serious, earn success by thoroughness and dependability. Practical, matter-of-fact, realistic, and responsible. Decide logically what should be done and work toward it steadily, regardless of distractions. Take pleasure in making everything orderly and organized - their work, their home, their life. Value traditions and loyalty.",
        "['ISFJ']": "Quiet, friendly, responsible, and conscientious. Committed and steady in meeting their obligations. Thorough, painstaking, and accurate. Loyal, considerate, notice and remember specifics about people who are important to them, concerned with how others feel. Strive to create an orderly and harmonious environment at work and at home.",
        "['INFJ']": "Seek meaning and connection in ideas, relationships, and material possessions. Want to understand what motivates people and are insightful about others. Conscientious and committed to their firm values. Develop a clear vision about how best to serve the common good. Organized and decisive in implementing their vision.",
        "['INTJ']": "Have original minds and great drive for implementing their ideas and achieving their goals. Quickly see patterns in external events and develop long-range explanatory perspectives. When committed, organize a job and carry it through. Skeptical and independent, have high standards of competence and performance - for themselves and others.",
        "['ISTP']": "Tolerant and flexible, quiet observers until a problem appears, then act quickly to find workable solutions. Analyze what makes things work and readily get through large amounts of data to isolate the core of practical problems. Interested in cause and effect, organize facts using logical principles, value efficiency.",
        "['ISFP']": "Quiet, friendly, sensitive, and kind. Enjoy the present moment, what's going on around them. Like to have their own space and to work within their own time frame. Loyal and committed to their values and to people who are important to them. Dislike disagreements and conflicts, do not force their opinions or values on others.",
        "['INFP']": "Idealistic, loyal to their values and to people who are important to them. Want an external life that is congruent with their values. Curious, quick to see possibilities, can be catalysts for implementing ideas. Seek to understand people and to help them fulfill their potential. Adaptable, flexible, and accepting unless a value is threatened.",
        "['INTP']": "Seek to develop logical explanations for everything that interests them. Theoretical and abstract, interested more in ideas than in social interaction. Quiet, contained, flexible, and adaptable. Have unusual ability to focus in depth to solve problems in their area of interest. Skeptical, sometimes critical, always analytical.",
        "['ESTP']": "Flexible and tolerant, they take a pragmatic approach focused on immediate results. Theories and conceptual explanations bore them - they want to act energetically to solve the problem. Focus on the here-and-now, spontaneous, enjoy each moment that they can be active with others. Enjoy material comforts and style. Learn best through doing.",
        "['ESFP']": "Outgoing, friendly, and accepting. Exuberant lovers of life, people, and material comforts. Enjoy working with others to make things happen. Bring common sense and a realistic approach to their work, and make work fun. Flexible and spontaneous, adapt readily to new people and environments. Learn best by trying a new skill with other people.",
        "['ENFP']": "Warmly enthusiastic and imaginative. See life as full of possibilities. Make connections between events and information very quickly, and confidently proceed based on the patterns they see. Want a lot of affirmation from others, and readily give appreciation and support. Spontaneous and flexible, often rely on their ability to improvise and their verbal fluency.",
        "['ENTP']": "Quick, ingenious, stimulating, alert, and outspoken. Resourceful in solving new and challenging problems. Adept at generating conceptual possibilities and then analyzing them strategically. Good at reading other people. Bored by routine, will seldom do the same thing the same way, apt to turn to one new interest after another.",
        "['ESTJ']": "Practical, realistic, matter-of-fact. Decisive, quickly move to implement decisions. Organize projects and people to get things done, focus on getting results in the most efficient way possible. Take care of routine details. Have a clear set of logical standards, systematically follow them and want others to also. Forceful in implementing their plans.",
        "['ESFJ']": "Warmhearted, conscientious, and cooperative. Want harmony in their environment, work with determination to establish it. Like to work with others to complete tasks accurately and on time. Loyal, follow through even in small matters. Notice what others need in their day-by-day lives and try to provide it. Want to be appreciated for who they are and for what they contribute.",
        "['ENFJ']": "Warm, empathetic, responsive, and responsible. Highly attuned to the emotions, needs, and motivations of others. Find potential in everyone, want to help others fulfill their potential. May act as catalysts for individual and group growth. Loyal, responsive to praise and criticism. Sociable, facilitate others in a group, and provide inspiring leadership.",
        "['ENTJ']": "Frank, decisive, assume leadership readily. Quickly see illogical and inefficient procedures and policies, develop and implement comprehensive systems to solve organizational problems. Enjoy long-term planning and goal setting. Usually well informed, well read, enjoy expanding their knowledge and passing it on to others. Forceful in presenting their ideas.",
    }
    val = '('+', '.join(str(e) for e in df.values[0])+')'
    return (val, y_pred.replace("['", "").replace("']", ""), descr[y_pred])
    
    
    


app = Flask(__name__)

@app.route('/', methods=['GET',"POST"])
def index():
    if request.method == "POST":
        about = request.form.get('about') #calculate HEXACO score and MBTI type and store them in variables
        (hexaco, mbti, descr) = backend_call(about)
        return render_template('index.html', scroll='result', hexaco=hexaco, mbti=mbti, descr=descr) #return the variables
    return render_template('index.html')

if __name__ == "__main__":
    app.run()
