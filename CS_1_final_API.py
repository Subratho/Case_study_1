import pickle 
import lightgbm as lgb
import pandas as pd
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm
import time
from flask import request, jsonify
from flask_restplus import  Resource, Namespace, fields

ns1 = Namespace('final', description='Case study 1')

mod = ns1.model('fin_model', {
    'comment': fields.String(required=True)
})

## helper function
def fe(df):
  punctuation = string.punctuation
  df['char_count'] = df['Description'].apply(len)
  df['word_count'] = df['Description'].apply(lambda x: len(x.split()))
  df['word_density'] = df['char_count'] / (df['word_count']+1)
  df['punctuation_count'] = df['Description'].apply(lambda x: len("".join(_ for _ in x if _ in punctuation)))
  df['upper_case_word_count'] = df['Description'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))
  return df

def stop_word(sent):
  en_stops = set(stopwords.words('english'))

  filter = []
  for word in sent.split(" "): 
    if word.lower() not in en_stops or word.lower() == 'not':
        filter.append(word)
  
  return " ".join(filter)

def binary_dataframes(root_dir, category):
  '''
  Will return data specific to one of the category

  category = 'commenting' or 'groping' or 'ogling'
  '''

  dir = root_dir+'binary_classification/'
  train = pd.read_csv(dir+category+'_data/train.csv')
  cv = pd.read_csv(dir+category+'_data/dev.csv')
  test = pd.read_csv(dir+category+'_data/test.csv') 

  return pd.concat([train, cv], axis=0, ignore_index=True) , test


## main function

def final_1(X):
  '''
  Will return data specific to one of the category

  X = 1 D vector

  '''
  X = pd.DataFrame({'Description': [X]})
  X = fe(X)
  
  X['Description'] = X['Description'].map(stop_word)
  categories = ['commenting', 'groping','ogling']

  out = {
      'commenting': "",
      'ogling': "",
      'groping': "",
  }
  for category in categories:
    with open('safecity_model/vectorizer_'+category+'.pkl', "rb") as input_file:
        vec = pickle.load(input_file)
    
    trans = vec.transform(X['Description'])
    dp = pd.concat([pd.DataFrame(trans.toarray(), columns=vec.vocabulary_), X.drop(['Description'], axis=1)], axis=1)
    
    clf = lgb.Booster(model_file='safecity_model/lgb_'+category+'.txt')

    out[category] = "No" if clf.predict(dp)<=0.5 else "Yes"
 
  return out



@ns1.route("/")
class CS1(Resource):

    @ns1.expect(mod)
    def post(self):
        body = request.get_json()
        ans = final_1(body['comment'])
        return jsonify(ans)
