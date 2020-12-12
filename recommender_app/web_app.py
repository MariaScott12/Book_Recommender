import webbrowser
import flask
import pickle
import pandas as pd 

from miscScripts import Recommender

#############    PREPARE Model  ###############
with open(f'./files/df_Hp_novel_20_topics.pkl','rb') as fin:
    df_Hp = pickle.load(fin)

with open(f'./files/vectorizer_novel.pkl','rb') as fin:
    vectorizer = pickle.load(fin)

with open(f'./files/nmf_novel_20_topics.pkl','rb') as fin:
    nmf = pickle.load(fin)

##############################################


# Initialize the app
app = flask.Flask(__name__)

#loads the page
@app.route("/")
def viz_page():
    with open("index.html", 'r') as viz_file:
        return viz_file.read()
    
#listens
@app.route("/gof", methods=["POST"])
def score():
    """
    When A POST request with json data is made to this url,
    Read the grid from the json, update and send it back
    """
    #html "posts" a request and python gets the json  from that request 
    data = flask.request.json
    query = data['grid']
    print(query[0])

    df_output, _ = Recommender(df_Hp, query[0], 'next novel', vectorizer, nmf, top_n=10)
    titles = df_output['title'].tolist() 
    print(titles)

    return flask.jsonify({'words':titles})

#--------- RUN WEB APP SERVER ------------#

# (The default website port)

if __name__ == '__main__':
    url = 'http://127.0.0.1:5000'
    webbrowser.open_new(url)
    app.run(debug=True)