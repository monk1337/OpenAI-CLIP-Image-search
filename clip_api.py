from flask import Flask, jsonify, request
from flask_cors import CORS
from load_model import find_best_matches
from time import gmtime, strftime

app = Flask(__name__)
CORS(app)
@app.route('/checking',methods=['GET'])
def check():
    return jsonify({"status":"working"})


@app.route('/get_url',methods=['POST'])
def _inference():
    try:
        print(request.json['query'])
        raw_query    = str(request.json['query']["text_query"])
        how_many     = int(request.json['query']["n"])
        key          = str(request.json['query']["key"])
        
        return jsonify({"result" : find_best_matches(raw_query, how_many, key)})
    except Exception as e:
        print(e)
        return jsonify({"result":"-1"})

if __name__ == '__main__':
    app.run(host= '0.0.0.0', port= 5002, debug=True)
