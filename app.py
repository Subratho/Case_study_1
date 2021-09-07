from flask_restplus import Api
from CS_1_final_API import ns1
from flask import Flask

api = Api(
    title='Case study 1',
    version='1.0',
    description='Final case study'
)

api.add_namespace(ns1)


app = Flask(__name__)
api.init_app(app)

app.run(debug=True)
