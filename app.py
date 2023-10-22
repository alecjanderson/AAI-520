
from flask import Flask, request, render_template

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
         text = request.form['question']
         text = text + "THIS is will be where the model do stuff"
        
         return render_template('page.html', input = text)
    if request.method == "GET":
        return render_template('page.html')
         
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)