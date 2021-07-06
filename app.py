import pickle
from flask import Flask, render_template , request, redirect

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
cv = pickle.load(open('transformer.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        
        y_pred = model.predict(vect)
        return render_template('index.html', pred = y_pred)


if __name__=="__main__":
    app.run(debug=True)