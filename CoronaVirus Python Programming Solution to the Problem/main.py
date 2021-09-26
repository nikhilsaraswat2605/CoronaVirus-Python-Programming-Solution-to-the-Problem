from flask import Flask, render_template, request

app = Flask(__name__)
import pickle

# open a file where you store your pickled data
file = open("model.pkl","rb")
clf = pickle.load(file)
file.close()

@app.route("/", methods = ["GET", "POST"])
def hello_world():
    if request.method=="POST":
        # print(request.form)
        myDict = request.form
        fever = float(myDict['fever'])
        age = int(myDict['age'])
        bodyPain = int(myDict['bodyPain'])
        runnyNose = int(myDict['runnyNose'])
        diffBreath = int(myDict['diffBreath'])
        # Code for Inference
        inputFeatures = [fever,age,bodyPain,runnyNose,diffBreath]
        infProb = clf.predict_proba([inputFeatures])[0][1]
        # print(infProb)
        return render_template("show.html", inf = round(infProb)*100)
    # return "<p>Hello, World!</p> " + str(infProb)
    return render_template("index.html")

if __name__ == '__main__': 
    app.run(debug=True)
    