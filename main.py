from pydantic import BaseModel

from fastapi import FastAPI
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from deta import Deta

deta = Deta("b01hp2uzru3_BPCmdxE65SYHWocdLGcYmek66woozV7x")

drive = deta.Drive("anxdep")


comment_clf_stream = drive.get('comment_clf.joblib')
score_clf_stream = drive.get('score_clf.joblib')

with open("comment_clf.joblib", "wb+") as f:
    for chunk in comment_clf_stream.iter_chunks(4096):
        f.write(chunk)
    comment_clf_stream.close()

with open("score_clf.joblib", "wb+") as f:
    for chunk in score_clf_stream.iter_chunks(4096):
        f.write(chunk)
    score_clf_stream.close()


class UserScore(BaseModel):
    bdi: int
    bai: int
    age: int
    gender: int


loaded_comment_clf = joblib.load("comment_clf.joblib")
comment_clf: RandomForestClassifier = loaded_comment_clf['model']
vectorizer: CountVectorizer = loaded_comment_clf['vectorizer']
comment_labels = ['none', 'moderate', 'severe']

score_clf: RandomForestClassifier = joblib.load("score_clf.joblib")

app = FastAPI()


@app.get("/")
def root():
    return {"result": "Success negus"}


@app.post("/models/rf/comment")
def classify_comment(comment: str):
    y_pred = comment_clf.predict_proba(vectorizer.transform([comment]))[0]
    print(y_pred)
    return {"predicted": comment_labels[np.argmax(y_pred)], "none": y_pred[0], "moderate": y_pred[1], "severe": y_pred[2]}


@app.post("/models/rf/score")
def classify_score(user_score: UserScore):
    print(score_clf.classes_)
    result = score_clf.predict_proba(np.array(
        [[user_score.bdi, user_score.bai, user_score.age, user_score.gender],]))[0]
    print(result)
    return {"predicted": score_clf.classes_[np.argmax(result)],
            "severe anxiety": result[0],
            "both severe": result[1],
            "severe depression": result[2],
            "no attention needed": result[3]}
