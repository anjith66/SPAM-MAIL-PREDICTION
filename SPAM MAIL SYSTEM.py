# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 17:53:02 2026

@author: anjith
"""
import pickle

load_model=pickle.load(open("D:/projects ml/SPAM MAIL PREDICTION/spam_mail_prediction.sav","rb"))
feature_extraction1=pickle.load(open("D:/projects ml/SPAM MAIL PREDICTION/features_extraction.sav","rb"))
input_mail = ["Congratulations! You have won a free iPhone. Click here now!"]

input_features = feature_extraction1.transform(input_mail)

prediction = load_model.predict(input_features)

if prediction[0] == 0:
    print("Spam Mail")
else:
    print("Ham Mail")