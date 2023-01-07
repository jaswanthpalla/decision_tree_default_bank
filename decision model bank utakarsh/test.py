import pickle


predict_save_model=pickle.load(open('model_8_d.pkl','rb'))

predictions=predict_save_model.predict()
print(predictions)