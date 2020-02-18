import streamlit as st
import joblib, os

from wordcloud import WordCloud
from PIL import Image

import spacy
nlp = spacy.load('en_core_web_sm')

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')



# vectorizer
news_vectorizer = open("models/final_news_cv_vectorizer.pkl", "rb")
news_cv = joblib.load(news_vectorizer)

def load_prediction_models(model_file):
	loaded_models = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_models
	
def get_keys(val, my_dict):
	for key, value in my_dict.items():
		if val == value:
			return key


def main():
	""" News classifier app with Streamlit"""
	
	st.title("News Classifier ML APP")
	st.subheader("NLP and ML App with Streamlit")
	
	
	activities = ["Prediction", "NLP"]
	
	choice = st.sidebar.selectbox("Choose activity", activities)
	
	if choice == "Prediction":
		st.info("Prediction with ML")
		news_text = st.text_area("Enter Text", "Type Here")
		all_ml_models = ["LR", "RF", "DT"]
		model_choice = st.selectbox("Choose ML Model", all_ml_models)
		prediction_labels = {'business':0, 'tech':1, 'sport':2, 'health':3, 'politics':4, 'entertainment':5}
		
		if st.button("Classify"):
			st.text("Original text:: \n {} ".format(news_text)) 
			
			vect_text = news_cv.transform([news_text]).toarray()
			
			if model_choice == 'LR':
				predictor = load_prediction_models("models/newsclassifier_Logit_model.pkl")
				prediction = predictor.predict(vect_text)
				
			elif model_choice == "DT":
				predictor = load_prediction_models("models/newsclassifier_RFOREST_model.pkl")
				prediction = predictor.predict(vect_text)
				
			elif model_choice == "RF":
				predictor = load_prediction_models("models/newsclassifier_CART_model.pkl")
				prediction = predictor.predict(vect_text)
				
			st.info(get_keys(prediction, prediction_labels))
				
		
		
	if choice == "NLP":
		st.info("Natural Language Processing")
		news_text = st.text_area("Enter Text", "Type Here")
		nlp_task = ['Tokenization','NER','Lemmatization', "POS Tag"]
		task_choice = st.selectbox("Choose NLP Task",nlp_task)
		if st.button("Analyze"):
			st.info("original text: \n {}".format(news_text))
			
			docs = nlp(news_text)
			if task_choice == "Tokenization":
				result = [token.text for token in docs]
			
			elif task_choice == "NER":
				result = [(entity.text, entity.label_) for entity in docs.ents]
					
			elif task_choice == "Lemmatization":
				result = ["Token: {}, Lemma: {}".format(token.text, token.lemma_) for token in docs]
			
			elif task_choice == "POS Tag":
				result = ["Token: {}, POS: {}, Dependency: {}".format(word.text, word.tag_, word.dep_) for word in docs]
				
			st.json(result)
			
			
		if st.checkbox("WordCloud"):
			wordcloud = WordCloud().generate(news_text)
			plt.imshow(wordcloud, interpolation = 'bilinear')
			plt.axis("off")  # Remove the axis numbers
			st.pyplot()
				
		
		
		
	
	
if __name__ == '__main__':
	main()
