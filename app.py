import speech_recognition as sr
from flask import logging, Flask, render_template, request, flash
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex
import joblib
app = Flask(__name__)
app.secret_key = "mentalhealth"
nltk.download('vader_lexicon')



def classify_emotion(word_emotions):
    # Classify emotion based on the word-level emotions
    if "joy" in word_emotions:
        return "Joy"
    elif "anger" in word_emotions:
        return "Anger"
    elif "fear" in word_emotions or "sadness" in word_emotions:
        return "Sad"
    elif "surprise" in word_emotions or "disgust" in word_emotions:
        return "Stressed"
    else:
        return "Neutral"


def get_mental_health_advice(category, emotion):
    advice_dict = {
        "home_problems": {
            "anger": "Try to communicate your feelings with your family members and find a peaceful resolution.",
            "sad": "Consider talking to a friend or a therapist about what's bothering you at home.",
            "neutral": "Create a comfortable and positive environment at home. Engage in activities that bring you joy.",
            "joy": "Celebrate the positive moments at home and strengthen your bonds with your loved ones.",
            "stressed": "Identify the sources of stress at home and work towards finding solutions or coping mechanisms."
        },
        "school_stress": {
            "anger": "Speak to your teachers or a counselor about the issues causing anger in school.",
            "sad": "Reach out to a teacher or a trusted adult for support and guidance.",
            "neutral": "Maintain a healthy balance between academics and leisure. Take breaks when needed.",
            "joy": "Celebrate your achievements and positive experiences in school. Stay motivated.",
            "stressed": "Break down tasks into manageable parts and prioritize. Seek help from teachers if needed."
        },
        "workplace_issues": {
            "anger": "Sorry to hear that, It's okay to be frustrated. Try addressing your concerns with your supervisor or HR in a professional manner.",
            "sad": "Consider talking to a colleague or seeking support from your workplace's mental health resources.",
            "neutral": "Maintain a work-life balance. Take breaks and engage in activities you enjoy outside of work.",
            "joy": "Acknowledge and celebrate your successes at work. Share positive experiences with colleagues.",
            "stressed": "Prioritize tasks, delegate when possible, and communicate with your team about workload concerns."
        }
    }

    if category in advice_dict and emotion in advice_dict[category]:
        return advice_dict[category][emotion]
    else:
        return "No specific advice available."

# def perform_semantic_analysis(text):
#     sia = SentimentIntensityAnalyzer()
#
#     # Perform sentiment analysis on the given text
#     sentiment_score = sia.polarity_scores(text)['compound']
#     sentiment=""
#     # Classify the sentiment based on the compound score
#     if sentiment_score >= 0.05:
#         sentiment = 'Positive'
#     elif sentiment_score <= -0.05:
#         sentiment = 'Negative'
#     else:
#         sentiment = 'Neutral'
#
#     return sentiment, sentiment_score


@app.route('/')
def index():
    flash("If you're struggling, reach out to someone you trust or seek professional support. You are not burdening others by sharing your feelings. Let's break the stigma surrounding mental health together.")
    return render_template('index.html')

@app.route('/audio_to_text/')
def audio_to_text():
    flash(" Press Start to start recording audio and press Stop to end recording audio")
    return render_template('audio_to_text.html')

@app.route('/audio', methods=['POST'])
def audio():
    r = sr.Recognizer()
    with open('upload/audio.wav', 'wb') as f:
        f.write(request.data)

  
    with sr.AudioFile('upload/audio.wav') as source:
        audio_data = r.record(source)
        text = r.recognize_google(audio_data, language='en-IN', show_all=True)
        print(text)
        flash("Please wait for a while")


        return_text = " Did you say : <br> "
        try:
            for num, texts in enumerate(text['alternative']):
                return_text += str(num+1) +") " + texts['transcript']  + " <br> "


        except:
            return_text = " Sorry!!!! Voice not Detected "
            return return_text
        emotion=''
        print(return_text)
        # Predict the categories for the test set
        loaded_model = joblib.load('text_classification_model.joblib')

        # Predict the category using the trained model
        predicted_category = loaded_model.predict([return_text])[0]
        print(f"Predicted Category: {predicted_category}")

        # Analyze emotion using NRCLex
        nrc_obj = NRCLex(return_text)
        word_emotions = nrc_obj.raw_emotion_scores
        predominant_emotion = classify_emotion(word_emotions)

        print(f"Predominant Emotion: {predominant_emotion}")

        # Get mental health advice based on category and emotion
        advice = get_mental_health_advice(predicted_category, predominant_emotion.lower())

        print(f"Mental Health Advice: {advice}")
        
    return "Category: "+str(predicted_category)+ "<br>Emotion: "+str(predominant_emotion)+"<br>Wellness Advice: "+str(advice)




if __name__ == "__main__":
    app.run(debug=True)
