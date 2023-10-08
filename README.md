# SpeakUp
A web application which uses AI to provide emotional support to individuals experiencing mental health issues.
The app is develop using python and the flask framework. Users can record their voice basically their rants or feelings and we will convert their speech to text on which we use NTLK to analyse the speech and detect underlying emotions. 
We have developing a custom made SKLearn model, to determine the problem category of either home_problems, school_stress, or workplace_issues  based on the speech patterns. Based on the category and emotions, our app will provide wellness advice. We aim to provide people a channel for letting their emotions out and provide some form of comfort in return. 

To run the web application, please install the software packages stated in the requirements.txt file 
and run the app.py file either using VsCode or PyCharm
