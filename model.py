from nrclex import NRCLex
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Sample data for training
workplace_texts = [
    "I have a meeting with my boss.",
    "Work is stressing me out.",
    "Collaborating with team members on a new project.",
    "Received positive feedback from a colleague.",
    "Working on an exciting project.",
    "Preparing for a client presentation.",
    "Dealing with a tight deadline.",
    "Attending a professional development workshop.",
    "Organizing a team-building event.",
    "Balancing work tasks and personal responsibilities.",
    "Negotiating a new contract with a client.",
    "Handling a challenging project with tight timelines.",
    "Leading a brainstorming session with the team.",
    "Participating in a work-related conference.",
    "Adapting to a new work environment.",
    "Addressing communication challenges within the team.",
    "Providing mentorship to a junior colleague.",
    "Planning and executing a successful marketing campaign.",
    "Managing a cross-functional project team.",
    "Balancing work-related travel and personal commitments.",
]
home_problems_texts = [
    "There's a leak in the roof.",
    "Family issues are bothering me.",
    "Dealing with a plumbing problem at home.",
    "Home renovation causing temporary inconvenience.",
    "Coping with the challenges of remote learning for kids.",
    "Planning to address electrical issues at home.",
    "Adjusting to a new neighborhood.",
    "Discussing home budgeting with family members.",
    "Handling a pest control situation at home.",
    "Balancing household chores with personal time.",
    "Exploring sustainable living practices at home.",
    "Managing home security concerns.",
    "Organizing a family event at home.",
    "Dealing with noisy neighbors.",
    "Setting up a home office for remote work.",
    "Creating a home garden and facing challenges.",
    "Adopting a pet and adjusting to new responsibilities.",
    "Planning a home improvement project.",
    "Handling unexpected home repairs.",
    "Adapting to a minimalist lifestyle at home.",
]

school_stress_texts = [
    "Exams are coming up, and I'm feeling stressed.",
    "I have a lot of homework to do.",
    "Preparing for a class presentation.",
    "Struggling with understanding a difficult subject.",
    "Balancing extracurricular activities and schoolwork.",
    "Joining a study group for better collaboration.",
    "Feeling overwhelmed by the workload.",
    "Adjusting to a new school environment.",
    "Seeking guidance from a teacher on challenging topics.",
    "Planning for college applications and entrance exams.",
    "Participating in a science fair project.",
    "Working on a group project with classmates.",
    "Feeling nervous about a school performance.",
    "Exploring career options during school counseling sessions.",
    "Attending a college fair to gather information.",
    "Dealing with peer pressure and social dynamics.",
    "Preparing for a debate competition.",
    "Taking on a leadership role in a school club.",
    "Participating in a sports event and managing time.",
    "Adapting to a new curriculum format.",
]
# Labels for the training data
workplace_labels = ["workplace_issues"] * len(workplace_texts)
home_problems_labels = ["home_problems"] * len(home_problems_texts)
school_stress_labels = ["school_stress"] * len(school_stress_texts)

# Combine samples and labels
all_texts = workplace_texts + home_problems_texts + school_stress_texts
all_labels = workplace_labels + home_problems_labels + school_stress_labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(all_texts, all_labels, test_size=0.2, random_state=42)

# Build a text classification pipeline using Naive Bayes
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(X_train, y_train)
joblib.dump(model, 'text_classification_model.joblib')
loaded_model = joblib.load('text_classification_model.joblib')

# Predict the categories for the test set
y_pred = loaded_model.predict(X_test)

# Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

classification_rep = classification_report(y_test, y_pred)
print("Classification Report:")
print(classification_rep)



