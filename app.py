import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Preprocessing
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [ps.stem(word) for word in tokens]
    return " ".join(tokens)

# Streamlit UI
st.title("📧 Spam Message Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if input_sms.strip() != "":
        # Preprocess
        transformed = transform_text(input_sms)

        # Vectorize
        vector_input = tfidf.transform([transformed])

        # Predict
        prediction = model.predict(vector_input)[0]

        # Output
        if prediction == 1:
            st.error("🚨 Spam")
        else:
            st.success("✅ Not Spam")
    else:
        st.warning("Please enter a message to classify.")

# --- Sample Messages Section ---
st.subheader("📌 Try with these sample messages")

spam_examples = [
    "Congratulations! You've won a free iPhone. Click the link to claim now.",
    "Get rich quick! Work from home and earn $5000 weekly.",
    "Exclusive deal! Limited time offer, buy now and save 70%.",
    "Claim your lottery prize today. Reply with your bank details.",
    "Urgent! Your account has been suspended. Verify immediately.",
    "You have been selected for a $1000 Walmart gift card. Click here.",
    "This is not a joke! You have won a cruise trip to the Bahamas.",
    "Final notice: Pay your overdue bill to avoid penalty.",
    "Earn money fast! No experience required. Sign up now.",
    "Hot singles in your area waiting to meet you!"
]
ham_examples = [
    "Hey, are we still meeting for lunch tomorrow?",   # kept same
    "Let me know once you reach home safely.",
    "I’ll text you after the meeting is over.",
    "Can we shift our call to the evening instead?",
    "Mom asked if you can bring some bread and milk.",
    "Don’t forget to email the teacher about the project.",
    "We are going hiking this weekend, want to join?",
    "Did you watch the cricket match last night?",
    "Looking forward to seeing you at the seminar.",
    "Thanks for your help yesterday, really appreciate it."   # kept same
]



st.write("### 🚨 Spam Examples")
for msg in spam_examples:
    st.code(msg)

st.write("### ✅ Ham Examples")
for msg in ham_examples:
    st.code(msg)
