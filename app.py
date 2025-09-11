import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

import nltk

# Ensure necessary NLTK data is downloaded
for resource in ["punkt", "stopwords"]:
    try:
        nltk.data.find(resource if resource != "punkt" else "tokenizers/punkt")
    except LookupError:
        nltk.download(resource)


# Load models
tfidf = pickle.load(open('Requirs/vectorizer.pkl', 'rb'))
model = pickle.load(open('Requirs/model.pkl', 'rb'))

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

# --- Sidebar Info ---
st.sidebar.title("📊 Model Information")
st.sidebar.markdown("""
**ExtraTreesClassifier Results:**

- Accuracy:  `0.9865`  
- Precision: `0.9875`  
- Recall:    `0.9866`  
- F1-Score:  `0.9870`  
""")
st.sidebar.markdown("---")
st.sidebar.subheader("🌐 Connect with Me")
st.sidebar.write("[LinkedIn](https://www.linkedin.com/in/hasnainyaqoob/)")
st.sidebar.write("[GitHub](https://github.com/hasnainyaqub)")
st.sidebar.write("[Kaggle](https://www.kaggle.com/hasnainyaqooob)")
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

# --- Sample Messages ---
st.subheader("📌 Try with these sample messages")

col1, col2 = st.columns(2)

spam_examples = [
    "Get rich quick! Work from home and earn $5000 weekly.",
    "Exclusive deal! Limited time offer, buy now and save 70%.",
    "Claim your lottery prize today. Reply with your bank details.",
    "Urgent! Your account has been suspended. Verify immediately.",
    "Congratulations! You've won a free iPhone. Click the link to claim now.",
    "You have been selected for a $1000 Walmart gift card. Click here.",
    "This is not a joke! You have won a cruise trip to the Bahamas.",
    "Final notice: Pay your overdue bill to avoid penalty.",
    "Earn money fast! No experience required. Sign up now.",
    "Hot singles in your area waiting to meet you!"
]
ham_examples = [
    "Hey, are we still meeting for lunch tomorrow?",   
    "Let me know once you reach home safely.",
    "I’ll send you the meeting notes later today.",
    "Are you free for dinner tomorrow night?",
    "Please remind me to call the doctor in the morning.",
    "Let’s go jogging together this weekend.",
    "The parcel should arrive by Friday afternoon.",
    "Are you coming to the family gathering on Sunday?",
    "The shop was closed today, I’ll go again tomorrow.",
    "Thanks for your help yesterday, really appreciate it." 
]



with col1:
    st.write("### ✅ Ham Examples")
    for msg in ham_examples:
        st.code(msg)

with col2:
    st.write("### 🚨 Spam Examples")
    for msg in spam_examples:
        st.code(msg)

# --- Dataset & Disclaimer ---
st.subheader("ℹ️ About this Project")
st.markdown("""
This model was trained on the [Email Spam Classification Dataset](https://www.kaggle.com/datasets/purusinghvi/email-spam-classification-dataset) from Kaggle.  

### 📊 Dataset Information
- **RangeIndex:** 83,448 entries (0 to 83,447)  
- **Columns (2):**  
  - `label` → 83,448 non-null, `int64`  
  - `text` → 83,448 non-null, `object`  
- **Memory usage:** ~1.3 MB  

**Label meaning:**  
- `1` → Spam email (43,910 entries)  
- `0` → Ham/legitimate email (39,538 entries)  

**Text column:**  
Contains the actual email message content used for classification.  

---

### ⚠️ Disclaimer
While the model achieves **high accuracy** on test data, it is **not a large-scale production model**.  
It may sometimes **predict wrongly** and should **not** be used for critical filtering tasks (like banking, security, or healthcare).  

This project is for **learning and demonstration purposes only**.  
""")
