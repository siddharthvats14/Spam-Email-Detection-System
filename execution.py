import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved model
with open("new_file.pkl", "rb") as file:
    model, feature_extraction = pickle.load(file)
print("Enter an email: ")
input_mail = [input()]

# Transform input data using the same vectorizer used during training
input_data_feature = feature_extraction.transform(input_mail)

# Make predictions
prediction = model.predict(input_data_feature)
print("Prediction:", prediction)
# Transform input data using the same vectorizer used during training
input_data_feature = feature_extraction.transform(input_mail)
