import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data from the pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Extract labels as a NumPy array
labels = np.asarray(data_dict['labels'])

# Handle inconsistent shapes in 'data'
data_raw = data_dict['data']

try:
    # Try to convert to NumPy array directly
    data = np.asarray(data_raw)
except ValueError as e:
    print(f"Inconsistent data shapes found: {e}")
    
    # Find the maximum length of sequences in the data
    max_length = max([len(item) for item in data_raw])
    
    # Pad all sequences to the maximum length
    data_padded = [np.pad(item, (0, max_length - len(item)), mode='constant') for item in data_raw]
    
    # Convert the padded sequences to a NumPy array
    data = np.asarray(data_padded)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize the RandomForestClassifier model
model = RandomForestClassifier()

# Train the model on the training data
model.fit(x_train, y_train)

# Predict the labels for the test set
y_predict = model.predict(x_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_predict)

# Output the model's accuracy
print(f"Model Accuracy: {accuracy * 100:.2f}%")

f=open('model.p','wb')
pickle.dump({'model':model},f)
f.close()
