import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# Step 1: Prepare the labeled dataset
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
image_paths = ['dataset/a.png', 'dataset/b.png', 'dataset/c.png', 'dataset/d.png', 'dataset/e.png', 'dataset/f.png', 'dataset/g.png', 'dataset/h.png', 'dataset/i.png']

# Step 2: Feature Extraction - Contour Comparison
X = []
y = []

for letter, path in zip(letters, image_paths):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, threshold = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_count = len(contours)

    X.append(contour_count)
    y.append(letter)

# Convert to numpy arrays
X = np.array(X).reshape(-1, 1)
y = np.array(y)

# Step 3: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Normalize the features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Train the KNN model
k = 3  # Number of neighbors
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = knn.predict(X_test)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Step 10: Predict on new, unseen data
new_image_paths = ['dataset/g.png', 'dataset/b.png']
new_data = []

for path in new_image_paths:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, threshold = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_count = len(contours)

    new_data.append(contour_count)

new_data = np.array(new_data).reshape(-1, 1)
new_data = scaler.transform(new_data)
predicted_classes = knn.predict(new_data)
print(f"Predicted classes for new data: {predicted_classes}")
