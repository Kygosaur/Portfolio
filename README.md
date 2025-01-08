# Academic Projects Collection

A comprehensive collection of projects, implementations, and code samples developed during my academic journey. This repository serves as a reference and archive of various technologies and concepts explored during coursework, assignments, and final year project.

## Table of Contents
- [Firebase Projects](#firebase-projects)
- [Machine Learning Implementations](#machine-learning-implementations)
- [Final Year Project - OCR Implementation](#final-year-project---ocr-implementation)
- [Course Assignments](#course-assignments)

## Firebase Projects

### Real-time Database Implementation
Implementation of Firebase Realtime Database for dynamic data storage and retrieval.

```javascript
// Initialize Firebase
const firebaseConfig = {
  // Configuration details
};

// Real-time data listener
firebase.database().ref('users').on('value', (snapshot) => {
  const data = snapshot.val();
  updateUI(data);
});

// Data writing function
function writeUserData(userId, name, email) {
  firebase.database().ref('users/' + userId).set({
    username: name,
    email: email
  });
}
```

### Firebase Authentication
User authentication implementation using Firebase Auth.

```javascript
// Email/Password Authentication
function signUp(email, password) {
  firebase.auth().createUserWithEmailAndPassword(email, password)
    .then((userCredential) => {
      const user = userCredential.user;
      // Handle successful signup
    })
    .catch((error) => {
      // Handle errors
    });
}

// Google Sign-in
const provider = new firebase.auth.GoogleAuthProvider();
function googleSignIn() {
  firebase.auth().signInWithPopup(provider)
    .then((result) => {
      // Handle successful sign-in
    });
}
```

## Machine Learning Implementations

### K-Nearest Neighbors (KNN)
Implementation of KNN algorithm for classification tasks.

```python
import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)
    
    def _predict(self, x):
        # Compute distances
        distances = [self.euclidean_distance(x, x_train) 
                    for x_train in self.X_train]
        
        # Get indices of k nearest samples
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Majority vote
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
```

### Principal Component Analysis (PCA)
Implementation of PCA for dimensionality reduction.

```python
class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        
    def fit(self, X):
        # Center the data
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        
        # Compute covariance matrix
        cov = np.cov(X.T)
        
        # Compute eigenvectors & eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # Sort eigenvectors
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Store first n eigenvectors
        self.components = eigenvectors[:, :self.n_components]
        
    def transform(self, X):
        # Project data
        X = X - self.mean
        return np.dot(X, self.components)
```

## Final Year Project - OCR Implementation

### Text Detection and Recognition
Implementation of OCR using Tesseract and OpenCV for text extraction.

```python
import cv2
import pytesseract
from PIL import Image

def preprocess_image(image_path):
    # Read image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    return binary

def extract_text(image_path):
    # Preprocess image
    processed_img = preprocess_image(image_path)
    
    # Convert to PIL Image
    pil_img = Image.fromarray(processed_img)
    
    # Extract text
    text = pytesseract.image_to_string(pil_img)
    
    return text

# Text extraction with bounding boxes
def extract_text_regions(image_path):
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    
    # Get bounding boxes
    boxes = pytesseract.image_to_boxes(img)
    
    for b in boxes.splitlines():
        b = b.split()
        img = cv2.rectangle(img, 
                          (int(b[1]), h - int(b[2])), 
                          (int(b[3]), h - int(b[4])), 
                          (0, 255, 0), 2)
    
    return img
```

## Course Assignments

### Data Structures Implementation
Common data structures implemented during coursework.

```python
# Binary Search Tree Implementation
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None
        
    def insert(self, value):
        if not self.root:
            self.root = Node(value)
        else:
            self._insert_recursive(self.root, value)
            
    def _insert_recursive(self, node, value):
        if value < node.value:
            if node.left is None:
                node.left = Node(value)
            else:
                self._insert_recursive(node.left, value)
        else:
            if node.right is None:
                node.right = Node(value)
            else:
                self._insert_recursive(node.right, value)
```

### Algorithm Implementations
Various algorithms implemented during algorithm courses.

```python
# Merge Sort Implementation
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
        
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
            
    result.extend(left[i:])
    result.extend(right[j:])
    return result
```

## Usage and Setup

### Prerequisites
- Python 3.7+
- Node.js (for Firebase projects)
- OpenCV
- Tesseract
- Required Python packages: numpy, scipy, pytorch, tensorflow

### Installation
```bash
# Create virtual environment
python -m venv env
source env/bin/activate  # Unix
env\Scripts\activate     # Windows

# Install requirements
pip install -r requirements.txt

# Install Node packages (for Firebase projects)
npm install
```

## Project Structure
```
academic-projects/
├── firebase/
│   ├── auth/
│   └── database/
├── machine-learning/
│   ├── knn/
│   └── pca/
├── fyp/
│   └── ocr/
└── assignments/
    ├── data-structures/
    └── algorithms/
```

## Documentation
Each project directory contains:
- README.md with specific setup instructions
- Requirements file
- Source code with comments
- Example usage
- Test cases (where applicable)

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing
Feel free to fork this repository and submit pull requests for improvements.

## Contact
For any queries regarding the implementations, please reach out through the repository issues section.
