## 🖼️ Streamlit Image Similarity Search
### 📘 Overview

Streamlit Image Similarity Search is a deep learning–powered web app that finds and displays the most visually similar images from a dataset.
It uses MobileNetV2 (pretrained on ImageNet) as a feature extractor and cosine similarity to compare images.
The app allows users to upload an image or select one from the database, instantly retrieving top visually similar matches.
The dataset is hosted on Kaggle
(https://www.kaggle.com/datasets/venkatesh2410/similar-images).

### Dataset

Due to large file sizes, the full dataset is **hosted on Kaggle** instead of GitHub.  
You can access and download the dataset directly using the following link:

🔗 **Kaggle Dataset:** [Similar Images Dataset](https://www.kaggle.com/datasets/venkatesh2410/similar-images)
### 🧠 Key Features

🧩 Deep Feature Extraction using MobileNetV2

🔍 Cosine Similarity to measure image similarity

⚡ Cached Embeddings for fast retrieval (features.npy, filenames.npy)

🖥️ Streamlit Web App with upload and selection options

📊 Displays top 5 similar images with similarity scores

### ⚙️ How It Works

* Feature Extraction:
The model extracts 1280-dimensional deep features from each image using MobileNetV2’s convolutional layers.

* Normalization:
Feature vectors are normalized using L2 normalization.

* Similarity Search:
Cosine similarity is computed between the query image and database images.

* Results:
The app returns the Top 5 most similar images, along with similarity scores.

### Streamlit

Streamlit web app for deep-learning-based image similarity search.
Uses MobileNetV2 to extract image embeddings and cosine similarity for comparison.
Allows users to upload or select an image and view top visually similar matches.
Built with TensorFlow, scikit-learn, and Streamlit for interactive, real-time search.

### Use the Interface

Select an image from the existing dataset

Or upload a new image

View the top visually similar images and their scores

### 🚀 Future Improvements

Integrate FAISS or Annoy for faster similarity search on large datasets

Add image upload previews and progress bars

Extend to video frame search or multimodal search

### 📜 License

This project is released under the MIT License.

## 👤 Author

Venktesh
Deep Learning & Computer Vision Enthusiast
📧 [venkateshvarada56@gmail.com]
🔗 []
