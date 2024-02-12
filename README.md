# Flower Image Clustering

This Flask application demonstrates clustering of flower images using machine learning techniques. It uses the VGG16 model for feature extraction, followed by KMeans clustering to group similar images together. The resulting clusters are then visualized in an interactive webpage.

## Demo

Explore the clustered flower images live at [Flower Image Clustering](https://images-clustering-uxff.onrender.com).

## Setup

### Prerequisites

Make sure you have the following installed:

- Python
- Flask
- Keras
- Scikit-learn
- Numpy
- Pickle

You can install the required Python packages using:

```bash
pip install Flask Keras scikit-learn numpy
```

### Running the Application

1. Clone the repository:

```bash
git clone https://github.com/Kasparov2000/news-clustering.git
```

2. Change into the project directory:

```bash
cd news-clustering
```

3. Run the Flask application:

```bash
python app.py
```

The application will be accessible at [https://images-clustering-uxff.onrender.com](https://images-clustering-uxff.onrender.com) in your web browser.

## Features

- **Feature Extraction:** Uses the VGG16 model for extracting features from flower images.
- **Clustering:** Applies KMeans clustering to group images based on visual similarities.
- **Dimensionality Reduction:** Reduces the dimensionality of feature vectors using PCA.
- **Interactive Visualization:** Provides an interactive webpage to explore clustered images.

## File Structure

- `app.py`: The main Flask application.
- `templates/index.html`: HTML template for rendering clustered images.
- `flower_images/`: Folder containing the flower images dataset.
- `static/`: Static files, including clustered images organized in directories.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
