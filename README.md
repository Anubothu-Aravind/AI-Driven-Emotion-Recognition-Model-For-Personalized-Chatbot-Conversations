# AI-Driven Emotion Recognition Model for Personalized Chatbot Conversations

This project provides a real-time emotion and gender recognition system using Convolutional Neural Networks (CNNs). It can detect faces from images or video streams and classify them into different emotion categories and genders. This technology is designed to be integrated into personalized chatbots to enhance user interaction by understanding their emotional state.

## Features

- **Real-time Emotion Recognition**: Classifies faces into emotions such as Happy, Sad, Angry, Surprise, Neutral, etc.
- **Gender Classification**: Detects the gender of the individuals in the frame.
- **Multiple Interfaces**: 
  - Image-based inference
  - Real-time video/webcam inference
  - Web-based server for remote viewing
- **Pre-trained Models**: Includes pre-trained mini-XCEPTION and simple CNN models for fast inference.

## Project Structure

- `src/`: Core source code for models, utilities, and demos.
- `trained_models/`: Pre-trained weights for face detection, emotion, and gender classification.
- `datasets/`: Dataset handling scripts.
- `images/`: Sample images for testing.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/LasyWorks/AI-Driven-Emotion-Recognition-Model-For-Personalized-Chatbot-Conversations.git
   cd AI-Driven-Emotion-Recognition-Model-For-Personalized-Chatbot-Conversations
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   pip install -r REQUIREMENTS.txt
   ```

## Usage

### 1. Image Emotion/Gender Recognition
To detect emotion and gender in a static image:
```bash
python src/image_emotion_gender_demo.py images/test_image.jpg
```

### 2. Video Emotion/Gender Recognition (Webcam)
To run real-time recognition using your webcam:
```bash
python src/video_emotion_gender_demo.py
```

### 3. Web Demo
To start a Flask-based web server that streams the webcam output with recognition labels:
```bash
python src/web_video_server.py
```
Then open your browser and navigate to `http://localhost:5000`.

## Training
If you wish to train your own models, you can use the following scripts:
- `python src/train_emotion_classifier.py`
- `python src/train_gender_classifier.py`

Note: You will need to download the fer2013 and IMDB datasets and place them in the `datasets` folder.

## References
This project utilizes XCEPTION architectures for efficient and accurate facial expression recognition.
