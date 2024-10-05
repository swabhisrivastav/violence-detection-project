import clip
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image


class Model:
    def __init__(self, settings_path: str = './settings.yaml'):
        with open(settings_path, "r") as file:
            self.settings = yaml.safe_load(file)

        self.device = self.settings['model-settings']['device']
        self.model_name = self.settings['model-settings']['model-name']
        self.threshold = self.settings['model-settings']['prediction-threshold']
        self.model, self.preprocess = clip.load(self.model_name,
                                                device=self.device)
        self.labels = self.settings['label-settings']['labels']
        self.labels_ = []
        for label in self.labels:
            text = 'a photo of ' + label  # will increase model's accuracy
            self.labels_.append(text)

        self.text_features = self.vectorize_text(self.labels_)
        self.default_label = self.settings['label-settings']['default-label']

    @torch.no_grad()
    def transform_image(self, image_batch: list):
        '''
        Transforms a batch of images to tensors using CLIP's preprocess.

        Args:
            image_batch (list): List of numpy arrays (frames).

        Returns:
            torch.Tensor: Batched tensor of transformed images.
        '''
        # Convert each frame from numpy to PIL, preprocess, and stack into batch
        tf_images = [self.preprocess(Image.fromarray(image).convert('RGB')).unsqueeze(0).to(self.device)
                     for image in image_batch]
        return torch.cat(tf_images)  # Stack all tensors along batch dimension

    @torch.no_grad()
    def tokenize(self, text: list):
        text = clip.tokenize(text).to(self.device)
        return text

    @torch.no_grad()
    def vectorize_text(self, text: list):
        tokens = self.tokenize(text=text)
        text_features = self.model.encode_text(tokens)
        return text_features

    @torch.no_grad()
    def predict_(self, text_features: torch.Tensor,
                 image_features: torch.Tensor):
        # Pick the top most similar label for each image in the batch
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = image_features @ text_features.T
        values, indices = similarity.topk(1, dim=-1)
        return values, indices

    @torch.no_grad()
    def predict_batch(self, image_batch: list) -> list:
        '''
        Does prediction on a batch of images.

        Args:
            image_batch (list): List of numpy image arrays with RGB channel
                                ordering.

        Returns:
            (list): List of dictionaries, each containing predictions:
                    {
                    'label': 'some_label',
                    'confidence': 0.X
                    }
        '''
        # Transform the batch of images
        tf_images = self.transform_image(image_batch)
        
        # Get image features for the batch
        image_features = self.model.encode_image(tf_images)

        # Predict labels for the batch
        predictions = []
        for image_feature in image_features:
            values, indices = self.predict_(text_features=self.text_features,
                                            image_features=image_feature.unsqueeze(0))
            label_index = indices[0].cpu().item()
            label_text = self.default_label
            model_confidence = abs(values[0].cpu().item())

            if model_confidence >= self.threshold:
                label_text = self.labels[label_index]

            predictions.append({
                'label': label_text,
                'confidence': model_confidence
            })

        return predictions

    @torch.no_grad()
    def predict(self, image: np.array) -> dict:
        '''
        Makes a prediction for a single image.

        Args:
            image (np.array): Input image in numpy format.

        Returns:
            dict: Dictionary containing:
                  {
                      'label': 'predicted_label',
                      'confidence': confidence_score
                  }
        '''
        # Transform the image to tensor
        tf_image = self.transform_image([image])  # Transform single image into a batch of size 1

        # Get image features
        image_features = self.model.encode_image(tf_image)

        # Pick the most similar label
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
        similarity = image_features @ text_features.T
        values, indices = similarity.topk(1, dim=-1)

        # Extract label and confidence
        label_index = indices[0].cpu().item()
        label_text = self.default_label
        model_confidence = abs(values[0].cpu().item())

        if model_confidence >= self.threshold:
            label_text = self.labels[label_index]

        return {
            'label': label_text,
            'confidence': model_confidence
        }

    @staticmethod
    def plot_image(image: np.array, title_text: str):
        plt.figure(figsize=[13, 13])
        plt.title(title_text)
        plt.axis('off')
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)


# Code for video capture and batch processing
def process_video_batch(model, video_source=0, batch_size=4):
    cap = cv2.VideoCapture(video_source)  # 0 for webcam, or replace with video file path
    frame_batch = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the frame to RGB if using OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Add the frame to the batch
        frame_batch.append(frame)

        # Check if we have enough frames for a batch
        if len(frame_batch) == batch_size:
            # Process the batch of frames
            predictions = model.predict_batch(frame_batch)
            
            for idx, prediction in enumerate(predictions):
                print(f"Frame {idx+1}: {prediction}")
            
            # Clear the batch
            frame_batch = []

        # Optionally display the frame (processed frame in RGB)
        cv2.imshow("Webcam", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close windows
    cap.release()
    cv2.destroyAllWindows()


# Example usage for single image processing:
def process_image(model, image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    prediction = model.predict(image=image)  # Using the predict method for a single image
    print(f'Predicted label for image: {prediction["label"]}, Confidence: {prediction["confidence"]}')


# Example usage:
if __name__ == "__main__":
    # Initialize model
    model = Model()

    # Process video or webcam feed in batches
    process_video_batch(model, video_source=0, batch_size=4)

    # Example for single image processing
    # Uncomment the line below and provide the correct image path to test
    # process_image(model, 'path_to_your_image.jpg')
