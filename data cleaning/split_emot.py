import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('../data/fer2013.csv')

# Extracting pixel values and converting them to numpy arrays
df['pixels'] = df['pixels'].apply(lambda x: np.fromstring(x, dtype=int, sep=' '))

# Function to plot images for each emotion
def plot_emotion_images(df, emotion_label, num_images=9):
    emotion_df = df[df['emotion'] == emotion_label]
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fig.suptitle(f'Emotion: {emotion_label}', fontsize=20)
    
    for i, ax in enumerate(axes.flatten()):
        if i < len(emotion_df):
            pixels = emotion_df.iloc[i]['pixels'].reshape(48, 48)
            ax.imshow(pixels, cmap='gray')
            ax.axis('off')
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.show()

# List of emotions
emotions = df['emotion'].unique()

# Plot images for each emotion
for emotion in emotions:
    plot_emotion_images(df, emotion_label=emotion, num_images=9)
