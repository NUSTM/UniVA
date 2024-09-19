from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


for dataset in ['M3ED', 'MOSEI']:
    if dataset == 'MOSEI':
        label_va = {
            'happy': [1, 0.735] ,
            'sad': [0.225 , 0.333] ,
            'anger': [0.167, 0.865],
            'surprise': [0.875, 0.875],
            'disgust': [0.052, 0.775],
            'fear': [0.073, 0.840]
        }
        emotion_vectors = np.array([label_va['happy'], label_va['sad'], label_va['anger'], label_va['surprise'], label_va['disgust'], label_va['fear']])
    elif dataset == 'M3ED':
        label_va = {
            'happy': [1, 0.735] ,
            'surprise': [0.875, 0.875],
            'sad': [0.225 , 0.333] ,
            'disgust': [0.052, 0.775],
            'anger': [0.167, 0.865],
            'fear': [0.073, 0.840],
            'neutral': [0.5, 0.5]
        }
        emotion_vectors = np.array([label_va['happy'], label_va['surprise'], label_va['sad'], label_va['disgust'], label_va['anger'], label_va['fear'], label_va['neutral']])

    emotion_vectors_normalize = (emotion_vectors - emotion_vectors.min(axis=0)) / (emotion_vectors.max(axis=0) - emotion_vectors.min(axis=0)) * 2 - 1
    # print(emotion_vectors_normalize)
    label_similarity_matrix = np.array(cosine_similarity(emotion_vectors_normalize))
    print(label_similarity_matrix)

    np.save(f'{dataset}_label_similarity_matrix.npy', label_similarity_matrix)