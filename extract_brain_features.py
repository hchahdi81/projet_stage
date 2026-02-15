import os
import numpy as np
from descriptor import bit_glcm_haralick_beta
import cv2

def process_dataset(folder_path, output_file):
    """
    Traite un dataset à partir d'un dossier donné et génère une signature avec des labels basés sur les noms des dossiers.
    
    Args:
        folder_path (str): Chemin du dossier contenant les images.
        output_file (str): Nom du fichier de sortie pour sauvegarder les caractéristiques.
    """
    globalfeatures = []  # Liste pour stocker les caractéristiques

    # Vérifier si le dossier existe
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return

    # Parcourir les sous-dossiers
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # Vérifier les extensions d'images
                image_path = os.path.join(root, file)
                label = os.path.basename(root)  # Le nom du dossier parent devient le label
                print(f"Processing: {image_path} with label: {label}")

                # Chargement de l'image
                image = cv2.imread(image_path, 0)
                if image is None:
                    print(f"Failed to load image: {image_path}")
                    continue
                else:
                    print(f"Image loaded successfully: {image_path}, Shape: {image.shape}")

                # Extraction des caractéristiques
                features = bit_glcm_haralick_beta(image_path)
                if features is not None:
                    features = features + [label]  # Ajouter le label basé sur le dossier
                    globalfeatures.append(features)
                    print(f"Extracted features: {features}")
                else:
                    print(f"No features extracted for image: {image_path}")

    # Sauvegarder les caractéristiques si elles ont été extraites
    if globalfeatures:
        globalfeatures = np.array(globalfeatures, dtype=object)
        np.save(output_file, globalfeatures)
        print(f"Features saved to {output_file}")
    else:
        print(f"No features extracted for folder: {folder_path}")


# Chemin vers les datasets
brain_tumor_folder = "C:/Users/hatim/Desktop/2025/medical_project/multimodal_analysis/brain_dataset/Training"
no_tumor_brain_folder = "C:/Users/hatim/Desktop/2025/medical_project/multimodal_analysis/brain_dataset/Training_no_tumor"
brain_tumor_testing_folder = "C:/Users/hatim/Desktop/2025/medical_project/multimodal_analysis/brain_dataset/Testing"

# Appeler la fonction pour traiter les dossiers
process_dataset(brain_tumor_testing_folder, output_file="test_features.npy")
