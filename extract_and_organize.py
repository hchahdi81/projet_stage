import os
import numpy as np
from descriptor import bit_glcm_haralick_beta
import cv2

# Fonction pour traiter chaque sous-dossier et générer une signature
def process_datasets_by_type(folder_paths):
    for root_folder in folder_paths:  # Parcourir chaque chemin fourni
        print(f"\n--- Starting processing for folder: {root_folder} ---\n")

        if not os.path.exists(root_folder):  # Vérifier si le chemin existe
            print(f"Folder not found: {root_folder}")
            continue

        for folder in os.listdir(root_folder):  # Parcourir chaque sous-dossier
            folder_path = os.path.join(root_folder, folder)
            if os.path.isdir(folder_path):  # Vérifier si c'est un dossier
                print(f"Processing folder: {folder}")

                globalfeatures = []  # Liste pour stocker les caractéristiques de ce type de cancer
                for root, _, files in os.walk(folder_path):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            image_path = os.path.join(root, file)
                            label = os.path.basename(folder_path)  # Le nom du dossier est le label
                            print(f"Processing: {image_path}")

                            # Vérification si le fichier existe
                            if not os.path.exists(image_path):
                                print(f"File not found: {image_path}")
                                continue

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
                                features = features + [label, image_path]  # Ajouter le label et le chemin à la signature
                                globalfeatures.append(features)
                                print(f"Extracted features with bit glcm haralick: {features}")
                            else:
                                print(f"No features extracted for image: {image_path}")

                # Conversion en tableau NumPy et sauvegarde
                if globalfeatures:  # Sauvegarder uniquement si des caractéristiques ont été extraites
                    globalfeatures = np.array(globalfeatures)
                    output_file = f"{folder}_features.npy"  # Nom du fichier basé sur le dossier
                    np.save(output_file, globalfeatures)
                    print(f"Features for {folder} saved to {output_file}")
                else:
                    print(f"No features extracted for folder: {folder}")

        print(f"\n--- Finished processing for folder: {root_folder} ---\n")

# Liste des chemins de datasets à traiter
dataset_paths = [
    "C:/Users/hatim/Desktop/2025/medical_project/multimodal_analysis/datasets/Multi_Cancer/Brain Cancer",
    "C:/Users/hatim/Desktop/2025/medical_project/multimodal_analysis/datasets/Multi_Cancer/Breast Cancer",
    "C:/Users/hatim/Desktop/2025/medical_project/multimodal_analysis/datasets/Multi_Cancer/Cervical Cancer",
    "C:/Users/hatim/Desktop/2025/medical_project/multimodal_analysis/datasets/Multi_Cancer/Kidney Cancer",
    "C:/Users/hatim/Desktop/2025/medical_project/multimodal_analysis/datasets/Multi_Cancer/Lung and Colon Cancer",
    "C:/Users/hatim/Desktop/2025/medical_project/multimodal_analysis/datasets/Multi_Cancer/Lymphoma",
    "C:/Users/hatim/Desktop/2025/medical_project/multimodal_analysis/datasets/Multi_Cancer/Oral Cancer",

]

# Appeler la fonction pour traiter les datasets
process_datasets_by_type(dataset_paths)
