import cv2
import torch
import numpy as np
import json
from ultralytics import YOLO
import os
from datetime import datetime
import pandas as pd
from PIL import Image
from transformers import DPTImageProcessor, DPTForDepthEstimation
import math

# ======================= KONFIGURASI =======================
# Path ke folder dataset Anda
DATASET_FOLDER = "pear_seaweed_augmented"  # Folder berisi gambar
CALIBRATION_FILE = "calibration.json"
OUTPUT_CSV = "feature_extracted_augment_dataset.csv"  # File output hasil ekstraksi fitur
YOLO_MODEL_PATH = "pear_seaweed_final.pt"

# ===========================================================

class SeaweedFeatureExtractor:
    """Class untuk ekstraksi fitur dari citra rumput laut"""
    
    def __init__(self, yolo_model_path, calibration_file):
        """Inisialisasi model dan kalibrasi"""
        print("[INFO] Memuat model YOLO dan DPT...")
        
        # Load YOLO
        self.yolo_model = YOLO(yolo_model_path)
        
        # Load DPT untuk depth estimation
        dpt_model_name = "Intel/dpt-large"
        self.dpt_processor = DPTImageProcessor.from_pretrained(dpt_model_name)
        self.dpt_model = DPTForDepthEstimation.from_pretrained(dpt_model_name)
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo_model.to(self.device)
        self.dpt_model.to(self.device)
        self.dpt_model.eval()
        
        # Load camera calibration
        self.camera_matrix = self.load_calibration(calibration_file)
        
        print(f"[INFO] Model berhasil dimuat. Device: {self.device}")
    
    def load_calibration(self, filepath):
        """Load camera calibration matrix"""
        try:
            with open(filepath, 'r') as f:
                return np.array(json.load(f)['camera_matrix'])
        except FileNotFoundError:
            print(f"[ERROR] File kalibrasi tidak ditemukan: {filepath}")
            return None
    
    def estimate_distance(self, w, h):
        """Estimasi jarak kamera berdasarkan ukuran bounding box standar"""
        size = w + h
        if size == 0:
            return None
        a, b = 21550, 16.159
        distance = a / size + b
        return round(distance, 2)
    
    def get_depth_map(self, image_bgr):
        """Mendapatkan depth map dari DPT"""
        # Convert BGR to RGB PIL Image
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # Process dengan DPT
        inputs = self.dpt_processor(images=img_pil, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.dpt_model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Resize ke ukuran asli
        depth_map = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=img_pil.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()
        
        return depth_map
    
    def extract_features_from_image(self, image_path):
        """
        Ekstraksi 10 fitur dari satu gambar
        
        Returns:
            list: List of dictionaries berisi fitur untuk setiap objek terdeteksi
        """
        # Baca gambar
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"[ERROR] Gagal membaca gambar: {image_path}")
            return []
        
        # Deteksi dengan YOLO
        yolo_results = self.yolo_model(frame, conf=0.5, verbose=False)
        
        if not yolo_results or yolo_results[0].masks is None:
            print(f"[WARNING] Tidak ada objek terdeteksi di: {image_path}")
            return []
        
        # Dapatkan depth map
        depth_map = self.get_depth_map(frame)
        
        # List untuk menyimpan fitur semua objek
        features_list = []
        
        num_objects = len(yolo_results[0].masks.data)
        print(f"[INFO] Memproses {num_objects} objek dari {os.path.basename(image_path)}")
        
        for i in range(num_objects):
            try:
                features = self.extract_single_object_features(
                    frame, yolo_results, i, depth_map, image_path
                )
                if features:
                    features_list.append(features)
            except Exception as e:
                print(f"[ERROR] Gagal ekstraksi objek {i}: {e}")
                continue
        
        return features_list
    
    def extract_single_object_features(self, frame, yolo_results, obj_idx, depth_map, image_path):
        """Ekstraksi fitur untuk satu objek menggunakan STANDARD BOUNDING BOX"""
        
        # 1. Ambil mask
        mask_tensor = yolo_results[0].masks.data[obj_idx]
        mask = cv2.resize(
            mask_tensor.cpu().numpy().astype(np.uint8),
            (frame.shape[1], frame.shape[0])
        )
        
        # 2. Temukan kontur
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 5:
            return None
        
        # 3. Hitung STANDARD BOUNDING BOX (Axis-Aligned Bounding Box)
        x, y, w_px, h_px = cv2.boundingRect(contour)
        
        # Validasi ukuran bounding box
        if w_px <= 0 or h_px <= 0:
            return None
        
        # 4. FITUR 4: Jarak Kamera (cm)
        distance_cm = self.estimate_distance(w_px, h_px)
        if not distance_cm or distance_cm <= 0:
            return None
        
        # 5. Konversi pixel ke cm menggunakan camera matrix
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        
        # FITUR 2: Panjang Bounding Box (cm) - ini adalah WIDTH dari standard bbox
        panjang_cm = w_px * distance_cm / fx
        
        # FITUR 3: Lebar Bounding Box (cm) - ini adalah HEIGHT dari standard bbox
        lebar_cm = h_px * distance_cm / fy
        
        # 6. FITUR 1: Area Segmentasi (cm²)
        pixel_area = np.sum(mask > 0)
        bbox_area_px = w_px * h_px
        cm_per_pixel = (panjang_cm * lebar_cm) / bbox_area_px if bbox_area_px > 0 else 0
        area_cm2 = pixel_area * cm_per_pixel
        
        # 7. FITUR 5: Ketebalan Terukur (cm) dari depth map
        depth_values = depth_map[mask > 0]
        valid_depths = depth_values[depth_values > 0]
        
        if len(valid_depths) == 0:
            return None
        
        avg_relative_depth = np.mean(valid_depths)
        scale_constant = distance_cm / (avg_relative_depth + 1e-6)
        scaled_depths = valid_depths * scale_constant
        ketebalan_cm = np.mean(scaled_depths) * 0.1  # Faktor konversi ke ketebalan
        
        # 8. FITUR 6: Volume Estimasi (cm³)
        volume_cm3 = area_cm2 * ketebalan_cm
        
        # 9. FITUR 7: Aspect Ratio (width/height dari standard bbox)
        aspect_ratio = panjang_cm / lebar_cm if lebar_cm > 0 else 0
        
        # 10. FITUR 8: Perimeter (cm)
        perimeter_px = cv2.arcLength(contour, True)
        perimeter_cm = perimeter_px * math.sqrt(cm_per_pixel)
        
        # 11. FITUR 9: Solidity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = pixel_area / hull_area if hull_area > 0 else 0
        
        # 12. FITUR 10: Compactness
        compactness = (perimeter_px ** 2) / (4 * math.pi * pixel_area) if pixel_area > 0 else 0
        
        # Kembalikan dictionary fitur
        features = {
            'image_file': os.path.basename(image_path),
            'object_id': obj_idx,
            'bbox_x': x,
            'bbox_y': y,
            'bbox_width_px': w_px,
            'bbox_height_px': h_px,
            'area_cm2': round(area_cm2, 2),
            'panjang_cm': round(panjang_cm, 2),
            'lebar_cm': round(lebar_cm, 2),
            'jarak_kamera_cm': round(distance_cm, 2),
            'ketebalan_cm': round(ketebalan_cm, 2),
            'volume_cm3': round(volume_cm3, 2),
            'aspect_ratio': round(aspect_ratio, 3),
            'perimeter_cm': round(perimeter_cm, 2),
            'solidity': round(solidity, 3),
            'compactness': round(compactness, 3)
        }
        
        return features


def main():
    """Main function untuk ekstraksi fitur dari semua gambar"""
    
    # 1. Inisialisasi extractor
    extractor = SeaweedFeatureExtractor(YOLO_MODEL_PATH, CALIBRATION_FILE)
    
    if extractor.camera_matrix is None:
        print("[ERROR] Kalibrasi kamera gagal. Program dihentikan.")
        return
    
    # 2. Dapatkan semua file gambar
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [
        os.path.join(DATASET_FOLDER, f) 
        for f in os.listdir(DATASET_FOLDER) 
        if os.path.splitext(f)[1].lower() in image_extensions
    ]
    
    print(f"\n[INFO] Ditemukan {len(image_files)} gambar di folder {DATASET_FOLDER}")
    
    # 3. Ekstraksi fitur dari semua gambar
    all_features = []
    
    for idx, img_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] Processing: {os.path.basename(img_path)}")
        features = extractor.extract_features_from_image(img_path)
        all_features.extend(features)
    
    # 4. Simpan ke CSV
    if all_features:
        df = pd.DataFrame(all_features)
        # Gunakan comma sebagai separator (standar CSV)
        df.to_csv(OUTPUT_CSV, index=False, sep=',')
        print(f"\n[SUCCESS] Ekstraksi selesai! {len(all_features)} objek berhasil diproses.")
        print(f"[INFO] Fitur tersimpan di: {OUTPUT_CSV}")
        print(f"\nPreview data:")
        print(df.head())
        print(f"\nStatistik fitur:")
        print(df.describe())
    else:
        print("[WARNING] Tidak ada fitur yang berhasil diekstraksi.")


if __name__ == "__main__":
    main()