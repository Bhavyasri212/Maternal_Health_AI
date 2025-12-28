import pandas as pd
import numpy as np
import os
import glob
import cv2  # You might need to run: pip install opencv-python
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from data_loader import DataLoader

class DataPreprocessor:
    def __init__(self):
        self.loader = DataLoader()
        self.scaler_clinical = StandardScaler()
        self.encoders = {} 
        
    def clean_clinical_data(self, df):
        print("   ... Processing Enhanced Clinical/Lifestyle Data")
        
        # 1. Map Target
        risk_map = {'low risk': 0, 'mid risk': 1, 'high risk': 2}
        df['RiskLevel'] = df['RiskLevel'].str.lower().str.strip()
        df['RiskLevel_Encoded'] = df['RiskLevel'].map(risk_map)
        df = df.dropna(subset=['RiskLevel_Encoded'])
        
        # 2. Categorical Encoding
        categorical_cols = ['education', 'income_category', 'urban_rural', 'diet_quality', 'diet_adherence']
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
        
        # 3. Numerical Features
        feature_cols = [
            'Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate',
            'sleep_hours', 'phys_activity_level', 'stress_score',
            'education', 'income_category', 'urban_rural',
            'diet_quality', 'hemoglobin', 'iron_suppl', 'folic_suppl', 'diet_adherence'
        ]
        available_cols = [c for c in feature_cols if c in df.columns]
        
        X_clinical = df[available_cols].values
        y_risk = df['RiskLevel_Encoded'].values
        
        # Simulate BirthWeight
        np.random.seed(42)
        base_weight = 3300
        risk_penalty = np.where(y_risk == 2, -600, np.where(y_risk == 1, -200, 0))
        hemo_penalty = np.where(df['hemoglobin'] < 11, -300, 0) if 'hemoglobin' in df.columns else 0
        noise = np.random.normal(0, 200, len(df))
        y_weight = base_weight + risk_penalty + hemo_penalty + noise
        
        X_clinical = self.scaler_clinical.fit_transform(X_clinical)
        
        return df, X_clinical, y_risk, y_weight

    def clean_ctg_data(self, df):
        print("   ... Processing CTG Data")
        df['NSP_Mapped'] = df['NSP'] - 1 
        ts_cols = ['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'ASTV', 'MSTV', 'ALTV', 'MLTV']
        imputer = SimpleImputer(strategy='mean')
        X_ctg = imputer.fit_transform(df[ts_cols])
        y_ctg = df['NSP_Mapped'].values
        return X_ctg, y_ctg

    def get_activity_window(self, df_mhealth, window_size=50):
        if df_mhealth is None or len(df_mhealth) < window_size:
            return np.zeros((window_size, 3))
        start_idx = np.random.randint(0, len(df_mhealth) - window_size)
        return df_mhealth.iloc[start_idx : start_idx + window_size].values

    def load_ultrasound_images(self, num_samples, img_size=(128, 128)):
        """ Loads images or generates placeholders if missing. """
        print("   ... Processing Ultrasound Images")
        # Try to find images in data/Ultrasound
        img_path = os.path.join(self.loader.data_path, "Ultrasound", "*.*")
        files = glob.glob(img_path)
        
        images = []
        use_real_images = len(files) > 0
        
        if use_real_images:
            print(f"   ℹ️ Found {len(files)} real images.")
        else:
            print("   ⚠️ No images found in data/Ultrasound. Generating synthetic noise placeholders.")

        for i in range(num_samples):
            if use_real_images:
                f = files[i % len(files)]
                img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                if img is None: 
                    img = np.zeros(img_size)
                else:
                    img = cv2.resize(img, img_size)
            else:
                # Generate black image with noise
                img = np.zeros(img_size, dtype=np.float32)
                noise = np.random.rand(*img_size) * 0.2
                img = img + noise
                
            img = img / 255.0  # Normalize
            img = np.expand_dims(img, axis=-1)
            images.append(img)
                
        return np.array(images)

    def fuse_datasets(self):
        print("🔄 Starting Quad-Modal Fusion (Clinical + CTG + Activity + Image)...")
        
        # Load
        clinical_raw = self.loader.load_maternal_risk_data()
        ctg_raw = self.loader.load_ctg_data()
        mhealth_raw = self.loader.load_mhealth_activity_sample(subject_id=1)
        
        # Clean
        df_clin, X_clin, y_risk, y_weight = self.clean_clinical_data(clinical_raw)
        X_ctg_pool, y_ctg_pool = self.clean_ctg_data(ctg_raw)
        
        # Load Images (Matched to Clinical Size)
        X_images = self.load_ultrasound_images(len(df_clin))
        
        X_final_ctg = []
        X_final_act = []
        
        # Fuse Loop
        for i in range(len(df_clin)):
            current_risk = int(y_risk[i])
            
            # CTG
            match_idxs = np.where(y_ctg_pool == current_risk)[0]
            if len(match_idxs) > 0:
                idx = np.random.choice(match_idxs)
                X_final_ctg.append(X_ctg_pool[idx])
            else:
                X_final_ctg.append(X_ctg_pool[0])
            
            # Activity
            X_final_act.append(self.get_activity_window(mhealth_raw))
            
        X_final_ctg = np.array(X_final_ctg).reshape((-1, 11, 1))
        X_final_act = np.array(X_final_act)
        
        print(f"✅ Fusion Complete. Images Shape: {X_images.shape}")
        
        return [X_clin, X_final_ctg, X_final_act, X_images], [y_risk, y_weight]