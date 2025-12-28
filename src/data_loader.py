import pandas as pd
import os

class DataLoader:
    def __init__(self):
        # Get absolute path of current file (data_loader.py)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Move one level up → project root
        project_root = os.path.abspath(os.path.join(current_dir, ".."))
        
        # Data directory
        self.data_path = os.path.join(project_root, "data")

    def load_ctg_data(self):
        """
        Loads the UCI Cardiotocography dataset (Excel).
        We specifically extract the 'Data' sheet which contains the processed features.
        """
        file_path = os.path.join(self.data_path, 'CTG.xls')
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CTG.xls not found in {self.data_path}. Please download it from UCI.")

        # Load specific sheet 'Data', skipping header rows to get to the actual table
        # The UCI Excel file usually has 1 row of descriptions, then headers. 
        # We drop rows with missing 'LB' (Baseline Heart Rate) as they are empty footer lines.
        df = pd.read_excel(file_path, sheet_name='Data', header=1)
        
        # Drop empty rows (the dataset often has empty lines at the end)
        df = df.dropna(subset=['LB'])
        
        print(f"✅ CTG Data Loaded. Shape: {df.shape}")
        return df

    def load_maternal_risk_data(self):
        """
        Loads the ENHANCED Maternal Health Dataset (with Lifestyle/Socio-economic data).
        """
        # Changed filename to the new one you downloaded
        file_path = os.path.join(self.data_path, 'Maternal_Health_Risk_Augmented.csv')
        
        # Fallback for the old name if the new one isn't there yet
        if not os.path.exists(file_path):
            print(f"⚠️ Enhanced file not found at {file_path}. Checking for standard dataset...")
            file_path = os.path.join(self.data_path, 'Maternal Health Risk Data Set.csv')
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No Maternal Risk CSV found in {self.data_path}")
            
        df = pd.read_csv(file_path)
        print(f"✅ Enhanced Maternal Data Loaded. Shape: {df.shape}")
        return df

    def load_mhealth_activity_sample(self, subject_id=1):
        """
        Loads a sample subject from MHEALTH to simulate 'Lifestyle/Activity' stream.
        We only take columns 0,1,2 (Chest Acceleration) as a proxy for movement.
        """
        # Note: MHEALTH files are usually named 'mHealth_subject1.log'
        filename = f'mHealth_subject{subject_id}.log'
        folder_path = os.path.join(self.data_path, 'MHEALTHDATASET')
        file_path = os.path.join(folder_path, filename)
        
        if not os.path.exists(file_path):
            print(f"⚠️ Warning: MHEALTH file {filename} not found. Skipping.")
            return None
            
        # MHEALTH is whitespace separated, no header
        # Use sep='\s+' for newer pandas versions to handle whitespace
        df = pd.read_csv(file_path, sep='\s+', header=None)
        
        # Rename vital columns (Chest Accel X, Y, Z)
        df = df.iloc[:, [0, 1, 2]]
        df.columns = ['acc_x', 'acc_y', 'acc_z']
        
        print(f"✅ MHEALTH Activity Sample Loaded for Subject {subject_id}. Shape: {df.shape}")
        return df

# --- Test Execution ---
if __name__ == "__main__":
    loader = DataLoader()
    try:
        ctg = loader.load_ctg_data()
        risk = loader.load_maternal_risk_data()
        
        print("\n--- CTG Preview ---")
        print(ctg.head())
        
        print("\n--- Maternal Risk Preview ---")
        print(risk.head())
        print(f"Columns in Risk Data: {risk.columns.tolist()}")
        
    except Exception as e:
        print(f"Error: {e}")