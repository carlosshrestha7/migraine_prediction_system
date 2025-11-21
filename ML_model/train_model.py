# train_model.py
"""
Training script for migraine prediction model
"""

import os
import sys
import pandas as pd

# Add current directory to path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from migraine_model import ModelTrainer


def main():
    print("\n" + "=" * 60)
    print("🧠 MIGRAINE PREDICTION ML MODEL - TRAINING MODULE")
    print("=" * 60 + "\n")
    
    # Check if data file exists
    data_file = 'event_dump.csv'
    if not os.path.exists(data_file):
        print(f"❌ Data file '{data_file}' not found!")
        print("Please make sure your CSV data file is in the same directory.")
        return
    
    try:
        # Train with your CSV file
        model = ModelTrainer.train_model_from_csv(
            csv_filepath=data_file,
            gender='female',
            save_path='migraine_model.pkl'
        )
        
        print("\n" + "=" * 60)
        print("✅ TRAINING COMPLETE!")
        print("=" * 60)
        print("📦 Model saved and ready for use.")
        print("💡 You can now run app.py to use the prediction system.")
        print("=" * 60)
        
    except FileNotFoundError:
        print(f"❌ Could not find data file: {data_file}")
        print("Please check that the file exists and try again.")
    except pd.errors.EmptyDataError:
        print(f"❌ Data file '{data_file}' is empty!")
        print("Please provide a valid CSV file with data.")
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check that your CSV file has the required columns")
        print("2. Make sure the file is not corrupted")
        print("3. Verify there are enough data samples (at least 100 recommended)")
        print("4. Check that the file format is correct CSV")
        
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()