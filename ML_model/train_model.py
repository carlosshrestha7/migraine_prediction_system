# train_model.py
"""
Training script for migraine prediction model
"""

from migraine_model import ModelTrainer


def main():
    print("=== Migraine Prediction ML Model - Training Module ===\n")
    
    try:
        # Train with your CSV file
        model = ModelTrainer.train_model_from_csv(
            csv_filepath='event_dump.csv',
            gender='female',
            save_path='migraine_model.pkl'
        )
        
        print("\n" + "=" * 60)
        print("✅ Training complete! Model saved and ready for use.")
        print("=" * 60)
        
    except FileNotFoundError:
        print("❌ CSV file 'event_dump.csv' not found.")
        print("💡 Please make sure your data file exists in the current directory")
    except Exception as e:
        print(f"❌ Error during training: {e}")
        print("💡 Check your CSV file format and data quality")


if __name__ == "__main__":
    main()