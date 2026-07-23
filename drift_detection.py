import pandas as pd
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset

def load_data():
    reference_data = pd.read_csv("datasets_preprocessing/X_train_clean.csv")
    current_data = pd.read_csv("datasets_preprocessing/X_test_clean.csv")
    return reference_data, current_data

def main():
    print("Memuat dataset untuk pengujian Data Drift")
    reference_data, current_data = load_data()

    data_drift_report = Report(metrics=[
        DataDriftPreset()
    ])

    print("Mengalkulasi Data Drift")
    data_drift_report.run(reference_data=reference_data, current_data=current_data)

    output_html = "data_drift_report.html"
    data_drift_report.save_html(output_html)
    
    print(f"Laporan Data Drift berhasil dibuat: {output_html}")

if __name__ == "__main__":
    main()
