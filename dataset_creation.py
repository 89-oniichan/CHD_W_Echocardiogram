import pandas as pd
import numpy as np

def generate_data(start_id, num_patients, disease_type):
    # Data generation based on the disease type: 'Normal', 'ASD', 'VSD'
    data = {
        "Patient_ID": np.arange(start_id, start_id + num_patients),
        "Left_Atrium": np.random.uniform(4.0, 4.5, num_patients).round(1) if disease_type != 'Normal' else np.random.uniform(2.0, 4.0, num_patients).round(1),
        "Interventricular_Septum": np.random.uniform(0.7, 1.2, num_patients).round(1) if disease_type != 'Normal' else np.random.uniform(0.6, 1.0, num_patients).round(1),
        "Pulse_Wave_Doppler": np.random.uniform(1.0, 2.5, num_patients).round(1) if disease_type != 'Normal' else np.random.uniform(0.6, 1.0, num_patients).round(1),
        "LV_EDD": np.random.uniform(5.9, 10.0, num_patients).round(1) if disease_type != 'Normal' else np.random.uniform(3.5, 5.7, num_patients).round(1),
        "LV_ESD": np.random.uniform(4.0, 10.0, num_patients).round(1) if disease_type != 'Normal' else np.random.uniform(2.6, 3.9, num_patients).round(1),
        "Ejection_Fraction": np.random.randint(0, 50, num_patients) if disease_type != 'Normal' else np.random.randint(55, 70, num_patients),
        "Pulmonary_Jet": np.random.uniform(1.0, 3.1, num_patients).round(1) if disease_type != 'Normal' else np.random.uniform(0.6, 1.0, num_patients).round(1),
        "Aortic_Jet": np.random.uniform(1.7, 3.1, num_patients).round(1) if disease_type != 'Normal' else np.random.uniform(1.0, 1.7, num_patients).round(1)
    }
    return pd.DataFrame(data)

# Number of patients for each category
num_patients_normal = 20
num_patients_asd = 0
num_patients_vsd = 0

# Generate datasets
data_normal = generate_data(1, num_patients_normal, 'Normal')
data_asd = generate_data(num_patients_normal + 1, num_patients_asd, 'ASD')
data_vsd = generate_data(num_patients_normal + num_patients_asd + 1, num_patients_vsd, 'VSD')

# Combine the datasets
combined_data = pd.concat([data_normal, data_asd, data_vsd], ignore_index=True)

# Save to CSV
csv_file_path = 'your_dataset.csv'
combined_data.to_csv(csv_file_path, index=False)

print("Dataset saved to 'your_dataset.csv'")
