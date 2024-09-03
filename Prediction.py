import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Step 1: Data Preparation
data = pd.read_csv('your_dataset.csv')

# Step 2: Defining Fuzzy Variables and Membership Functions
left_atrium = ctrl.Antecedent(np.arange(0, 10.1, 0.1), 'left_atrium')
interventricular_septum = ctrl.Antecedent(np.arange(0, 2.1, 0.1), 'interventricular_septum')
pulse_wave_doppler = ctrl.Antecedent(np.arange(0, 3.1, 0.1), 'pulse_wave_doppler')
lv_edd = ctrl.Antecedent(np.arange(0, 10.1, 0.1), 'lv_edd')
lv_esd = ctrl.Antecedent(np.arange(0, 10.1, 0.1), 'lv_esd')
ejection_fraction = ctrl.Antecedent(np.arange(0, 100.1, 1), 'ejection_fraction')
pulmonary_jet = ctrl.Antecedent(np.arange(0, 4.1, 0.1), 'pulmonary_jet')
aortic_jet = ctrl.Antecedent(np.arange(0, 4.1, 0.1), 'aortic_jet')
disease_likelihood = ctrl.Consequent(np.arange(0, 100.1, 1), 'disease_likelihood')

# Define membership functions based on provided parameter ranges
# Membership functions for 'left_atrium'
left_atrium['normal'] = fuzz.trimf(left_atrium.universe, [0, 4, 10])
left_atrium['ASD'] = fuzz.trimf(left_atrium.universe, [4, 4, 4.5])
left_atrium['VSD'] = fuzz.trimf(left_atrium.universe, [0, 0, 4])

# Define membership functions for 'interventricular_septum'
interventricular_septum['normal'] = fuzz.trimf(interventricular_septum.universe, [0, 1, 2])
interventricular_septum['ASD'] = fuzz.trimf(interventricular_septum.universe, [0.7, 1.2, 2])
interventricular_septum['VSD'] = fuzz.trimf(interventricular_septum.universe, [0.7, 1.2, 2])

# Define membership functions for 'pulse_wave_doppler'
pulse_wave_doppler['normal'] = fuzz.trimf(pulse_wave_doppler.universe, [0, 1.5, 3])
pulse_wave_doppler['ASD'] = fuzz.trimf(pulse_wave_doppler.universe, [1.0, 2.5, 3])
pulse_wave_doppler['VSD'] = fuzz.trimf(pulse_wave_doppler.universe, [1.0, 2.5, 3])

# Define membership functions for 'lv_edd'
lv_edd['normal'] = fuzz.trimf(lv_edd.universe, [3.5, 5.7, 10])
lv_edd['ASD'] = fuzz.trimf(lv_edd.universe, [5.9, 10, 10])
lv_edd['VSD'] = fuzz.trimf(lv_edd.universe, [5.9, 10, 10])

# Define membership functions for 'lv_esd'
lv_esd['normal'] = fuzz.trimf(lv_esd.universe, [2.6, 3.9, 10])
lv_esd['ASD'] = fuzz.trimf(lv_esd.universe, [4.0, 10, 10])
lv_esd['VSD'] = fuzz.trimf(lv_esd.universe, [4.0, 10, 10])

# Define membership functions for 'ejection_fraction'
ejection_fraction['normal'] = fuzz.trimf(ejection_fraction.universe, [55, 70, 100])
ejection_fraction['ASD'] = fuzz.trimf(ejection_fraction.universe, [0, 50, 70])
ejection_fraction['VSD'] = fuzz.trimf(ejection_fraction.universe, [0, 50, 70])

# Define membership functions for 'pulmonary_jet'
pulmonary_jet['normal'] = fuzz.trimf(pulmonary_jet.universe, [0, 1.0, 2.5])
pulmonary_jet['ASD'] = fuzz.trimf(pulmonary_jet.universe, [1.0, 2.5, 4])
pulmonary_jet['VSD'] = fuzz.trimf(pulmonary_jet.universe, [1.0, 2.5, 4])

# Define membership functions for 'aortic_jet'
aortic_jet['normal'] = fuzz.trimf(aortic_jet.universe, [0, 1.7, 4])
aortic_jet['ASD'] = fuzz.trimf(aortic_jet.universe, [1.7, 4, 4])
aortic_jet['VSD'] = fuzz.trimf(aortic_jet.universe, [1.7, 4, 4])

# Define membership functions for 'disease_likelihood'
disease_likelihood['normal'] = fuzz.trimf(disease_likelihood.universe, [0, 30, 60])
disease_likelihood['ASD'] = fuzz.trimf(disease_likelihood.universe, [30, 60, 90])
disease_likelihood['VSD'] = fuzz.trimf(disease_likelihood.universe, [60, 90, 100])


# Step 3: Fuzzy Rules
# Define rules based on membership functions and parameter ranges for each variable
rules = [
    ctrl.Rule(left_atrium['normal'] & interventricular_septum['normal'] & pulse_wave_doppler['normal'] & lv_edd['normal'] & lv_esd['normal'] & ejection_fraction['normal'] & pulmonary_jet['normal'] & aortic_jet['normal'], disease_likelihood['normal']),

    # Define rules for ASD
    ctrl.Rule(left_atrium['ASD'] | interventricular_septum['ASD'] | pulse_wave_doppler['ASD'] | lv_edd['ASD'] | lv_esd['ASD'] | ejection_fraction['ASD'] | pulmonary_jet['ASD'] | aortic_jet['ASD'], disease_likelihood['ASD']),

    # Define rules for VSD
    ctrl.Rule(left_atrium['VSD'] | interventricular_septum['VSD'] | pulse_wave_doppler['VSD'] | lv_edd['VSD'] | lv_esd['VSD'] | ejection_fraction['VSD'] | pulmonary_jet['VSD'] | aortic_jet['VSD'], disease_likelihood['VSD'])
]


# Step 4: Model Building
disease_model = ctrl.ControlSystem(rules)
disease_prediction = ctrl.ControlSystemSimulation(disease_model)

# Step 5: Model Validation and Prediction (for each patient in the dataset)
for index, row in data.iterrows():
    # Set input values for the patient
    left_atrium_value = row['Left_Atrium']
    ivs_value = row['Interventricular_Septum']
    pulse_wave_doppler_value = row['Pulse_Wave_Doppler']
    lv_edd_value = row['LV_EDD']
    lv_esd_value = row['LV_ESD']
    ef_value = row['Ejection_Fraction']
    pulmonary_jet_value = row['Pulmonary_Jet']
    aortic_jet_value = row['Aortic_Jet']

    # Compute disease likelihood
    disease_prediction.input['left_atrium'] = left_atrium_value
    disease_prediction.input['interventricular_septum'] = ivs_value
    disease_prediction.input['pulse_wave_doppler'] = pulse_wave_doppler_value
    disease_prediction.input['lv_edd'] = lv_edd_value
    disease_prediction.input['lv_esd'] = lv_esd_value
    disease_prediction.input['ejection_fraction'] = ef_value
    disease_prediction.input['pulmonary_jet'] = pulmonary_jet_value
    disease_prediction.input['aortic_jet'] = aortic_jet_value

    disease_prediction.compute()

    # Get the disease likelihood for this patient
    likelihood = disease_prediction.output['disease_likelihood']

    # Determine the disease classification
    if likelihood <= 50:
        classification = "Normal"
    elif 50 < likelihood <= 65:
        classification = "ASD"
    else:
        classification = "VSD"

    # Prints the results for this patient
    print(f"Patient {row['Patient_ID']} - Disease Classification: {classification} (Likelihood: {likelihood}%)")
