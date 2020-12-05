# ORN_Classification
This repository was made to classify Osteoradionecrosis (ORN) after H&amp;N radiotherapy


Osteoradionecrosis (ORN), or bone death, is a multifactorial late complication caused by radiotherapy (RT),
that diminishes the bone’s ability to withstand trauma and avoid infection. In head and neck cancer (HNC), 
ORN most commonly manifests in the mandible. It is influenced by RT-induced factors and patient-related parameters.
The most effective way to limit RT-induced ORN is to reduce mandibular volumes receiving high RT doses.
However, this strategy may also result in a reduced dose to the tumor due to its proximity to the mandible,
and therefore must be restricted to only the patients deemed most vulnerable to mandibular ORN. 

The objective of this study was to design a prognostic model based on RT-planning CT-derived radiomic features
extracted from mandible contours along with the patient’s clinical features to predict the probability of mandibular
ORN from the end of RT to the onset of ORN in HNC patients. We hypothesized that these features are related to mandibular ORN
and that incorporating them into a prediction model will help to identify patients at risk of mandibular ORN after HNC RT. 

Patient data was retrospectively collected from the Princess Margaret Cancer Centre, University Health Network and 
based on the following inclusion criteria: patients had (1) had status regarding radiation-induced bone toxicity,
(2) the time to the toxicity event was recorded, and (3) head RT-planning CT images in addition to mandible contours.
Then, the patient’s history was reviewed by radiation oncologists to collect clinical features. 
Quantitative image features were then extracted from the segmented mandible for each patient. 
Finally, multivariable models, a binary classifier, and a regressor were independently trained on three sets of features (radiomic, demo-clinical, and both)
to predict the patient’s risk of ORN and the time between the end of RT and the start of ORN respectively. 

In total, we analyzed CT images from 336 OPC patients with known ORN status (175 positive, 161 negative).
We extracted a total of 1877 radiomic features from the manually-segmented Mandible from each patient. 
Initially, cases were labelled with their status regarding presence and time to radiation toxicity. 
Top 50, most relevant and least redundant, features were with mRMRe were used for both binary classification and regression models to predict time to the mandibular ORN. 
Model training upon radiomics and clinical features resulted in the higher accuracy value of 0.93 (AUCROC) compared to the only radiomics features (0.92) and clinical model (0.74). The distribution of AUC values was significantly wider in models trained on clinical features than radiomics or radiomics plus clinical features. 
