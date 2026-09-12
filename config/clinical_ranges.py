NORMAL_RANGES = {
    "heartrate":{"low":60,"high":100},"systolic_bp":{"low":90,"high":140},
    "diastolic_bp":{"low":60,"high":90},"respiratory_rate":{"low":10,"high":22},
    "spo2":{"low":95,"high":100},"glucose":{"low":70,"high":140},
    "potassium":{"low":3.5,"high":5.0},"sodium":{"low":136,"high":145},
    "hemoglobin":{"low":12.0,"high":17.5},"creatinine":{"low":0.6,"high":1.2},
    "blood_urea_nitro":{"low":7,"high":20},"age":{"low":18,"high":89},
}
REALITY_LIMITS = {
    "age":(0,120),"glucose":(5,2600),"sodium":(90,200),"spo2":(30,100),
    "respiratory_rate":(5,100),"heartrate":(15,260),"systolic_bp":(30,300),
    "diastolic_bp":(10,230),"creatinine":(0.1,30),"blood_urea_nitro":(1,400),
    "potassium":(1.0,12),"hemoglobin":(1.5,25),
}
EXTREME_THRESHOLDS = {
    "heartrate":{"low":40,"high":150},"systolic_bp":{"low":60,"high":200},
    "respiratory_rate":{"low":8,"high":35},"spo2":{"low":80},
    "glucose":{"low":50,"high":400},"potassium":{"low":2.5,"high":7.5},
    "sodium":{"low":120,"high":155},"hemoglobin":{"low":5.0,"high":18.0},
    "creatinine":{"high":8.0},"blood_urea_nitro":{"high":80},
}
FATAL_THRESHOLDS = {
    "heartrate":{"low":20,"high":200},"systolic_bp":{"low":40},
    "respiratory_rate":{"low":3,"high":90},"spo2":{"low":40},
    "glucose":{"low":20,"high":1500},"potassium":{"high":10},
    "sodium":{"low":100,"high":180},"hemoglobin":{"low":2.5},
    "creatinine":{"high":20},"blood_urea_nitro":{"high":180},
}
CATEGORICAL = {
    "admission_type":["ELECTIVE","EMERGENCY","URGENT"],
    "first_careunit":["CCU","CSRU","MICU","SICU","TSICU"],
    "ethnicity":["ASIAN","BLACK","HISPANIC","MISSING","OTHER","UNOBTAINABLE","WHITE"],
    "icd9":["Blood disorders","Circulatory system","Congenital","Digestive","Endocrine/metabolic","Genitourinary","Infectious diseases","Injury/poisoning","Mental disorders","Musculoskeletal","Neoplasms","Nervous system","Pregnancy","Respiratory","Skin","Supplementary factors (V-codes)","Symptoms/ill-defined","UNKNOWN"],
    "bmi":["Missing","Normal","Underweight","High","Obese","Morbidly obese"],
    "nt-probnp":["Missing","Normal","Acute","Critical"],
    "cholesterol":["Highrisk","Intermediate","Lowrisk","Missing"],
    "albumin":["High","Low","Missing","Normal"],
    "readmission":[0,1,"0","1",True,False,"True","False","TRUE","FALSE"],
    "prior_icu":[0,1,"0","1",True,False,"True","False","TRUE","FALSE"],
    "gender":["M","F"],
}
RULE_PARAMS = {
    "PP_IMPOSSIBLE_MAX":2,"PP_SUSPICIOUS_MAX":5,"PP_WIDE_MIN":150,"PP_WIDE_SBP_MAX":200,
    "PP_WIDE_AGE_MAX":60,"PP_HTN_SBP_MIN":200,"PP_HTN_PP_MAX":10,
    "BUNCR_RATIO_IMPOSSIBLE_LOW":1.0,"BUNCR_RATIO_HIGH":100,"BUNCR_HIGH_BUN_MIN":80,"BUNCR_HIGH_HB_MIN":11,
    "LOW_CR_MAX":0.1,"COMP_ANEMIA_HB_MAX":5.0,"COMP_ANEMIA_SBP_MAX":90,"COMP_SHOCK_SBP_MAX":50,
    "COMP_HYPERK_K_MIN":7.5,"COMP_HYPERK_CR_MAX":1.0,"COMP_HYPONA_NA_MAX":110,"COMP_HYPONA_SBP_MAX":90,
    "EXTREME_COUNT_MIN":5,"ELDERLY_AGE_MIN":60,"ELDERLY_HB_HIGH_MIN":18,
    "PREG_AGE_MIN":8,"PREG_AGE_MAX":60,"PREG_HB_MAX":15,
    "NUT_IMPOSSIBLE_HB_MIN":17,"NUT_ALBNORMAL_HB_MAX":6,"NUT_CHOLMALN_HB_MAX":7,
    "NUT_OBESEHYPO_SBP_MAX":90,"ALB_OBESE_ANEMIA_HB_MAX":5,
    # "GLUNA_GLUCOSE_MIN": 300,
    # "GLUNA_NA_MIN": 135,
    "HYPOK_K_MAX": 2.0,
    # "ALLNORMAL_SERIOUS_ICD9": 1,
}

# SERIOUS_ICD9 = ["Circulatory system", "Respiratory", "Infectious diseases", "Injury/poisoning", "Neoplasms"]

PROTEIN_WASTING_CHAPTERS = ["Neoplasms","Genitourinary","Digestive","Blood disorders"]
ELEVATED_BMI = ["High","Obese","Morbidly obese"]
