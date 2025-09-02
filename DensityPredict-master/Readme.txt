1. Collect data based on JB-2008 and NRLMSISE-00 model from folder "jb" and "nrl". The collecting process based on the accelerometer-derived density database, in the providing code the satellite is CHAMP.
2. Collect necessary data based on the framework in file "CollectData.m".
	2.1 Combine database use "AssambleData.m" if necessary.
3. Standardize data use "StandData.m".
4. Train model use "EvidentialModel.py" based on Python.
5. Evaluate model performance based on "ModelPerformance.m".