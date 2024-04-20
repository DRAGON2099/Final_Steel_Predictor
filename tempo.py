# Load the CSV data into a DataFrame
df = pd.read_csv('your_training_data_conc.csv')

# Data Preprocessing
# Assuming the columns 'carbon_percent', 'chromium_percent', 'aluminium_percent', etc., are features, and 'uts', 'elongation', 'yield_strength', 'reduction_area' are target variables
X = df[['uts', 'elongation', 'yield_strength', 'reduction_area']]  # Features
y_carbon_percent = df['carbon_percent']  # Target variable for carbon percent prediction
y_chromium_percent = df['chromium_percent']  # Target variable for chromium percent prediction
y_aluminium_percent = df['aluminium_percent']  # Target variable for aluminium percent prediction
y_manganese_percent = df['manganese_percent']  # Target variable for manganese percent prediction
y_silicon_percent = df['silicon_percent']  # Target variable for silicon percent prediction
y_nickel_percent = df['nickel_percent']  # Target variable for nickel percent prediction
y_cobalt_percent = df['cobalt_percent']  # Target variable for cobalt percent prediction
y_molybdenum_percent = df['molybdenum_percent']  # Target variable for molybdenum percent prediction
y_tungsten_percent = df['tungsten_percent']  # Target variable for tungsten percent prediction
y_niobium_percent = df['niobium_percent']  # Target variable for niobium percent prediction
y_phosphorous_percent = df['phosphorous_percent']  # Target variable for phosphorous percent prediction
y_copper_percent = df['copper_percent']  # Target variable for copper percent prediction
y_titanium_percent = df['titanium_percent']  # Target variable for titanium percent prediction
y_tantalum_percent = df['tantalum_percent']  # Target variable for tantalum percent prediction
y_hafnium_percent = df['hafnium_percent']  # Target variable for hafnium percent prediction
y_rhenium_percent = df['rhenium_percent']  # Target variable for rhenium percent prediction
y_vanadium_percent = df['vanadium_percent']  # Target variable for vanadium percent prediction
y_boron_percent = df['boron_percent']  # Target variable for boron percent prediction
y_nitrogen_percent = df['nitrogen_percent']  # Target variable for nitrogen percent prediction
y_oxygen_percent = df['oxygen_percent']  # Target variable for oxygen percent prediction
y_sulphur_percent = df['sulphur_percent']  # Target variable for sulphur percent prediction

# Splitting the data into training and testing sets
X_train, X_test, y_carbon_percent_train, y_carbon_percent_test = train_test_split(X, y_carbon_percent, test_size=0.2, random_state=42)
X_train, X_test, y_chromium_percent_train, y_chromium_percent_test = train_test_split(X, y_chromium_percent, test_size=0.2, random_state=42)
X_train, X_test, y_aluminium_percent_train, y_aluminium_percent_test = train_test_split(X, y_aluminium_percent, test_size=0.2, random_state=42)
X_train, X_test, y_manganese_percent_train, y_manganese_percent_test = train_test_split(X, y_manganese_percent, test_size=0.2, random_state=42)
X_train, X_test, y_silicon_percent_train, y_silicon_percent_test = train_test_split(X, y_silicon_percent, test_size=0.2, random_state=42)
X_train, X_test, y_nickel_percent_train, y_nickel_percent_test = train_test_split(X, y_nickel_percent, test_size=0.2, random_state=42)
X_train, X_test, y_cobalt_percent_train, y_cobalt_percent_test = train_test_split(X, y_cobalt_percent, test_size=0.2, random_state=42)
X_train, X_test, y_molybdenum_percent_train, y_molybdenum_percent_test = train_test_split(X, y_molybdenum_percent, test_size=0.2, random_state=42)
X_train, X_test, y_tungsten_percent_train, y_tungsten_percent_test = train_test_split(X, y_tungsten_percent, test_size=0.2, random_state=42)
X_train, X_test, y_niobium_percent_train, y_niobium_percent_test = train_test_split(X, y_niobium_percent, test_size=0.2, random_state=42)
X_train, X_test, y_phosphorous_percent_train, y_phosphorous_percent_test = train_test_split(X, y_phosphorous_percent, test_size=0.2, random_state=42)
X_train, X_test, y_copper_percent_train, y_copper_percent_test = train_test_split(X, y_copper_percent, test_size=0.2, random_state=42)
X_train, X_test, y_titanium_percent_train, y_titanium_percent_test = train_test_split(X, y_titanium_percent, test_size=0.2, random_state=42)
X_train, X_test, y_tantalum_percent_train, y_tantalum_percent_test = train_test_split(X, y_tantalum_percent, test_size=0.2, random_state=42)
X_train, X_test, y_hafnium_percent_train, y_hafnium_percent_test = train_test_split(X, y_hafnium_percent, test_size=0.2, random_state=42)
X_train, X_test, y_rhenium_percent_train, y_rhenium_percent_test = train_test_split(X, y_rhenium_percent, test_size=0.2, random_state=42)
X_train, X_test, y_vanadium_percent_train, y_vanadium_percent_test = train_test_split(X, y_vanadium_percent, test_size=0.2, random_state=42)
X_train, X_test, y_boron_percent_train, y_boron_percent_test = train_test_split(X, y_boron_percent, test_size=0.2, random_state=42)
X_train, X_test, y_nitrogen_percent_train, y_nitrogen_percent_test = train_test_split(X, y_nitrogen_percent, test_size=0.2, random_state=42)
X_train, X_test, y_oxygen_percent_train, y_oxygen_percent_test = train_test_split(X, y_oxygen_percent, test_size=0.2, random_state=42)
X_train, X_test, y_sulphur_percent_train, y_sulphur_percent_test = train_test_split(X, y_sulphur_percent, test_size=0.2, random_state=42)

# Model Training
model_carbon_percent = LinearRegression()
model_carbon_percent.fit(X_train, y_carbon_percent_train)

model_chromium_percent = LinearRegression()
model_chromium_percent.fit(X_train, y_chromium_percent_train)

model_aluminium_percent = LinearRegression()
model_aluminium_percent.fit(X_train, y_aluminium_percent_train)

model_manganese_percent = LinearRegression()
model_manganese_percent.fit(X_train, y_manganese_percent_train)

model_silicon_percent = LinearRegression()
model_silicon_percent.fit(X_train, y_silicon_percent_train)

model_nickel_percent = LinearRegression()
model_nickel_percent.fit(X_train, y_nickel_percent_train)

model_cobalt_percent = LinearRegression()
model_cobalt_percent.fit(X_train, y_cobalt_percent_train)

model_molybdenum_percent = LinearRegression()
model_molybdenum_percent.fit(X_train, y_molybdenum_percent_train)

model_tungsten_percent = LinearRegression()
model_tungsten_percent.fit(X_train, y_tungsten_percent_train)

model_niobium_percent = LinearRegression()
model_niobium_percent.fit(X_train, y_niobium_percent_train)

model_phosphorous_percent = LinearRegression()
model_phosphorous_percent.fit(X_train, y_phosphorous_percent_train)

model_copper_percent = LinearRegression()
model_copper_percent.fit(X_train, y_copper_percent_train)

model_titanium_percent = LinearRegression()
model_titanium_percent.fit(X_train, y_titanium_percent_train)

model_tantalum_percent = LinearRegression()
model_tantalum_percent.fit(X_train, y_tantalum_percent_train)

model_hafnium_percent = LinearRegression()
model_hafnium_percent.fit(X_train, y_hafnium_percent_train)

model_rhenium_percent = LinearRegression()
model_rhenium_percent.fit(X_train, y_rhenium_percent_train)

model_vanadium_percent = LinearRegression()
model_vanadium_percent.fit(X_train, y_vanadium_percent_train)

model_boron_percent = LinearRegression()
model_boron_percent.fit(X_train, y_boron_percent_train)

model_nitrogen_percent = LinearRegression()
model_nitrogen_percent.fit(X_train, y_nitrogen_percent_train)

model_oxygen_percent = LinearRegression()
model_boron_percent.fit(X_train, y_boron_percent_train)

model_sulphur_percent = LinearRegression()
model_sulphur_percent.fit(X_train, y_sulphur_percent_train)

# Model Evaluation (Optional)
carbon_percent_score = model_carbon_percent.score(X_test, y_carbon_percent_test)
chromium_percent_score = model_chromium_percent.score(X_test, y_chromium_percent_test)
aluminium_percent_score = model_aluminium_percent.score(X_test, y_aluminium_percent_test)
manganese_percent_score = model_manganese_percent.score(X_test, y_manganese_percent_test)
silicon_percent_score = model_silicon_percent.score(X_test, y_silicon_percent_test)
nickel_percent_score = model_nickel_percent.score(X_test, y_nickel_percent_test)
cobalt_percent_score = model_cobalt_percent.score(X_test, y_cobalt_percent_test)
molybdenum_percent_score = model_molybdenum_percent.score(X_test, y_molybdenum_percent_test)
tungsten_percent_score = model_tungsten_percent.score(X_test, y_tungsten_percent_test)
niobium_percent_score = model_niobium_percent.score(X_test, y_niobium_percent_test)
phosphorous_percent_score = model_phosphorous_percent.score(X_test, y_phosphorous_percent_test)
copper_percent_score = model_copper_percent.score(X_test, y_copper_percent_test)
titanium_percent_score = model_titanium_percent.score(X_test, y_titanium_percent_test)
tantalum_percent_score = model_tantalum_percent.score(X_test, y_tantalum_percent_test)
hafnium_percent_score = model_hafnium_percent.score(X_test, y_hafnium_percent_test)
rhenium_percent_score = model_rhenium_percent.score(X_test, y_rhenium_percent_test)
vanadium_percent_score = model_vanadium_percent.score(X_test, y_vanadium_percent_test)
boron_percent_score = model_boron_percent.score(X_test, y_boron_percent_test)
nitrogen_percent_score = model_nitrogen_percent.score(X_test, y_nitrogen_percent_test)
oxygen_percent_score = model_oxygen_percent.score(X_test, y_oxygen_percent_test)
sulphur_percent_score = model_sulphur_percent.score(X_test, y_sulphur_percent_test)

# Save Trained Models
joblib.dump(model_carbon_percent, 'model_carbon_percent.pkl')
joblib.dump(model_chromium_percent, 'model_chromium_percent.pkl')
joblib.dump(model_aluminium_percent, 'model_aluminium_percent.pkl')
joblib.dump(model_manganese_percent, 'model_manganese_percent.pkl')
joblib.dump(model_silicon_percent, 'model_silicon_percent.pkl')
joblib.dump(model_nickel_percent, 'model_nickel_percent.pkl')
joblib.dump(model_cobalt_percent, 'model_cobalt_percent.pkl')
joblib.dump(model_molybdenum_percent, 'model_molybdenum_percent.pkl')
joblib.dump(model_tungsten_percent, 'model_tungsten_percent.pkl')
joblib.dump(model_niobium_percent, 'model_niobium_percent.pkl')
joblib.dump(model_phosphorous_percent, 'model_phosphorous_percent.pkl')
joblib.dump(model_copper_percent, 'model_copper_percent.pkl')
joblib.dump(model_titanium_percent, 'model_titanium_percent.pkl')
joblib.dump(model_tantalum_percent, 'model_tantalum_percent.pkl')
joblib.dump(model_hafnium_percent, 'model_hafnium_percent.pkl')
joblib.dump(model_rhenium_percent, 'model_rhenium_percent.pkl')
joblib.dump(model_vanadium_percent, 'model_vanadium_percent.pkl')
joblib.dump(model_boron_percent, 'model_boron_percent.pkl')
joblib.dump(model_nitrogen_percent, 'model_nitrogen_percent.pkl')
joblib.dump(model_oxygen_percent, 'model_oxygen_percent.pkl')
joblib.dump(model_sulphur_percent, 'model_sulphur_percent.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json

    # Extract input features
    uts = input_data['uts']
    elongation = input_data['elongation']
    yield_strength = input_data['yield_strength']
    reduction_area = input_data['reduction_area']
    # Extract more input features as needed

    # Load the trained models
    model_carbon_percent = joblib.load('model_carbon_percent.pkl')
    model_chromium_percent = joblib.load('model_chromium_percent.pkl')
    model_aluminium_percent = joblib.load('model_aluminium_percent.pkl')
    model_manganese_percent = joblib.load('model_manganese_percent.pkl')
    model_silicon_percent = joblib.load('model_silicon_percent.pkl')  # Machine learning model for silicon percent prediction
    model_nickel_percent = joblib.load('model_nickel_percent.pkl')  # Machine learning model for nickel percent prediction
    model_cobalt_percent = joblib.load('model_cobalt_percent.pkl')  # Machine learning model for cobalt percent prediction
    model_molybdenum_percent = joblib.load('model_molybdenum_percent.pkl')  # Machine learning model for molybdenum percent prediction
    model_tungsten_percent = joblib.load('model_tungsten_percent.pkl')  # Machine learning model for tungsten percent prediction
    model_niobium_percent = joblib.load('model_niobium_percent.pkl')  # Machine learning model for niobium percent prediction
    model_phosphorous_percent = joblib.load('model_phosphorous_percent.pkl')  # Machine learning model for phosphorous percent prediction
    model_copper_percent = joblib.load('model_copper_percent.pkl')  # Machine learning model for copper percent prediction
    model_titanium_percent = joblib.load('model_titanium_percent.pkl')  # Machine learning model for titanium percent prediction
    model_tantalum_percent = joblib.load('model_tantalum_percent.pkl')  # Machine learning model for tantalum percent prediction
    model_hafnium_percent = joblib.load('model_hafnium_percent.pkl')  # Machine learning model for hafnium percent prediction
    model_rhenium_percent = joblib.load('model_rhenium_percent.pkl')  # Machine learning model for rhenium percent prediction
    model_vanadium_percent = joblib.load('model_vanadium_percent.pkl')  # Machine learning model for vanadium percent prediction
    model_boron_percent = joblib.load('model_boron_percent.pkl')  # Machine learning model for boron percent prediction
    model_nitrogen_percent = joblib.load('model_nitrogen_percent.pkl')  # Machine learning model for nitrogen percent prediction
    model_oxygen_percent = joblib.load('model_oxygen_percent.pkl')  # Machine learning model for oxygen percent prediction
    model_sulphur_percent = joblib.load('model_sulphur_percent.pkl')  # Machine learning model for sulphur percent prediction


    # Make predictions using the loaded models
    carbon_percent_prediction = model_carbon_percent.predict([[uts, elongation, yield_strength,reduction_area]])[0]
    chromium_percent_prediction = model_chromium_percent.predict([[uts, elongation, yield_strength,reduction_area]])[0]
    aluminium_percent_prediction = model_aluminium_percent.predict([[uts, elongation, yield_strength,reduction_area]])[0]
    manganese_percent_prediction = model_manganese_percent.predict([[uts, elongation, yield_strength,reduction_area]])[0]
    silicon_percent_prediction = model_silicon_percent.predict([[uts, elongation, yield_strength, reduction_area]])[0]
    nickel_percent_prediction = model_nickel_percent.predict([[uts, elongation, yield_strength, reduction_area]])[0]
    cobalt_percent_prediction = model_cobalt_percent.predict([[uts, elongation, yield_strength, reduction_area]])[0]
    molybdenum_percent_prediction = model_molybdenum_percent.predict([[uts, elongation, yield_strength, reduction_area]])[0]
    tungsten_percent_prediction = model_tungsten_percent.predict([[uts, elongation, yield_strength, reduction_area]])[0]
    niobium_percent_prediction = model_niobium_percent.predict([[uts, elongation, yield_strength, reduction_area]])[0]
    phosphorous_percent_prediction = model_phosphorous_percent.predict([[uts, elongation, yield_strength, reduction_area]])[0]
    copper_percent_prediction = model_copper_percent.predict([[uts, elongation, yield_strength, reduction_area]])[0]
    titanium_percent_prediction = model_titanium_percent.predict([[uts, elongation, yield_strength, reduction_area]])[0]
    tantalum_percent_prediction = model_tantalum_percent.predict([[uts, elongation, yield_strength, reduction_area]])[0]
    hafnium_percent_prediction = model_hafnium_percent.predict([[uts, elongation, yield_strength, reduction_area]])[0]
    rhenium_percent_prediction = model_rhenium_percent.predict([[uts, elongation, yield_strength, reduction_area]])[0]
    vanadium_percent_prediction = model_vanadium_percent.predict([[uts, elongation, yield_strength, reduction_area]])[0]
    boron_percent_prediction = model_boron_percent.predict([[uts, elongation, yield_strength, reduction_area]])[0]
    nitrogen_percent_prediction = model_nitrogen_percent.predict([[uts, elongation, yield_strength, reduction_area]])[0]
    oxygen_percent_prediction = model_oxygen_percent.predict([[uts, elongation, yield_strength, reduction_area]])[0]
    sulphur_percent_prediction = model_sulphur_percent.predict([[uts, elongation, yield_strength, reduction_area]])[0]


    # Calculate accuracy (R2 score) - Optional
    # accuracy = ...

    # Prepare response data
    response_data = {
        'carbon_percent_Prediction': carbon_percent_prediction,
        'chromium_percent_Prediction': chromium_percent_prediction,
        'aluminium_percent_Prediction': aluminium_percent_prediction,
        'manganese_percent_Prediction': manganese_percent_prediction,
        'silicon_percent_Prediction': silicon_percent_prediction,
        'nickel_percent_Prediction': nickel_percent_prediction,
        'cobalt_percent_Prediction': cobalt_percent_prediction,
        'molybdenum_percent_Prediction': molybdenum_percent_prediction,
        'tungsten_percent_Prediction': tungsten_percent_prediction,
        'niobium_percent_Prediction': niobium_percent_prediction,
        'phosphorous_percent_Prediction': phosphorous_percent_prediction,
        'copper_percent_Prediction': copper_percent_prediction,
        'titanium_percent_Prediction': titanium_percent_prediction,
        'tantalum_percent_Prediction': tantalum_percent_prediction,
        'hafnium_percent_Prediction': hafnium_percent_prediction,
        'rhenium_percent_Prediction': rhenium_percent_prediction,
        'vanadium_percent_Prediction': vanadium_percent_prediction,
        'boron_percent_Prediction': boron_percent_prediction,
        'nitrogen_percent_Prediction': nitrogen_percent_prediction,
        'oxygen_percent_Prediction': oxygen_percent_prediction,
        'sulphur_percent_Prediction': sulphur_percent_prediction,
        'carbon_percentScore': carbon_percent_score,
        'chromium_percentScore': chromium_percent_score,
        'aluminium_percentScore': aluminium_percent_score,
        'manganese_percentScore': manganese_percent_score,
        # Add accuracy if needed
    }

    return jsonify(response_data)
