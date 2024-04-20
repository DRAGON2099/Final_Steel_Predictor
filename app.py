import pandas as pd
from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

app = Flask(__name__)

# Read data from the Excel file
excel_data = pd.read_excel('your_excel_file.xlsx')

@app.route('/')
def index():
    return render_template('codes.html')

@app.route('/search', methods=['POST'])
def search():
    # Get search parameters from the form
    search_type = request.form.get('search_type')
    value_to_search = request.form.get('value_to_search')

    # Define the column name based on the selected search type
    column_name = get_column_name(search_type)

    if column_name:
        # Create a condition for the selected column and search value
        condition = (excel_data[column_name] == float(value_to_search))

        # Add optional filters to the condition
        optional_filters = ["reduction", "uts", "elongation", "yield"]
        for filter_type in optional_filters:
            optional_filter_value = request.form.get('optional_filter_' + filter_type)
            if optional_filter_value:
                optional_column_name = get_column_name(filter_type)
                if optional_column_name:
                    condition &= (excel_data[optional_column_name] == float(optional_filter_value))

        # Filter data based on the condition
        filtered_data = excel_data[condition]

        # Convert filtered data to HTML table
        table_html = filtered_data.to_html(classes='table', index=False) if not filtered_data.empty else "<p>No matching records found.</p>"

        return render_template('codes.html', table_html=table_html)

    return render_template('codes.html', error_message="Invalid search type.")

def get_column_name(search_type):
    # Map search type to the corresponding column name
    column_mapping = {
        'uts': 'Ultimate Tensile Stress, [MPa]',
        'elongation': 'Tensile elongation [%]',
        'reduction': 'Reduction area [%]',
        'yield': 'Yield Stress, [MPa]'
    }
    return column_mapping.get(search_type)

# Load the CSV data into a DataFrame
df = pd.read_csv('your_training_data.csv')

# Data Preprocessing
# Assuming the columns 'carbon_percent', 'chromium_percent', 'aluminium_percent', etc., are features, and 'uts', 'elongation', 'yield_strength', 'reduction_area' are target variables
X = df[['carbon_percent', 'chromium_percent', 'aluminium_percent', 'manganese_percent','silicon_percent','nickel_percent','cobalt_percent','molybdenum_percent','tungsten_percent','niobium_percent','phosphorous_percent','copper_percent','titanium_percent','tantalum_percent','hafnium_percent','rhenium_percent','vanadium_percent','boron_percent','nitrogen_percent','oxygen_percent','sulphur_percent']]  # Features
y_uts = df['uts']  # Target variable for UTS prediction
y_elongation = df['elongation']  # Target variable for Elongation prediction
y_yield_strength = df['yield_strength']  # Target variable for Yield Strength prediction
y_reduction_area = df['reduction_area']  # Target variable for Reduction in Area prediction

# Splitting the data into training and testing sets
X_train, X_test, y_uts_train, y_uts_test = train_test_split(X, y_uts, test_size=0.2, random_state=42)
X_train, X_test, y_elongation_train, y_elongation_test = train_test_split(X, y_elongation, test_size=0.2, random_state=42)
X_train, X_test, y_yield_strength_train, y_yield_strength_test = train_test_split(X, y_yield_strength, test_size=0.2, random_state=42)
X_train, X_test, y_reduction_area_train, y_reduction_area_test = train_test_split(X, y_reduction_area, test_size=0.2, random_state=42)

# Model Training
model_uts = LinearRegression()
model_uts.fit(X_train, y_uts_train)

model_elongation = LinearRegression()
model_elongation.fit(X_train, y_elongation_train)

model_yield_strength = LinearRegression()
model_yield_strength.fit(X_train, y_yield_strength_train)

model_reduction_area = LinearRegression()
model_reduction_area.fit(X_train, y_reduction_area_train)

# Model Evaluation (Optional)
uts_score = model_uts.score(X_test, y_uts_test)
elongation_score = model_elongation.score(X_test, y_elongation_test)
yield_strength_score = model_yield_strength.score(X_test, y_yield_strength_test)
reduction_area_score = model_reduction_area.score(X_test, y_reduction_area_test)

# Save Trained Models
joblib.dump(model_uts, 'model_uts.pkl')
joblib.dump(model_elongation, 'model_elongation.pkl')
joblib.dump(model_yield_strength, 'model_yield_strength.pkl')
joblib.dump(model_reduction_area, 'model_reduction_area.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json

    # Extract input features
    carbon_percent = input_data['carbonPercent']
    chromium_percent = input_data['chromiumPercent']
    aluminium_percent = input_data['aluminiumPercent']
    manganese_percent = input_data['manganesePercent']
    silicon_percent = input_data['siliconPercent']
    nickel_percent = input_data['nickelPercent']
    cobalt_percent = input_data['cobaltPercent']
    molybdenum_percent = input_data['molybdenumPercent']
    tungsten_percent = input_data['tungstenPercent']
    niobium_percent = input_data['niobiumPercent']
    phosphorous_percent = input_data['phosphorousPercent']
    copper_percent = input_data['copperPercent']
    titanium_percent = input_data['titaniumPercent']
    tantalum_percent = input_data['tantalumPercent']
    hafnium_percent = input_data['hafniumPercent']
    rhenium_percent = input_data['rheniumPercent']
    vanadium_percent = input_data['vanadiumPercent']
    boron_percent = input_data['boronPercent']
    nitrogen_percent = input_data['nitrogenPercent']
    oxygen_percent = input_data['oxygenPercent']
    sulphur_percent = input_data['sulphurPercent']
    # Extract more input features as needed

    # Load the trained models
    model_uts = joblib.load('model_uts.pkl')
    model_elongation = joblib.load('model_elongation.pkl')
    model_yield_strength = joblib.load('model_yield_strength.pkl')
    model_reduction_area = joblib.load('model_reduction_area.pkl')

    # Make predictions using the loaded models
    uts_prediction = model_uts.predict([[carbon_percent, chromium_percent, aluminium_percent,manganese_percent,silicon_percent,nickel_percent,cobalt_percent,molybdenum_percent,tungsten_percent,niobium_percent,phosphorous_percent,copper_percent,titanium_percent,tantalum_percent,hafnium_percent,rhenium_percent,vanadium_percent,boron_percent,nitrogen_percent,oxygen_percent,sulphur_percent]])[0]
    elongation_prediction = model_elongation.predict([[carbon_percent, chromium_percent, aluminium_percent,manganese_percent,silicon_percent,nickel_percent,cobalt_percent,molybdenum_percent,tungsten_percent,niobium_percent,phosphorous_percent,copper_percent,titanium_percent,tantalum_percent,hafnium_percent,rhenium_percent,vanadium_percent,boron_percent,nitrogen_percent,oxygen_percent,sulphur_percent]])[0]
    yield_strength_prediction = model_yield_strength.predict([[carbon_percent, chromium_percent, aluminium_percent,manganese_percent,silicon_percent,nickel_percent,cobalt_percent,molybdenum_percent,tungsten_percent,niobium_percent,phosphorous_percent,copper_percent,titanium_percent,tantalum_percent,hafnium_percent,rhenium_percent,vanadium_percent,boron_percent,nitrogen_percent,oxygen_percent,sulphur_percent]])[0]
    reduction_area_prediction = model_reduction_area.predict([[carbon_percent, chromium_percent, aluminium_percent,manganese_percent,silicon_percent,nickel_percent,cobalt_percent,molybdenum_percent,tungsten_percent,niobium_percent,phosphorous_percent,copper_percent,titanium_percent,tantalum_percent,hafnium_percent,rhenium_percent,vanadium_percent,boron_percent,nitrogen_percent,oxygen_percent,sulphur_percent]])[0]

    # Calculate accuracy (R2 score) - Optional
    # accuracy = ...

    # Prepare response data
    response_data = {
        'utsPrediction': uts_prediction,
        'elongationPrediction': elongation_prediction,
        'yieldStrengthPrediction': yield_strength_prediction,
        'reductionAreaPrediction': reduction_area_prediction,
        'utsScore': uts_score,
        'elongationScore': elongation_score,
        'yieldStrengthScore': yield_strength_score,
        'reductionAreaScore': reduction_area_score,
        # Add accuracy if needed
    }

    return jsonify(response_data)

#Load the CSV data into a DataFrame
df = pd.read_csv('your_training_data_conc.csv')

# Data Preprocessing
# Assuming the columns 'carbon_percent', 'chromium_percent', 'aluminium_percent', etc., are TARGET VARIABLES, and 'uts', 'elongation', 'yield_strength', 'reduction_area' are FEATURES
X = df[['yield_strength','uts','elongation','reduction_area']]  # Features
y_carbon_percent = df['carbon_percent']  # Target variable for carbon percent prediction
y_chromium_percent = df['chromium_percent']  # Target variable for chromium percent prediction
y_manganese_percent = df['manganese_percent']  # Target variable for manganese percent prediction
y_silicon_percent = df['silicon_percent']  # Target variable for silicon percent prediction
y_nickel_percent = df['nickel_percent']  # Target variable for nickel percent prediction
y_cobalt_percent = df['cobalt_percent']  # Target variable for cobalt percent prediction
y_molybdenum_percent = df['molybdenum_percent']  # Target variable for molybdenum percent prediction
y_tungsten_percent = df['tungsten_percent']  # Target variable for tungsten percent prediction
y_niobium_percent = df['niobium_percent']  # Target variable for niobium percent prediction
y_aluminium_percent = df['aluminium_percent']  # Target variable for aluminium percent prediction
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
X_train, X_test, y_silicon_percent_train, y_silicon_percent_test = train_test_split(X, y_silicon_percent, test_size=0.2, random_state=42)
X_train, X_test, y_nickel_percent_train, y_nickel_percent_test = train_test_split(X, y_nickel_percent, test_size=0.2, random_state=42)
X_train, X_test, y_cobalt_percent_train, y_cobalt_percent_test = train_test_split(X, y_cobalt_percent, test_size=0.2, random_state=42)
X_train, X_test, y_molybdenum_percent_train, y_molybdenum_percent_test = train_test_split(X, y_molybdenum_percent, test_size=0.2, random_state=42)
X_train, X_test, y_tungsten_percent_train, y_tungsten_percent_test = train_test_split(X, y_tungsten_percent, test_size=0.2, random_state=42)
X_train, X_test, y_niobium_percent_train, y_niobium_percent_test = train_test_split(X, y_niobium_percent, test_size=0.2, random_state=42)
X_train, X_test, y_aluminium_percent_train, y_aluminium_percent_test = train_test_split(X, y_aluminium_percent, test_size=0.2, random_state=42)
X_train, X_test, y_manganese_percent_train, y_manganese_percent_test = train_test_split(X, y_manganese_percent, test_size=0.2, random_state=42)
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
model_oxygen_percent.fit(X_train, y_oxygen_percent_train)

model_sulphur_percent = LinearRegression()
model_sulphur_percent.fit(X_train, y_sulphur_percent_train)

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


@app.route('/steel_search', methods=['POST'])
def predict_comp():
    input_data = request.json

    # Extract input features
    yield_strength = input_data.get('yield_strength')
    uts = input_data.get('uts')
    elongation = input_data.get('elongation')
    reduction_area = input_data.get('reduction_area')

    # Check if all required fields are present
    if None in [yield_strength, uts, elongation, reduction_area]:
        return jsonify({'error': 'Incomplete data'})

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
    carbon_percent_prediction = model_carbon_percent.predict([[yield_strength,uts, elongation,reduction_area]])[0]
    chromium_percent_prediction = model_chromium_percent.predict([[yield_strength,uts, elongation,reduction_area]])[0]
    aluminium_percent_prediction = model_aluminium_percent.predict([[yield_strength,uts, elongation,reduction_area]])[0]
    manganese_percent_prediction = model_manganese_percent.predict([[yield_strength,uts, elongation,reduction_area]])[0]
    silicon_percent_prediction = model_silicon_percent.predict([[yield_strength,uts, elongation,reduction_area]])[0]
    nickel_percent_prediction = model_nickel_percent.predict([[yield_strength,uts, elongation,reduction_area]])[0]
    cobalt_percent_prediction = model_cobalt_percent.predict([[yield_strength,uts, elongation,reduction_area]])[0]
    molybdenum_percent_prediction = model_molybdenum_percent.predict([[yield_strength,uts, elongation,reduction_area]])[0]
    tungsten_percent_prediction = model_tungsten_percent.predict([[yield_strength,uts, elongation,reduction_area]])[0]
    niobium_percent_prediction = model_niobium_percent.predict([[yield_strength,uts, elongation,reduction_area]])[0]
    phosphorous_percent_prediction = model_phosphorous_percent.predict([[yield_strength,uts, elongation,reduction_area]])[0]
    copper_percent_prediction = model_copper_percent.predict([[yield_strength,uts, elongation,reduction_area]])[0]
    titanium_percent_prediction = model_titanium_percent.predict([[yield_strength,uts, elongation,reduction_area]])[0]
    tantalum_percent_prediction = model_tantalum_percent.predict([[yield_strength,uts, elongation,reduction_area]])[0]
    hafnium_percent_prediction = model_hafnium_percent.predict([[yield_strength,uts, elongation,reduction_area]])[0]
    rhenium_percent_prediction = model_rhenium_percent.predict([[yield_strength,uts, elongation,reduction_area]])[0]
    vanadium_percent_prediction = model_vanadium_percent.predict([[yield_strength,uts, elongation,reduction_area]])[0]
    boron_percent_prediction = model_boron_percent.predict([[yield_strength,uts, elongation,reduction_area]])[0]
    nitrogen_percent_prediction = model_nitrogen_percent.predict([[yield_strength,uts, elongation,reduction_area]])[0]
    oxygen_percent_prediction = model_oxygen_percent.predict([[yield_strength,uts, elongation,reduction_area]])[0]
    sulphur_percent_prediction = model_sulphur_percent.predict([[yield_strength,uts, elongation,reduction_area]])[0]


    # Calculate accuracy (R2 score) - Optional
    # accuracy = ...
    carbon_percent_score = r2_score(y_carbon_percent_test, model_carbon_percent.predict(X_test))
    chromium_percent_score = r2_score(y_chromium_percent_test, model_chromium_percent.predict(X_test))
    aluminium_percent_score = r2_score(y_aluminium_percent_test, model_aluminium_percent.predict(X_test))
    manganese_percent_score = r2_score(y_manganese_percent_test, model_manganese_percent.predict(X_test))
    silicon_percent_score = r2_score(y_silicon_percent_test, model_silicon_percent.predict(X_test))
    nickel_percent_score = r2_score(y_nickel_percent_test, model_nickel_percent.predict(X_test))
    cobalt_percent_score = r2_score(y_cobalt_percent_test, model_cobalt_percent.predict(X_test))
    molybdenum_percent_score = r2_score(y_molybdenum_percent_test, model_molybdenum_percent.predict(X_test))
    tungsten_percent_score = r2_score(y_tungsten_percent_test, model_tungsten_percent.predict(X_test))
    niobium_percent_score = r2_score(y_niobium_percent_test, model_niobium_percent.predict(X_test))
    phosphorous_percent_score = r2_score(y_phosphorous_percent_test, model_phosphorous_percent.predict(X_test))
    copper_percent_score = r2_score(y_copper_percent_test, model_copper_percent.predict(X_test))
    titanium_percent_score = r2_score(y_titanium_percent_test, model_titanium_percent.predict(X_test))
    tantalum_percent_score = r2_score(y_tantalum_percent_test, model_tantalum_percent.predict(X_test))
    hafnium_percent_score = r2_score(y_hafnium_percent_test, model_hafnium_percent.predict(X_test))
    rhenium_percent_score = r2_score(y_rhenium_percent_test, model_rhenium_percent.predict(X_test))
    vanadium_percent_score = r2_score(y_vanadium_percent_test, model_vanadium_percent.predict(X_test))
    boron_percent_score = r2_score(y_boron_percent_test, model_boron_percent.predict(X_test))
    nitrogen_percent_score = r2_score(y_nitrogen_percent_test, model_nitrogen_percent.predict(X_test))
    oxygen_percent_score = r2_score(y_oxygen_percent_test, model_oxygen_percent.predict(X_test))
    sulphur_percent_score = r2_score(y_sulphur_percent_test, model_sulphur_percent.predict(X_test))

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
        'copper_percentScore': copper_percent_score,
        'nickel_percentScore': nickel_percent_score,
        'silicon_percentScore': silicon_percent_score,
        'titanium_percentScore': titanium_percent_score,
        'phosphorous_percentScore': phosphorous_percent_score,
        'sulphur_percentScore': sulphur_percent_score,
        'nitrogen_percentScore': nitrogen_percent_score,
        'vanadium_percentScore': vanadium_percent_score,
        'boron_percentScore': boron_percent_score,
        'oxygen_percentScore': oxygen_percent_score,
        'rhenium_percentScore': rhenium_percent_score,
        'hafnium_percentScore': hafnium_percent_score,
        'tantalum_percentScore': tantalum_percent_score,
        'cobalt_percentScore': cobalt_percent_score,
        'molybdenum_percentScore': molybdenum_percent_score,
        'tungsten_percentScore': tungsten_percent_score,
        'niobium_percentScore': niobium_percent_score
        # Add accuracy if needed
    }

    return jsonify(response_data)


if __name__ == '__main__':
    app.run(debug=True)
