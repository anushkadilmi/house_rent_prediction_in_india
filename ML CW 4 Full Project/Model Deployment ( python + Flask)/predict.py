from flask import Flask, request, render_template
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import re

app = Flask(__name__)

# Load trained files
model = joblib.load('best_xgboost_random.joblib')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('columns.pkl')

# Initialize LabelEncoder
label_encoder = LabelEncoder()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        bhk = int(request.form['BHK'].strip())
        size = int(request.form['Size'].strip())
        area_locality = request.form['Area Locality'].strip()
        city = request.form['City'].strip()
        furnishing_status = request.form['Furnishing Status'].strip()
        tenant_preferred = request.form['Tenant Preferred'].strip()
        bathroom = int(request.form['Bathroom'].strip())
        point_of_contact = request.form['Point of Contact'].strip()
        floor_input = request.form['Floor'].strip()

       
        pattern = r'(?P<CurrentFloor>\w+)\s*out\s*of\s*(?P<TotalFloors>\d+)'
        match = re.match(pattern, floor_input)
        if match:
            current_floor = match.group('CurrentFloor').strip()
            total_floors = match.group('TotalFloors').strip()

            # Replace non-numeric floor labels
            floor_replacements = {
                'Ground': 0,
                'Basement': -1  
            }
            current_floor = floor_replacements.get(current_floor, current_floor)

          
            try:
                current_floor = pd.to_numeric(current_floor, errors='coerce')
                total_floors = pd.to_numeric(total_floors, errors='coerce')
            except ValueError:
                current_floor = None
                total_floors = None
        else:
            current_floor = None
            total_floors = None

        
        new_data = pd.DataFrame({
            'BHK': [bhk],
            'Size': [size],
            'Area Locality': [area_locality],
            'City': [city],
            'Furnishing Status': [furnishing_status],
            'Tenant Preferred': [tenant_preferred],
            'Bathroom': [bathroom],
            'Point of Contact': [point_of_contact],
            'CurrentFloor': [current_floor],
            'TotalFloors': [total_floors]
        })

       
        print("Received data for prediction:")
        print(new_data)

  
        for column in columns:
            if column in new_data.columns:
                if new_data[column].dtype == 'object':
                    new_data[column] = label_encoder.fit_transform(new_data[column].astype(str))

    
        print("Data after transformation:")
        print(new_data)

        # Predict
        try:
           
            new_data_scaled = scaler.transform(new_data)
            new_data_scaled = pd.DataFrame(new_data_scaled, columns=new_data.columns)

           
            print("Data after scaling:")
            print(new_data_scaled)

            prediction = model.predict(new_data_scaled)
            prediction_text = f'Predicted Rent: {prediction[0]}'
        except ValueError as e:
            prediction_text = f'Error: {e}'

        return render_template('index.html', prediction=prediction_text)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
