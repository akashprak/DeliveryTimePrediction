from flask import Flask, request, render_template
from src.pipeline.prediction_pipeline import PredictPipeline, CustomData

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method=='GET':
        return render_template('form.html')
    else:
        data = CustomData(request.form)
        data_df = data.as_dataframe()
        predictor_obj = PredictPipeline()
        pred, distance = predictor_obj.predict(data_df)
        
        pred=int(pred)
        distance=round(distance,1)
        return render_template('result.html', results= pred, distance=distance)
    
if __name__=='__main__':
    app.run(host='0.0.0.0', port=7000)
