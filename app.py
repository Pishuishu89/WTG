from flask import Flask, render_template, request, jsonify
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
from userMachine import run_model  # type: ignore # This module contains your converted notebook code

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/update', methods=['POST'])
def update():
    try:
        # Retrieve user input values from the form and convert them to floats.
        pts = float(request.form.get('pts', 0))
        ast = float(request.form.get('ast', 0))
        trb = float(request.form.get('trb', 0))
        blk = float(request.form.get('blk', 0))
        
        # Package the input values as a dictionary.
        user_input_percentages = {
            'PTS': pts,
            'AST': ast,
            'TRB': trb,
            'BLK': blk
        }
        
        # Validate that the percentages sum to 100.
        if sum(user_input_percentages.values()) != 100:
            return jsonify({'error': 'The total of percentages must equal 100.'})
        
        # Run your ML model with the user inputs.
        # run_model returns a dictionary of figure objects.
        fig_dict = run_model(user_input_percentages)
        
        # In this example, we choose to send the 'top_5_mvp' figure.
        fig = fig_dict['top_5_mvp']
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        return jsonify({'img_base64': img_base64})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)