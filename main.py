from flask import Flask, request, render_template
from make_prediction import recidivism

# create a flask object
app = Flask(__name__)

# creates an association between the / page and the entry_page function (defaults to GET)
@app.route('/')
def entry_page():
    return render_template('index.html')

# creates an association between the /predict_recipe page and the render_message function
# (includes POST requests which allow users to enter in data via form)
@app.route('/predict_recid/', methods=['GET', 'POST'])
def render_message():

    # user-entered ingredients
    attributes = ['priors_count', 'age']

    # error messages to ensure correct units of measure
    messages = ["The amount of priors must be an integer",
                "The age must be an integer."]

    # hold all amounts as floats
    amounts = []

    # takes user input and ensures it can be turned into a floats
    for i, ing in enumerate(attributes):
        user_input = request.form[ing]
        try:
            float_attribute = float(user_input)
        except:
            return render_template('index.html', message=messages[i])
        amounts.append(float_attribute)

    # show user final message
    final_message = recidivism(amounts)
    return render_template('index.html', message=final_message)

if __name__ == '__main__':
    app.run(debug=True)