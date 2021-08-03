from flask import Flask,jsonify,request
from flask import render_template

app = Flask(__name__)
labels = []
values = []

@app.route("/")
def get_chart_page():
    global labels,values
    labels = []
    values = []
    return render_template('chart.html', values=values, labels=labels)


@app.route('/refreshData')
def refresh_graph_data():
    global labels, values
    print("labels now: " + str(labels))
    print("data now: " + str(values))
    return jsonify(sLabel=labels, sData=values)


@app.route('/updateData', methods=['POST'])
def update_data():
    global labels, values
    request_data = request.get_json()
    print(f'request data is: {request_data}')
    print(f'request data type is: {type(request_data)}')
    if not request_data or 'data' not in request_data:
        return "error", 400
    labels = request_data['data']['labels']
    values = request_data['data']['values']
    print("labels received: " + str(labels))
    print("data received: " + str(values))
    return "success", 201


if __name__ == "__main__":
    app.run(host='localhost', port=5001)