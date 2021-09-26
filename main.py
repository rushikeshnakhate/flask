from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

stores = [
    {
        'name': 'My Store',
        'items': {
            'name': 'item1',
            'price': 10
        }
    }
]


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/store', methods=['POST'])
def create_store():
    request_data = request.get_json()
    new_store = {
        'name': request_data['name'],
        'items': []
    }
    stores.append(new_store)
    return jsonify(new_store)


@app.route('/store/<string:name>')
def get_store(name):
    for store in stores:
        if store['name'] == name:
            return jsonify({'item': store[name]})
    return jsonify({'message': 'store=${store} not found'})


@app.route('/store/<string:name>/item', methods=['POST'])
def create_item_in_store(name):
    request_data = request.get_json()
    for store in stores:
        if store['name'] == name:
            new_item = {
                'item': request_data['name'],
                'price': request_data['price']
            }
            store['items'].append(new_item)
            return jsonify(new_item)
    return jsonify({'message': 'store= not found'})


if __name__ == "__main__":
    app.run()
