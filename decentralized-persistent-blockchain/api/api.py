from flask import Flask, request, jsonify
from blockchain.blockchain import Blockchain
from blockchain.pow import ProofOfWork
from synchronization.sync import Synchronizer

app = Flask(__name__)
blockchain = Blockchain()
pow = ProofOfWork()
synchronizer = Synchronizer(blockchain)

@app.route('/add_node', methods=['POST'])
def add_node():
    data = request.get_json()
    node_address = data.get('node_address')

    if not node_address:
        return jsonify({"message": "Missing node_address"}), 400

    blockchain.add_node(node_address)
    return jsonify({"message": f"Node {node_address} added"}), 200

@app.route('/remove_node', methods=['POST'])
def remove_node():
    data = request.get_json()
    node_address = data.get('node_address')

    if not node_address:
        return jsonify({"message": "Missing node_address"}), 400

    blockchain.remove_node(node_address)
    return jsonify({"message": f"Node {node_address} removed"}), 200

@app.route('/mine_block', methods=['POST'])
def mine_block():
    data = request.get_json()
    block_data = data.get('data')

    if not block_data:
        return jsonify({"message": "Missing block data"}), 400

    previous_hash = blockchain.chain[-1].hash
    nonce = pow.mine(block_data, previous_hash)

    if blockchain.add_block(block_data, nonce):
        return jsonify({"message": "Block mined successfully", "nonce": nonce}), 200
    return jsonify({"message": "Failed to mine block"}), 500

@app.route('/sync', methods=['GET'])
def sync_chain():
    synchronizer.synchronize()
    return jsonify({"message": "Blockchain synchronized"}), 200

@app.route('/chain', methods=['GET'])
def get_chain():
    chain_data = [{"index": block.index, "timestamp": block.timestamp, "data": block.data, "hash": block.hash, "nonce": block.nonce} for block in blockchain.chain]
    return jsonify({"chain": chain_data}), 200