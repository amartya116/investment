# marketTransactionsledger.py

import hashlib
import json
import time
import sys
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Set, Optional
import requests
from uuid import uuid4
from threading import Lock
from collections import Counter
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from fastapi.responses import JSONResponse

# Database setup
DATABASE_URL = "postgresql://postgres:amartya@localhost:5432/stock_analysis"
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# SQLAlchemy Base
Base = declarative_base()

# Define the TradeDB model
class TradeDB(Base):
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    price = Column(Float)
    quantity = Column(Integer)
    buyer_order_id = Column(String)
    seller_order_id = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

def get_db_session():
    """Create a new database session"""
    return SessionLocal()

def save_trades_to_db(transactions: List[dict]):
    """Save trades to database with proper session management"""
    if not transactions:
        return
        
    db = get_db_session()
    try:
        for tx in transactions:
            try:
                trade = TradeDB(
                    symbol=tx["symbol"],
                    price=tx["price"],
                    quantity=tx["quantity"],
                    buyer_order_id=tx["buyer_order_id"],
                    seller_order_id=tx["seller_order_id"]
                )
                db.add(trade)
                print(f"Adding trade to DB: {tx['symbol']} - {tx['quantity']} @ {tx['price']}")
            except Exception as e:
                print(f"Failed to create trade object: {e}")
        
        db.commit()
        print(f"Successfully saved {len(transactions)} trades to database")
        
    except Exception as e:
        print(f"Failed to save trades to database: {e}")
        db.rollback()
    finally:
        db.close()
def mine_block_and_process():
    print("Mining started...")
    proof = blockchain.proof_of_work(blockchain.last_block['proof'])
    block = blockchain.new_block(proof)

    print(f"Block mined with {len(block['transactions'])} transactions")

    if block['transactions']:
        print(f"Saving {len(block['transactions'])} transactions from mined block to database")
        save_trades_to_db(block['transactions'])
    else:
        print("No transactions in mined block to save to database")

    # Force-send the new block to all peers
    broadcast_results = force_push_block_to_peers(block)

    return {
        "block": block,
        "broadcast_results": broadcast_results,
        "transactions_saved": len(block['transactions']),
        "message": f"Block mined, saved to DB, and forcefully synced to {broadcast_results['successful']} peers"
    }
def force_push_block_to_peers(block):
    results = {"successful": 0, "failed": 0, "errors": []}
    print(PEERS)
    for peer in PEERS:
        try:
            print(f"Pushing block to {peer}/blocks/receive")
            res = requests.post(f"{peer}/blocks/receive", json=block, timeout=5)
            if res.status_code == 200:
                results["successful"] += 1
            else:
                results["failed"] += 1
                results["errors"].append(f"{peer}: {res.status_code} {res.text}")
        except Exception as e:
            results["failed"] += 1
            results["errors"].append(f"{peer}: {str(e)}")
    
    return results
# Blockchain logic
class Blockchain:
    def __init__(self):
        self.chain: List[dict] = []
        self.current_transactions: List[dict] = []
        self.lock = Lock()
        # Create genesis block
        self.new_block(proof=100, previous_hash='1')

   
    def new_transaction(self, symbol, price, quantity, buyer_order_id, seller_order_id):
        self.current_transactions.append({
            'symbol': symbol,
            'price': str(price),
            'quantity': str(quantity),
            'buyer_order_id': buyer_order_id,
            'seller_order_id': seller_order_id
        })
        return self.last_block['index'] + 1

    def new_block(self, proof, previous_hash=None):
        block = {
            'index': len(self.chain) + 1,
            'timestamp': str(datetime.utcnow()),
            'transactions': self.current_transactions,  # 💡 Important
            'proof': proof,
            'previous_hash': previous_hash or self.hash(self.chain[-1]),
        }
        self.current_transactions = []  # Reset the transaction pool
        self.chain.append(block)
        return block

    def add_block(self, block: dict) -> bool:
        """Add a received block to the chain if valid"""
        with self.lock:
            # Validate the block
            if not self.is_valid_block(block):
                print(f"Block {block.get('index')} failed validation and was not added.")
                return False
                
            # Add the block
            self.chain.append(block)
            
            # Remove any pending transactions that are now in the block
            block_transactions = block.get('transactions', [])
            for block_tx in block_transactions:
                # Remove matching transactions from pending
                self.current_transactions = [
                    tx for tx in self.current_transactions 
                    if not self.transactions_match(tx, block_tx)
                ]
            
            return True

    def is_valid_block(self, block: dict) -> bool:
        last_block = self.last_block

        # 1. Index check
        expected_index = last_block['index'] + 1
        if block['index'] != expected_index:
            print(f"Invalid index: expected {expected_index}, got {block['index']}")
            return False

        # 2. Previous hash check
        expected_prev = self.hash(last_block)
        if block['previous_hash'] != expected_prev:
            print(f"Invalid previous_hash: expected {expected_prev}, got {block['previous_hash']}")
            return False

        # 3. Proof of work check
        if not self.valid_proof(last_block['proof'], block['proof']):
            print(f"Invalid proof: block proof {block['proof']} does not satisfy proof-of-work for last proof {last_block['proof']}")
            return False

        return True

    def transactions_match(self, tx1: dict, tx2: dict) -> bool:
        """Check if two transactions are the same"""
        return (tx1.get('symbol') == tx2.get('symbol') and
                tx1.get('price') == tx2.get('price') and
                tx1.get('quantity') == tx2.get('quantity') and
                tx1.get('buyer_order_id') == tx2.get('buyer_order_id') and
                tx1.get('seller_order_id') == tx2.get('seller_order_id'))

    @staticmethod
    def hash(block: dict) -> str:
        block_string = json.dumps(block, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

    @property
    def last_block(self) -> dict:
        return self.chain[-1]

    def valid_proof(self, last_proof: int, proof: int) -> bool:
        guess = f"{last_proof}{proof}".encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:4] == "0000"

    def proof_of_work(self, last_proof: int) -> int:
        proof = 0
        while not self.valid_proof(last_proof, proof):
            proof += 1
        return proof

    def valid_chain(self, chain: List[dict]) -> bool:
        for i in range(1, len(chain)):
            prev = chain[i-1]
            curr = chain[i]
            if curr['previous_hash'] != self.hash(prev):
                return False
            if not self.valid_proof(prev['proof'], curr['proof']):
                return False
        return True

# FastAPI setup
app = FastAPI()
blockchain = Blockchain()
PEERS: Set[str] = set()
NODE_ID = str(uuid4()).replace('-', '')

# Pydantic model for transactions
class Transaction(BaseModel):
    symbol: str
    price: float
    quantity: int
    buyer_order_id: str
    seller_order_id: str

# Pydantic model for blocks
class Block(BaseModel):
    index: int
    timestamp: float
    transactions: List[dict]
    proof: int
    previous_hash: str

# Pydantic model for peer registration
class PeerNode(BaseModel):
    address: str

def broadcast_block_to_peers(block: dict):
    """Broadcast a newly mined block to all peers"""
    print(f"Starting broadcast to {len(PEERS)} peers: {list(PEERS)}")
    
    if not PEERS:
        print("No peers to broadcast to")
        return {"successful": 0, "failed": 0}
    
    successful_broadcasts = 0
    failed_broadcasts = 0
    print(PEERS)
    for peer in PEERS:
        try:
            print(f"Attempting to broadcast to peer: {peer}")
            response = requests.post(
                f"{peer}/blocks/receive",
                json=block,
                timeout=5
            )
            if response.status_code == 200:
                successful_broadcasts += 1
                print(f"✓ Successfully broadcast block to {peer}")
            else:
                failed_broadcasts += 1
                print(f"✗ Failed to broadcast block to {peer}: HTTP {response.status_code}")
                print(f"Response: {response.text}")
        except requests.exceptions.ConnectionError as e:
            failed_broadcasts += 1
            print(f"✗ Connection error broadcasting to {peer}: {str(e)}")
        except requests.exceptions.Timeout as e:
            failed_broadcasts += 1
            print(f"✗ Timeout error broadcasting to {peer}: {str(e)}")
        except Exception as e:
            failed_broadcasts += 1
            print(f"✗ Unknown error broadcasting to {peer}: {str(e)}")
    
    print(f"Broadcast complete: {successful_broadcasts} successful, {failed_broadcasts} failed")
    return {"successful": successful_broadcasts, "failed": failed_broadcasts}

@app.post("/transactions/new")
def add_transaction(tx: Transaction):
    idx = blockchain.new_transaction(
        tx.symbol, tx.price, tx.quantity, 
        tx.buyer_order_id, tx.seller_order_id
    )
    return {"message": f"Transaction will be added to block {idx}"}

@app.get("/mine")
def mine():
    return mine_block_and_process()

@app.post("/blocks/receive")
def receive_block(block: dict):
    """
    Force-replace or append the incoming block:
      - If a block at this index already exists, overwrite it.
      - Otherwise, append it.
    """
    idx = block.get("index", len(blockchain.chain) + 1)
    # 1) Ensure the chain list is big enough
    if idx <= len(blockchain.chain):
        # overwrite existing
        blockchain.chain[idx - 1] = block
    else:
        # append new
        blockchain.chain.append(block)

    # 2) Persist its transactions
    if block.get("transactions"):
        save_trades_to_db(block["transactions"])

    return {"message": f"Block {idx} replaced/appended successfully"}


@app.get("/chain")
def full_chain():
    response = {
        'chain': blockchain.chain,
        'length': len(blockchain.chain),
    }
    return response

@app.post("/nodes/register")
def register_peer(peer: PeerNode):
    """Register a new peer node"""
    if peer.address in PEERS:
        return {"message": f"Peer {peer.address} already registered", "total_peers": len(PEERS)}
    
    PEERS.add(peer.address)
    print(f"Registered new peer: {peer.address}")
    return {"message": f"Peer {peer.address} registered successfully", "total_peers": len(PEERS)}

@app.delete("/nodes/register/{peer_address}")
def unregister_peer(peer_address: str):
    """Unregister a peer node"""
    # URL decode the peer address
    import urllib.parse
    decoded_address = urllib.parse.unquote(peer_address)
    
    if decoded_address in PEERS:
        PEERS.remove(decoded_address)
        print(f"Unregistered peer: {decoded_address}")
        return {"message": f"Peer {decoded_address} unregistered successfully", "total_peers": len(PEERS)}
    else:
        return {"message": f"Peer {decoded_address} not found", "total_peers": len(PEERS)}

@app.get("/nodes/register")
def get_peers():
    return {"nodes": list(PEERS), "total_peers": len(PEERS)}

@app.get("/nodes/status")
def get_node_status():
    """Get detailed node status including peers and blockchain info"""
    return {
        "node_id": NODE_ID,
        "peers": list(PEERS),
        "total_peers": len(PEERS),
        "blockchain_length": len(blockchain.chain),
        "pending_transactions": len(blockchain.current_transactions),
        "last_block_index": blockchain.last_block['index']
    }

@app.get("/nodes/synchronize")
def synchronize():
    def clean_chain(chain):
        """Remove timestamp and hashes for comparison purposes"""
        return json.dumps([
            {k: v for k, v in block.items() if k not in ["timestamp", "previous_hash"]} for block in chain
        ], sort_keys=True)

    # Start with our local chain
    local_chain_str = clean_chain(blockchain.chain)
    chain_votes = {local_chain_str: {"count": 1, "chain": blockchain.chain}}
    
    # Total nodes in network = this node + all peers
    total_nodes = len(PEERS)
    
    # Get chains from all peers
    for peer in PEERS:
        try:
            response = requests.get(f"{peer}/chain", timeout=5)
            if response.status_code == 200:
                peer_chain = response.json()["chain"]
                peer_chain_str = clean_chain(peer_chain)
                
                if peer_chain_str in chain_votes:
                    chain_votes[peer_chain_str]["count"] += 1
                else:
                    chain_votes[peer_chain_str] = {"count": 1, "chain": peer_chain}
        except Exception as e:
            print(f"Could not contact peer {peer}: {str(e)}")
            continue
    
    # Find the chain with the most votes
    majority_chain_str = None
    max_votes = 0
    majority_chain = None
    
    for chain_str, data in chain_votes.items():
        if data["count"] > max_votes:
            max_votes = data["count"]
            majority_chain_str = chain_str
            majority_chain = data["chain"]
    
    # Check if we have a majority (more than half of total nodes)
    required_majority = (total_nodes // 2) + 1
    
    if max_votes >= required_majority:
        # We have a majority, check if our chain needs updating
        if majority_chain_str != local_chain_str:
            # Validate the majority chain before updating
            if blockchain.valid_chain(majority_chain):
                with blockchain.lock:
                    blockchain.chain = majority_chain
                    # Clear pending transactions since we're adopting a new chain
                    blockchain.current_transactions.clear()
                
                return {
                    "updated": True, 
                    "message": f"Chain updated by majority consensus ({max_votes}/{total_nodes} nodes agree)",
                    "total_nodes": total_nodes,
                    "majority_votes": max_votes
                }
            else:
                return {
                    "updated": False, 
                    "message": "Majority chain is invalid, keeping local chain",
                    "total_nodes": total_nodes,
                    "majority_votes": max_votes
                }
        else:
            return {
                "updated": False, 
                "message": f"Chain already matches majority ({max_votes}/{total_nodes} nodes agree)",
                "total_nodes": total_nodes,
                "majority_votes": max_votes
            }
    else:
        return {
            "updated": False, 
            "message": f"No majority consensus reached ({max_votes}/{total_nodes} nodes agree, need {required_majority})",
            "total_nodes": total_nodes,
            "majority_votes": max_votes
        }


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    peers_arg = sys.argv[2] if len(sys.argv) > 2 else ''
    if peers_arg:
        PEERS.update(peers_arg.split(','))
    print(f"Node starting on port {port}, peers={PEERS}")
    uvicorn.run(app, host='127.0.0.1', port=port)