import json
import os
from pathlib import Path

def create_sample_dataset():
    # 1. Create the raw document directory
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Document 1
    doc1 = """# QuantumFlow Innovations
    
QuantumFlow Innovations was a pioneering quantum computing startup founded in 2018 by Dr. Aris Thorne and Dr. Elena Rostova. The company specialized in room-temperature superconducting qubits. 

After several major breakthroughs in 2022, the company struggled with manufacturing scaling. In March 2024, QuantumFlow Innovations was fully acquired by NexGen Tech for $1.2 Billion. Following the acquisition, Dr. Thorne became the Chief Quantum Architect at NexGen.
"""
    (raw_dir / "quantumflow.md").write_text(doc1)

    # Document 2
    doc2 = """# NexGen Tech

NexGen Tech is a multinational technology conglomerate headquartered in Austin, Texas. Founded in 2005, it originally focused on enterprise cloud solutions but has since expanded into artificial intelligence and quantum computing.

The company is currently led by CEO Sarah Jenkins, who took over in 2021. Under her leadership, NexGen has aggressively pursued strategic acquisitions to bolster its hardware capabilities. The company's main flagship product is the 'NexCloud Enterprise Suite'.
"""
    (raw_dir / "nexgen.md").write_text(doc2)

    # 2. Create the evaluation dataset directory
    eval_dir = Path("data/eval")
    eval_dir.mkdir(parents=True, exist_ok=True)

    # 3. Create the QA pairs
    qa_pairs = [
        {
            "query": "Who is the CEO of NexGen Tech?",
            "answer": "Sarah Jenkins",
            "type": "single_hop"
        },
        {
            "query": "Who founded QuantumFlow Innovations?",
            "answer": "Dr. Aris Thorne and Dr. Elena Rostova",
            "type": "single_hop"
        },
        {
            "query": "Who is the CEO of the company that acquired QuantumFlow Innovations?",
            "answer": "Sarah Jenkins",
            "type": "multi_hop"
        },
        {
            "query": "Where is the headquarters of the company that bought Dr. Aris Thorne's startup?",
            "answer": "Austin, Texas",
            "type": "multi_hop"
        }
    ]

    with open(eval_dir / "qa_pairs.json", "w") as f:
        json.dump(qa_pairs, f, indent=2)

    print(f"Created 2 sample documents in {raw_dir}/")
    print(f"Created {len(qa_pairs)} QA pairs in {eval_dir}/qa_pairs.json")
    print("\nYou can now test the system by running:")
    print("1. curl -X POST http://localhost:8000/ingest -H 'Content-Type: application/json' -d '{\"directory\": \"data/raw/\"}'")
    print("2. curl -X POST http://localhost:8000/query -H 'Content-Type: application/json' -d '{\"question\": \"Who is the CEO of the company that acquired QuantumFlow Innovations?\"}'")

if __name__ == "__main__":
    create_sample_dataset()
