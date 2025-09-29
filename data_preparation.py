import pandas as pd
import json
import os
from pathlib import Path

# Sample data about different topics
sample_data = [
    {
        "id": "1",
        "text": "The theory of relativity was developed by Albert Einstein in 1905. It consists of special and general relativity, fundamentally changing our understanding of space and time.",
        "source": "physics",
        "metadata": {"author": "Science History", "year": 1905}
    },
    {
        "id": "2",
        "text": "Quantum mechanics is a fundamental theory in physics that describes the behavior of matter and energy at atomic and subatomic scales.",
        "source": "physics",
        "metadata": {"author": "Quantum Physics Journal", "year": 1920}
    },
    {
        "id": "3",
        "text": "The Renaissance was a period in European history marking the transition from the Middle Ages to modernity, spanning the 14th to the 17th century.",
        "source": "history",
        "metadata": {"author": "Historical Studies", "year": 1400}
    },
    {
        "id": "4",
        "text": "Machine learning is a field of artificial intelligence that uses statistical techniques to give computer systems the ability to learn from data.",
        "source": "computer_science",
        "metadata": {"author": "AI Research", "year": 1959}
    },
    {
        "id": "5",
        "text": "The human brain contains approximately 86 billion neurons, each connected to thousands of other neurons, forming an incredibly complex network.",
        "source": "neuroscience",
        "metadata": {"author": "Neuroscience Today", "year": 2012}
    }
]

def prepare_data():
    """Prepare and save sample data as JSON."""
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Save as JSON
    output_path = data_dir / "sample_data.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"Sample data saved to {output_path}")
    return str(output_path)

if __name__ == "__main__":
    prepare_data()
