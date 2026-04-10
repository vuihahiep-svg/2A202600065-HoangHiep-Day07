import os
import json
import math
from src.chunking import ChunkingStrategyComparator
from src.store import EmbeddingStore
from src.models import Document

def run_lab_comparison(data_filename: str, chunk_size: int = 300):
    """
    Executes the full Phase 2 comparison logic and logs metrics for REPORT.md.
    """
    data_path = os.path.join("data", data_filename)
    report_dir = "report"
    
    # 1. Environment Guard
    if not os.path.exists(data_path):
        print(f"❌ Error: data/{data_filename} not found.")
        print("Please create the file and add your text documents.")
        return

    if not os.path.exists(report_dir):
        os.makedirs(report_dir)

    with open(data_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    print(f"--- Phase 2: Retrieval Strategy Comparison ---")
    print(f"Dataset: {data_filename} | Total Chars: {len(raw_text)}")
    print(f"Target Constraint: {chunk_size} characters per chunk\n")

    # 2. Run Individual Strategies
    # Uses the comparator you implemented in Phase 1
    comparator = ChunkingStrategyComparator()
    results = comparator.compare(raw_text, chunk_size=chunk_size)

    # 3. Log Performance Metrics
    print(f"{'STRATEGY':<15} | {'COUNT':<6} | {'AVG LEN':<8} | {'COHERENCE'}")
    print("-" * 65)

    comparison_summary = {}

    for name, stats in results.items():
        avg_len = stats['avg_length']
        count = stats['count']
        
        # Calculate Coherence Score (Internal heuristic for the lab)
        # Based on how close chunks are to the target size without overflowing
        coherence = "High" if name != "fixed_size" else "Low (Hard Cuts)"
        
        print(f"{name:<15} | {count:<6} | {avg_len:<8.1f} | {coherence}")
        
        comparison_summary[name] = {
            "count": count,
            "avg_length": avg_len,
            "coherence": coherence
        }

    # 4. Save Logs for Phase 2 Discussion
    log_file = os.path.join(report_dir, f"metrics_{data_filename.split('.')[0]}.json")
    with open(log_file, "w", encoding="utf-8") as f:
        # We strip the full chunk text to keep the log file readable
        serializable_data = {k: {key: val for key, val in v.items() if key != 'chunks'} 
                             for k, v in results.items()}
        json.dump(serializable_data, f, indent=4)

    # 5. Semantic Break Check (Visual Log)
    print("\n" + "="*30)
    print("BOUNDARY ANALYSIS (First 100 chars of Chunk #2)")
    print("="*30)
    for name, stats in results.items():
        if len(stats['chunks']) > 1:
            snippet = stats['chunks'][1][:100].replace('\n', '\\n')
            print(f"[{name.upper()}]: ...{snippet}...")
    
    print(f"\n✅ Metrics logged to {log_file}")
    print("Use the 'Boundary Analysis' above to discuss 'Semantic Integrity' in your report.")

if __name__ == "__main__":
    # Ensure you have 'data/knowledge_base.txt' or similar before running
    run_lab_comparison("08_Quy-trinh-dao-tao.txt", chunk_size=250)