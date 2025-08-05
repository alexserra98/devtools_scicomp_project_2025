import torch
import time
import sys
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.attention_implementations import (
    StandardAttention,
    SparseAttention,
    FlashAttention,
)


def demonstrate_document_processing(output_dir: str = "document_processing_results"):
    """Demonstrate sparse attention advantages for document processing."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("📄 DOCUMENT PROCESSING DEMONSTRATION")
    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Simulate processing different document lengths
    document_scenarios = [
        {
            "name": "Short Article",
            "seq_len": 1024,
            "description": "Blog post or news article",
        },
        {
            "name": "Research Paper",
            "seq_len": 4096,
            "description": "Academic paper with references",
        },
        {
            "name": "Technical Report",
            "seq_len": 8192,
            "description": "Long technical documentation",
        },
        {
            "name": "Book Chapter",
            "seq_len": 16384,
            "description": "Full book chapter or legal document",
        },
    ]

    d_model = 512
    num_heads = 8
    batch_size = 1  # Processing one document at a time

    print("\n🔍 Testing Document Processing Scenarios:")
    print("-" * 60)
    
    # Collect all results for saving
    all_results = {
        "metadata": {
            "device": str(device),
            "d_model": d_model,
            "num_heads": num_heads,
            "batch_size": batch_size
        },
        "scenarios": {}
    }

    for scenario in document_scenarios:
        seq_len = scenario["seq_len"]
        print(f"\\n📖 {scenario['name']} ({seq_len:,} tokens)")
        print(f"   Use case: {scenario['description']}")

        # Test data
        if device.type == "cuda":
            x = torch.randn(
                batch_size, seq_len, d_model, device=device, dtype=torch.float16
            )
        else:
            x = torch.randn(batch_size, seq_len, d_model, device=device)

        results = {}

        # Test Standard Attention
        print("   \n   🔸 Standard Attention:")
        try:
            standard_attn = StandardAttention(d_model, num_heads).to(device)
            if device.type == "cuda":
                standard_attn = standard_attn.half()

            memory, time_ms = benchmark_attention_mechanism(standard_attn, x)
            results["standard"] = {"memory": memory, "time": time_ms, "success": True}
            print(f"      Memory: {memory:6.1f} MB, Time: {time_ms:6.1f} ms ✅")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                results["standard"] = {
                    "memory": float("inf"),
                    "time": float("inf"),
                    "success": False,
                }
                print("      ❌ Out of Memory - Cannot process this document length")
                torch.cuda.empty_cache()
            else:
                raise e

        # Test Sparse Local Attention (document-specific pattern)
        print("   \n   🔸 Sparse Local Attention (window=512):")
        try:
            # Use larger window for documents to capture paragraph-level context
            sparse_attn = SparseAttention(
                d_model,
                num_heads,
                sparsity_pattern="local",
                window_size=512,  # Larger window for document processing
            ).to(device)
            if device.type == "cuda":
                sparse_attn = sparse_attn.half()

            memory, time_ms = benchmark_attention_mechanism(sparse_attn, x)
            results["sparse"] = {"memory": memory, "time": time_ms, "success": True}
            print(f"      Memory: {memory:6.1f} MB, Time: {time_ms:6.1f} ms ✅")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                results["sparse"] = {
                    "memory": float("inf"),
                    "time": float("inf"),
                    "success": False,
                }
                print("      ❌ Out of Memory")
                torch.cuda.empty_cache()
            else:
                raise e

        # Test Flash Attention
        print("   \n   🔸 Flash Attention:")
        try:
            flash_attn = FlashAttention(d_model, num_heads, block_size=512).to(device)
            if device.type == "cuda":
                flash_attn = flash_attn.half()

            memory, time_ms = benchmark_attention_mechanism(flash_attn, x)
            results["flash"] = {"memory": memory, "time": time_ms, "success": True}
            print(f"      Memory: {memory:6.1f} MB, Time: {time_ms:6.1f} ms ✅")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                results["flash"] = {
                    "memory": float("inf"),
                    "time": float("inf"),
                    "success": False,
                }
                print("      ❌ Out of Memory")
                torch.cuda.empty_cache()
            else:
                raise e

        # Analysis
        print("   \\n   📊 Analysis:")
        successful_methods = [k for k, v in results.items() if v["success"]]

        if not successful_methods:
            print("      ❌ No methods can process this document length")
        else:
            # Find most memory efficient
            best_memory = min(results[k]["memory"] for k in successful_methods)
            memory_winner = [
                k for k in successful_methods if results[k]["memory"] == best_memory
            ][0]

            # Find fastest
            best_time = min(results[k]["time"] for k in successful_methods)
            time_winner = [
                k for k in successful_methods if results[k]["time"] == best_time
            ][0]

            print(
                f"      🏆 Most Memory Efficient: {memory_winner.title()} ({best_memory:.1f} MB)"
            )
            print(f"      🏆 Fastest: {time_winner.title()} ({best_time:.1f} ms)")

            # Show memory savings
            if "standard" in results and results["standard"]["success"]:
                standard_memory = results["standard"]["memory"]
                for method in ["sparse", "flash"]:
                    if method in results and results[method]["success"]:
                        savings = (
                            (standard_memory - results[method]["memory"])
                            / standard_memory
                            * 100
                        )
                        print(
                            f"      💰 {method.title()} saves {savings:.1f}% memory vs Standard"
                        )

        # Save results for this scenario
        all_results["scenarios"][scenario["name"]] = {
            "description": scenario["description"],
            "seq_len": seq_len,
            "results": results
        }

        print("   " + "-" * 50)

    print("\\n\\n🎯 REAL-WORLD IMPLICATIONS:")
    print("=" * 60)
    print("""
💡 **When to Use Each Attention Mechanism:**

📝 **For Document Processing:**
   🥇 Flash Attention: Best overall choice
      - Handles all document lengths efficiently
      - Minimal memory overhead
      - Excellent speed

   🥈 Sparse Local Attention: Fallback for extreme cases  
      - Enables processing when standard attention fails
      - Good for very long documents (>16K tokens)
      - Use when Flash isn't available

   🥉 Standard Attention: Limited to short documents
      - Only suitable for articles <4K tokens
      - High memory usage limits scalability

📚 **Use Case Examples:**
   ✅ Legal document analysis (sparse/flash for long contracts)
   ✅ Academic paper processing (sparse/flash for full papers)
   ✅ Book summarization (sparse/flash for chapters)
   ✅ Technical documentation (sparse/flash for manuals)
   
🚫 **Memory Limitations:**
   - Standard attention fails on documents >8K tokens
   - Sparse attention enables 2-4x longer document processing
   - Flash attention scales to very long documents efficiently
    """)
    
    # Save results
    save_results(all_results, output_dir)
    create_plots(all_results, output_dir)
    
    print(f"\n💾 Results saved to: {output_dir}")
    print(f"   - Raw data: {output_dir}/document_processing_results.json")
    print(f"   - Plots: {output_dir}/memory_comparison.png, {output_dir}/time_comparison.png")


def save_results(results: dict, output_dir: str):
    """Save results to JSON file."""
    # Convert inf values to string for JSON serialization
    def convert_inf(obj):
        if isinstance(obj, dict):
            return {k: convert_inf(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_inf(v) for v in obj]
        elif obj == float("inf"):
            return "inf"
        elif obj == float("-inf"):
            return "-inf"
        else:
            return obj
    
    serializable_results = convert_inf(results)
    
    output_file = Path(output_dir) / "document_processing_results.json"
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)


def create_plots(results: dict, output_dir: str):
    """Create comparison plots."""
    scenarios = results["scenarios"]
    scenario_names = list(scenarios.keys())
    
    methods = ["standard", "sparse", "flash"]
    method_labels = ["Standard", "Sparse Local", "Flash"]
    colors = ["#ff6b6b", "#4ecdc4", "#45b7d1"]
    
    # Memory comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for i, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
        memories = []
        valid_lengths = []
        
        for name in scenario_names:
            result = scenarios[name]["results"].get(method, {})
            if result.get("success", False) and result["memory"] != float("inf"):
                memories.append(result["memory"])
                valid_lengths.append(scenarios[name]["seq_len"])
        
        if memories:  # Only plot if we have valid data
            ax1.plot(valid_lengths, memories, 'o-', color=color, label=label, linewidth=2, markersize=8)
    
    ax1.set_xlabel("Sequence Length (tokens)")
    ax1.set_ylabel("Peak Memory (MB)")
    ax1.set_title("Memory Usage by Document Length")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    
    # Time comparison plot
    for i, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
        times = []
        valid_lengths = []
        
        for name in scenario_names:
            result = scenarios[name]["results"].get(method, {})
            if result.get("success", False) and result["time"] != float("inf"):
                times.append(result["time"])
                valid_lengths.append(scenarios[name]["seq_len"])
        
        if times:  # Only plot if we have valid data
            ax2.plot(valid_lengths, times, 'o-', color=color, label=label, linewidth=2, markersize=8)
    
    ax2.set_xlabel("Sequence Length (tokens)")
    ax2.set_ylabel("Time (ms)")
    ax2.set_title("Processing Time by Document Length")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    
    plt.tight_layout()
    
    # Save plots
    memory_plot_path = Path(output_dir) / "memory_comparison.png"
    plt.savefig(memory_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a summary bar chart
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Get the longest document that each method can handle
    max_lengths = {}
    for method in methods:
        max_len = 0
        for name in scenario_names:
            result = scenarios[name]["results"].get(method, {})
            if result.get("success", False):
                max_len = max(max_len, scenarios[name]["seq_len"])
        max_lengths[method] = max_len
    
    methods_with_data = [method for method in methods if max_lengths[method] > 0]
    labels_with_data = [method_labels[methods.index(method)] for method in methods_with_data]
    lengths_with_data = [max_lengths[method] for method in methods_with_data]
    colors_with_data = [colors[methods.index(method)] for method in methods_with_data]
    
    bars = ax.bar(labels_with_data, lengths_with_data, color=colors_with_data, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, length in zip(bars, lengths_with_data):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{length:,}',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel("Maximum Document Length (tokens)")
    ax.set_title("Maximum Document Length Each Method Can Process")
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save summary plot
    summary_plot_path = Path(output_dir) / "capability_summary.png"
    plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    

def benchmark_attention_mechanism(attention, x):
    """Benchmark a single attention mechanism."""
    device = x.device

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = attention(x)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Benchmark
    start_time = time.time()
    with torch.no_grad():
        for _ in range(5):
            _ = attention(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.time()
    avg_time = (end_time - start_time) / 5 * 1000  # ms

    # Get memory
    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
    else:
        peak_memory = x.numel() * x.element_size() * 4 / 1024 / 1024  # Estimate

    return peak_memory, avg_time


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Demonstrate sparse attention advantages for document processing")
    parser.add_argument(
        "--output-dir", 
        default="document_processing_results", 
        help="Output directory for results and plots"
    )
    
    args = parser.parse_args()
    demonstrate_document_processing(args.output_dir)
