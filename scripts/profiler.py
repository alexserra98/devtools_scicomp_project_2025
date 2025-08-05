import os
import sys
import time
import json
import subprocess
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import argparse

import torch
import torch.profiler
import numpy as np
import psutil
import matplotlib.pyplot as plt
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.attention_implementations import StandardAttention, SparseAttention, FlashAttention
from src.transformer_model import TransformerLM


class GPUProfiler:
    """GPU profiling utilities using PyTorch profiler."""
    
    def __init__(self, device: str = 'auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.profiles = {}
    
    def profile_attention_mechanism(
        self,
        attention_type: str,
        batch_size: int = 4,
        seq_len: int = 512,
        d_model: int = 512,
        num_heads: int = 8,
        num_iterations: int = 100,
        **attention_kwargs
    ) -> Dict[str, Any]:
        """Profile a specific attention mechanism."""
        
        print(f"Profiling {attention_type} attention...")
        
        # Create attention layer
        if attention_type == 'standard':
            attention = StandardAttention(d_model, num_heads).to(self.device)
        elif attention_type == 'sparse':
            attention = SparseAttention(d_model, num_heads, **attention_kwargs).to(self.device)
        elif attention_type == 'flash':
            attention = FlashAttention(d_model, num_heads, **attention_kwargs).to(self.device)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        # Create input
        x = torch.randn(batch_size, seq_len, d_model, device=self.device, requires_grad=True)
        
        # Warmup
        for _ in range(10):
            output = attention(x)
            loss = output.sum()
            loss.backward()
            attention.zero_grad()
        
        # Profile with PyTorch profiler
        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.device.type == 'cuda':
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            for _ in range(num_iterations):
                output = attention(x)
                loss = output.sum()
                loss.backward()
                attention.zero_grad()
        
        # Store profile
        self.profiles[attention_type] = prof
        
        # Extract key metrics
        metrics = self._extract_metrics(prof, attention_type)
        
        return metrics
    
    def _extract_metrics(self, prof: torch.profiler.profile, attention_type: str) -> Dict[str, Any]:
        """Extract key metrics from profiler output."""
        
        # Get events table
        events = prof.key_averages()
        
        # Calculate totals
        total_cpu_time = sum(event.cpu_time_total for event in events)
        total_cuda_time = sum(event.cuda_time_total for event in events) if self.device.type == 'cuda' else 0
        total_cpu_memory = sum(event.cpu_memory_usage for event in events if event.cpu_memory_usage > 0)
        total_cuda_memory = sum(event.cuda_memory_usage for event in events if event.cuda_memory_usage > 0)
        
        # Find attention-specific operations
        attention_ops = [event for event in events if any(
            keyword in event.key.lower() for keyword in ['attention', 'matmul', 'softmax', 'linear']
        )]
        
        attention_cpu_time = sum(event.cpu_time_total for event in attention_ops)
        attention_cuda_time = sum(event.cuda_time_total for event in attention_ops) if self.device.type == 'cuda' else 0
        
        metrics = {
            'total_cpu_time_us': total_cpu_time,
            'total_cuda_time_us': total_cuda_time,
            'total_cpu_memory_bytes': total_cpu_memory,
            'total_cuda_memory_bytes': total_cuda_memory,
            'attention_cpu_time_us': attention_cpu_time,
            'attention_cuda_time_us': attention_cuda_time,
            'attention_cpu_percentage': (attention_cpu_time / total_cpu_time * 100) if total_cpu_time > 0 else 0,
            'attention_cuda_percentage': (attention_cuda_time / total_cuda_time * 100) if total_cuda_time > 0 else 0,
            'top_operations': []
        }
        
        # Get top 10 most expensive operations
        sorted_events = sorted(events, key=lambda x: x.cuda_time_total if self.device.type == 'cuda' else x.cpu_time_total, reverse=True)
        for i, event in enumerate(sorted_events[:10]):
            metrics['top_operations'].append({
                'name': event.key,
                'cpu_time_us': event.cpu_time_total,
                'cuda_time_us': event.cuda_time_total,
                'cpu_memory_bytes': event.cpu_memory_usage,
                'cuda_memory_bytes': event.cuda_memory_usage,
                'count': event.count
            })
        
        return metrics
    
    def save_profiles(self, output_dir: str):
        """Save profiler traces for detailed analysis."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for attention_type, prof in self.profiles.items():
            # Save Chrome trace for detailed visualization
            trace_file = output_path / f"{attention_type}_trace.json"
            prof.export_chrome_trace(str(trace_file))
            
            # Save summary table
            summary_file = output_path / f"{attention_type}_summary.txt"
            with open(summary_file, 'w') as f:
                f.write(f"Profile Summary for {attention_type} Attention\\n")
                f.write("="*50 + "\\n\\n")
                f.write(prof.key_averages().table(sort_by="cuda_time_total" if self.device.type == 'cuda' else "cpu_time_total"))
        
        print(f"Profiles saved to {output_dir}")


class MemoryProfiler:
    """Memory usage profiling for different attention mechanisms."""
    
    def __init__(self, device: str = 'auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
    
    def profile_memory_usage(
        self,
        seq_lengths: List[int] = [128, 256, 512, 1024, 2048],
        d_model: int = 512,
        num_heads: int = 8,
        batch_size: int = 4
    ) -> Dict[str, Dict[str, List[float]]]:
        """Profile memory usage across different sequence lengths."""
        
        results = {
            'standard': {'peak_memory': [], 'allocated_memory': []},
            'sparse_local': {'peak_memory': [], 'allocated_memory': []},
            'sparse_strided': {'peak_memory': [], 'allocated_memory': []},
            'flash': {'peak_memory': [], 'allocated_memory': []}
        }
        
        attention_configs = {
            'standard': (StandardAttention, {}),
            'sparse_local': (SparseAttention, {'sparsity_pattern': 'local', 'window_size': 64}),
            'sparse_strided': (SparseAttention, {'sparsity_pattern': 'strided', 'window_size': 32, 'stride': 8}),
            'flash': (FlashAttention, {'block_size': 64})
        }
        
        for seq_len in seq_lengths:
            print(f"\\nTesting sequence length: {seq_len}")
            
            for attention_name, (attention_class, kwargs) in attention_configs.items():
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                
                try:
                    # Create attention layer
                    attention = attention_class(d_model, num_heads, **kwargs).to(self.device)
                    
                    # Create input
                    x = torch.randn(batch_size, seq_len, d_model, device=self.device, requires_grad=True)
                    
                    # Forward pass
                    output = attention(x)
                    loss = output.sum()
                    
                    # Backward pass
                    loss.backward()
                    
                    # Measure memory
                    if self.device.type == 'cuda':
                        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
                        allocated_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                    else:
                        # For CPU, use approximate measurement
                        process = psutil.Process()
                        memory_info = process.memory_info()
                        peak_memory = memory_info.rss / 1024 / 1024  # MB
                        allocated_memory = peak_memory  # Approximate
                    
                    results[attention_name]['peak_memory'].append(peak_memory)
                    results[attention_name]['allocated_memory'].append(allocated_memory)
                    
                    print(f"  {attention_name}: Peak={peak_memory:.1f}MB, Allocated={allocated_memory:.1f}MB")
                    
                except Exception as e:
                    print(f"  {attention_name}: Error - {str(e)}")
                    results[attention_name]['peak_memory'].append(float('nan'))
                    results[attention_name]['allocated_memory'].append(float('nan'))
                
                # Cleanup
                del attention, output, loss, x
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        return results
    
    def plot_memory_usage(self, results: Dict[str, Dict[str, List[float]]], seq_lengths: List[int], output_dir: str):
        """Plot memory usage results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Peak memory plot
        for attention_type, metrics in results.items():
            peak_memory = [m for m in metrics['peak_memory'] if not np.isnan(m)]
            seq_lengths_filtered = seq_lengths[:len(peak_memory)]
            
            if peak_memory:
                ax1.plot(seq_lengths_filtered, peak_memory, marker='o', label=attention_type)
        
        ax1.set_xlabel('Sequence Length')
        ax1.set_ylabel('Peak Memory (MB)')
        ax1.set_title('Peak Memory Usage vs Sequence Length')
        ax1.legend()
        ax1.set_xscale('log', base=2)
        ax1.set_yscale('log')
        ax1.grid(True)
        
        # Allocated memory plot
        for attention_type, metrics in results.items():
            allocated_memory = [m for m in metrics['allocated_memory'] if not np.isnan(m)]
            seq_lengths_filtered = seq_lengths[:len(allocated_memory)]
            
            if allocated_memory:
                ax2.plot(seq_lengths_filtered, allocated_memory, marker='s', label=attention_type)
        
        ax2.set_xlabel('Sequence Length')
        ax2.set_ylabel('Allocated Memory (MB)')
        ax2.set_title('Allocated Memory vs Sequence Length')
        ax2.legend()
        ax2.set_xscale('log', base=2)
        ax2.set_yscale('log')
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'memory_usage.png'), dpi=300, bbox_inches='tight')
        plt.show()


class ComputationalComplexityAnalyzer:
    """Analyze computational complexity of attention mechanisms."""
    
    def __init__(self, device: str = 'auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
    
    def analyze_scaling(
        self,
        seq_lengths: List[int] = [64, 128, 256, 512, 1024],
        d_model: int = 512,
        num_heads: int = 8,
        batch_size: int = 4,
        num_iterations: int = 20
    ) -> Dict[str, Dict[str, List[float]]]:
        """Analyze how runtime scales with sequence length."""
        
        results = {
            'standard': {'runtime': [], 'flops': []},
            'sparse_local': {'runtime': [], 'flops': []},
            'flash': {'runtime': [], 'flops': []}
        }
        
        attention_configs = {
            'standard': (StandardAttention, {}),
            'sparse_local': (SparseAttention, {'sparsity_pattern': 'local', 'window_size': 64}),
            'flash': (FlashAttention, {'block_size': 64})
        }
        
        for seq_len in seq_lengths:
            print(f"\\nAnalyzing sequence length: {seq_len}")
            
            for attention_name, (attention_class, kwargs) in attention_configs.items():
                try:
                    # Create attention layer
                    attention = attention_class(d_model, num_heads, **kwargs).to(self.device)
                    
                    # Create input
                    x = torch.randn(batch_size, seq_len, d_model, device=self.device)
                    
                    # Warmup
                    for _ in range(5):
                        _ = attention(x)
                    
                    # Benchmark runtime
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    start_time = time.time()
                    
                    for _ in range(num_iterations):
                        output = attention(x)
                        if self.device.type == 'cuda':
                            torch.cuda.synchronize()
                    
                    end_time = time.time()
                    avg_runtime = (end_time - start_time) / num_iterations * 1000  # ms
                    
                    # Estimate FLOPs (simplified)
                    # Standard attention: O(n^2 * d) for attention matrix computation
                    # Plus O(n^2 * d) for attention-value multiplication
                    if attention_name == 'standard':
                        flops = 2 * seq_len * seq_len * d_model * num_heads * batch_size
                    elif attention_name == 'sparse_local':
                        # Sparse attention with local window
                        window_size = kwargs.get('window_size', 64)
                        effective_length = min(seq_len, window_size)
                        flops = 2 * seq_len * effective_length * d_model * num_heads * batch_size
                    else:  # flash
                        # Flash attention has same FLOPs as standard but better memory access
                        flops = 2 * seq_len * seq_len * d_model * num_heads * batch_size
                    
                    results[attention_name]['runtime'].append(avg_runtime)
                    results[attention_name]['flops'].append(flops)
                    
                    print(f"  {attention_name}: {avg_runtime:.2f}ms, {flops/1e9:.2f} GFLOPs")
                    
                except Exception as e:
                    print(f"  {attention_name}: Error - {str(e)}")
                    results[attention_name]['runtime'].append(float('nan'))
                    results[attention_name]['flops'].append(float('nan'))
                
                # Cleanup
                del attention, output, x
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        return results
    
    def plot_scaling_analysis(self, results: Dict[str, Dict[str, List[float]]], seq_lengths: List[int], output_dir: str):
        """Plot scaling analysis results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Runtime scaling
        for attention_type, metrics in results.items():
            runtime = [r for r in metrics['runtime'] if not np.isnan(r)]
            seq_lengths_filtered = seq_lengths[:len(runtime)]
            
            if runtime:
                ax1.loglog(seq_lengths_filtered, runtime, marker='o', label=attention_type)
        
        ax1.set_xlabel('Sequence Length')
        ax1.set_ylabel('Runtime (ms)')
        ax1.set_title('Runtime Scaling with Sequence Length')
        ax1.legend()
        ax1.grid(True)
        
        # Add theoretical complexity lines
        x_theory = np.array(seq_lengths)
        y_quadratic = x_theory**2 / seq_lengths[0]**2 * results['standard']['runtime'][0] if results['standard']['runtime'] and not np.isnan(results['standard']['runtime'][0]) else x_theory**2
        y_linear = x_theory / seq_lengths[0] * results['sparse_local']['runtime'][0] if results['sparse_local']['runtime'] and not np.isnan(results['sparse_local']['runtime'][0]) else x_theory
        
        ax1.loglog(x_theory, y_quadratic, '--', alpha=0.5, label='O(n²) reference')
        ax1.loglog(x_theory, y_linear, '--', alpha=0.5, label='O(n) reference')
        
        # Throughput (GFLOPs/s)
        for attention_type, metrics in results.items():
            throughput = []
            seq_lengths_filtered = []
            
            for i, (runtime, flops) in enumerate(zip(metrics['runtime'], metrics['flops'])):
                if not (np.isnan(runtime) or np.isnan(flops)) and runtime > 0:
                    throughput.append(flops / 1e9 / (runtime / 1000))  # GFLOPs/s
                    seq_lengths_filtered.append(seq_lengths[i])
            
            if throughput:
                ax2.semilogx(seq_lengths_filtered, throughput, marker='s', label=attention_type)
        
        ax2.set_xlabel('Sequence Length')
        ax2.set_ylabel('Throughput (GFLOPs/s)')
        ax2.set_title('Computational Throughput')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'scaling_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()


def run_comprehensive_profiling(output_dir: str = "profiling_results", device: str = 'auto'):
    """Run comprehensive profiling suite."""
    
    print("Starting comprehensive profiling suite...")
    print("="*60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. GPU Profiling
    print("\\n1. GPU PROFILING")
    print("-" * 30)
    
    gpu_profiler = GPUProfiler(device)
    gpu_results = {}
    
    attention_configs = {
        'standard': {},
        'sparse_local': {'sparsity_pattern': 'local', 'window_size': 64},
        'flash': {'block_size': 64}
    }
    
    for attention_type, kwargs in attention_configs.items():
        try:
            metrics = gpu_profiler.profile_attention_mechanism(
                attention_type=attention_type,
                batch_size=4,
                seq_len=512,
                d_model=512,
                num_heads=8,
                num_iterations=50,
                **kwargs
            )
            gpu_results[attention_type] = metrics
        except Exception as e:
            print(f"Error profiling {attention_type}: {str(e)}")
            gpu_results[attention_type] = {'error': str(e)}
    
    # Save GPU profiling results
    gpu_profiler.save_profiles(os.path.join(output_dir, "gpu_profiles"))
    
    with open(os.path.join(output_dir, "gpu_metrics.json"), 'w') as f:
        json.dump(gpu_results, f, indent=2)
    
    # 2. Memory Profiling
    print("\\n2. MEMORY PROFILING")
    print("-" * 30)
    
    memory_profiler = MemoryProfiler(device)
    memory_results = memory_profiler.profile_memory_usage(
        seq_lengths=[128, 256, 512, 1024],
        d_model=512,
        num_heads=8,
        batch_size=4
    )
    
    memory_profiler.plot_memory_usage(memory_results, [128, 256, 512, 1024], output_dir)
    
    with open(os.path.join(output_dir, "memory_results.json"), 'w') as f:
        json.dump(memory_results, f, indent=2)
    
    # 3. Computational Complexity Analysis
    print("\\n3. COMPUTATIONAL COMPLEXITY ANALYSIS")
    print("-" * 40)
    
    complexity_analyzer = ComputationalComplexityAnalyzer(device)
    scaling_results = complexity_analyzer.analyze_scaling(
        seq_lengths=[64, 128, 256, 512, 1024],
        d_model=512,
        num_heads=8,
        batch_size=4,
        num_iterations=20
    )
    
    complexity_analyzer.plot_scaling_analysis(scaling_results, [64, 128, 256, 512, 1024], output_dir)
    
    with open(os.path.join(output_dir, "scaling_results.json"), 'w') as f:
        json.dump(scaling_results, f, indent=2)
    
    # 4. Generate Summary Report
    print("\\n4. GENERATING SUMMARY REPORT")
    print("-" * 35)
    
    generate_profiling_report(gpu_results, memory_results, scaling_results, output_dir)
    
    print(f"\\nProfiling complete! Results saved to: {output_dir}")


def generate_profiling_report(gpu_results, memory_results, scaling_results, output_dir):
    """Generate a comprehensive profiling report."""
    
    report_file = os.path.join(output_dir, "profiling_report.md")
    
    with open(report_file, 'w') as f:
        f.write("# Attention Mechanisms Profiling Report\\n\\n")
        
        # GPU Profiling Section
        f.write("## GPU Profiling Results\\n\\n")
        
        valid_gpu_results = {k: v for k, v in gpu_results.items() if 'error' not in v}
        
        if valid_gpu_results:
            f.write("| Attention Type | CPU Time (μs) | CUDA Time (μs) | Peak Memory (MB) | Attention % of Total |\\n")
            f.write("|----------------|---------------|----------------|------------------|---------------------|\\n")
            
            for attention_type, metrics in valid_gpu_results.items():
                cpu_time = metrics.get('total_cpu_time_us', 0)
                cuda_time = metrics.get('total_cuda_time_us', 0)
                memory_mb = metrics.get('total_cuda_memory_bytes', 0) / (1024 * 1024)
                attention_pct = metrics.get('attention_cuda_percentage', 0)
                
                f.write(f"| {attention_type} | {cpu_time:.0f} | {cuda_time:.0f} | {memory_mb:.1f} | {attention_pct:.1f}% |\\n")
        
        # Memory Profiling Section
        f.write("\\n## Memory Usage Analysis\\n\\n")
        
        f.write("Memory usage scales differently for each attention mechanism:\\n\\n")
        
        for attention_type, metrics in memory_results.items():
            peak_memories = [m for m in metrics['peak_memory'] if not np.isnan(m)]
            if peak_memories:
                f.write(f"- **{attention_type}**: Peak memory ranges from {min(peak_memories):.1f}MB to {max(peak_memories):.1f}MB\\n")
        
        # Scaling Analysis Section
        f.write("\\n## Computational Scaling Analysis\\n\\n")
        
        f.write("Runtime scaling with sequence length:\\n\\n")
        
        for attention_type, metrics in scaling_results.items():
            runtimes = [r for r in metrics['runtime'] if not np.isnan(r)]
            if len(runtimes) >= 2:
                # Calculate scaling factor
                scaling_factor = runtimes[-1] / runtimes[0]
                f.write(f"- **{attention_type}**: {scaling_factor:.2f}x slowdown from shortest to longest sequence\\n")
        
        # Recommendations Section
        f.write("\\n## Recommendations\\n\\n")
        
        # Find best performing attention mechanism
        best_runtime = float('inf')
        best_attention = None
        
        for attention_type, metrics in scaling_results.items():
            if metrics['runtime'] and not np.isnan(metrics['runtime'][-1]):
                if metrics['runtime'][-1] < best_runtime:
                    best_runtime = metrics['runtime'][-1]
                    best_attention = attention_type
        
        if best_attention:
            f.write(f"- **Fastest attention mechanism**: {best_attention} ({best_runtime:.2f}ms for longest sequence)\\n")
        
        # Memory efficiency
        min_memory = float('inf')
        most_efficient = None
        
        for attention_type, metrics in memory_results.items():
            peak_memories = [m for m in metrics['peak_memory'] if not np.isnan(m)]
            if peak_memories and max(peak_memories) < min_memory:
                min_memory = max(peak_memories)
                most_efficient = attention_type
        
        if most_efficient:
            f.write(f"- **Most memory efficient**: {most_efficient} ({min_memory:.1f}MB peak memory)\\n")
        
        f.write("\\n## Files Generated\\n\\n")
        f.write("- `gpu_profiles/`: Detailed GPU profiling traces\\n")
        f.write("- `gpu_metrics.json`: GPU profiling metrics\\n")
        f.write("- `memory_results.json`: Memory usage data\\n")
        f.write("- `memory_usage.png`: Memory usage plots\\n")
        f.write("- `scaling_results.json`: Computational scaling data\\n")
        f.write("- `scaling_analysis.png`: Scaling analysis plots\\n")
    
    print(f"Report generated: {report_file}")


def main():
    """Main function for profiling script."""
    parser = argparse.ArgumentParser(description="Comprehensive profiling of attention mechanisms")
    
    parser.add_argument("--output-dir", default="profiling_results",
                       help="Output directory for profiling results")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                       help="Device to use for profiling")
    
    args = parser.parse_args()
    
    run_comprehensive_profiling(args.output_dir, args.device)


if __name__ == "__main__":
    main()
