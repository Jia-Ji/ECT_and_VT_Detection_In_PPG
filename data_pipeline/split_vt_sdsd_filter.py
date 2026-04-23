"""
Split VT SDSD Filter

This module filters VT (ventricular tachycardia) samples from split datasets based on log(SDSD) threshold.
It takes a pre-split dataset (train/val/test numpy arrays) and outputs two datasets:
- One with all non-VT samples + VT samples where log(SDSD) > threshold (high)
- One with all non-VT samples + VT samples where log(SDSD) <= threshold (low)

Input: Split dataset directory containing numpy arrays (e.g., splitted_data_sqi0/sqi_0_0)
Output: Two new directories with filtered datasets
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
import argparse
from typing import Dict, List, Tuple


class SplitVTSDSDFilter:
    """
    Filter class for splitting VT samples from pre-split datasets based on log(SDSD) threshold.
    """
    
    def __init__(self, config: DictConfig):
        """
        Initialize the filter with configuration.
        
        Args:
            config: DictConfig containing filter parameters
        """
        self.config = config
        self.filter_cfg = config.split_vt_sdsd_filter
        
        # Load parameters from config
        self.input_dir = self.filter_cfg.get('input_dir')
        self.output_dir_high = self.filter_cfg.get('output_dir_high')
        self.output_dir_low = self.filter_cfg.get('output_dir_low')
        
        # VT label encoding (default: 2 for VT in standard encoding NORM=0, ECT=1, VT=2)
        self.vt_label = self.filter_cfg.get('vt_label', 2)
        
        # Threshold configuration
        self.log_sdsd_threshold = self.filter_cfg.get('log_sdsd_threshold', 4.0)
        
        # Plot configuration (optional)
        self.save_plot = self.filter_cfg.get('save_plot', True)
        plot_config = self.filter_cfg.get('plot', {})
        self.plot_output_dir = plot_config.get('output_dir', './PPG_data/analysis_plots')
        self.plot_filename = plot_config.get('filename', 'split_vt_log_sdsd_distribution.png')
        self.figsize = plot_config.get('figsize', [12, 8])
        self.bins = plot_config.get('bins', 50)
        self.dpi = plot_config.get('dpi', 150)
        
        # Validate required paths
        if not self.input_dir:
            raise ValueError("input_dir must be specified in config.split_vt_sdsd_filter")
        if not self.output_dir_high:
            raise ValueError("output_dir_high must be specified in config.split_vt_sdsd_filter")
        if not self.output_dir_low:
            raise ValueError("output_dir_low must be specified in config.split_vt_sdsd_filter")
    
    def get_array_files(self) -> Dict[str, List[str]]:
        """
        Get list of array files for each split (train, val, test).
        
        Returns:
            Dictionary mapping split names to list of file paths
        """
        splits = ['train', 'val', 'test']
        array_files = {split: [] for split in splits}
        
        # List all .npy files in input directory
        all_files = os.listdir(self.input_dir)
        npy_files = [f for f in all_files if f.endswith('.npy')]
        csv_files = [f for f in all_files if f.endswith('.csv')]
        txt_files = [f for f in all_files if f.endswith('.txt')]
        json_files = [f for f in all_files if f.endswith('.json')]
        
        # Categorize files by split
        for split in splits:
            for f in npy_files:
                if f.endswith(f'_{split}.npy'):
                    array_files[split].append(f)
            for f in csv_files:
                if f.endswith(f'_{split}.csv'):
                    array_files[split].append(f)
        
        # Add non-split files (like domain_stats.json, data_distribution.txt)
        array_files['metadata'] = txt_files + json_files
        
        return array_files
    
    def load_split_data(self, split: str) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
        """
        Load all numpy arrays for a given split.
        
        Args:
            split: Split name ('train', 'val', or 'test')
            
        Returns:
            Tuple of (data dict, file_patterns dict) where:
            - data: Dictionary mapping array names to numpy arrays
            - file_patterns: Dictionary mapping array names to original filename patterns
        """
        data = {}
        file_patterns = {}  # Track original file naming pattern for each array
        
        # Load all .npy files for this split
        # Pattern 1: files ending with _{split}.npy (e.g., labels_train.npy)
        npy_files = [f for f in os.listdir(self.input_dir) 
                     if f.endswith(f'_{split}.npy')]
        
        for f in npy_files:
            # Extract array name (e.g., 'labels' from 'labels_train.npy')
            array_name = f.replace(f'_{split}.npy', '')
            file_path = os.path.join(self.input_dir, f)
            data[array_name] = np.load(file_path, allow_pickle=True)
            file_patterns[array_name] = 'end'  # Split name at end
        
        # Pattern 2: files with _{split}_ in middle (e.g., ppg_train_normalized.npy)
        npy_files_middle = [f for f in os.listdir(self.input_dir) 
                           if f'_{split}_' in f and f.endswith('.npy')]
        
        for f in npy_files_middle:
            # Extract array name (e.g., 'ppg_normalized' from 'ppg_train_normalized.npy')
            array_name = f.replace(f'_{split}_', '_').replace('.npy', '')
            file_path = os.path.join(self.input_dir, f)
            data[array_name] = np.load(file_path, allow_pickle=True)
            # Store the original filename template (replace split with placeholder)
            file_patterns[array_name] = f.replace(f'_{split}_', '_{split}_')  # middle pattern
            
        # Load CSV files (like bm_stats)
        csv_files = [f for f in os.listdir(self.input_dir) 
                     if f.endswith(f'_{split}.csv')]
        
        for f in csv_files:
            array_name = f.replace(f'_{split}.csv', '')
            file_path = os.path.join(self.input_dir, f)
            df = pd.read_csv(file_path)
            data[array_name] = df.to_numpy()
            file_patterns[array_name] = 'csv'
        
        return data, file_patterns
    
    def compute_log_sdsd(self, sdsd_array: np.ndarray) -> np.ndarray:
        """
        Compute log(SDSD) for each sample.
        
        Args:
            sdsd_array: Array of SDSD values
            
        Returns:
            Array of log(SDSD) values (NaN for invalid values)
        """
        log_sdsd = np.full_like(sdsd_array, np.nan, dtype=np.float64)
        
        # Handle valid values (positive and not NaN)
        valid_mask = (~np.isnan(sdsd_array)) & (sdsd_array > 0)
        log_sdsd[valid_mask] = np.log(sdsd_array[valid_mask])
        
        return log_sdsd
    
    def filter_by_vt_threshold(self, data: Dict[str, np.ndarray], 
                                log_sdsd: np.ndarray) -> Tuple[Dict[str, np.ndarray], 
                                                                Dict[str, np.ndarray],
                                                                Dict[str, int]]:
        """
        Filter data into high and low log(SDSD) groups for VT samples.
        Non-VT samples are kept in both groups.
        
        Args:
            data: Dictionary of arrays for a split
            log_sdsd: Array of log(SDSD) values
            
        Returns:
            Tuple of (data_high, data_low, stats) where:
            - data_high: Data with non-VT + VT samples where log(SDSD) > threshold
            - data_low: Data with non-VT + VT samples where log(SDSD) <= threshold
            - stats: Dictionary with filtering statistics
        """
        labels = data['labels']
        n_samples = len(labels)
        
        # Create masks
        vt_mask = labels == self.vt_label
        non_vt_mask = ~vt_mask
        
        # For VT samples, create high/low masks based on log(SDSD)
        vt_high_mask = vt_mask & (log_sdsd > self.log_sdsd_threshold)
        vt_low_mask = vt_mask & (log_sdsd <= self.log_sdsd_threshold)
        
        # Handle VT samples with invalid log(SDSD) - include them in low group
        vt_invalid_mask = vt_mask & np.isnan(log_sdsd)
        vt_low_mask = vt_low_mask | vt_invalid_mask
        
        # Final masks: non-VT + filtered VT
        high_mask = non_vt_mask | vt_high_mask
        low_mask = non_vt_mask | vt_low_mask
        
        # Filter all arrays
        data_high = {}
        data_low = {}
        
        for array_name, array in data.items():
            if len(array) == n_samples:
                data_high[array_name] = array[high_mask]
                data_low[array_name] = array[low_mask]
            else:
                # Skip arrays that don't match the sample count
                print(f"  Warning: Skipping {array_name} (length {len(array)} != {n_samples})")
        
        # Calculate statistics
        stats = {
            'total_samples': n_samples,
            'non_vt_samples': non_vt_mask.sum(),
            'vt_samples': vt_mask.sum(),
            'vt_high_samples': vt_high_mask.sum(),
            'vt_low_samples': vt_low_mask.sum(),
            'vt_invalid_sdsd': vt_invalid_mask.sum(),
            'high_total': high_mask.sum(),
            'low_total': low_mask.sum()
        }
        
        return data_high, data_low, stats
    
    def save_split_data(self, data: Dict[str, np.ndarray], output_dir: str, 
                        split: str, file_patterns: Dict[str, str]) -> None:
        """
        Save filtered data arrays to output directory.
        
        Args:
            data: Dictionary of arrays to save
            output_dir: Output directory path
            split: Split name ('train', 'val', or 'test')
            file_patterns: Dictionary mapping array names to file patterns
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for array_name, array in data.items():
            pattern = file_patterns.get(array_name, 'end')
            
            if pattern == 'csv':
                # Save as CSV
                file_path = os.path.join(output_dir, f"{array_name}_{split}.csv")
                df = pd.DataFrame(array)
                df.to_csv(file_path, index=False)
            elif pattern == 'end':
                # Standard pattern: array_name_{split}.npy
                file_path = os.path.join(output_dir, f"{array_name}_{split}.npy")
                np.save(file_path, array)
            else:
                # Middle pattern: e.g., ppg_normalized -> ppg_{split}_normalized.npy
                # array_name like 'ppg_normalized' -> split into parts
                parts = array_name.split('_')
                if len(parts) >= 2:
                    # Insert split after first part (e.g., ppg_normalized -> ppg_train_normalized)
                    filename = f"{parts[0]}_{split}_{'_'.join(parts[1:])}.npy"
                else:
                    filename = f"{array_name}_{split}.npy"
                file_path = os.path.join(output_dir, filename)
                np.save(file_path, array)
    
    def copy_metadata_files(self, output_dir: str) -> None:
        """
        Copy metadata files (data_distribution.txt, domain_stats.json) to output directory.
        """
        metadata_files = ['data_distribution.txt', 'domain_stats.json']
        
        for f in metadata_files:
            src_path = os.path.join(self.input_dir, f)
            if os.path.exists(src_path):
                dst_path = os.path.join(output_dir, f)
                with open(src_path, 'r') as src:
                    content = src.read()
                with open(dst_path, 'w') as dst:
                    dst.write(content)
    
    def save_distribution_report(self, stats_all: Dict[str, Dict[str, int]], 
                                  output_dir: str, group_name: str) -> None:
        """
        Save distribution report for filtered dataset.
        
        Args:
            stats_all: Dictionary of statistics for each split
            output_dir: Output directory path
            group_name: Name of the group ('high' or 'low')
        """
        report_path = os.path.join(output_dir, 'filter_distribution.txt')
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"VT SDSD FILTER DISTRIBUTION REPORT ({group_name.upper()})\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Filter Configuration:\n")
            f.write(f"  log(SDSD) threshold: {self.log_sdsd_threshold}\n")
            f.write(f"  VT label encoding: {self.vt_label}\n")
            if group_name == 'high':
                f.write(f"  Filter criteria: VT samples with log(SDSD) > {self.log_sdsd_threshold}\n")
            else:
                f.write(f"  Filter criteria: VT samples with log(SDSD) <= {self.log_sdsd_threshold}\n")
            f.write(f"\nSource directory: {self.input_dir}\n")
            f.write("\n" + "=" * 80 + "\n\n")
            
            for split, stats in stats_all.items():
                f.write(f"{split.upper()} SPLIT\n")
                f.write("-" * 80 + "\n")
                f.write(f"  Original total samples: {stats['total_samples']}\n")
                f.write(f"  Non-VT samples: {stats['non_vt_samples']}\n")
                f.write(f"  VT samples (original): {stats['vt_samples']}\n")
                f.write(f"  VT samples (high log_sdsd > {self.log_sdsd_threshold}): {stats['vt_high_samples']}\n")
                f.write(f"  VT samples (low log_sdsd <= {self.log_sdsd_threshold}): {stats['vt_low_samples']}\n")
                f.write(f"  VT samples with invalid SDSD: {stats['vt_invalid_sdsd']}\n")
                f.write(f"\n  Final samples in this dataset: {stats['high_total'] if group_name == 'high' else stats['low_total']}\n")
                f.write(f"    (Non-VT: {stats['non_vt_samples']} + VT: {stats['vt_high_samples'] if group_name == 'high' else stats['vt_low_samples']})\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
        
        print(f"  ✓ Saved distribution report to: {report_path}")
    
    def plot_distribution(self, log_sdsd_all: Dict[str, np.ndarray], 
                          labels_all: Dict[str, np.ndarray]) -> str:
        """
        Plot the distribution of log(SDSD) values for VT samples across all splits.
        
        Args:
            log_sdsd_all: Dictionary mapping split names to log(SDSD) arrays
            labels_all: Dictionary mapping split names to label arrays
            
        Returns:
            Path to the saved plot
        """
        if not self.save_plot:
            return ""
        
        # Collect all VT log(SDSD) values
        vt_log_sdsd = []
        for split in ['train', 'val', 'test']:
            if split in log_sdsd_all and split in labels_all:
                log_sdsd = log_sdsd_all[split]
                labels = labels_all[split]
                vt_mask = labels == self.vt_label
                vt_values = log_sdsd[vt_mask]
                vt_values = vt_values[~np.isnan(vt_values)]
                vt_log_sdsd.extend(vt_values)
        
        vt_log_sdsd = np.array(vt_log_sdsd)

        # ensure log(SDSD) values are larger than 0
        vt_log_sdsd = vt_log_sdsd[vt_log_sdsd > 0]

        if len(vt_log_sdsd) == 0:
            print("Warning: No valid VT log(SDSD) values to plot")
            return ""
        
        # Create output directory if it doesn't exist
        os.makedirs(self.plot_output_dir, exist_ok=True)
        output_path = os.path.join(self.plot_output_dir, self.plot_filename)
        
        # Create figure
        fig, ax = plt.subplots(figsize=[14,8])
        
        # Split values by threshold
        low_values = vt_log_sdsd[vt_log_sdsd <= self.log_sdsd_threshold]
        high_values = vt_log_sdsd[vt_log_sdsd > self.log_sdsd_threshold]
        
        # Determine bin edges for consistent binning
        all_min, all_max = vt_log_sdsd.min(), vt_log_sdsd.max()
        bin_edges = np.linspace(all_min, all_max, self.bins + 1)
        
        ax.hist(low_values, bins=bin_edges, color='#F5DDB5', edgecolor='black', 
                alpha=0.7, label=f'log(SDSD) ≤ {self.log_sdsd_threshold} (n={len(low_values)})')
        ax.hist(high_values, bins=bin_edges, color='#30859D', edgecolor='black', 
                alpha=0.7, label=f'log(SDSD) > {self.log_sdsd_threshold} (n={len(high_values)})')
        
        # Add threshold line
        ax.axvline(x=self.log_sdsd_threshold, color='black', linestyle='--', 
                   linewidth=2, label=f'Threshold = {self.log_sdsd_threshold}')
        
        # Add labels and title
        ax.set_xlabel('log(SDSD)', fontsize=32, fontweight='bold')
        ax.set_ylabel('VT Frequency', fontsize=32, fontweight='bold')
        ax.set_xticklabels(ax.get_xticks(), fontsize=24, fontweight='bold')
        # set y-ticks to show integer counts
        ax.set_yticks(np.arange(0, int(max(ax.get_yticks())) + 1, max(1, int(max(ax.get_yticks()) // 8))))
        ax.set_yticklabels(ax.get_yticks(), fontsize=24, fontweight='bold')
        # ax.set_title('Distribution of VT Samples by log(SDSD)', 
        #              fontsize=32, fontweight='bold', pad=15)
        ax.legend(fontsize=28, loc='upper left')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # # Add statistics text box
        # stats_text = (
        #     f"VT Statistics (n={len(vt_log_sdsd)})\n"
        #     f"{'─' * 25}\n"
        #     f"Mean: {np.mean(vt_log_sdsd):.3f}\n"
        #     f"Median: {np.median(vt_log_sdsd):.3f}\n"
        #     f"Std: {np.std(vt_log_sdsd):.3f}\n"
        #     f"Min: {np.min(vt_log_sdsd):.3f}\n"
        #     f"Max: {np.max(vt_log_sdsd):.3f}\n"
        #     f"\n"
        #     f"Low (≤{self.log_sdsd_threshold}): {len(low_values)} ({len(low_values)/len(vt_log_sdsd)*100:.1f}%)\n"
        #     f"High (>{self.log_sdsd_threshold}): {len(high_values)} ({len(high_values)/len(vt_log_sdsd)*100:.1f}%)"
        # )
        # ax.text(
        #     0.02, 0.98, stats_text,
        #     transform=ax.transAxes,
        #     fontsize=9,
        #     verticalalignment='top',
        #     horizontalalignment='left',
        #     bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8)
        # )
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        
        print(f"\n✓ Saved distribution plot to: {output_path}")
        return output_path
    
    def run(self) -> Tuple[str, str]:
        """
        Run the complete filtering pipeline.
        
        Returns:
            Tuple of (output_dir_high, output_dir_low) paths
        """
        print("=" * 80)
        print("SPLIT VT SDSD FILTER")
        print("=" * 80)
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory (high): {self.output_dir_high}")
        print(f"Output directory (low): {self.output_dir_low}")
        print(f"VT label encoding: {self.vt_label}")
        print(f"log(SDSD) threshold: {self.log_sdsd_threshold}")
        print("=" * 80)
        
        # Verify input directory exists
        if not os.path.exists(self.input_dir):
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
        
        # Get array files
        array_files = self.get_array_files()
        print(f"\nFound array files:")
        for split, files in array_files.items():
            if files:
                print(f"  {split}: {len(files)} files")
        
        # Process each split
        splits = ['train', 'val', 'test']
        stats_all = {}
        log_sdsd_all = {}
        labels_all = {}
        
        for split in splits:
            print(f"\n[Processing {split.upper()} split]")
            
            # Load data with file patterns
            data, file_patterns = self.load_split_data(split)
            print(f"  Loaded {len(data)} arrays")
            
            # Check required arrays
            if 'sdsd' not in data:
                raise ValueError(f"Required 'sdsd' array not found in {split} split")
            if 'labels' not in data:
                raise ValueError(f"Required 'labels' array not found in {split} split")
            
            # Compute log(SDSD)
            log_sdsd = self.compute_log_sdsd(data['sdsd'])
            log_sdsd_all[split] = log_sdsd.copy()
            labels_all[split] = data['labels'].copy()
            
            # Print VT statistics
            vt_mask = data['labels'] == self.vt_label
            vt_log_sdsd = log_sdsd[vt_mask]
            valid_vt_log_sdsd = vt_log_sdsd[~np.isnan(vt_log_sdsd)]
            print(f"  VT samples: {vt_mask.sum()}")
            print(f"  Valid VT log(SDSD) values: {len(valid_vt_log_sdsd)}")
            if len(valid_vt_log_sdsd) > 0:
                print(f"  VT log(SDSD) range: [{valid_vt_log_sdsd.min():.3f}, {valid_vt_log_sdsd.max():.3f}]")
                print(f"  VT log(SDSD) mean: {valid_vt_log_sdsd.mean():.3f}")
            
            # # Filter by threshold
            # data_high, data_low, stats = self.filter_by_vt_threshold(data, log_sdsd)
            # stats_all[split] = stats
            
            # print(f"  Filtered high (VT log_sdsd > {self.log_sdsd_threshold}): {stats['high_total']} samples")
            # print(f"  Filtered low (VT log_sdsd <= {self.log_sdsd_threshold}): {stats['low_total']} samples")
            
            # # Save filtered data
            # print(f"  Saving high group...")
            # self.save_split_data(data_high, self.output_dir_high, split, file_patterns)
            # print(f"  Saving low group...")
            # self.save_split_data(data_low, self.output_dir_low, split, file_patterns)
        
        # Copy metadata files
        print("\n[Copying metadata files]")
        self.copy_metadata_files(self.output_dir_high)
        self.copy_metadata_files(self.output_dir_low)
        
        # Save distribution reports
        print("\n[Saving distribution reports]")
        self.save_distribution_report(stats_all, self.output_dir_high, 'high')
        self.save_distribution_report(stats_all, self.output_dir_low, 'low')
        
        # Plot distribution
        if self.save_plot:
            print("\n[Plotting VT distribution]")
            self.plot_distribution(log_sdsd_all, labels_all)
        
        # Print summary
        print("\n" + "=" * 80)
        print("FILTERING COMPLETE")
        print("=" * 80)
        print(f"\nOutput directories:")
        print(f"  High (log_sdsd > {self.log_sdsd_threshold}): {self.output_dir_high}")
        print(f"  Low (log_sdsd <= {self.log_sdsd_threshold}): {self.output_dir_low}")
        
        print(f"\nSummary across all splits:")
        total_vt_high = sum(s['vt_high_samples'] for s in stats_all.values())
        total_vt_low = sum(s['vt_low_samples'] for s in stats_all.values())
        total_non_vt = sum(s['non_vt_samples'] for s in stats_all.values())
        print(f"  Total non-VT samples (in both): {total_non_vt}")
        print(f"  Total VT high samples: {total_vt_high}")
        print(f"  Total VT low samples: {total_vt_low}")
        
        return self.output_dir_high, self.output_dir_low


def main():
    """Main entry point for the Split VT SDSD filter."""
    parser = argparse.ArgumentParser(
        description="Filter VT samples from split datasets by log(SDSD) threshold"
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='./config/split_vt_sdsd_filter_config.yaml',
        help='Path to the configuration YAML file'
    )
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    config = OmegaConf.load(args.config)
    print(f"Loaded configuration from: {args.config}")
    
    # Create filter and run
    vt_filter = SplitVTSDSDFilter(config)
    vt_filter.run()


if __name__ == "__main__":
    main()
