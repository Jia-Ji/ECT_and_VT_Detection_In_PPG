"""
VT SDSD Filter

This module filters VT (ventricular tachycardia) samples based on log(SDSD) threshold
and outputs two separate pickle files. All non-VT samples are kept in both files,
only VT samples are filtered by log(SDSD):
- One with all non-VT samples + VT samples where log(SDSD) < threshold
- One with all non-VT samples + VT samples where log(SDSD) >= threshold

SDSD (Standard Deviation of Successive Differences) is an HRV metric extracted from PPG signals.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
import argparse


class VTSDSDFilter:
    """
    Filter class for splitting VT samples based on log(SDSD) threshold.
    """
    
    def __init__(self, config: DictConfig):
        """
        Initialize the filter with configuration.
        
        Args:
            config: DictConfig containing filter parameters
        """
        self.config = config
        self.filter_cfg = config.filter_vt_sdsd
        
        # Load parameters from config
        self.input_path = self.filter_cfg.get('input_path')
        self.output_dir = self.filter_cfg.get('output_dir', './PPG_data/processed_df')
        self.vt_labels = self.filter_cfg.get('vt_labels', ['VT'])
        
        # Threshold configuration
        self.log_sdsd_threshold = self.filter_cfg.get('log_sdsd_threshold', 4.0)
        
        # Output filenames
        self.output_low_filename = self.filter_cfg.get('output_low_filename', 'vt_log_sdsd_low.pkl')
        self.output_high_filename = self.filter_cfg.get('output_high_filename', 'vt_log_sdsd_high.pkl')
        
        # Outlier removal configuration
        outlier_config = self.filter_cfg.get('outlier_removal', {})
        self.outlier_enable = outlier_config.get('enable', False)
        self.outlier_method = outlier_config.get('method', 'percentile')
        self.iqr_k = outlier_config.get('iqr_k', 1.5)
        self.lower_percentile = outlier_config.get('lower_percentile', 1)
        self.upper_percentile = outlier_config.get('upper_percentile', 99)
        
        # Plot configuration (optional)
        self.save_plot = self.filter_cfg.get('save_plot', True)
        plot_config = self.filter_cfg.get('plot', {})
        self.plot_output_dir = plot_config.get('output_dir', './PPG_data/analysis_plots')
        self.plot_filename = plot_config.get('filename', 'vt_log_sdsd_distribution.png')
        self.figsize = plot_config.get('figsize', [12, 8])
        self.bins = plot_config.get('bins', 50)
        self.dpi = plot_config.get('dpi', 150)
        
        # Validate required paths
        if not self.input_path:
            raise ValueError("input_path must be specified in config.filter_vt_sdsd")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load the pickle file from preprocessing.
        
        Returns:
            DataFrame containing the processed data
        """
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        
        df = pd.read_pickle(self.input_path)
        print(f"Loaded dataframe with shape {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Label distribution:\n{df['label'].value_counts()}")
        
        return df
    
    def separate_vt_and_non_vt(self, df: pd.DataFrame) -> tuple:
        """
        Separate dataframe into VT and non-VT samples.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (df_vt, df_non_vt, matched_labels)
        """
        # Convert vt_labels to list if needed
        vt_labels = list(self.vt_labels) if not isinstance(self.vt_labels, list) else self.vt_labels
        
        # Case-insensitive matching for VT labels
        available_labels = df['label'].unique()
        matched_labels = []
        
        for vt_label in vt_labels:
            # Try exact match first
            if vt_label in available_labels:
                matched_labels.append(vt_label)
            # Try case-insensitive match
            else:
                for avail_label in available_labels:
                    if str(avail_label).upper() == str(vt_label).upper():
                        matched_labels.append(avail_label)
                        break
        
        if not matched_labels:
            print(f"Warning: No VT labels found. Available labels: {available_labels}")
            print(f"Searched for: {vt_labels}")
            return pd.DataFrame(), df.copy(), []
        
        print(f"\nSeparating VT samples with labels: {matched_labels}")
        df_vt = df[df['label'].isin(matched_labels)].copy()
        df_non_vt = df[~df['label'].isin(matched_labels)].copy()
        
        print(f"Found {len(df_vt)} VT samples")
        print(f"Found {len(df_non_vt)} non-VT samples")
        
        return df_vt, df_non_vt, matched_labels
    
    def compute_log_sdsd_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute log(SDSD) for each row and add as a new column.
        
        Args:
            df: DataFrame containing hrv column
            
        Returns:
            DataFrame with added 'log_sdsd' column
        """
        log_sdsd_values = []
        
        for idx in range(len(df)):
            hrv_dict = df.iloc[idx]['hrv']
            if hrv_dict is not None and 'SDSD' in hrv_dict:
                sdsd_val = hrv_dict['SDSD']
                if sdsd_val is not None and not np.isnan(sdsd_val) and sdsd_val > 0:
                    log_sdsd_values.append(np.log(sdsd_val))
                else:
                    log_sdsd_values.append(np.nan)
            else:
                log_sdsd_values.append(np.nan)
        
        df = df.copy()
        df['log_sdsd'] = log_sdsd_values
        
        valid_count = df['log_sdsd'].notna().sum()
        print(f"\nComputed log(SDSD) for {valid_count} out of {len(df)} samples")
        print(f"Samples with invalid/missing SDSD: {len(df) - valid_count}")
        
        return df
    
    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove outliers from the dataframe based on log_sdsd values.
        This is only applied for plotting, not for saving filtered data.
        
        Args:
            df: DataFrame with log_sdsd column
            
        Returns:
            Filtered DataFrame with outliers removed
        """
        if not self.outlier_enable:
            print("Outlier removal for plotting: Disabled")
            return df
        
        # Only consider rows with valid log_sdsd
        valid_mask = df['log_sdsd'].notna()
        values = df.loc[valid_mask, 'log_sdsd'].values
        original_count = len(values)
        
        if self.outlier_method == 'iqr':
            # IQR-based outlier removal
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            lower_bound = q1 - self.iqr_k * iqr
            upper_bound = q3 + self.iqr_k * iqr
            
            print(f"IQR outlier removal (k={self.iqr_k}):")
            print(f"  Bounds: [{lower_bound:.3f}, {upper_bound:.3f}]")
            
        elif self.outlier_method == 'percentile':
            # Percentile-based outlier removal
            lower_bound = np.percentile(values, self.lower_percentile)
            upper_bound = np.percentile(values, self.upper_percentile)
            
            print(f"Percentile outlier removal ({self.lower_percentile}%-{self.upper_percentile}%):")
            print(f"  Bounds: [{lower_bound:.3f}, {upper_bound:.3f}]")
        
        else:
            print(f"Warning: Unknown outlier method '{self.outlier_method}', skipping outlier removal")
            return df
        
        # Create mask for non-outliers (keep NaN values for now, they'll be filtered later)
        outlier_mask = (
            (df['log_sdsd'] >= lower_bound) & 
            (df['log_sdsd'] <= upper_bound)
        ) | df['log_sdsd'].isna()
        
        df_filtered = df[outlier_mask].copy()
        
        removed_count = len(df) - len(df_filtered)
        print(f"  Removed {removed_count} outliers ({removed_count/len(df)*100:.1f}%)")
        print(f"  Remaining samples: {len(df_filtered)}")
        
        return df_filtered
    
    def split_by_threshold(self, df: pd.DataFrame) -> tuple:
        """
        Split dataframe into two based on log_sdsd threshold.
        
        Args:
            df: DataFrame with log_sdsd column
            
        Returns:
            Tuple of (df_low, df_high) where:
            - df_low: samples with log_sdsd < threshold
            - df_high: samples with log_sdsd >= threshold
        """
        # Remove rows with invalid log_sdsd
        df_valid = df[df['log_sdsd'].notna()].copy()
        invalid_count = len(df) - len(df_valid)
        if invalid_count > 0:
            print(f"\nExcluded {invalid_count} samples with invalid log(SDSD)")
        
        # Split by threshold
        df_low = df_valid[df_valid['log_sdsd'] < self.log_sdsd_threshold].copy()
        df_high = df_valid[df_valid['log_sdsd'] >= self.log_sdsd_threshold].copy()
        
        print(f"\nSplit by log(SDSD) threshold = {self.log_sdsd_threshold}:")
        print(f"  Low  (log_sdsd < {self.log_sdsd_threshold}): {len(df_low)} samples")
        print(f"  High (log_sdsd >= {self.log_sdsd_threshold}): {len(df_high)} samples")
        
        # Print statistics for each group
        if len(df_low) > 0:
            print(f"\n  Low group statistics:")
            print(f"    Mean log(SDSD): {df_low['log_sdsd'].mean():.3f}")
            print(f"    Min log(SDSD): {df_low['log_sdsd'].min():.3f}")
            print(f"    Max log(SDSD): {df_low['log_sdsd'].max():.3f}")
        
        if len(df_high) > 0:
            print(f"\n  High group statistics:")
            print(f"    Mean log(SDSD): {df_high['log_sdsd'].mean():.3f}")
            print(f"    Min log(SDSD): {df_high['log_sdsd'].min():.3f}")
            print(f"    Max log(SDSD): {df_high['log_sdsd'].max():.3f}")
        
        return df_low, df_high
    
    def save_filtered_data(self, df_vt_low: pd.DataFrame, df_vt_high: pd.DataFrame, 
                           df_non_vt: pd.DataFrame) -> tuple:
        """
        Save the filtered dataframes to pickle files.
        Each output file contains all non-VT samples plus the filtered VT samples.
        
        Args:
            df_vt_low: DataFrame with VT samples having low log_sdsd values
            df_vt_high: DataFrame with VT samples having high log_sdsd values
            df_non_vt: DataFrame with all non-VT samples
            
        Returns:
            Tuple of (path_low, path_high)
        """
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Combine non-VT with low VT group
        df_low_combined = pd.concat([df_non_vt, df_vt_low], ignore_index=True)
        path_low = os.path.join(self.output_dir, self.output_low_filename)
        df_low_combined.to_pickle(path_low)
        print(f"\n✓ Saved low log(SDSD) dataset to: {path_low}")
        print(f"    Total samples: {len(df_low_combined)} (non-VT: {len(df_non_vt)}, VT: {len(df_vt_low)})")
        
        # Combine non-VT with high VT group
        df_high_combined = pd.concat([df_non_vt, df_vt_high], ignore_index=True)
        path_high = os.path.join(self.output_dir, self.output_high_filename)
        df_high_combined.to_pickle(path_high)
        print(f"✓ Saved high log(SDSD) dataset to: {path_high}")
        print(f"    Total samples: {len(df_high_combined)} (non-VT: {len(df_non_vt)}, VT: {len(df_vt_high)})")
        
        return path_low, path_high
    
    def plot_distribution(self, df: pd.DataFrame) -> str:
        """
        Plot the distribution of log(SDSD) values with threshold line.
        
        Args:
            df: DataFrame with log_sdsd column
            
        Returns:
            Path to the saved plot
        """
        if not self.save_plot:
            return ""
        
        # Get valid log_sdsd values
        log_sdsd = df['log_sdsd'].dropna().values
        
        if len(log_sdsd) == 0:
            print("Warning: No valid log(SDSD) values to plot")
            return ""
        
        # Create output directory if it doesn't exist
        os.makedirs(self.plot_output_dir, exist_ok=True)
        output_path = os.path.join(self.plot_output_dir, self.plot_filename)
        
        # Create figure
        fig, ax = plt.subplots(figsize=tuple(self.figsize))
        
        # Plot histogram with different colors for low and high
        low_values = log_sdsd[log_sdsd < self.log_sdsd_threshold]
        high_values = log_sdsd[log_sdsd >= self.log_sdsd_threshold]
        
        # Determine bin edges for consistent binning
        all_min, all_max = log_sdsd.min(), log_sdsd.max()
        bin_edges = np.linspace(all_min, all_max, self.bins + 1)
        
        ax.hist(low_values, bins=bin_edges, color='#3498db', edgecolor='black', 
                alpha=0.7, label=f'log(SDSD) < {self.log_sdsd_threshold} (n={len(low_values)})')
        ax.hist(high_values, bins=bin_edges, color='#e74c3c', edgecolor='black', 
                alpha=0.7, label=f'log(SDSD) ≥ {self.log_sdsd_threshold} (n={len(high_values)})')
        
        # Add threshold line
        ax.axvline(x=self.log_sdsd_threshold, color='black', linestyle='--', 
                   linewidth=2, label=f'Threshold = {self.log_sdsd_threshold}')
        
        # Add labels and title
        ax.set_xlabel('log(SDSD)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of VT Samples by log(SDSD)', fontsize=14, fontweight='bold', pad=15)
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add statistics text box
        stats_text = self._generate_stats_text(log_sdsd, low_values, high_values)
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8)
        )
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        
        print(f"\n✓ Saved distribution plot to: {output_path}")
        return output_path
    
    def _generate_stats_text(self, all_values: np.ndarray, low_values: np.ndarray, 
                             high_values: np.ndarray) -> str:
        """
        Generate statistics text for the plot.
        """
        stats_text = (
            f"Overall Statistics (n={len(all_values)})\n"
            f"{'─' * 25}\n"
            f"Mean: {np.mean(all_values):.3f}\n"
            f"Median: {np.median(all_values):.3f}\n"
            f"Std: {np.std(all_values):.3f}\n"
            f"Min: {np.min(all_values):.3f}\n"
            f"Max: {np.max(all_values):.3f}\n"
            f"\n"
            f"Low group: {len(low_values)} ({len(low_values)/len(all_values)*100:.1f}%)\n"
            f"High group: {len(high_values)} ({len(high_values)/len(all_values)*100:.1f}%)"
        )
        return stats_text
    
    def run(self) -> tuple:
        """
        Run the complete filtering pipeline.
        
        Returns:
            Tuple of (path_low, path_high) for saved pickle files
        """
        print("=" * 80)
        print("VT SDSD FILTER")
        print("=" * 80)
        print(f"Input file: {self.input_path}")
        print(f"Output directory: {self.output_dir}")
        print(f"VT labels: {self.vt_labels}")
        print(f"log(SDSD) threshold: {self.log_sdsd_threshold}")
        print(f"Outlier removal (plotting only): {'Enabled' if self.outlier_enable else 'Disabled'}")
        if self.outlier_enable:
            if self.outlier_method == 'iqr':
                print(f"  Method: IQR (k={self.iqr_k})")
            else:
                print(f"  Method: Percentile ({self.lower_percentile}%-{self.upper_percentile}%)")
        print("=" * 80)
        
        # Step 1: Load data
        print("\n[Step 1] Loading data...")
        df = self.load_data()
        
        # Step 2: Separate VT and non-VT samples
        print("\n[Step 2] Separating VT and non-VT samples...")
        df_vt, df_non_vt, matched_labels = self.separate_vt_and_non_vt(df)
        
        if len(df_vt) == 0:
            print("\nError: No VT samples found in the dataset.")
            return "", ""
        
        # Step 3: Compute log(SDSD) column for VT samples
        print("\n[Step 3] Computing log(SDSD) for VT samples...")
        df_vt = self.compute_log_sdsd_column(df_vt)
        
        # Step 4: Split VT samples by threshold
        print("\n[Step 4] Splitting VT samples by threshold...")
        df_vt_low, df_vt_high = self.split_by_threshold(df_vt)
        
        # Step 5: Save filtered data (non-VT + filtered VT)
        print("\n[Step 5] Saving filtered data...")
        path_low, path_high = self.save_filtered_data(df_vt_low, df_vt_high, df_non_vt)
        
        # Step 6: Plot distribution (optional, with outlier removal for VT samples only)
        if self.save_plot:
            print("\n[Step 6] Plotting VT distribution...")
            # Combine VT samples for plotting
            df_vt_combined = pd.concat([df_vt_low, df_vt_high], ignore_index=True)
            # Remove outliers only for plotting
            df_for_plot = self.remove_outliers(df_vt_combined)
            self.plot_distribution(df_for_plot)
        
        print("\n" + "=" * 80)
        print("FILTERING COMPLETE")
        print("=" * 80)
        print(f"\nOutput files:")
        print(f"  Low log(SDSD):  {path_low}")
        print(f"  High log(SDSD): {path_high}")
        
        return path_low, path_high


def main():
    """Main entry point for the VT SDSD filter."""
    parser = argparse.ArgumentParser(
        description="Filter VT samples by log(SDSD) threshold"
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='./config/filter_vt_sdsd_config.yaml',
        help='Path to the configuration YAML file'
    )
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    config = OmegaConf.load(args.config)
    print(f"Loaded configuration from: {args.config}")
    
    # Create filter and run
    vt_filter = VTSDSDFilter(config)
    vt_filter.run()


if __name__ == "__main__":
    main()
