"""
Filter by Collection Site

This script filters the combined dataframe into two separate pickle files based on
the collection site determined by the 'ID0' column:
- Cath Lab: ID0 starts with 'ID'
- Cardiac Theatre: ID0 starts with 'Theatre'

Usage:
    python -m data_pipeline.filter_by_collection_site
    
    Or with custom paths:
    python -m data_pipeline.filter_by_collection_site \
        --input PPG_data/combined_df/PPGECG_all_renamed_mo_v1c_complete_labelled.pkl \
        --output_dir PPG_data/combined_df
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        plt.style.use('ggplot')

# Custom color palette for SQI ranges
SQI_COLORS = {
    'Low (0-70%)': '#ED8D5A',      # Red
    'Medium (70-90%)': '#ECB66C',   # Orange/Yellow
    'High (90-100%)': '#51999F'     # Green
}

# Colors for collection sites
SITE_COLORS = {
    'Cath Lab': '#4198AC',
    'Cardiac Theatre': '#ED8D5A'
}


def filter_by_collection_site(
    input_path: str,
    output_dir: str,
    cath_lab_filename: str = "cath_lab_samples.pkl",
    cardiac_theatre_filename: str = "cardiac_theatre_samples.pkl"
) -> tuple:
    """
    Filter the dataframe by collection site based on ID0 column.
    
    Args:
        input_path: Path to the input pickle file
        output_dir: Directory to save the output files
        cath_lab_filename: Filename for cath lab samples
        cardiac_theatre_filename: Filename for cardiac theatre samples
        
    Returns:
        Tuple of (cath_lab_df, cardiac_theatre_df)
    """
    # Load the data
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    print(f"Loading data from: {input_path}")
    df = pd.read_pickle(input_path)
    print(f"Loaded dataframe with shape: {df.shape}")
    
    # Check if ID0 column exists
    if 'ID0' not in df.columns:
        raise ValueError("Column 'ID0' not found in the dataframe")
    
    # Convert ID0 to string for filtering
    df['ID0'] = df['ID0'].astype(str)
    
    # Filter by collection site
    # Cath Lab: ID0 starts with 'ID'
    cath_lab_mask = df['ID0'].str.startswith('ID')
    cath_lab_df = df[cath_lab_mask].copy()
    
    # Cardiac Theatre: ID0 starts with 'Theatre'
    cardiac_theatre_mask = df['ID0'].str.startswith('Theatre')
    cardiac_theatre_df = df[cardiac_theatre_mask].copy()
    
    # Check for samples that don't match either pattern
    unmatched_mask = ~(cath_lab_mask | cardiac_theatre_mask)
    unmatched_count = unmatched_mask.sum()
    if unmatched_count > 0:
        print(f"\nWarning: {unmatched_count} samples have ID0 that doesn't start with 'ID' or 'Theatre'")
        unmatched_ids = df.loc[unmatched_mask, 'ID0'].unique()
        print(f"Unmatched ID0 prefixes: {unmatched_ids[:10]}...")  # Show first 10
    
    # Print statistics
    print(f"\n{'='*50}")
    print("Filtering Results:")
    print(f"{'='*50}")
    print(f"Total samples: {len(df)}")
    print(f"Cath Lab samples (ID0 starts with 'ID'): {len(cath_lab_df)}")
    print(f"Cardiac Theatre samples (ID0 starts with 'Theatre'): {len(cardiac_theatre_df)}")
    print(f"Unmatched samples: {unmatched_count}")
    
    # Print label distribution if label column exists
    if 'label' in df.columns:
        print(f"\n{'='*50}")
        print("Label Distribution:")
        print(f"{'='*50}")
        
        print("\nCath Lab:")
        print(cath_lab_df['label'].value_counts().to_string())
        
        print("\nCardiac Theatre:")
        print(cardiac_theatre_df['label'].value_counts().to_string())
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the filtered dataframes
    cath_lab_path = os.path.join(output_dir, cath_lab_filename)
    cardiac_theatre_path = os.path.join(output_dir, cardiac_theatre_filename)
    
    cath_lab_df.to_pickle(cath_lab_path)
    cardiac_theatre_df.to_pickle(cardiac_theatre_path)
    
    print(f"\n{'='*50}")
    print("Output Files:")
    print(f"{'='*50}")
    print(f"Cath Lab samples saved to: {cath_lab_path}")
    print(f"Cardiac Theatre samples saved to: {cardiac_theatre_path}")
    
    return cath_lab_df, cardiac_theatre_df


def plot_sqi_distribution_by_class(
    df: pd.DataFrame,
    site_name: str,
    output_dir: str,
    sqi_bins: list = [0, 70, 90, 100],
    sqi_labels: list = ['Low (0-70%)', 'Medium (70-90%)', 'High (90-100%)'],
    class_order: list = ['SR', 'ECT', 'VT'],
    figsize: tuple = (10, 7)
) -> None:
    """
    Plot a stacked bar chart showing the count of each class (SR, ECT, VT)
    with SQI distribution proportions within each class.
    
    Args:
        df: DataFrame containing 'label' and 'sqi' columns
        site_name: Name of the collection site (for title)
        output_dir: Directory to save the figure
        sqi_bins: Bin edges for SQI ranges
        sqi_labels: Labels for SQI ranges
        class_order: Order of classes to display
        figsize: Figure size
    """
    # Check required columns
    if 'label' not in df.columns:
        raise ValueError("Column 'label' not found in dataframe")
    if 'sqi' not in df.columns:
        raise ValueError("Column 'sqi' not found in dataframe")
    
    # Map labels (handle different naming conventions)
    label_mapping = {
        'NORM': 'SR',
        'ECT': 'ECT', 
        'PAC': 'ECT',
        'PVC': 'ECT',
        'VT': 'VT'
    }
    df = df.copy()
    df['class_label'] = df['label'].map(lambda x: label_mapping.get(x, x))
    
    # Filter only classes we want to plot
    df_filtered = df[df['class_label'].isin(class_order)].copy()
    
    if len(df_filtered) == 0:
        print(f"Warning: No valid samples found for {site_name}")
        return
    
    # Bin the SQI values
    df_filtered['sqi_range'] = pd.cut(
        df_filtered['sqi'], 
        bins=sqi_bins, 
        labels=sqi_labels, 
        include_lowest=True
    )
    
    # Create a cross-tabulation for counts
    cross_tab = pd.crosstab(df_filtered['class_label'], df_filtered['sqi_range'])
    
    # Reindex to ensure correct order
    cross_tab = cross_tab.reindex(index=class_order, columns=sqi_labels, fill_value=0)
    
    # Calculate proportions for stacking
    proportions = cross_tab.div(cross_tab.sum(axis=1), axis=0)
    
    # Get total counts per class
    total_counts = cross_tab.sum(axis=1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set background
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('#FFFFFF')
    
    # Bar positions
    x = np.arange(len(class_order))
    bar_width = 0.6
    
    # Plot stacked bars
    bottom = np.zeros(len(class_order))
    bars_dict = {}
    
    for sqi_label in sqi_labels:
        heights = proportions[sqi_label].values
        bars = ax.bar(
            x, heights, bar_width, 
            bottom=bottom, 
            label=sqi_label,
            color=SQI_COLORS[sqi_label],
            edgecolor='white',
            linewidth=1.5
        )
        bars_dict[sqi_label] = bars
        bottom += heights
    
    # Add count annotations on bars
    for i, class_name in enumerate(class_order):
        total = total_counts[class_name]
        # Add total count above bar
        ax.annotate(
            f'n={total}',
            xy=(i, 1.02),
            ha='center', va='bottom',
            fontsize=12, fontweight='bold',
            color='#2C3E50'
        )
        
        # Add percentage labels inside each segment
        cumulative = 0
        for sqi_label in sqi_labels:
            count = cross_tab.loc[class_name, sqi_label]
            prop = proportions.loc[class_name, sqi_label]
            if prop > 0.08:  # Only show label if segment is large enough
                mid_point = cumulative + prop / 2
                ax.annotate(
                    f'{prop*100:.1f}%',
                    xy=(i, mid_point),
                    ha='center', va='center',
                    fontsize=10, fontweight='bold',
                    color='white'
                )
            cumulative += prop
    
    # Customize plot
    ax.set_xlabel('Class', fontsize=14, fontweight='bold', color='#2C3E50')
    ax.set_ylabel('Proportion', fontsize=14, fontweight='bold', color='#2C3E50')
    ax.set_title(
        f'{site_name}\nSQI Distribution by Class',
        fontsize=16, fontweight='bold', color='#2C3E50', pad=20
    )
    
    ax.set_xticks(x)
    ax.set_xticklabels(class_order, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
    
    # Add legend
    ax.legend(
        title='SQI Range',
        loc='lower right',
        fontsize=10,
        title_fontsize=11,
        framealpha=0.9,
        edgecolor='#BDC3C7'
    )
    
    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#BDC3C7')
    ax.spines['bottom'].set_color('#BDC3C7')
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    filename = f"sqi_distribution_{site_name.lower().replace(' ', '_')}.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved figure to: {save_path}")
    
    # Print detailed statistics
    print(f"\n{site_name} - SQI Distribution Statistics:")
    print("-" * 70)
    print(f"{'Class':<10} {'Total':<10} {'Low SQI':<15} {'Medium SQI':<15} {'High SQI':<15}")
    print("-" * 70)
    for class_name in class_order:
        total = total_counts[class_name]
        low = cross_tab.loc[class_name, 'Low (0-70%)']
        med = cross_tab.loc[class_name, 'Medium (70-90%)']
        high = cross_tab.loc[class_name, 'High (90-100%)']
        print(f"{class_name:<10} {total:<10} {low} ({low/total*100:.1f}%){'':>5} {med} ({med/total*100:.1f}%){'':>5} {high} ({high/total*100:.1f}%)")


def plot_class_prevalence_comparison(
    cath_lab_df: pd.DataFrame,
    cardiac_theatre_df: pd.DataFrame,
    output_dir: str,
    class_order: list = ['SR', 'ECT', 'VT'],
    figsize: tuple = (12, 7)
) -> None:
    """
    Plot a bar chart comparing class frequency and prevalence between Cath Lab and Cardiac Theatre.
    Shows Cath Lab classes on the left and Cardiac Theatre classes on the right.
    
    Args:
        cath_lab_df: DataFrame for Cath Lab samples
        cardiac_theatre_df: DataFrame for Cardiac Theatre samples
        output_dir: Directory to save the figure
        class_order: Order of classes to display
        figsize: Figure size
    """
    # Map labels (handle different naming conventions)
    label_mapping = {
        'NORM': 'SR',
        'ECT': 'ECT', 
        'PAC': 'ECT',
        'PVC': 'ECT',
        'VT': 'VT'
    }
    
    def get_class_counts(df, label_col='label'):
        if label_col not in df.columns:
            raise ValueError(f"Column '{label_col}' not found in dataframe")
        df = df.copy()
        df['class_label'] = df[label_col].map(lambda x: label_mapping.get(x, x))
        counts = df['class_label'].value_counts()
        return counts
    
    # Get counts for each site
    cath_counts = get_class_counts(cath_lab_df)
    theatre_counts = get_class_counts(cardiac_theatre_df)
    
    # Calculate total only for SR, ECT, VT classes
    cath_total = sum(cath_counts.get(cls, 0) for cls in class_order)
    theatre_total = sum(theatre_counts.get(cls, 0) for cls in class_order)
    
    cath_prevalence = {cls: cath_counts.get(cls, 0) / cath_total * 100 for cls in class_order}
    theatre_prevalence = {cls: theatre_counts.get(cls, 0) / theatre_total * 100 for cls in class_order}
    
    cath_raw_counts = {cls: cath_counts.get(cls, 0) for cls in class_order}
    theatre_raw_counts = {cls: theatre_counts.get(cls, 0) for cls in class_order}
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set background
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('#FFFFFF')
    
    # Bar positions: Cath Lab (3 bars) on left, gap, Cardiac Theatre (3 bars) on right
    n_classes = len(class_order)
    bar_width = 0.7
    gap = 0.6 # Gap between the two groups
    
    # Cath Lab positions: 0, 1, 2
    cath_positions = np.arange(n_classes)
    # Cardiac Theatre positions: 3.5, 4.5, 5.5 (with gap)
    theatre_positions = np.arange(n_classes) + n_classes + gap
    
    # Plot Cath Lab bars (frequency/count)
    cath_values = [cath_raw_counts[cls] for cls in class_order]
    bars_cath = ax.bar(
        cath_positions, cath_values, bar_width,
        color=SITE_COLORS['Cath Lab'],
        edgecolor='white',
        linewidth=1.5,
        label=f'Cath Lab (n={cath_total})'
    )
    
    # Plot Cardiac Theatre bars (frequency/count)
    theatre_values = [theatre_raw_counts[cls] for cls in class_order]
    bars_theatre = ax.bar(
        theatre_positions, theatre_values, bar_width,
        color=SITE_COLORS['Cardiac Theatre'],
        edgecolor='white',
        linewidth=1.5,
        label=f'Cardiac Theatre (n={theatre_total})'
    )
    
    # Add labels on Cath Lab bars
    for bar, cls in zip(bars_cath, class_order):
        height = bar.get_height()
        count = cath_raw_counts[cls]
        prevalence = cath_prevalence[cls]
        # Add count and prevalence on top
        ax.annotate(
            f'n={count}\n({prevalence:.1f}%)',
            xy=(bar.get_x() + bar.get_width()/2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=22, fontweight='bold',
            color='#2C3E50'
        )
    
    # Add labels on Cardiac Theatre bars
    for bar, cls in zip(bars_theatre, class_order):
        height = bar.get_height()
        count = theatre_raw_counts[cls]
        prevalence = theatre_prevalence[cls]
        # Add count and prevalence on top
        ax.annotate(
            f'n={count}\n({prevalence:.1f}%)',
            xy=(bar.get_x() + bar.get_width()/2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=22, fontweight='bold',
            color='#2C3E50'
        )
    
    # Customize plot
    # ax.set_xlabel('Class', fontsize=28, fontweight='bold', color='#2C3E50')
    ax.set_ylabel('Frequency', fontsize=32, fontweight='bold', color='#2C3E50')
    ax.set_yticks(np.arange(0, max(max(cath_values), max(theatre_values)) * 1.25, step=1000))
    ax.set_yticklabels([f'{int(y)}' for y in ax.get_yticks()], fontsize=20, fontweight='bold')
    # ax.set_title(
    #     'Class Frequency and Prevalence: Cath Lab vs Cardiac Theatre',
    #     fontsize=16, fontweight='bold', color='#2C3E50', pad=20
    # )
    
    # Set x-axis ticks and labels
    all_positions = np.concatenate([cath_positions, theatre_positions])
    all_labels = class_order + class_order
    ax.set_xticks(all_positions)
    ax.set_xticklabels(all_labels, fontsize=24, fontweight='bold')
    
    # Add site labels below x-axis
    cath_center = np.mean(cath_positions)
    theatre_center = np.mean(theatre_positions)
    
    # Add site group labels
    ax.annotate(
        'Cath Lab',
        xy=(cath_center, 0),
        xytext=(0, -35),
        textcoords="offset points",
        ha='center', va='top',
        fontsize=32, fontweight='bold',
        color=SITE_COLORS['Cath Lab']
    )
    ax.annotate(
        'Cardiac Theatre',
        xy=(theatre_center, 0),
        xytext=(0, -35),
        textcoords="offset points",
        ha='center', va='top',
        fontsize=32, fontweight='bold',
        color=SITE_COLORS['Cardiac Theatre']
    )
    
    # Set y-axis limit with headroom for labels
    max_val = max(max(cath_values), max(theatre_values))
    ax.set_ylim(0, max_val * 1.25)
    
    # Add legend
    ax.legend(
        loc='upper right',
        fontsize=24,
        framealpha=0.9,
        edgecolor='#BDC3C7'
    )
    
    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#BDC3C7')
    ax.spines['bottom'].set_color('#BDC3C7')
    
    # Add vertical line to separate the two groups
    separator_x = (cath_positions[-1] + theatre_positions[0]) / 2
    ax.axvline(x=separator_x, color='#BDC3C7', linestyle='--', linewidth=1.5, alpha=0.7)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for site labels
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "class_prevalence_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved prevalence comparison figure to: {save_path}")
    
    # Print detailed statistics
    print(f"\nClass Prevalence Comparison:")
    print("-" * 70)
    print(f"{'Class':<10} {'Cath Lab':<25} {'Cardiac Theatre':<25}")
    print("-" * 70)
    for cls in class_order:
        cath_str = f"{cath_raw_counts[cls]} ({cath_prevalence[cls]:.1f}%)"
        theatre_str = f"{theatre_raw_counts[cls]} ({theatre_prevalence[cls]:.1f}%)"
        print(f"{cls:<10} {cath_str:<25} {theatre_str:<25}")
    print("-" * 70)
    print(f"{'Total':<10} {cath_total:<25} {theatre_total:<25}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter samples by collection site (Cath Lab vs Cardiac Theatre)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="PPG_data/combined_df/PPGECG_all_renamed_mo_v1c_complete_labelled.pkl",
        help="Path to input pickle file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="PPG_data/combined_df",
        help="Directory to save output files"
    )
    parser.add_argument(
        "--cath_lab_filename",
        type=str,
        default="cath_lab_samples.pkl",
        help="Filename for cath lab samples"
    )
    parser.add_argument(
        "--cardiac_theatre_filename",
        type=str,
        default="cardiac_theatre_samples.pkl",
        help="Filename for cardiac theatre samples"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate SQI distribution plots for each site"
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="PPG_data/analysis_plots",
        help="Directory to save plots"
    )
    
    args = parser.parse_args()
    
    cath_lab_df, cardiac_theatre_df = filter_by_collection_site(
        input_path=args.input,
        output_dir=args.output_dir,
        cath_lab_filename=args.cath_lab_filename,
        cardiac_theatre_filename=args.cardiac_theatre_filename
    )
    
    # Generate plots if requested
    if args.plot:
        print(f"\n{'='*50}")
        print("Generating SQI Distribution Plots")
        print(f"{'='*50}")
        
        # Plot for Cath Lab
        if len(cath_lab_df) > 0:
            plot_sqi_distribution_by_class(
                df=cath_lab_df,
                site_name="Cath Lab",
                output_dir=args.plot_dir
            )
        
        # Plot for Cardiac Theatre
        if len(cardiac_theatre_df) > 0:
            plot_sqi_distribution_by_class(
                df=cardiac_theatre_df,
                site_name="Cardiac Theatre",
                output_dir=args.plot_dir
            )
        
        # Plot class prevalence comparison
        print(f"\n{'='*50}")
        print("Generating Class Prevalence Comparison Plot")
        print(f"{'='*50}")
        
        if len(cath_lab_df) > 0 and len(cardiac_theatre_df) > 0:
            plot_class_prevalence_comparison(
                cath_lab_df=cath_lab_df,
                cardiac_theatre_df=cardiac_theatre_df,
                output_dir=args.plot_dir
            )


if __name__ == "__main__":
    main()
