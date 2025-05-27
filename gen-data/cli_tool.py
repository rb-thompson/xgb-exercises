import argparse
from DataBuilder import DataBuilder

# CLI tool for generating random datasets using DataBuilder
#
# Examples: 
# python cli_tool.py --category Nature --rows 5 --cols 3 --missing --output dataset.csv
# python cli_tool.py --category Finance --rows 10 --output finance_data.json --format json

def main():
    parser = argparse.ArgumentParser(description="Generate random datasets.")
    parser.add_argument('--category', default='Nature', help='Data category')
    parser.add_argument('--rows', type=int, default=10, help='Number of rows')
    parser.add_argument('--cols', type=int, help='Number of columns')
    parser.add_argument('--missing', action='store_true', help='Include missing values')
    parser.add_argument('--output', default='dataset.csv', help='Output file')
    parser.add_argument('--format', default='csv', choices=['csv', 'json'], help='Output format')
    args = parser.parse_args()

    try:
        builder = DataBuilder(category=args.category, rows=args.rows, cols=args.cols, 
                             missing_values=args.missing)
        df = builder.generate_data()
        builder.export_data(df, args.output, format=args.format)
        print(f"Dataset successfully saved to {args.output} in {args.format} format")
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)

if __name__ == '__main__':
    main()