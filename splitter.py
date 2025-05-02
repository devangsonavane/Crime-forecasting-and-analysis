import csv
import os

def split_csv(input_file, output_dir, rows_per_file):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        with open(input_file, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            # Read the header separately so we can add it to every split file
            header = next(reader)
            
            file_count = 1
            rows = []
            for row in reader:
                rows.append(row)
                # When the collected rows reach the limit, write them to a new CSV file
                if len(rows) >= rows_per_file:
                    output_file = os.path.join(output_dir, f'split_{file_count}.csv')
                    with open(output_file, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(header)
                        writer.writerows(rows)
                    print(f'Written {len(rows)} rows to {output_file}')
                    file_count += 1
                    rows = []
                    
            # Write any remaining rows to a final CSV file
            if rows:
                output_file = os.path.join(output_dir, f'split_{file_count}.csv')
                with open(output_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    writer.writerows(rows)
                print(f'Written {len(rows)} rows to {output_file}')
    except Exception as e:
        print("Error occurred:", e)

if __name__ == '__main__':
    # Set your parameters here
    input_file = 'E:\Programs\SEM 6\BDA\Project\Crime_Data_from_2020_to_Present.csv'     # Replace with your CSV file path
    output_dir = 'E:\Programs\SEM 6\BDA\Project\Data'   # Replace with your desired output directory
    rows_per_file = 10000        # Replace with the number of rows per file

    print(f"Splitting '{input_file}' into parts of {rows_per_file} rows each, saving to '{output_dir}'...")
    split_csv(input_file, output_dir, rows_per_file)
