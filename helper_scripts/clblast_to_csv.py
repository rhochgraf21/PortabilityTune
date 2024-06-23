import json
import csv
import sys

def convert_json_to_csv(input_json_file, output_csv_file):
    # Parse the JSON file
    with open(input_json_file, 'r') as f:
        data = json.load(f)

    # Extract relevant fields and prepare CSV data
    csv_data = [] # key, val {kernel_family, results}

    for key, value in data.items():
        # print(value)
        device_name = value.get("clblast_device_name", "")
        device_core_clock = value.get("device_core_clock", "")
        label = f"{device_name} {device_core_clock}"
        family = value.get("kernel_family", "")

        if family == "xgemm":
            # continue

            for result in value.get("results_formatted", []):
                row = {
                    "m": value.get("arg_m", ""),
                    "n": value.get("arg_n", ""),
                    "k": value.get("arg_k", ""),
                    "batch": "1",
                    "label": label,
                    "mean_ms": result.get("time", ""),
                    "rate_s": 1000/float(result.get("time", "")) if result.get("time","") != "" else "",
                    "GEMMK": result.get("GEMMK", ""),
                    "KREG": result.get("KREG", ""),
                    "KWG": result.get("KWG", ""),
                    "KWI": result.get("KWI", ""),
                    "MDIMA": result.get("MDIMA", ""),
                    "MDIMC": result.get("MDIMC", ""),
                    "MWG": result.get("MWG", ""),
                    "NDIMB": result.get("NDIMB", ""),
                    "NDIMC": result.get("NDIMC", ""),
                    "NWG": result.get("NWG", ""),
                    "SA": result.get("SA", ""),
                    "SB": result.get("SB", ""),
                    "STRM": result.get("STRM", ""),
                    "STRN": result.get("STRN", ""),
                    "VWM": result.get("VWM", ""),
                    "VWN": result.get("VWN", ""),
                    "alpha": result.get("arg_alpha", ""),
                    "beta": result.get("arg_beta", ""),
                    "precision": result.get("precision", "")
                }
                csv_data.append(row)

    # Write to CSV file
    with open(output_csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
        writer.writeheader()
        for row in csv_data:
            writer.writerow(row)

# Example usage
convert_json_to_csv(sys.argv[1], sys.argv[2])