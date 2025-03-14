import argparse
from loguru import logger
import csv

from tools.csv_preprocessor import is_national_exam, normalize_question, read_csv


def extract_national_exam(csv_path: str):
    data = read_csv(csv_path)

    segments_list = []
    for row in data:
        question_id, question_text = row[0], row[1]
        segments = normalize_question(question_text)
        segments["id"] = question_id
        segments_list.append(segments)

    return segments_list


def extract_patient_exam(csv_path: str):
    data = read_csv(csv_path)

    segments_list = []
    for row in data:
        segments = {
            "id": row[0],
            "question": row[1],
        }
        segments_list.append(segments)

    return segments_list


def save_segments_list(segments_list: list, csv_path: str):
    keys = segments_list[0].keys()

    with open(csv_path, "w") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()

        for segments in segments_list:
            writer.writerow(segments)


def main(input_csv_path: str, output_csv_path: str):
    if is_national_exam(input_csv_path):
        logger.info(f"{input_csv_path}: 国試データです")
        segments_list = extract_national_exam(input_csv_path)
    else:
        logger.info(f"{input_csv_path}: 国試データではありません")
        segments_list = extract_patient_exam(input_csv_path)

    save_segments_list(segments_list, output_csv_path)
    logger.info(f"Saved to {output_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize a question")
    parser.add_argument("csv_path", help="Path to the input CSV file")
    parser.add_argument("output_csv_path", help="Path to the output CSV file")

    args = parser.parse_args()
    main(args.csv_path, args.output_csv_path)
