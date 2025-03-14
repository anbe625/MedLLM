import csv
from loguru import logger

def read_csv(csv_path: str):
    """
    指定されたCSVファイルを読み込み、ヘッダー行を除いたデータ行のリストを返します。

    手順:
      1. CSVファイルを開き、各行をリストとして読み込みます。
      2. 最初の行をヘッダーとして扱い、リストから削除します（削除したヘッダーはログ出力されます）。

    Parameters:
        csv_path (str): 読み込み対象のCSVファイルのパス

    Returns:
        list: CSVファイルの各行（ヘッダーは除く）のリスト
    """
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        data = [row for row in reader]

        # 最初の行はヘッダーなので削除し、ログ出力する
        logger.warning(f'Removed header: {data.pop(0)}')

    return data
